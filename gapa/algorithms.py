from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import pickle as pkl
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional, Sequence
import uuid

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from gapa.config import get_results_dir
from gapa.workflow import Algorithm, DataLoader, Monitor


class _CaptureSaveMixin:
    captured_result: Optional[Dict[str, Any]] = None

    def save(self, dataset, gene, best_metric, time_list, method, **kwargs):  # type: ignore[override]
        self.captured_result = {
            "dataset": dataset,
            "best_gene": deepcopy(gene),
            "best_metric": list(best_metric) if isinstance(best_metric, (list, tuple)) else [best_metric],
            "time_list": list(time_list),
            "method": method,
            "extra": dict(kwargs),
        }
        capture_path = getattr(self, "capture_path", None)
        if capture_path:
            try:
                Path(str(capture_path)).write_text(
                    json.dumps(_json_safe(self.captured_result), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
        return super().save(dataset, gene, best_metric, time_list, method, **kwargs)


def _datasets_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1] / "datasets"
    if repo_root.exists():
        return repo_root
    return Path(__file__).resolve().parent / "datasets"


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class _LegacyAlgorithm(Algorithm):
    supports_incremental = False

    def create_evaluator(self, data_loader: DataLoader) -> nn.Module:
        raise NotImplementedError("Legacy algorithms use run_full() and do not expose incremental evaluators.")

    def create_controller(self, data_loader: DataLoader, mode: str, device: torch.device):
        raise NotImplementedError("Legacy algorithms use run_full() and do not expose incremental controllers.")

    def build_distributed_evaluator(self, data_loader: DataLoader, device: torch.device, pop_size: int) -> nn.Module:
        raise NotImplementedError("Legacy algorithm does not expose distributed fitness setup.")

    @staticmethod
    def _seed_distributed_setup(seed: int = 1024) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _results_dir(name: str) -> Path:
        root = get_results_dir(Path(__file__).resolve().parents[1])
        path = root / "builtins" / name.lower()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _clone_loader(data_loader: DataLoader, **overrides: Any) -> Any:
        payload = dict(getattr(data_loader, "__dict__", {}))
        payload.update(overrides)
        return SimpleNamespace(**payload)

    @staticmethod
    def _observer(monitor: Monitor):
        def on_iter(gen: int, max_gen: int, payload: Any) -> None:
            monitor._generation = int(gen)
            if isinstance(payload, dict):
                metrics = {"generation": int(gen), "max_generation": int(max_gen), **payload}
                monitor._extra_history.append(metrics)
                if isinstance(payload.get("fitness"), (int, float)):
                    monitor._fitness_history.append(float(payload["fitness"]))
            elif isinstance(payload, (int, float)):
                monitor._fitness_history.append(float(payload))
        return on_iter

    @classmethod
    def _observer_spec(cls, workflow: Any, name: str):
        if getattr(workflow, "mode", None) in ("m", "mnm"):
            path = cls._results_dir(name) / f"observer_{uuid.uuid4().hex}.jsonl"
            return {"type": "jsonl", "path": str(path)}
        return cls._observer(workflow.monitor)

    @staticmethod
    def _capture_path(name: str) -> Path:
        return _LegacyAlgorithm._results_dir(name) / f"capture_{uuid.uuid4().hex}.json"

    @staticmethod
    def _load_capture(path: Optional[Path]) -> Optional[Dict[str, Any]]:
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    @classmethod
    def _consume_observer_spec(cls, monitor: Monitor, observer: Any) -> None:
        if not isinstance(observer, dict) or observer.get("type") != "jsonl" or not observer.get("path"):
            return
        path = Path(str(observer.get("path")))
        if not path.exists():
            return
        try:
            rows = []
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
        except Exception:
            return
        if not rows:
            return
        monitor._extra_history = []
        monitor._fitness_history = []
        last_gen = 0
        for row in rows:
            gen = int(row.get("generation") or 0)
            last_gen = max(last_gen, gen)
            metrics = row.get("metrics")
            if isinstance(metrics, dict):
                enriched = {"generation": gen, "max_generation": int(row.get("max_generation") or 0), **metrics}
                monitor._extra_history.append(enriched)
                fit = metrics.get("fitness")
                if isinstance(fit, (int, float)):
                    monitor._fitness_history.append(float(fit))
            else:
                fit = row.get("best_fitness")
                if isinstance(fit, (int, float)):
                    monitor._fitness_history.append(float(fit))
        monitor._generation = last_gen

    @staticmethod
    def _as_tensor(gene: Any) -> Optional[torch.Tensor]:
        if gene is None:
            return None
        if isinstance(gene, torch.Tensor):
            return gene.detach().cpu()
        try:
            return torch.as_tensor(gene)
        except Exception:
            return None

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except Exception:
            return None

    def _apply_capture(self, monitor: Monitor, capture: Optional[Dict[str, Any]], metric_names: Sequence[str]) -> None:
        if not capture:
            return
        best_metric = list(capture.get("best_metric") or [])
        metrics: Dict[str, Any] = {}
        for idx, name in enumerate(metric_names):
            if idx < len(best_metric):
                value = self._to_float(best_metric[idx])
                if value is not None:
                    metrics[name] = value
        extra = capture.get("extra") if isinstance(capture.get("extra"), dict) else {}
        for key, value in extra.items():
            if key.startswith("best") and isinstance(value, list) and value:
                tail = self._to_float(value[-1])
                if tail is not None:
                    metrics[key] = tail
        elapsed = None
        if len(best_metric) > len(metric_names):
            elapsed = self._to_float(best_metric[len(metric_names)])
        if elapsed is None:
            time_list = capture.get("time_list") if isinstance(capture.get("time_list"), list) else []
            if time_list:
                elapsed = self._to_float(time_list[-1])
        gene = self._as_tensor(capture.get("best_gene"))
        primary = None
        if metric_names:
            primary = metrics.get(metric_names[0])
        if primary is None and best_metric:
            primary = self._to_float(best_metric[0])
        monitor._best_fitness = primary
        monitor._best_solution = gene
        if elapsed is not None:
            monitor._local_timing = {
                "iter_seconds": elapsed,
                "iter_avg_ms": (elapsed / max(1, int(monitor._generation or 1))) * 1000.0,
                "throughput_ips": None,
            }
        monitor._remote_result = {"best_metrics": metrics}
        if primary is not None and not monitor._fitness_history:
            monitor._fitness_history.append(primary)

    @staticmethod
    def _load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    @classmethod
    def _comm_path(cls, name: str) -> Path:
        return cls._results_dir(name) / f"comm_{uuid.uuid4().hex}.json"

    @staticmethod
    def _attach_runtime_report(
        monitor: Monitor,
        *,
        evaluator: Optional[nn.Module] = None,
        comm_path: Optional[Path] = None,
    ) -> None:
        result = dict(monitor._remote_result or {})
        comm = None
        comm_detailed = None
        if evaluator is not None and hasattr(evaluator, "comm_stats"):
            try:
                payload = evaluator.comm_stats()
                if isinstance(payload, dict):
                    comm = payload
            except Exception:
                pass
        if evaluator is not None and hasattr(evaluator, "detailed_stats"):
            try:
                payload = evaluator.detailed_stats()
                if isinstance(payload, dict):
                    comm_detailed = payload
            except Exception:
                pass
        if comm is None:
            comm = _LegacyAlgorithm._load_json(comm_path)
        if comm:
            result["comm"] = comm
        if comm_detailed:
            result["comm_detailed"] = comm_detailed
            if not result.get("comm") and isinstance(comm_detailed.get("avg_ms"), (int, float)):
                result["comm"] = {
                    "type": comm_detailed.get("type"),
                    "avg_ms": comm_detailed.get("avg_ms"),
                    "total_ms": comm_detailed.get("total_comm_ms"),
                    "calls": comm_detailed.get("calls"),
                }
        monitor._remote_result = result

    @staticmethod
    def _maybe_wrap_evaluator(workflow: Any, evaluator: nn.Module) -> nn.Module:
        if getattr(workflow, "mode", None) == "mnm" and getattr(workflow, "servers", None):
            wrapped = workflow._wrap_for_mnm(evaluator)
            setattr(wrapped, "_is_distributed", True)
            workflow._evaluator = wrapped
            return wrapped
        return evaluator

    @staticmethod
    def _mnm_wrap_factory(workflow: Any):
        if getattr(workflow, "mode", None) == "mnm" and getattr(workflow, "servers", None):
            def _wrap(evaluator: nn.Module) -> nn.Module:
                wrapped = workflow._wrap_for_mnm(evaluator)
                setattr(wrapped, "_is_distributed", True)
                workflow._evaluator = wrapped
                return wrapped
            return _wrap
        return None


class CaptureSixDSTController(_CaptureSaveMixin):
    pass


_CAPTURE_CONTROLLER_SPECS = {
    "CaptureSixDSTControllerRuntime": ("gapa.algorithm.CND.SixDST", "SixDSTController"),
    "CaptureCutoffControllerRuntime": ("gapa.algorithm.CND.Cutoff", "CutoffController"),
    "CaptureTDEControllerRuntime": ("gapa.algorithm.CND.TDE", "TDEController"),
    "CaptureCGNControllerRuntime": ("gapa.algorithm.CDA.CGN", "CGNController"),
    "CaptureQAttackControllerRuntime": ("gapa.algorithm.CDA.QAttack", "QAttackController"),
    "CaptureCDAEDAControllerRuntime": ("gapa.algorithm.CDA.EDA", "EDAController"),
    "CaptureLPAGAControllerRuntime": ("gapa.algorithm.LPA.LPA_GA", "GAController"),
    "CaptureLPAEDAControllerRuntime": ("gapa.algorithm.LPA.EDA", "EDAController"),
    "CaptureNCAGAControllerRuntime": ("gapa.algorithm.NCA.NCA_GA", "NCA_GAController"),
}


def _capture_controller_class(name: str, module_name: str, base_name: str) -> type:
    existing = globals().get(name)
    if isinstance(existing, type):
        return existing
    module = __import__(module_name, fromlist=[base_name])
    base = getattr(module, base_name)
    cls = type(name, (_CaptureSaveMixin, base), {})
    cls.__module__ = __name__
    globals()[name] = cls
    return cls


def __getattr__(name: str):
    spec = _CAPTURE_CONTROLLER_SPECS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _capture_controller_class(name, spec[0], spec[1])


class SixDSTAlgorithm(_LegacyAlgorithm):
    def __init__(self, pop_size: int = 80, crossover_rate: float = 0.6, mutate_rate: float = 0.2, cutoff_tag: str = "popGreedy_cutoff_"):
        super().__init__()
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutate_rate = mutate_rate
        self.cutoff_tag = cutoff_tag
        self.side = "min"

    def run_full(self, workflow, steps: int) -> None:
        from gapa.algorithm.CND.SixDST import SixDST, SixDSTController, SixDSTEvaluator
        from gapa.utils.functions import set_seed
        CaptureController = _capture_controller_class("CaptureSixDSTControllerRuntime", "gapa.algorithm.CND.SixDST", "SixDSTController")

        set_seed(1024)
        loader = self._clone_loader(workflow.data_loader, mode=workflow.mode, world_size=workflow.world_size)
        observer = self._observer_spec(workflow, "sixdst")
        capture_path = self._capture_path("sixdst")
        comm_path = self._comm_path("sixdst") if workflow.mode == "m" else None
        controller = CaptureController(
            path=str(self._results_dir("sixdst")) + "/",
            pattern="write",
            cutoff_tag=self.cutoff_tag,
            data_loader=loader,
            loops=1,
            crossover_rate=self.crossover_rate,
            mutate_rate=self.mutate_rate,
            pop_size=self.pop_size,
            device=workflow.device,
        )
        controller.observer = observer
        controller.capture_path = str(capture_path)
        if comm_path is not None:
            controller.comm_path = str(comm_path)
        evaluator = SixDSTEvaluator(pop_size=self.pop_size, adj=loader.A, device=workflow.device)
        evaluator = self._maybe_wrap_evaluator(workflow, evaluator)
        SixDST(workflow.mode, int(steps), loader, controller, evaluator, workflow.world_size, verbose=workflow.verbose)
        self._consume_observer_spec(workflow.monitor, observer)
        self._apply_capture(workflow.monitor, controller.captured_result or self._load_capture(capture_path), ("PCG", "MCN"))
        self._attach_runtime_report(workflow.monitor, evaluator=evaluator, comm_path=comm_path)

    def build_distributed_evaluator(self, data_loader: DataLoader, device: torch.device, pop_size: int) -> nn.Module:
        from gapa.algorithm.CND.SixDST import SixDSTController, SixDSTEvaluator
        from gapa.utils.functions import set_seed

        loader = self._clone_loader(data_loader, mode="s", world_size=1)
        controller = SixDSTController(
            path=None,
            pattern=None,
            cutoff_tag=self.cutoff_tag,
            data_loader=loader,
            loops=1,
            crossover_rate=self.crossover_rate,
            mutate_rate=self.mutate_rate,
            pop_size=int(pop_size),
            device=device,
        )
        evaluator = SixDSTEvaluator(pop_size=int(pop_size), adj=loader.A, device=device)
        set_seed(1024)
        return controller.setup(data_loader=loader, evaluator=evaluator)


class CutOffAlgorithm(_LegacyAlgorithm):
    def __init__(self, pop_size: int = 40, crossover_rate: float = 0.7, mutate_rate: float = 0.2, cutoff_tag: str = "popGreedy_cutoff_"):
        super().__init__()
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutate_rate = mutate_rate
        self.cutoff_tag = cutoff_tag
        self.side = "min"

    def run_full(self, workflow, steps: int) -> None:
        from gapa.algorithm.CND.Cutoff import Cutoff, CutoffController, CutoffEvaluator
        CaptureController = _capture_controller_class("CaptureCutoffControllerRuntime", "gapa.algorithm.CND.Cutoff", "CutoffController")

        loader = self._clone_loader(workflow.data_loader, mode=workflow.mode, world_size=workflow.world_size)
        observer = self._observer_spec(workflow, "cutoff")
        capture_path = self._capture_path("cutoff")
        comm_path = self._comm_path("cutoff") if workflow.mode == "m" else None
        controller = CaptureController(
            path=str(self._results_dir("cutoff")) + "/",
            pattern="write",
            cutoff_tag=self.cutoff_tag,
            data_loader=loader,
            loops=1,
            crossover_rate=self.crossover_rate,
            mutate_rate=self.mutate_rate,
            pop_size=self.pop_size,
            device=workflow.device,
        )
        controller.observer = observer
        controller.capture_path = str(capture_path)
        if comm_path is not None:
            controller.comm_path = str(comm_path)
        evaluator = CutoffEvaluator(pop_size=self.pop_size, graph=loader.G, nodes=loader.nodes, device=workflow.device)
        evaluator = self._maybe_wrap_evaluator(workflow, evaluator)
        Cutoff(workflow.mode, int(steps), loader, controller, evaluator, workflow.world_size, verbose=workflow.verbose)
        self._consume_observer_spec(workflow.monitor, observer)
        self._apply_capture(workflow.monitor, controller.captured_result or self._load_capture(capture_path), ("PCG", "MCN"))
        self._attach_runtime_report(workflow.monitor, evaluator=evaluator, comm_path=comm_path)

    def build_distributed_evaluator(self, data_loader: DataLoader, device: torch.device, pop_size: int) -> nn.Module:
        from gapa.algorithm.CND.Cutoff import CutoffController, CutoffEvaluator

        self._seed_distributed_setup()
        loader = self._clone_loader(data_loader, mode="s", world_size=1)
        controller = CutoffController(
            path=None,
            pattern=None,
            cutoff_tag=self.cutoff_tag,
            data_loader=loader,
            loops=1,
            crossover_rate=self.crossover_rate,
            mutate_rate=self.mutate_rate,
            pop_size=int(pop_size),
            device=device,
        )
        evaluator = CutoffEvaluator(pop_size=int(pop_size), graph=loader.G, nodes=loader.nodes, device=device)
        return controller.setup(data_loader=loader, evaluator=evaluator)


class TDEAlgorithm(_LegacyAlgorithm):
    def __init__(self, pop_size: int = 40, crossover_rate: float = 0.7, mutate_rate: float = 0.2):
        super().__init__()
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutate_rate = mutate_rate
        self.side = "max"

    def run_full(self, workflow, steps: int) -> None:
        from gapa.algorithm.CND.TDE import TDE, TDEController, TDEEvaluator
        CaptureController = _capture_controller_class("CaptureTDEControllerRuntime", "gapa.algorithm.CND.TDE", "TDEController")

        loader = self._clone_loader(workflow.data_loader, mode=workflow.mode, world_size=workflow.world_size)
        observer = self._observer_spec(workflow, "tde")
        capture_path = self._capture_path("tde")
        comm_path = self._comm_path("tde") if workflow.mode == "m" else None
        controller = CaptureController(
            path=str(self._results_dir("tde")) + "/",
            pattern="write",
            data_loader=loader,
            loops=1,
            crossover_rate=self.crossover_rate,
            mutate_rate=self.mutate_rate,
            pop_size=self.pop_size,
            device=workflow.device,
        )
        controller.observer = observer
        controller.capture_path = str(capture_path)
        if comm_path is not None:
            controller.comm_path = str(comm_path)
        evaluator = TDEEvaluator(pop_size=self.pop_size, graph=loader.G, budget=loader.k, device=workflow.device)
        evaluator = self._maybe_wrap_evaluator(workflow, evaluator)
        TDE(workflow.mode, int(steps), loader, controller, evaluator, workflow.world_size, verbose=workflow.verbose)
        self._consume_observer_spec(workflow.monitor, observer)
        self._apply_capture(workflow.monitor, controller.captured_result or self._load_capture(capture_path), ("PCG", "MCN"))
        self._attach_runtime_report(workflow.monitor, evaluator=evaluator, comm_path=comm_path)

    def build_distributed_evaluator(self, data_loader: DataLoader, device: torch.device, pop_size: int) -> nn.Module:
        from gapa.algorithm.CND.TDE import TDEController, TDEEvaluator

        self._seed_distributed_setup()
        loader = self._clone_loader(data_loader, mode="s", world_size=1)
        controller = TDEController(
            path=None,
            pattern=None,
            data_loader=loader,
            loops=1,
            crossover_rate=self.crossover_rate,
            mutate_rate=self.mutate_rate,
            pop_size=int(pop_size),
            device=device,
        )
        evaluator = TDEEvaluator(pop_size=int(pop_size), graph=loader.G, budget=loader.k, device=device)
        return controller.setup(data_loader=loader, evaluator=evaluator)


class CGNAlgorithm(_LegacyAlgorithm):
    def __init__(self, pop_size: int = 40, crossover_rate: float = 0.7, mutate_rate: float = 0.01):
        super().__init__()
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutate_rate = mutate_rate
        self.side = "min"

    def run_full(self, workflow, steps: int) -> None:
        from gapa.algorithm.CDA.CGN import CGN, CGNController, CGNEvaluator
        CaptureController = _capture_controller_class("CaptureCGNControllerRuntime", "gapa.algorithm.CDA.CGN", "CGNController")

        loader = self._clone_loader(workflow.data_loader, mode=workflow.mode, world_size=workflow.world_size)
        observer = self._observer_spec(workflow, "cgn")
        capture_path = self._capture_path("cgn")
        comm_path = self._comm_path("cgn") if workflow.mode == "m" else None
        controller = CaptureController(
            path=str(self._results_dir("cgn")) + "/",
            pattern="write",
            data_loader=loader,
            loops=1,
            crossover_rate=self.crossover_rate,
            mutate_rate=self.mutate_rate,
            pop_size=self.pop_size,
            device=workflow.device,
        )
        controller.observer = observer
        controller.capture_path = str(capture_path)
        if comm_path is not None:
            controller.comm_path = str(comm_path)
        evaluator = CGNEvaluator(pop_size=self.pop_size, graph=loader.G.copy(), device=workflow.device)
        evaluator = self._maybe_wrap_evaluator(workflow, evaluator)
        CGN(workflow.mode, int(steps), loader, controller, evaluator, workflow.world_size, verbose=workflow.verbose)
        self._consume_observer_spec(workflow.monitor, observer)
        self._apply_capture(workflow.monitor, controller.captured_result or self._load_capture(capture_path), ("Q", "NMI"))
        self._attach_runtime_report(workflow.monitor, evaluator=evaluator, comm_path=comm_path)

    def build_distributed_evaluator(self, data_loader: DataLoader, device: torch.device, pop_size: int) -> nn.Module:
        from gapa.algorithm.CDA.CGN import CGNController, CGNEvaluator

        self._seed_distributed_setup()
        loader = self._clone_loader(data_loader, mode="s", world_size=1)
        controller = CGNController(
            path=None,
            pattern=None,
            data_loader=loader,
            loops=1,
            crossover_rate=self.crossover_rate,
            mutate_rate=self.mutate_rate,
            pop_size=int(pop_size),
            device=device,
        )
        evaluator = CGNEvaluator(pop_size=int(pop_size), graph=loader.G.copy(), device=device)
        return controller.setup(data_loader=loader, evaluator=evaluator)


class QAttackAlgorithm(_LegacyAlgorithm):
    def __init__(self, pop_size: int = 40, crossover_rate: float = 0.8, mutate_rate: float = 0.1):
        super().__init__()
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutate_rate = mutate_rate
        self.side = "max"

    def run_full(self, workflow, steps: int) -> None:
        from gapa.algorithm.CDA.QAttack import QAttack, QAttackController, QAttackEvaluator
        CaptureController = _capture_controller_class("CaptureQAttackControllerRuntime", "gapa.algorithm.CDA.QAttack", "QAttackController")

        loader = self._clone_loader(workflow.data_loader, mode=workflow.mode, world_size=workflow.world_size)
        observer = self._observer_spec(workflow, "qattack")
        capture_path = self._capture_path("qattack")
        comm_path = self._comm_path("qattack") if workflow.mode == "m" else None
        controller = CaptureController(
            path=str(self._results_dir("qattack")) + "/",
            pattern="write",
            data_loader=loader,
            loops=1,
            crossover_rate=self.crossover_rate,
            mutate_rate=self.mutate_rate,
            pop_size=self.pop_size,
            device=workflow.device,
        )
        controller.observer = observer
        controller.capture_path = str(capture_path)
        if comm_path is not None:
            controller.comm_path = str(comm_path)
        evaluator = QAttackEvaluator(pop_size=self.pop_size, graph=loader.G.copy(), device=workflow.device)
        evaluator = self._maybe_wrap_evaluator(workflow, evaluator)
        QAttack(workflow.mode, int(steps), loader, controller, evaluator, workflow.world_size, verbose=workflow.verbose)
        self._consume_observer_spec(workflow.monitor, observer)
        self._apply_capture(workflow.monitor, controller.captured_result or self._load_capture(capture_path), ("Q", "NMI"))
        self._attach_runtime_report(workflow.monitor, evaluator=evaluator, comm_path=comm_path)

    def build_distributed_evaluator(self, data_loader: DataLoader, device: torch.device, pop_size: int) -> nn.Module:
        from gapa.algorithm.CDA.QAttack import QAttackController, QAttackEvaluator

        self._seed_distributed_setup()
        loader = self._clone_loader(data_loader, mode="s", world_size=1)
        controller = QAttackController(
            path=None,
            pattern=None,
            data_loader=loader,
            loops=1,
            crossover_rate=self.crossover_rate,
            mutate_rate=self.mutate_rate,
            pop_size=int(pop_size),
            device=device,
        )
        evaluator = QAttackEvaluator(pop_size=int(pop_size), graph=loader.G.copy(), device=device)
        return controller.setup(data_loader=loader, evaluator=evaluator)


class CDAEDAAlgorithm(_LegacyAlgorithm):
    def __init__(self, pop_size: int = 40, crossover_rate: float = 0.6, mutate_rate: float = 0.2):
        super().__init__()
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutate_rate = mutate_rate
        self.side = "max"

    def run_full(self, workflow, steps: int) -> None:
        from gapa.algorithm.CDA.EDA import EDA, EDAController, EDAEvaluator
        CaptureController = _capture_controller_class("CaptureCDAEDAControllerRuntime", "gapa.algorithm.CDA.EDA", "EDAController")

        loader = self._clone_loader(workflow.data_loader, mode=workflow.mode, world_size=workflow.world_size)
        observer = self._observer_spec(workflow, "cda_eda")
        capture_path = self._capture_path("cda_eda")
        comm_path = self._comm_path("cda_eda") if workflow.mode == "m" else None
        controller = CaptureController(
            path=str(self._results_dir("cda_eda")) + "/",
            pattern="write",
            data_loader=loader,
            loops=1,
            crossover_rate=self.crossover_rate,
            mutate_rate=self.mutate_rate,
            pop_size=self.pop_size,
            device=workflow.device,
        )
        controller.observer = observer
        controller.capture_path = str(capture_path)
        if comm_path is not None:
            controller.comm_path = str(comm_path)
        evaluator = EDAEvaluator(pop_size=self.pop_size, graph=loader.G.copy(), adj=loader.A, nodes_num=loader.nodes_num, device=workflow.device)
        evaluator = self._maybe_wrap_evaluator(workflow, evaluator)
        EDA(workflow.mode, int(steps), loader, controller, evaluator, workflow.world_size, verbose=workflow.verbose)
        self._consume_observer_spec(workflow.monitor, observer)
        self._apply_capture(workflow.monitor, controller.captured_result or self._load_capture(capture_path), ("Q", "NMI"))
        self._attach_runtime_report(workflow.monitor, evaluator=evaluator, comm_path=comm_path)

    def build_distributed_evaluator(self, data_loader: DataLoader, device: torch.device, pop_size: int) -> nn.Module:
        from gapa.algorithm.CDA.EDA import EDAController, EDAEvaluator

        self._seed_distributed_setup()
        loader = self._clone_loader(data_loader, mode="s", world_size=1)
        controller = EDAController(
            path=None,
            pattern=None,
            data_loader=loader,
            loops=1,
            crossover_rate=self.crossover_rate,
            mutate_rate=self.mutate_rate,
            pop_size=int(pop_size),
            device=device,
        )
        evaluator = EDAEvaluator(pop_size=int(pop_size), graph=loader.G.copy(), adj=loader.A, nodes_num=loader.nodes_num, device=device)
        return controller.setup(data_loader=loader, evaluator=evaluator)


class LPAGAAlgorithm(_LegacyAlgorithm):
    def __init__(self, pop_size: int = 40, crossover_rate: float = 0.7, mutate_rate: float = 0.1, attack_rate: float = 0.1):
        super().__init__()
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutate_rate = mutate_rate
        self.attack_rate = attack_rate
        self.side = "max"

    def run_full(self, workflow, steps: int) -> None:
        from gapa.algorithm.LPA.LPA_GA import GAEvaluator, GAController, LPA_GA
        CaptureController = _capture_controller_class("CaptureLPAGAControllerRuntime", "gapa.algorithm.LPA.LPA_GA", "GAController")

        edges = torch.tensor(list(workflow.data_loader.G.edges), device=workflow.device)
        observer = self._observer_spec(workflow, "lpa_ga")
        capture_path = self._capture_path("lpa_ga")
        comm_path = self._comm_path("lpa_ga") if workflow.mode == "m" else None
        loader = self._clone_loader(
            workflow.data_loader,
            k=float(self.attack_rate),
            edges=edges,
            edges_num=int(len(edges)),
            mode=workflow.mode,
            world_size=workflow.world_size,
        )
        controller = CaptureController(
            path=str(self._results_dir("lpa_ga")) + "/",
            pattern="write",
            data_loader=loader,
            loops=1,
            crossover_rate=self.crossover_rate,
            mutate_rate=self.mutate_rate,
            pop_size=self.pop_size,
            device=workflow.device,
        )
        controller.observer = observer
        controller.capture_path = str(capture_path)
        if comm_path is not None:
            controller.comm_path = str(comm_path)
        evaluator = GAEvaluator(pop_size=self.pop_size, graph=loader.G, ratio=0, device=workflow.device)
        evaluator = self._maybe_wrap_evaluator(workflow, evaluator)
        LPA_GA(workflow.mode, int(steps), loader, controller, evaluator, workflow.world_size, verbose=workflow.verbose)
        self._consume_observer_spec(workflow.monitor, observer)
        self._apply_capture(workflow.monitor, controller.captured_result or self._load_capture(capture_path), ("Pre", "AUC"))
        self._attach_runtime_report(workflow.monitor, evaluator=evaluator, comm_path=comm_path)

    def build_distributed_evaluator(self, data_loader: DataLoader, device: torch.device, pop_size: int) -> nn.Module:
        from gapa.algorithm.LPA.LPA_GA import GAEvaluator, GAController

        self._seed_distributed_setup()
        edges = torch.tensor(list(data_loader.G.edges), device=device)
        loader = self._clone_loader(
            data_loader,
            k=float(self.attack_rate),
            edges=edges,
            edges_num=int(len(edges)),
            mode="s",
            world_size=1,
        )
        controller = GAController(
            path=None,
            pattern=None,
            data_loader=loader,
            loops=1,
            crossover_rate=self.crossover_rate,
            mutate_rate=self.mutate_rate,
            pop_size=int(pop_size),
            device=device,
        )
        evaluator = GAEvaluator(pop_size=int(pop_size), graph=loader.G, ratio=0, device=device)
        return controller.setup(data_loader=loader, evaluator=evaluator)


class LPAEDAAlgorithm(_LegacyAlgorithm):
    def __init__(self, pop_size: int = 40, mutate_rate: float = 0.1, attack_rate: float = 0.1):
        super().__init__()
        self.pop_size = pop_size
        self.mutate_rate = mutate_rate
        self.attack_rate = attack_rate
        self.side = "max"

    def run_full(self, workflow, steps: int) -> None:
        from gapa.algorithm.LPA.EDA import EDA, EDAController, EDAEvaluator
        CaptureController = _capture_controller_class("CaptureLPAEDAControllerRuntime", "gapa.algorithm.LPA.EDA", "EDAController")

        edges = torch.tensor(list(workflow.data_loader.G.edges), device=workflow.device)
        observer = self._observer_spec(workflow, "lpa_eda")
        capture_path = self._capture_path("lpa_eda")
        comm_path = self._comm_path("lpa_eda") if workflow.mode == "m" else None
        loader = self._clone_loader(
            workflow.data_loader,
            k=float(self.attack_rate),
            edges=edges,
            edges_num=int(len(edges)),
            mode=workflow.mode,
            world_size=workflow.world_size,
        )
        controller = CaptureController(
            path=str(self._results_dir("lpa_eda")) + "/",
            pattern="write",
            data_loader=loader,
            loops=1,
            mutate_rate=self.mutate_rate,
            pop_size=self.pop_size,
            num_eda_pop=self.pop_size,
            device=workflow.device,
        )
        controller.observer = observer
        controller.capture_path = str(capture_path)
        if comm_path is not None:
            controller.comm_path = str(comm_path)
        evaluator = EDAEvaluator(pop_size=self.pop_size, graph=loader.G, ratio=0, device=workflow.device)
        evaluator = self._maybe_wrap_evaluator(workflow, evaluator)
        EDA(workflow.mode, int(steps), loader, controller, evaluator, workflow.world_size, verbose=workflow.verbose)
        self._consume_observer_spec(workflow.monitor, observer)
        self._apply_capture(workflow.monitor, controller.captured_result or self._load_capture(capture_path), ("Pre", "AUC"))
        self._attach_runtime_report(workflow.monitor, evaluator=evaluator, comm_path=comm_path)

    def build_distributed_evaluator(self, data_loader: DataLoader, device: torch.device, pop_size: int) -> nn.Module:
        from gapa.algorithm.LPA.EDA import EDAController, EDAEvaluator

        self._seed_distributed_setup()
        edges = torch.tensor(list(data_loader.G.edges), device=device)
        loader = self._clone_loader(
            data_loader,
            k=float(self.attack_rate),
            edges=edges,
            edges_num=int(len(edges)),
            mode="s",
            world_size=1,
        )
        controller = EDAController(
            path=None,
            pattern=None,
            data_loader=loader,
            loops=1,
            mutate_rate=self.mutate_rate,
            pop_size=int(pop_size),
            num_eda_pop=int(pop_size),
            device=device,
        )
        evaluator = EDAEvaluator(pop_size=int(pop_size), graph=loader.G, ratio=0, device=device)
        return controller.setup(data_loader=loader, evaluator=evaluator)


class NCAGAAlgorithm(_LegacyAlgorithm):
    def __init__(self, pop_size: int = 20, crossover_rate: float = 0.7, mutate_rate: float = 0.3, attack_rate: float = 0.025):
        super().__init__()
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutate_rate = mutate_rate
        self.attack_rate = attack_rate
        self.side = "max"

    @staticmethod
    def _pickle_load(pkl_file):
        if sys.version_info > (3, 0):
            return pkl.load(pkl_file, encoding="latin1")
        return pkl.load(pkl_file)

    @staticmethod
    def _parse_index_file(filename: Path) -> list[int]:
        return [int(line.strip()) for line in filename.read_text(encoding="utf-8").splitlines() if line.strip()]

    @staticmethod
    def _preprocess_features(features) -> np.ndarray:
        rowsum = np.asarray(features.sum(1)).reshape(-1)
        r_inv = np.power(rowsum, -1, where=rowsum != 0)
        r_inv[~np.isfinite(r_inv)] = 0.0
        return np.asarray(features.multiply(r_inv[:, None]).todense(), dtype=np.float32)

    @classmethod
    def _load_planetoid_payload(cls, dataset_name: str, device: torch.device) -> Optional[Any]:
        source_name = "cora_v2" if dataset_name == "cora" else dataset_name
        root = Path(__file__).resolve().parents[1] / "dataset" / dataset_name
        required = [root / f"ind.{source_name}.{name}" for name in ("x", "y", "tx", "ty", "allx", "ally", "graph")]
        index_path = root / f"ind.{source_name}.test.index"
        if not all(path.exists() for path in required) or not index_path.exists():
            return None

        try:
            from scipy import sparse as sp  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"NCA legacy Planetoid fallback requires scipy. {exc}") from exc

        objects = []
        for path in required:
            with path.open("rb") as handle:
                objects.append(cls._pickle_load(handle))
        x, y, tx, ty, allx, ally, graph = tuple(objects)

        test_idx_reorder = np.array(cls._parse_index_file(index_path), dtype=np.int64)
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_name == "citeseer":
            test_idx_range_full = range(int(min(test_idx_reorder)), int(max(test_idx_reorder)) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        feats = cls._preprocess_features(features)

        onehot_labels = np.vstack((ally, ty))
        onehot_labels[test_idx_reorder, :] = onehot_labels[test_idx_range, :]
        labels = np.argmax(onehot_labels, axis=1).astype(np.int64)

        train_index = np.arange(len(y), dtype=np.int64)
        val_index = np.arange(len(y), len(y) + 500, dtype=np.int64)
        test_index = test_idx_range.astype(np.int64)

        edges = [(_x, _y) for _x, _y_list in graph.items() for _y in _y_list if _x != _y]
        idx = torch.tensor(list(map(list, zip(*edges))), dtype=torch.long, device=device)
        value = torch.ones(idx.shape[1], dtype=torch.float32, device=device)
        ori_adj = torch.sparse_coo_tensor(indices=idx, values=value, dtype=torch.float32, device=device)
        adj = (ori_adj + ori_adj.t()).coalesce().bool().float()
        graph_nx = nx.Graph(adj.cpu().numpy())

        return SimpleNamespace(
            dataset=dataset_name,
            name=dataset_name,
            G=graph_nx,
            adj=adj.to_sparse_coo(),
            feats=torch.tensor(feats, dtype=torch.float32, device=device),
            labels=torch.tensor(labels, dtype=torch.long, device=device),
            train_index=torch.tensor(train_index, dtype=torch.long, device=device),
            val_index=torch.tensor(val_index, dtype=torch.long, device=device),
            test_index=torch.tensor(test_index, dtype=torch.long, device=device),
            num_nodes=int(feats.shape[0]),
            num_feats=int(feats.shape[1]),
            num_classes=int(np.max(labels) + 1),
            num_edge=int(len(edges)),
            k=None,
        )

    @staticmethod
    def _load_pt_payload(dataset_name: str, device: torch.device) -> Optional[Any]:
        root = Path(__file__).resolve().parents[1] / "dataset" / dataset_name
        required = [
            root / "adj.pt",
            root / "features.pt",
            root / "labels.pt",
            root / "train_index.pt",
            root / "val_index.pt",
            root / "test_index.pt",
        ]
        if not all(path.exists() for path in required):
            return None

        adj = torch.load(root / "adj.pt", map_location=device)
        feats = torch.load(root / "features.pt", map_location=device)
        labels = torch.load(root / "labels.pt", map_location=device)
        train_index = torch.load(root / "train_index.pt", map_location=device)
        val_index = torch.load(root / "val_index.pt", map_location=device)
        test_index = torch.load(root / "test_index.pt", map_location=device)
        if getattr(adj, "is_sparse", False):
            adj = adj.coalesce()
            graph_nx = nx.Graph(adj.to_dense().cpu().numpy())
        else:
            graph_nx = nx.Graph(adj.cpu().numpy())
            adj = adj.to_sparse_coo()

        num_nodes = int(feats.shape[0])
        num_edge = int(graph_nx.number_of_edges())
        return SimpleNamespace(
            dataset=dataset_name,
            name=dataset_name,
            G=graph_nx,
            adj=adj,
            feats=feats.float(),
            labels=labels.long(),
            train_index=train_index.long(),
            val_index=val_index.long(),
            test_index=test_index.long(),
            num_nodes=num_nodes,
            num_feats=int(feats.shape[1]),
            num_classes=int(torch.max(labels).item() + 1),
            num_edge=num_edge,
            k=None,
        )

    @staticmethod
    def _load_nca_payload(public_loader: DataLoader, device: torch.device) -> Any:
        data_path = Path(str(public_loader.meta.get("path") or ""))
        if not data_path.is_absolute():
            data_path = _datasets_root() / data_path
        if not data_path.exists():
            raise FileNotFoundError(f"NCA dataset file missing: {data_path}")
        raw = np.load(data_path, allow_pickle=True)
        required = {"edges", "node_features", "node_labels", "train_masks", "val_masks", "test_masks"}
        files = set(raw.files)
        if not required.issubset(files):
            csr_keys = {"adj_data", "adj_indices", "adj_indptr", "adj_shape", "attr_data", "attr_indices", "attr_indptr", "attr_shape", "labels"}
            key = public_loader.name
            if key.endswith("_filtered"):
                key = key.replace("_filtered", "")
            if csr_keys.issubset(files):
                payload = NCAGAAlgorithm._load_planetoid_payload(key, device)
                if payload is None:
                    payload = NCAGAAlgorithm._load_pt_payload(key, device)
                if payload is not None:
                    return payload
            raise RuntimeError(
                f"NCA public workflow expects filtered NPZ keys {sorted(required)} or a legacy Planetoid dataset with companion raw files. got={sorted(raw.files)}"
            )
        edges = raw["edges"]
        feats = raw["node_features"]
        labels = raw["node_labels"]
        train_index = np.where(np.array(raw["train_masks"][0]) == 1)[0]
        val_index = np.where(np.array(raw["val_masks"][0]) == 1)[0]
        test_index = np.where(np.array(raw["test_masks"][0]) == 1)[0]
        adj = torch.zeros((len(feats), len(feats)), dtype=torch.float32, device=device)
        for src, dst in edges:
            adj[int(src), int(dst)] = 1
            adj[int(dst), int(src)] = 1
        graph = nx.Graph(adj.cpu().numpy())
        key = public_loader.name
        if key.endswith("_filtered"):
            key = key.replace("_filtered", "")
        return SimpleNamespace(
            dataset=key,
            name=public_loader.name,
            G=graph,
            adj=adj.to_sparse_coo(),
            feats=torch.tensor(feats, dtype=torch.float32, device=device),
            labels=torch.tensor(labels, dtype=torch.long, device=device),
            train_index=torch.tensor(train_index, dtype=torch.long, device=device),
            val_index=torch.tensor(val_index, dtype=torch.long, device=device),
            test_index=torch.tensor(test_index, dtype=torch.long, device=device),
            num_nodes=int(len(feats)),
            num_feats=int(feats.shape[1]),
            num_classes=int(np.max(labels) + 1),
            num_edge=int(len(edges)),
            k=max(1, int(float(public_loader.meta.get("edges") or len(edges)) * self.attack_rate)) if False else None,
        )

    def run_full(self, workflow, steps: int) -> None:
        from gapa.algorithm.NCA.NCA_GA import NCA_GA, NCA_GAController, NCA_GAEvaluator
        from gapa.DeepLearning.Classifier import Classifier, load_set
        CaptureController = _capture_controller_class("CaptureNCAGAControllerRuntime", "gapa.algorithm.NCA.NCA_GA", "NCA_GAController")

        loader = self._load_nca_payload(workflow.data_loader, workflow.device)
        loader.k = max(1, int(self.attack_rate * loader.num_edge))
        observer = self._observer_spec(workflow, "nca_ga")
        capture_path = self._capture_path("nca_ga")
        comm_path = self._comm_path("nca_ga") if workflow.mode == "m" else None
        load_set(loader.dataset, "gcn", num_nodes=loader.num_nodes, num_edge=loader.num_edge)
        classifier = Classifier(model_name="gcn", input_dim=loader.num_feats, output_dim=loader.num_classes, device=workflow.device)
        classifier.initialize()
        classifier.fit(loader.feats, loader.adj, loader.labels, loader.train_index, loader.val_index, verbose=False)
        controller = CaptureController(
            path=str(self._results_dir("nca_ga")) + "/",
            pattern="write",
            data_loader=loader,
            classifier=classifier,
            loops=1,
            crossover_rate=self.crossover_rate,
            mutate_rate=self.mutate_rate,
            pop_size=self.pop_size,
            device=workflow.device,
        )
        controller.observer = observer
        controller.capture_path = str(capture_path)
        if comm_path is not None:
            controller.comm_path = str(comm_path)
        evaluator = NCA_GAEvaluator(
            classifier=classifier,
            feats=loader.feats,
            adj=loader.adj,
            test_index=loader.test_index,
            labels=loader.labels,
            pop_size=self.pop_size,
            device=workflow.device,
        )
        evaluator = self._maybe_wrap_evaluator(workflow, evaluator)
        NCA_GA(workflow.mode, int(steps), loader, controller, evaluator, workflow.world_size, verbose=workflow.verbose)
        self._consume_observer_spec(workflow.monitor, observer)
        self._apply_capture(workflow.monitor, controller.captured_result or self._load_capture(capture_path), ("Acc", "ASR"))
        self._attach_runtime_report(workflow.monitor, evaluator=evaluator, comm_path=comm_path)

    def build_distributed_evaluator(self, data_loader: DataLoader, device: torch.device, pop_size: int) -> nn.Module:
        from gapa.algorithm.NCA.NCA_GA import NCA_GAController, NCA_GAEvaluator
        from gapa.DeepLearning.Classifier import Classifier, load_set

        self._seed_distributed_setup()
        loader = self._load_nca_payload(data_loader, device)
        loader.k = max(1, int(self.attack_rate * loader.num_edge))
        load_set(loader.dataset, "gcn", num_nodes=loader.num_nodes, num_edge=loader.num_edge)
        classifier = Classifier(model_name="gcn", input_dim=loader.num_feats, output_dim=loader.num_classes, device=device)
        classifier.initialize()
        classifier.fit(loader.feats, loader.adj, loader.labels, loader.train_index, loader.val_index, verbose=False)
        controller = NCA_GAController(
            path=None,
            pattern=None,
            data_loader=loader,
            classifier=classifier,
            loops=1,
            crossover_rate=self.crossover_rate,
            mutate_rate=self.mutate_rate,
            pop_size=int(pop_size),
            device=device,
        )
        evaluator = NCA_GAEvaluator(
            classifier=classifier,
            feats=loader.feats,
            adj=loader.adj,
            test_index=loader.test_index,
            labels=loader.labels,
            pop_size=int(pop_size),
            device=device,
        )
        return controller.setup(data_loader=loader, evaluator=evaluator)


class GANIAlgorithm(_LegacyAlgorithm):
    def __init__(self, pop_size: int = 20, crossover_rate: float = 0.7, mutate_rate: float = 0.3, attack_rate: float = 0.05, homophily_ratio: float = 0.7):
        super().__init__()
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutate_rate = mutate_rate
        self.attack_rate = attack_rate
        self.homophily_ratio = homophily_ratio
        self.side = "min"

    def run_full(self, workflow, steps: int) -> None:
        from deeprobust.graph.defense import GCN as DRGCN
        from gapa.algorithm.NCA import SGA as sga_module
        from gapa.algorithm.NCA.SGA import SGA, SGAController
        from gapa.DeepLearning.Classifier import Classifier, load_set

        loader = NCAGAAlgorithm._load_nca_payload(workflow.data_loader, workflow.device)
        loader.k = max(1, int(self.attack_rate * loader.num_nodes))
        load_set(loader.dataset, "gcn", num_nodes=loader.num_nodes, num_edge=loader.num_edge)
        classifier = Classifier(model_name="gcn", input_dim=loader.num_feats, output_dim=loader.num_classes, device=workflow.device)
        classifier.initialize()
        classifier.fit(loader.feats, loader.adj, loader.labels, loader.train_index, loader.val_index, verbose=False)
        surrogate = DRGCN(
            nfeat=loader.num_feats,
            nclass=loader.num_classes,
            nhid=16,
            dropout=0.5,
            with_relu=False,
            with_bias=True,
            device=workflow.device,
        ).to(torch.device(workflow.device))
        surrogate.fit(loader.feats, loader.adj, loader.labels, loader.train_index, loader.val_index, patience=30)

        controller = SGAController(
            path=str(self._results_dir("gani")) + "/",
            pattern="write",
            data_loader=loader,
            classifier=classifier,
            loops=1,
            crossover_rate=self.crossover_rate,
            mutate_rate=self.mutate_rate,
            pop_size=self.pop_size,
            device=workflow.device,
        )
        controller.observer = self._observer_spec(workflow, "gani")
        comm_path = self._comm_path("gani") if workflow.mode == "m" else None
        if comm_path is not None:
            controller.comm_path = str(comm_path)
        capture: Dict[str, Any] = {}
        original_save = sga_module.SGAAlgorithm.save

        def patched_save(algo_self, controller_obj, dataset, gene, best_metric, time_list, method, **kwargs):
            capture.update(
                {
                    "dataset": dataset,
                    "best_gene": deepcopy(gene),
                    "best_metric": list(best_metric) if isinstance(best_metric, (list, tuple)) else [best_metric],
                    "time_list": list(time_list),
                    "method": method,
                    "extra": dict(kwargs),
                }
            )
            return original_save(algo_self, controller_obj, dataset, gene, best_metric, time_list, method, **kwargs)

        sga_module.SGAAlgorithm.save = patched_save
        try:
            SGA(
                workflow.mode,
                int(steps),
                loader,
                controller,
                surrogate=surrogate,
                classifier=classifier,
                homophily_ratio=self.homophily_ratio,
                world_size=workflow.world_size,
                wrap_evaluator=self._mnm_wrap_factory(workflow),
                verbose=workflow.verbose,
            )
        finally:
            sga_module.SGAAlgorithm.save = original_save
        self._apply_capture(workflow.monitor, capture or None, ("Acc", "ASR"))
        self._attach_runtime_report(workflow.monitor, comm_path=comm_path)


__all__ = [
    "CDAEDAAlgorithm",
    "CGNAlgorithm",
    "CutOffAlgorithm",
    "GANIAlgorithm",
    "LPAEDAAlgorithm",
    "LPAGAAlgorithm",
    "NCAGAAlgorithm",
    "QAttackAlgorithm",
    "SixDSTAlgorithm",
    "TDEAlgorithm",
]
