from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional, Sequence

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
        return super().save(dataset, gene, best_metric, time_list, method, **kwargs)


def _datasets_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1] / "datasets"
    if repo_root.exists():
        return repo_root
    return Path(__file__).resolve().parent / "datasets"


class _LegacyAlgorithm(Algorithm):
    supports_incremental = False

    def create_evaluator(self, data_loader: DataLoader) -> nn.Module:
        raise NotImplementedError("Legacy algorithms use run_full() and do not expose incremental evaluators.")

    def create_controller(self, data_loader: DataLoader, mode: str, device: torch.device):
        raise NotImplementedError("Legacy algorithms use run_full() and do not expose incremental controllers.")

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

        class CaptureController(_CaptureSaveMixin, SixDSTController):
            pass

        set_seed(1024)
        loader = self._clone_loader(workflow.data_loader, mode=workflow.mode, world_size=workflow.world_size)
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
        controller.observer = self._observer(workflow.monitor)
        evaluator = SixDSTEvaluator(pop_size=self.pop_size, adj=loader.A, device=workflow.device)
        evaluator = self._maybe_wrap_evaluator(workflow, evaluator)
        SixDST(workflow.mode, int(steps), loader, controller, evaluator, workflow.world_size, verbose=workflow.verbose)
        self._apply_capture(workflow.monitor, controller.captured_result, ("PCG", "MCN"))


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

        class CaptureController(_CaptureSaveMixin, CutoffController):
            pass

        loader = self._clone_loader(workflow.data_loader, mode=workflow.mode, world_size=workflow.world_size)
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
        controller.observer = self._observer(workflow.monitor)
        evaluator = CutoffEvaluator(pop_size=self.pop_size, graph=loader.G, nodes=loader.nodes, device=workflow.device)
        evaluator = self._maybe_wrap_evaluator(workflow, evaluator)
        Cutoff(workflow.mode, int(steps), loader, controller, evaluator, workflow.world_size, verbose=workflow.verbose)
        self._apply_capture(workflow.monitor, controller.captured_result, ("PCG", "MCN"))


class TDEAlgorithm(_LegacyAlgorithm):
    def __init__(self, pop_size: int = 40, crossover_rate: float = 0.7, mutate_rate: float = 0.2):
        super().__init__()
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutate_rate = mutate_rate
        self.side = "max"

    def run_full(self, workflow, steps: int) -> None:
        from gapa.algorithm.CND.TDE import TDE, TDEController, TDEEvaluator

        class CaptureController(_CaptureSaveMixin, TDEController):
            pass

        loader = self._clone_loader(workflow.data_loader, mode=workflow.mode, world_size=workflow.world_size)
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
        controller.observer = self._observer(workflow.monitor)
        evaluator = TDEEvaluator(pop_size=self.pop_size, graph=loader.G, budget=loader.k, device=workflow.device)
        evaluator = self._maybe_wrap_evaluator(workflow, evaluator)
        TDE(workflow.mode, int(steps), loader, controller, evaluator, workflow.world_size, verbose=workflow.verbose)
        self._apply_capture(workflow.monitor, controller.captured_result, ("PCG", "MCN"))


class CGNAlgorithm(_LegacyAlgorithm):
    def __init__(self, pop_size: int = 40, crossover_rate: float = 0.7, mutate_rate: float = 0.01):
        super().__init__()
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutate_rate = mutate_rate
        self.side = "min"

    def run_full(self, workflow, steps: int) -> None:
        from gapa.algorithm.CDA.CGN import CGN, CGNController, CGNEvaluator

        class CaptureController(_CaptureSaveMixin, CGNController):
            pass

        loader = self._clone_loader(workflow.data_loader, mode=workflow.mode, world_size=workflow.world_size)
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
        controller.observer = self._observer(workflow.monitor)
        evaluator = CGNEvaluator(pop_size=self.pop_size, graph=loader.G.copy(), device=workflow.device)
        evaluator = self._maybe_wrap_evaluator(workflow, evaluator)
        CGN(workflow.mode, int(steps), loader, controller, evaluator, workflow.world_size, verbose=workflow.verbose)
        self._apply_capture(workflow.monitor, controller.captured_result, ("Q", "NMI"))


class QAttackAlgorithm(_LegacyAlgorithm):
    def __init__(self, pop_size: int = 40, crossover_rate: float = 0.8, mutate_rate: float = 0.1):
        super().__init__()
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutate_rate = mutate_rate
        self.side = "max"

    def run_full(self, workflow, steps: int) -> None:
        from gapa.algorithm.CDA.QAttack import QAttack, QAttackController, QAttackEvaluator

        class CaptureController(_CaptureSaveMixin, QAttackController):
            pass

        loader = self._clone_loader(workflow.data_loader, mode=workflow.mode, world_size=workflow.world_size)
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
        controller.observer = self._observer(workflow.monitor)
        evaluator = QAttackEvaluator(pop_size=self.pop_size, graph=loader.G.copy(), device=workflow.device)
        evaluator = self._maybe_wrap_evaluator(workflow, evaluator)
        QAttack(workflow.mode, int(steps), loader, controller, evaluator, workflow.world_size, verbose=workflow.verbose)
        self._apply_capture(workflow.monitor, controller.captured_result, ("Q", "NMI"))


class CDAEDAAlgorithm(_LegacyAlgorithm):
    def __init__(self, pop_size: int = 40, crossover_rate: float = 0.6, mutate_rate: float = 0.2):
        super().__init__()
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutate_rate = mutate_rate
        self.side = "max"

    def run_full(self, workflow, steps: int) -> None:
        from gapa.algorithm.CDA.EDA import EDA, EDAController, EDAEvaluator

        class CaptureController(_CaptureSaveMixin, EDAController):
            pass

        loader = self._clone_loader(workflow.data_loader, mode=workflow.mode, world_size=workflow.world_size)
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
        controller.observer = self._observer(workflow.monitor)
        evaluator = EDAEvaluator(pop_size=self.pop_size, graph=loader.G.copy(), adj=loader.A, nodes_num=loader.nodes_num, device=workflow.device)
        evaluator = self._maybe_wrap_evaluator(workflow, evaluator)
        EDA(workflow.mode, int(steps), loader, controller, evaluator, workflow.world_size, verbose=workflow.verbose)
        self._apply_capture(workflow.monitor, controller.captured_result, ("Q", "NMI"))


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

        class CaptureController(_CaptureSaveMixin, GAController):
            pass

        edges = torch.tensor(list(workflow.data_loader.G.edges), device=workflow.device)
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
        controller.observer = self._observer(workflow.monitor)
        evaluator = GAEvaluator(pop_size=self.pop_size, graph=loader.G, ratio=0, device=workflow.device)
        evaluator = self._maybe_wrap_evaluator(workflow, evaluator)
        LPA_GA(workflow.mode, int(steps), loader, controller, evaluator, workflow.world_size, verbose=workflow.verbose)
        self._apply_capture(workflow.monitor, controller.captured_result, ("Pre", "AUC"))


class LPAEDAAlgorithm(_LegacyAlgorithm):
    def __init__(self, pop_size: int = 40, mutate_rate: float = 0.1, attack_rate: float = 0.1):
        super().__init__()
        self.pop_size = pop_size
        self.mutate_rate = mutate_rate
        self.attack_rate = attack_rate
        self.side = "max"

    def run_full(self, workflow, steps: int) -> None:
        from gapa.algorithm.LPA.EDA import EDA, EDAController, EDAEvaluator

        class CaptureController(_CaptureSaveMixin, EDAController):
            pass

        edges = torch.tensor(list(workflow.data_loader.G.edges), device=workflow.device)
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
        controller.observer = self._observer(workflow.monitor)
        evaluator = EDAEvaluator(pop_size=self.pop_size, graph=loader.G, ratio=0, device=workflow.device)
        evaluator = self._maybe_wrap_evaluator(workflow, evaluator)
        EDA(workflow.mode, int(steps), loader, controller, evaluator, workflow.world_size, verbose=workflow.verbose)
        self._apply_capture(workflow.monitor, controller.captured_result, ("Pre", "AUC"))


class NCAGAAlgorithm(_LegacyAlgorithm):
    def __init__(self, pop_size: int = 20, crossover_rate: float = 0.7, mutate_rate: float = 0.3, attack_rate: float = 0.025):
        super().__init__()
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutate_rate = mutate_rate
        self.attack_rate = attack_rate
        self.side = "max"

    @staticmethod
    def _load_nca_payload(public_loader: DataLoader, device: torch.device) -> Any:
        data_path = Path(str(public_loader.meta.get("path") or ""))
        if not data_path.is_absolute():
            data_path = _datasets_root() / data_path
        if not data_path.exists():
            raise FileNotFoundError(f"NCA dataset file missing: {data_path}")
        raw = np.load(data_path, allow_pickle=True)
        required = {"edges", "node_features", "node_labels", "train_masks", "val_masks", "test_masks"}
        if not required.issubset(set(raw.files)):
            raise RuntimeError(
                f"NCA public workflow currently expects filtered NPZ datasets with keys {sorted(required)}. got={sorted(raw.files)}"
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

        class CaptureController(_CaptureSaveMixin, NCA_GAController):
            pass

        loader = self._load_nca_payload(workflow.data_loader, workflow.device)
        loader.k = max(1, int(self.attack_rate * loader.num_edge))
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
        controller.observer = self._observer(workflow.monitor)
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
        self._apply_capture(workflow.monitor, controller.captured_result, ("Acc", "ASR"))


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
        controller.observer = self._observer(workflow.monitor)
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
