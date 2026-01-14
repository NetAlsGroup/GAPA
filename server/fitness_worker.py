from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple


torch = None


def _require_torch():
    global torch
    if torch is None:
        try:
            import torch as _torch  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"torch is required for fitness worker: {exc}") from exc
        torch = _torch
    return torch


def _select_device() -> str:
    torch = _require_torch()
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _dataset_dir() -> Path:
    return Path(os.getenv("GAPA_DATASET_DIR", str(_repo_root() / "dataset")))


def _find_dataset_file(name: str) -> Optional[Path]:
    if not name:
        return None
    dataset_dir = _dataset_dir()
    candidates: list[Path] = []
    candidates.append(dataset_dir / f"{name}.txt")
    candidates.append(dataset_dir / f"{name.lower()}.txt")
    candidates.append(dataset_dir / name / f"{name}.txt")
    candidates.append(dataset_dir / name.lower() / f"{name.lower()}.txt")
    norm = name.replace("_", "-")
    candidates.append(dataset_dir / f"{norm}.txt")
    candidates.append(dataset_dir / f"{norm.lower()}.txt")
    candidates.append(dataset_dir / norm / f"{norm}.txt")
    candidates.append(dataset_dir / norm.lower() / f"{norm.lower()}.txt")
    for p in candidates:
        if p.exists():
            return p
    target = f"{name}".lower()
    try:
        for p in dataset_dir.glob("**/*.txt"):
            if p.name.lower() in (f"{target}.txt", f"{norm.lower()}.txt"):
                return p
    except Exception:
        pass
    return None


def _find_dataset_gml(name: str) -> Optional[Path]:
    if not name:
        return None
    dataset_dir = _dataset_dir()
    candidates: list[Path] = []
    candidates.append(dataset_dir / name / f"{name}.gml")
    candidates.append(dataset_dir / name.lower() / f"{name.lower()}.gml")
    norm = name.replace("_", "-")
    candidates.append(dataset_dir / norm / f"{norm}.gml")
    candidates.append(dataset_dir / norm.lower() / f"{norm.lower()}.gml")
    for p in candidates:
        if p.exists():
            return p
    target = name.lower()
    target2 = norm.lower()
    try:
        for p in dataset_dir.glob("**/*.gml"):
            if p.name.lower() in (f"{target}.gml", f"{target2}.gml"):
                return p
    except Exception:
        pass
    return None


def _load_gml(name: str, *, sort_nodes: bool, rebuild_from_adj: bool, device: str) -> Dict[str, Any]:
    import networkx as nx  # type: ignore
    import torch  # type: ignore
    
    gml = _find_dataset_gml(name)
    if gml is None:
        dataset_dir = _dataset_dir()
        raise FileNotFoundError(f"dataset .gml not found for '{name}' under {dataset_dir}")
    
    G0 = nx.read_gml(str(gml), label="id")
    nodelist0 = sorted(list(G0.nodes())) if sort_nodes else list(G0.nodes())
    A0 = torch.tensor(nx.to_numpy_array(G0, nodelist=nodelist0), dtype=torch.float32)
    
    if rebuild_from_adj:
        G1 = nx.from_numpy_array(A0.cpu().numpy())
        return {"G": G1, "A": A0.to(device) if device != "cpu" else A0, "nodelist": list(G1.nodes())}
    
    return {"G": G0, "A": A0.to(device) if device != "cpu" else A0, "nodelist": nodelist0}



@dataclass(frozen=True)
class _ContextKey:
    algorithm: str
    dataset: str
    device: str


class _FitnessContext:
    def __init__(self, algorithm: str, dataset: str, device: str) -> None:
        self.algorithm = algorithm
        self.dataset = dataset
        self.device = device
        self.lock = threading.Lock()

        self._data: Any = None
        self._evaluator_setup: Any = None  # callable(pop_size)->evaluator
        self._evaluator_cache: Dict[int, Any] = {}

        self._prepare()

    def _prepare(self) -> None:
        torch = _require_torch()
        try:
            import networkx as nx  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"networkx is required for {self.algorithm}: {exc}") from exc
        import io
        import contextlib

        ds_file = _find_dataset_file(self.dataset)
        if ds_file is None:
            raise FileNotFoundError(f"dataset .txt not found for '{self.dataset}' under {_dataset_dir()}")

        algo = (self.algorithm or "").strip()

        def _adjlist(sort_nodes: bool) -> Dict[str, Any]:
            g0 = nx.read_adjlist(str(ds_file), nodetype=int)
            nodelist0 = sorted(list(g0.nodes())) if sort_nodes else list(g0.nodes())
            a0 = torch.tensor(nx.to_numpy_array(g0, nodelist=nodelist0), dtype=torch.float32)
            if sort_nodes:
                g1 = nx.from_numpy_array(a0.cpu().numpy())
                return {"G": g1, "A": a0.to(self.device) if self.device != "cpu" else a0, "nodelist": list(g1.nodes())}
            return {"G": g0, "A": a0.to(self.device) if self.device != "cpu" else a0, "nodelist": nodelist0}

        if algo == "SixDST":
            from gapa.algorithm.CND.SixDST import SixDSTController, SixDSTEvaluator  # type: ignore

            loaded = _adjlist(sort_nodes=False)
            data_loader = SimpleNamespace(dataset=self.dataset, device=self.device)
            data_loader.G = loaded["G"]
            data_loader.A = loaded["A"]
            data_loader.nodes_num = int(data_loader.A.shape[0])
            data_loader.nodes = torch.tensor(loaded["nodelist"], device=self.device)
            data_loader.selected_genes_num = int(0.4 * data_loader.nodes_num)
            data_loader.k = int(0.1 * data_loader.nodes_num)

            controller = SixDSTController(
                path=None,
                pattern=None,
                cutoff_tag="popGreedy_cutoff_",
                data_loader=data_loader,
                loops=1,
                crossover_rate=0.8,
                mutate_rate=0.2,
                pop_size=1,
                device=self.device,
            )

            def _setup(pop_size: int):
                evaluator = SixDSTEvaluator(pop_size=pop_size, adj=data_loader.A, device=self.device)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    return controller.setup(data_loader=data_loader, evaluator=evaluator)

            self._data = data_loader
            self._evaluator_setup = _setup
            return

        if algo == "CutOff":
            from gapa.algorithm.CND.Cutoff import CutoffController, CutoffEvaluator  # type: ignore

            loaded = _adjlist(sort_nodes=True)
            data_loader = SimpleNamespace(dataset=self.dataset, device=self.device)
            data_loader.G = loaded["G"]
            data_loader.A = loaded["A"]
            data_loader.nodes_num = int(data_loader.A.shape[0])
            data_loader.nodes = torch.tensor(loaded["nodelist"], device=self.device)
            data_loader.selected_genes_num = int(0.4 * data_loader.nodes_num)
            data_loader.k = int(0.1 * data_loader.nodes_num)

            controller = CutoffController(
                path=None,
                pattern=None,
                data_loader=data_loader,
                loops=1,
                crossover_rate=0.8,
                mutate_rate=0.2,
                pop_size=1,
                device=self.device,
            )

            def _setup(pop_size: int):
                evaluator = CutoffEvaluator(pop_size=pop_size, graph=data_loader.G, nodes=data_loader.nodes, device=self.device)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    return controller.setup(data_loader=data_loader, evaluator=evaluator)

            self._data = data_loader
            self._evaluator_setup = _setup
            return

        if algo == "TDE":
            from gapa.algorithm.CND.TDE import TDEController, TDEEvaluator  # type: ignore

            loaded = _adjlist(sort_nodes=True)
            data_loader = SimpleNamespace(dataset=self.dataset, device=self.device)
            data_loader.G = loaded["G"]
            data_loader.A = loaded["A"]
            data_loader.nodes_num = int(data_loader.A.shape[0])
            data_loader.nodes = torch.tensor(loaded["nodelist"], device=self.device)
            data_loader.selected_genes_num = int(0.4 * data_loader.nodes_num)
            data_loader.k = int(0.1 * data_loader.nodes_num)

            controller = TDEController(
                path=None,
                pattern=None,
                data_loader=data_loader,
                loops=1,
                crossover_rate=0.8,
                mutate_rate=0.2,
                pop_size=1,
                device=self.device,
            )

            def _setup(pop_size: int):
                evaluator = TDEEvaluator(pop_size=pop_size, graph=data_loader.G, budget=data_loader.k, device=self.device)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    return controller.setup(data_loader=data_loader, evaluator=evaluator)

            self._data = data_loader
            self._evaluator_setup = _setup
            self._data = data_loader
            self._evaluator_setup = _setup
            return

        if algo == "QAttack":
            from gapa.algorithm.CDA.QAttack import QAttackController, QAttackEvaluator  # type: ignore

            loaded = _load_gml(self.dataset, sort_nodes=True, rebuild_from_adj=False, device=self.device)
            data_loader = SimpleNamespace(dataset=self.dataset, device=self.device)
            data_loader.G = loaded["G"]
            data_loader.A = loaded["A"]
            data_loader.nodes_num = int(data_loader.A.shape[0])
            data_loader.nodes = torch.tensor(loaded["nodelist"], device=self.device)
            
            # Replicate ga_worker logic for attack rate
            attack_rate = float(os.getenv("GAPA_CDA_ATTACK_RATE", "0.1"))
            data_loader.selected_genes_num = int(attack_rate * 4 * data_loader.nodes_num)
            data_loader.k = int(attack_rate * data_loader.nodes_num)

            controller = QAttackController(
                path=None,
                pattern=None,
                data_loader=data_loader,
                loops=1,
                crossover_rate=0.8,
                mutate_rate=0.2,
                pop_size=1,
                device=self.device,
            )

            def _setup(pop_size: int):
                # QAttackEvaluator modifies graph, so pass a copy if needed, though here we create fresh one per worker usually
                # But fitness_worker reuses context. data_loader.G is shared.
                # QAttackEvaluator init: self.G = graph
                # Check QAttackEvaluator source if it modifies G. Usually Evaluators might.
                # ga_worker passes graph=data_loader.G.copy()
                evaluator = QAttackEvaluator(pop_size=pop_size, graph=data_loader.G.copy(), device=self.device)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    return controller.setup(data_loader=data_loader, evaluator=evaluator)

            self._data = data_loader
            self._evaluator_setup = _setup
            return

    def eval(self, population_cpu: Any) -> Tuple[Any, Dict[str, Any]]:
        torch = _require_torch()
        if not isinstance(population_cpu, torch.Tensor):
            raise TypeError("population must be a torch.Tensor")
        pop_size = int(population_cpu.shape[0])
        with self.lock:
            evaluator = self._evaluator_cache.get(pop_size)
            if evaluator is None:
                evaluator = self._evaluator_setup(pop_size)
                self._evaluator_cache[pop_size] = evaluator
            pop = population_cpu.to(self.device)
            out = evaluator(pop)
            return out.detach().to("cpu"), {"device": self.device, "pop_size": pop_size}


_CTX: Dict[_ContextKey, _FitnessContext] = {}
_CTX_LOCK = threading.Lock()


def clear_contexts() -> None:
    """Release cached fitness contexts and GPU memory."""
    global _CTX
    with _CTX_LOCK:
        _CTX = {}
    try:
        torch = _require_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def compute_fitness_batch(algorithm: str, dataset: str, population_cpu: Any, *, device: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
    """Compute fitness for a population chunk (CPU tensor in, CPU tensor out)."""
    torch = _require_torch()
    if device is None:
        device = _select_device()
    key = _ContextKey(algorithm=algorithm, dataset=dataset, device=device)
    with _CTX_LOCK:
        ctx = _CTX.get(key)
        if ctx is None:
            ctx = _FitnessContext(algorithm=algorithm, dataset=dataset, device=device)
            _CTX[key] = ctx
    if not isinstance(population_cpu, torch.Tensor):
        raise TypeError("population must be torch.Tensor")
    if population_cpu.device.type != "cpu":
        population_cpu = population_cpu.detach().to("cpu")
    return ctx.eval(population_cpu)
