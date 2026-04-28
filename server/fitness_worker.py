from __future__ import annotations

import os
import threading
import inspect
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

from gapa.config import get_dataset_dir

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
    return get_dataset_dir(_repo_root())


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
        import io
        import contextlib

        algo = (self.algorithm or "").strip()
        try:
            from .algorithm_registry import resolve_algorithm_id
            algo = resolve_algorithm_id(algo)
        except Exception:
            pass

        # Generic registry path:
        # Build distributed-fitness evaluator from the public algorithm wrapper,
        # so remote/MNM workers share the same setup logic as local execution.
        try:
            from .algorithm_registry import load_algorithm_entries, load_algorithm_registry
            from gapa import DataLoader
        except Exception:
            load_algorithm_entries = None  # type: ignore[assignment]
            load_algorithm_registry = None  # type: ignore[assignment]
            DataLoader = None  # type: ignore[assignment]

        if load_algorithm_registry is not None and DataLoader is not None:
            registry = load_algorithm_registry()
            algo_cls = registry.get(algo)
            if algo_cls is not None:
                # One dataset object can be reused; evaluator is created per pop_size.
                data_loader = DataLoader.load(self.dataset, device=self.device)
                torch_device = torch.device(self.device)
                init_kwargs_template: Dict[str, Any] = {}
                if load_algorithm_entries is not None:
                    for entry in load_algorithm_entries():
                        if not isinstance(entry, dict):
                            continue
                        if str(entry.get("id") or "").strip() == algo:
                            cfg = entry.get("init_kwargs")
                            if isinstance(cfg, dict):
                                init_kwargs_template = dict(cfg)
                            break

                def _build_algorithm(pop_size: int):
                    kwargs = dict(init_kwargs_template)
                    try:
                        sig = inspect.signature(algo_cls.__init__)
                        if "pop_size" in sig.parameters and "pop_size" not in kwargs:
                            kwargs["pop_size"] = int(pop_size)
                    except Exception:
                        pass
                    try:
                        inst = algo_cls(**kwargs)
                    except TypeError:
                        # Last-resort fallback for non-standard __init__ signatures.
                        inst = algo_cls()
                    # Keep pop_size aligned with current batch when algorithm exposes it.
                    if hasattr(inst, "pop_size"):
                        try:
                            setattr(inst, "pop_size", int(pop_size))
                        except Exception:
                            pass
                    return inst

                def _setup(pop_size: int):
                    inst = _build_algorithm(pop_size)
                    if hasattr(inst, "build_distributed_evaluator"):
                        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                            return inst.build_distributed_evaluator(data_loader, torch_device, int(pop_size))
                    evaluator = inst.create_evaluator(data_loader)
                    controller = inst.create_controller(data_loader, mode="s", device=torch_device)
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        return controller.setup(data_loader=data_loader, evaluator=evaluator)

                self._data = data_loader
                self._evaluator_setup = _setup
                return

        raise RuntimeError(
            f"unsupported algorithm for distributed fitness: '{algo}' "
            f"(raw='{self.algorithm}')"
        )

    def eval(self, population_cpu: Any, extra_context: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        torch = _require_torch()
        if not isinstance(population_cpu, torch.Tensor):
            raise TypeError("population must be a torch.Tensor")
        pop_size = int(population_cpu.shape[0])
        with self.lock:
            evaluator = self._evaluator_cache.get(pop_size)
            if evaluator is None:
                if not callable(self._evaluator_setup):
                    raise RuntimeError(
                        f"evaluator setup not ready for algorithm='{self.algorithm}'"
                    )
                evaluator = self._evaluator_setup(pop_size)
                self._evaluator_cache[pop_size] = evaluator
            
            # Apply generic task-specific context synchronization
            # Logic: If evaluator has attribute matching context key, override it.
            if extra_context:
                for key, val in extra_context.items():
                     if hasattr(evaluator, key):
                         # Handle tensor device texturing
                         target_val = val
                         if isinstance(target_val, torch.Tensor):
                             target_val = target_val.to(self.device)
                         setattr(evaluator, key, target_val)

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


def compute_fitness_batch(algorithm: str, dataset: str, population_cpu: Any, *, device: Optional[str] = None, extra_context: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
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
    # Auto-cast from float16 to float32 if compressed
    if population_cpu.dtype == torch.half:
        population_cpu = population_cpu.float()
    
    if population_cpu.device.type != "cpu":
        population_cpu = population_cpu.detach().to("cpu")
    return ctx.eval(population_cpu, extra_context=extra_context)
