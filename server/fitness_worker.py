from __future__ import annotations

import os
import time
import threading
import inspect
import gc
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from gapa.config import get_dataset_dir

torch = None
_CPU_POOL_EVALUATOR = None


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


def _canonical_algorithm_name(name: str) -> str:
    return str(name or "").strip().replace("-", "").replace("_", "").lower()


CPU_PARALLEL_ALGORITHMS = {"qattack", "cgn", "tde", "cutoff", "lpaeda", "lpaga"}


def _evaluator_cache_key(algorithm: str, device: str, pop_size: int) -> int:
    cache_by_pop = str(os.getenv("GAPA_FITNESS_CACHE_BY_POP_SIZE", "0") or "0").strip().lower()
    if cache_by_pop in ("1", "true", "yes", "on"):
        is_cpu_parallel = _canonical_algorithm_name(algorithm) in CPU_PARALLEL_ALGORITHMS and device == "cpu"
        return 0 if is_cpu_parallel else int(pop_size)
    return 0


def _init_cpu_pool(evaluator: Any) -> None:
    global _CPU_POOL_EVALUATOR
    _CPU_POOL_EVALUATOR = evaluator


def _cpu_pool_eval(pop_chunk_cpu: Any) -> Any:
    evaluator = _CPU_POOL_EVALUATOR
    if evaluator is None:
        raise RuntimeError("cpu pool evaluator not initialized")
    torch = _require_torch()
    if hasattr(evaluator, "pop_size"):
        try:
            evaluator.pop_size = int(pop_chunk_cpu.shape[0])
        except Exception:
            pass
    out = evaluator(pop_chunk_cpu)
    if isinstance(out, torch.Tensor):
        return out.detach().to("cpu")
    return torch.as_tensor(out).detach().to("cpu")


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
        self.created_at = time.time()
        self.last_used_at = self.created_at

        self._data: Any = None
        self._evaluator_setup: Any = None  # callable(pop_size)->evaluator
        self._evaluator_cache: Dict[int, Any] = {}
        self._cpu_pool: Optional[ProcessPoolExecutor] = None
        self._cpu_pool_workers: int = 0
        self._cpu_pool_evaluator_id: int = 0

        self._prepare()

    def close(self) -> None:
        if self._cpu_pool is not None:
            try:
                self._cpu_pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        self._cpu_pool = None
        self._cpu_pool_workers = 0
        self._cpu_pool_evaluator_id = 0
        self._evaluator_cache.clear()
        self._data = None

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
            self.last_used_at = time.time()
            is_cpu_parallel = _canonical_algorithm_name(self.algorithm) in CPU_PARALLEL_ALGORITHMS and self.device == "cpu"
            evaluator_key = _evaluator_cache_key(self.algorithm, self.device, pop_size)
            evaluator = self._evaluator_cache.get(evaluator_key)
            if evaluator is None:
                if not callable(self._evaluator_setup):
                    raise RuntimeError(
                        f"evaluator setup not ready for algorithm='{self.algorithm}'"
                    )
                evaluator = self._evaluator_setup(pop_size)
                self._evaluator_cache[evaluator_key] = evaluator
                self._trim_evaluator_cache(keep_key=evaluator_key)
            elif hasattr(evaluator, "pop_size"):
                try:
                    setattr(evaluator, "pop_size", int(pop_size))
                except Exception:
                    pass
            
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

            if is_cpu_parallel:
                t_forward_start = time.perf_counter()
                out = self._eval_cpu_parallel(evaluator, population_cpu)
                forward_ms = (time.perf_counter() - t_forward_start) * 1000.0
                return out.detach().to("cpu"), {
                    "device": self.device,
                    "pop_size": pop_size,
                    "copy_to_device_ms": 0.0,
                    "forward_ms": forward_ms,
                }

            t_copy_start = time.perf_counter()
            pop = population_cpu.to(self.device)
            if self.device.startswith("cuda"):
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            copy_to_device_ms = (time.perf_counter() - t_copy_start) * 1000.0

            t_forward_start = time.perf_counter()
            out = evaluator(pop)
            if self.device.startswith("cuda"):
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            forward_ms = (time.perf_counter() - t_forward_start) * 1000.0
            return out.detach().to("cpu"), {
                "device": self.device,
                "pop_size": pop_size,
                "copy_to_device_ms": copy_to_device_ms,
                "forward_ms": forward_ms,
            }

    def _eval_cpu_parallel(self, evaluator: Any, population_cpu: Any) -> Any:
        torch = _require_torch()
        pop_size = int(population_cpu.shape[0])
        workers = int(os.getenv("GAPA_CPU_FITNESS_PROCS", "0") or 0)
        if workers <= 0:
            workers = min(max(1, os.cpu_count() or 1), max(1, pop_size), 4)
        min_chunk = max(1, int(os.getenv("GAPA_CPU_FITNESS_MIN_CHUNK", "8") or 8))
        if workers <= 1 or pop_size <= 1 or pop_size < min_chunk:
            return evaluator(population_cpu)

        if (
            self._cpu_pool is None
            or self._cpu_pool_workers != workers
            or self._cpu_pool_evaluator_id != id(evaluator)
        ):
            if self._cpu_pool is not None:
                try:
                    self._cpu_pool.shutdown(wait=True, cancel_futures=False)
                except Exception:
                    pass
            ctx = None
            try:
                import multiprocessing as _mp
                ctx = _mp.get_context("fork")
            except Exception:
                ctx = None
            self._cpu_pool = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=ctx,
                initializer=_init_cpu_pool,
                initargs=(evaluator,),
            )
            self._cpu_pool_workers = workers
            self._cpu_pool_evaluator_id = id(evaluator)

        chunk_count = min(workers, pop_size)
        pop_chunks = [
            chunk.contiguous()
            for chunk in torch.chunk(population_cpu.detach().to("cpu"), chunk_count, dim=0)
            if chunk.numel() > 0
        ]
        if len(pop_chunks) <= 1 or self._cpu_pool is None:
            return evaluator(population_cpu)
        try:
            parts = list(self._cpu_pool.map(_cpu_pool_eval, pop_chunks))
            return torch.cat([part if isinstance(part, torch.Tensor) else torch.as_tensor(part) for part in parts], dim=0)
        except Exception:
            return evaluator(population_cpu)

    def _trim_evaluator_cache(self, *, keep_key: int) -> None:
        max_items = max(1, int(os.getenv("GAPA_FITNESS_EVALUATOR_CACHE_MAX", "4") or 4))
        if len(self._evaluator_cache) <= max_items:
            return
        for key in list(self._evaluator_cache.keys()):
            if key == keep_key:
                continue
            self._evaluator_cache.pop(key, None)
            if len(self._evaluator_cache) <= max_items:
                break

    def stats(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "dataset": self.dataset,
            "device": self.device,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "age_s": max(0.0, time.time() - self.created_at),
            "idle_s": max(0.0, time.time() - self.last_used_at),
            "evaluator_cache_size": len(self._evaluator_cache),
            "cpu_pool_active": self._cpu_pool is not None,
            "cpu_pool_workers": self._cpu_pool_workers,
        }


_CTX: Dict[_ContextKey, _FitnessContext] = {}
_CTX_LOCK = threading.Lock()


def _close_context(ctx: _FitnessContext) -> None:
    try:
        with ctx.lock:
            ctx.close()
    except Exception:
        pass


def _prune_contexts_locked(*, now: Optional[float] = None, force: bool = False) -> None:
    if not _CTX:
        return
    now = time.time() if now is None else float(now)
    ttl_s = float(os.getenv("GAPA_FITNESS_CONTEXT_TTL_S", "1800") or 1800)
    max_contexts = max(1, int(os.getenv("GAPA_FITNESS_CONTEXT_MAX", "4") or 4))

    stale_keys: List[_ContextKey] = []
    if force:
        stale_keys = list(_CTX.keys())
    elif ttl_s > 0:
        stale_keys = [key for key, ctx in _CTX.items() if now - ctx.last_used_at > ttl_s]
    for key in stale_keys:
        ctx = _CTX.pop(key, None)
        if ctx is not None:
            _close_context(ctx)

    if not force and len(_CTX) > max_contexts:
        victims = sorted(_CTX.items(), key=lambda item: item[1].last_used_at)
        for key, ctx in victims[: max(0, len(_CTX) - max_contexts)]:
            _CTX.pop(key, None)
            _close_context(ctx)


def context_stats() -> Dict[str, Any]:
    with _CTX_LOCK:
        return {
            "count": len(_CTX),
            "max_contexts": max(1, int(os.getenv("GAPA_FITNESS_CONTEXT_MAX", "4") or 4)),
            "ttl_s": float(os.getenv("GAPA_FITNESS_CONTEXT_TTL_S", "1800") or 1800),
            "contexts": [ctx.stats() for ctx in _CTX.values()],
        }


def clear_contexts() -> None:
    """Release cached fitness contexts and GPU memory."""
    global _CTX
    with _CTX_LOCK:
        _prune_contexts_locked(force=True)
        _CTX = {}
    gc.collect()
    try:
        torch = _require_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def warmup_context(
    algorithm: str,
    dataset: str,
    *,
    device: Optional[str] = None,
    pop_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Pre-create the cached context/evaluator used by remote fitness calls."""
    if device is None:
        device = _select_device()
    key = _ContextKey(algorithm=algorithm, dataset=dataset, device=device)
    created = False
    with _CTX_LOCK:
        _prune_contexts_locked()
        ctx = _CTX.get(key)
        if ctx is None:
            ctx = _FitnessContext(algorithm=algorithm, dataset=dataset, device=device)
            _CTX[key] = ctx
            created = True

    evaluator_created = False
    if pop_size is not None and int(pop_size) > 0:
        pop_size = int(pop_size)
        with ctx.lock:
            ctx.last_used_at = time.time()
            evaluator_key = _evaluator_cache_key(ctx.algorithm, ctx.device, pop_size)
            evaluator = ctx._evaluator_cache.get(evaluator_key)
            if evaluator is None:
                if not callable(ctx._evaluator_setup):
                    raise RuntimeError(
                        f"evaluator setup not ready for algorithm='{ctx.algorithm}'"
                    )
                evaluator = ctx._evaluator_setup(pop_size)
                ctx._evaluator_cache[evaluator_key] = evaluator
                ctx._trim_evaluator_cache(keep_key=evaluator_key)
                evaluator_created = True

    return {
        "warmed": True,
        "algorithm": algorithm,
        "dataset": dataset,
        "device": device,
        "pop_size": pop_size,
        "context_created": created,
        "evaluator_created": evaluator_created,
        "stats": ctx.stats(),
    }


def compute_fitness_batch(algorithm: str, dataset: str, population_cpu: Any, *, device: Optional[str] = None, extra_context: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
    """Compute fitness for a population chunk (CPU tensor in, CPU tensor out)."""
    torch = _require_torch()
    if device is None:
        device = _select_device()
    key = _ContextKey(algorithm=algorithm, dataset=dataset, device=device)
    with _CTX_LOCK:
        _prune_contexts_locked()
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
