from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

from .fitness_protocol import dumps, loads


def _require_requests():
    if requests is None:  # pragma: no cover
        raise RuntimeError("requests is required for distributed evaluator")
    return requests


def _require_torch():
    if torch is None or nn is None:  # pragma: no cover
        raise RuntimeError("torch is required for distributed evaluator")
    return torch, nn



@dataclass(frozen=True)
class Worker:
    server_id: str
    base_url: str


@dataclass(frozen=True)
class DeviceWorker:
    """Represents a single device (GPU or CPU) on a server for distributed fitness."""
    server_id: str      # e.g., "Server 163" or "local"
    base_url: str       # "" for local, "http://..." for remote
    device: str         # e.g., "cuda:0", "cuda:1", "cpu"
    is_local: bool      # True if this is the local node
    weight: float = 1.0 # Computation weight (default 1.0)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _servers_file() -> Path:
    return Path(os.getenv("GAPA_SERVERS_FILE", str(_repo_root() / "servers.json")))


def load_workers(exclude_local: bool = True, allowed_ids: Optional[List[str]] = None) -> List[Worker]:
    path = _servers_file()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(raw, dict):
        raw = raw.get("servers", [])
    if not isinstance(raw, list):
        return []
    out: List[Worker] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        sid = entry.get("id") or entry.get("name") or "remote"
        if exclude_local and sid == "local":
            continue
        if allowed_ids is not None and sid not in allowed_ids:
            continue
        base_url = entry.get("base_url") or ""
        if not base_url:
            host = entry.get("ip") or entry.get("host")
            port = entry.get("port")
            protocol = entry.get("protocol") or "http"
            if host:
                base_url = f"{protocol}://{host}{f':{port}' if port else ''}"
        if not base_url:
            continue
        out.append(Worker(server_id=str(sid), base_url=str(base_url).rstrip("/")))
    return out


def build_device_workers(
    workers: List[Worker],
    *,
    include_local: bool = False,
    local_device: Optional[str] = None,
    request_timeout: float = 5.0,
) -> List[DeviceWorker]:
    """Build a list of DeviceWorkers by querying each server's locked devices.
    
    Args:
        workers: List of server Workers to query
        include_local: Whether to include local CPU/GPU as a worker
        local_device: Device to use for local ("cpu", "cuda:0", etc.)
        request_timeout: Timeout for lock status query
    
    Returns:
        List of DeviceWorker, one per GPU (or CPU for CPU-only servers)
    """
    req = _require_requests()
    session = req.Session()
    session.trust_env = False
    
    device_workers: List[DeviceWorker] = []
    
    # Add local worker if requested
    if include_local:
        if local_device:
            device_workers.append(DeviceWorker(
                server_id="local",
                base_url="",
                device=local_device,
                is_local=True,
                weight=1.0,
            ))
    
    # Query each remote server for its locked devices
    for w in workers:
        try:
            resp = session.get(
                w.base_url + "/api/resource_lock/status",
                timeout=(2.0, request_timeout),
            )
            if not resp.ok:
                # Server unreachable, but still add as CPU fallback worker
                device_workers.append(DeviceWorker(
                    server_id=w.server_id,
                    base_url=w.base_url,
                    device="cpu",
                    is_local=False,
                    weight=0.5,
                ))
                continue
            lock_status = resp.json()
            if not lock_status.get("active"):
                # No active lock, skip this server
                continue
            
            devices = lock_status.get("devices") or []
            backend = lock_status.get("backend", "cpu")
            
            if devices:
                # Create one DeviceWorker per locked GPU
                for dev_id in devices:
                    device_workers.append(DeviceWorker(
                        server_id=w.server_id,
                        base_url=w.base_url,
                        device=f"cuda:{dev_id}",
                        is_local=False,
                        weight=1.0,
                    ))
            else:
                # CPU backend or unknown - add CPU worker
                device_workers.append(DeviceWorker(
                    server_id=w.server_id,
                    base_url=w.base_url,
                    device="cpu",
                    is_local=False,
                    weight=0.5 if backend == "cpu" else 1.0,
                ))
        except Exception:
            # On any error, add server as CPU fallback
            device_workers.append(DeviceWorker(
                server_id=w.server_id,
                base_url=w.base_url,
                device="cpu",
                is_local=False,
                weight=0.3,  # Lower weight for potentially unreliable server
            ))
    
    return device_workers


class AdaptiveWorkerPool:
    def __init__(
        self,
        workers: List[Worker],
        *,
        refresh_interval_s: float = 2.0,
        max_workers: int = 4,
        cpu_busy_threshold: float = 85.0,
        gpu_busy_threshold: float = 85.0,
        min_gpu_free_mb: int = 512,
        request_timeout_s: float = 30.0,
        use_strategy_plan: bool = True,
    ) -> None:
        self._workers = workers
        self._refresh_interval_s = float(refresh_interval_s)
        self._max_workers = int(max_workers)
        self._cpu_busy_threshold = float(cpu_busy_threshold)
        self._gpu_busy_threshold = float(gpu_busy_threshold)
        self._min_gpu_free_mb = int(min_gpu_free_mb)
        self._request_timeout_s = float(request_timeout_s)
        self._use_strategy_plan = bool(use_strategy_plan)

        self._last_refresh = 0.0
        self._scores: Dict[str, float] = {}
        self._ok: Dict[str, bool] = {}
        self._backend: Dict[str, str] = {}

        # Thread-local storage for Sessions (avoids cross-thread contention)
        import threading
        self._thread_local = threading.local()
    
    def _get_session(self):
        """Get or create a thread-local Session."""
        if not hasattr(self._thread_local, 'session'):
            req = _require_requests()
            self._thread_local.session = req.Session()
            self._thread_local.session.trust_env = False
        return self._thread_local.session

    def _score_from_snapshot(self, snap: Dict[str, Any]) -> float:
        cpu = snap.get("cpu") or {}
        cpu_usage = cpu.get("usage_percent")
        if cpu_usage is None:
            cpu_usage = 50.0
        cpu_ok = float(cpu_usage) <= self._cpu_busy_threshold

        gpus = snap.get("gpus") or []
        gpu_free = 0.0
        gpu_ok = True
        if isinstance(gpus, list) and gpus:
            for g in gpus:
                try:
                    free_mb = float(g.get("free_mb")) if g.get("free_mb") is not None else 0.0
                    gpu_free += max(0.0, free_mb)
                    util = g.get("gpu_util_percent")
                    if util is not None and float(util) > self._gpu_busy_threshold:
                        gpu_ok = False
                except Exception:
                    continue
            if gpu_free < float(self._min_gpu_free_mb):
                gpu_ok = False
        # CPU-only servers are allowed but get lower score.
        base = (gpu_free / 1024.0) if gpu_free > 0 else 0.5
        if not cpu_ok or not gpu_ok:
            base *= 0.1
        return max(base, 0.01)

    def refresh(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_refresh) < self._refresh_interval_s:
            return
        self._last_refresh = now

        scores: Dict[str, float] = {}
        ok: Dict[str, bool] = {}
        backend: Dict[str, str] = {}
        for w in self._workers:
            try:
                resp = self._session.get(w.base_url + "/api/resources", timeout=(2.0, 5.0))
                snap = resp.json() if resp.ok else {"error": f"HTTP {resp.status_code}"}
                if not isinstance(snap, dict) or snap.get("error"):
                    ok[w.server_id] = False
                    scores[w.server_id] = 0.01
                    continue
                base_score = self._score_from_snapshot(snap)
                if self._use_strategy_plan:
                    try:
                        pr = self._session.post(
                            w.base_url + "/api/strategy_plan",
                            json={"warmup": 0, "objective": "time", "multi_gpu": True},
                            timeout=(3.0, 15.0),
                        )
                        plan = pr.json() if pr.ok else {}
                        b = str(plan.get("backend") or plan.get("plan", {}).get("backend") or "").lower()
                        if b:
                            backend[w.server_id] = b
                            if b == "multi-gpu":
                                base_score *= 1.20
                            elif b == "cuda":
                                base_score *= 1.00
                            elif b == "cpu":
                                base_score *= 0.50
                    except Exception:
                        pass
                scores[w.server_id] = base_score
                ok[w.server_id] = True
            except Exception:
                ok[w.server_id] = False
                scores[w.server_id] = 0.01
        self._scores = scores
        self._ok = ok
        self._backend = backend

    def pick(self) -> List[Tuple[Worker, float]]:
        self.refresh()
        ranked = sorted(self._workers, key=lambda w: float(self._scores.get(w.server_id, 0.01)), reverse=True)
        picked: List[Tuple[Worker, float]] = []
        for w in ranked:
            if not self._ok.get(w.server_id, False):
                continue
            picked.append((w, float(self._scores.get(w.server_id, 0.01))))
            if len(picked) >= self._max_workers:
                break
        return picked

    def remote_fitness(self, worker: Worker, *, algorithm: str, dataset: str, population_cpu: Any, device: Optional[str] = None, extra_context: Optional[Dict[str, Any]] = None) -> "DetailedCallResult":
        """Execute remote fitness and return detailed timing breakdown."""
        # Phase 1: Serialize
        t_serialize_start = time.perf_counter()
        
        # Optimize: Cast float32 to float16 (half) for transport to reduce size by 50%
        # The worker will cast it back to float32.
        pop_payload = population_cpu
        if hasattr(pop_payload, "dtype") and pop_payload.dtype == torch.float32:
            pop_payload = pop_payload.half()
            
        payload_data = {"algorithm": algorithm, "dataset": dataset, "population": pop_payload}
        
        # Generalize Distributed Context Synchronization
        if extra_context:
            for k, v in extra_context.items():
                if hasattr(v, "cpu"):
                    # Ensure tensors are on CPU for serialization
                    payload_data[k] = v.cpu()
                else:
                    payload_data[k] = v

        if device:
            payload_data["device"] = device
        payload = dumps(payload_data)
        serialize_ms = (time.perf_counter() - t_serialize_start) * 1000.0
        payload_bytes = len(payload)
        
        # Phase 2: Network + Remote Compute
        # Use thread-local session for connection reuse
        session = self._get_session()
        t_network_start = time.perf_counter()
        resp = session.post(
            worker.base_url + "/api/fitness/batch",
            data=payload,
            headers={"Content-Type": "application/octet-stream"},
            timeout=(3.0, self._request_timeout_s),
        )
        network_total_ms = (time.perf_counter() - t_network_start) * 1000.0
        
        if not resp.ok:
            raise RuntimeError(f"remote fitness failed: {worker.server_id} HTTP {resp.status_code}")
        
        # Phase 3: Deserialize
        t_deserialize_start = time.perf_counter()
        data = loads(resp.content)
        deserialize_ms = (time.perf_counter() - t_deserialize_start) * 1000.0
        
        fit = data.get("fitness")
        if fit is None:
            raise RuntimeError(f"remote fitness bad response: {worker.server_id}")
        
        # Extract remote compute time if provided by worker
        compute_ms = float(data.get("compute_ms", 0.0))
        # Network time = total round-trip - remote compute time
        network_ms = max(0.0, network_total_ms - compute_ms)
        
        total_ms = serialize_ms + network_total_ms + deserialize_ms
        
        return DetailedCallResult(
            fitness=fit,
            worker_id=worker.server_id,
            serialize_ms=serialize_ms,
            network_ms=network_ms,
            compute_ms=compute_ms,
            deserialize_ms=deserialize_ms,
            total_ms=total_ms,
            payload_bytes=payload_bytes,
            chunk_size=int(population_cpu.shape[0]) if hasattr(population_cpu, 'shape') else 0,
        )


from dataclasses import dataclass

@dataclass
class DetailedCallResult:
    """Result of a single remote fitness call with timing breakdown."""
    fitness: Any
    worker_id: str
    serialize_ms: float
    network_ms: float
    compute_ms: float
    deserialize_ms: float
    total_ms: float
    payload_bytes: int
    chunk_size: int


class DistributedEvaluator(nn.Module):
    """Evaluator proxy that offloads fitness computation to other servers.
    
    Supports:
    - Multiple GPUs per server (each GPU is a separate worker)
    - Local node participation (CPU or GPU)
    - CPU-based algorithms (QAttack) that can use Local CPU
    """
    
    # Algorithms that are CPU-based (use NetworkX/iGraph, not tensor computation)
    CPU_ALGORITHMS = {"QAttack", "CGN"}

    def __init__(
        self,
        base_evaluator: nn.Module,
        *,
        algorithm: str,
        dataset: str,
        workers: Optional[List[Worker]] = None,
        allowed_server_ids: Optional[List[str]] = None,
        max_remote_workers: int = 16,
        refresh_interval_s: float = 2.0,
        use_strategy_plan: Optional[bool] = None,
        include_local: bool = False,
        local_device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._base = base_evaluator
        self.algorithm = algorithm
        self.dataset = dataset
        self._include_local = include_local
        self._local_device = local_device
        
        if use_strategy_plan is None:
            use_strategy_plan = bool(int(os.getenv("GAPA_MNM_USE_PLAN", "0") or 0))
        
        # Store workers for building DeviceWorkers later
        self._workers = workers or load_workers(exclude_local=True, allowed_ids=allowed_server_ids)
        
        # Legacy pool for backward compatibility
        self._pool = AdaptiveWorkerPool(
            self._workers,
            refresh_interval_s=refresh_interval_s,
            max_workers=max_remote_workers,
            use_strategy_plan=bool(use_strategy_plan),
        )
        
        # Device workers cache (rebuilt on each forward for fresh lock status)
        self._device_workers: List[DeviceWorker] = []
        
        # Is this a CPU-based algorithm?
        self._is_cpu_algorithm = algorithm in self.CPU_ALGORITHMS
        
        # Legacy stats
        self._comm_total_ms = 0.0  # Cumulative (sum of all calls)
        self._comm_wall_clock_ms = 0.0  # Wall clock (actual elapsed, parallel)
        self._comm_calls = 0
        
        # Detailed stats
        self._detailed_calls: List[DetailedCallResult] = []
        self._current_iter = 0
        self._per_worker_stats: Dict[str, Dict[str, float]] = {}
        self._per_iter_stats: List[Dict[str, Any]] = []
    
    def _build_device_workers(self) -> List[DeviceWorker]:
        """Build DeviceWorkers from all servers' locked devices."""
        # For CPU algorithms, always include local CPU
        include_local = self._include_local or self._is_cpu_algorithm
        local_device = self._local_device or ("cpu" if self._is_cpu_algorithm else None)
        
        return build_device_workers(
            self._workers,
            include_local=include_local,
            local_device=local_device,
        )

    def set_iteration(self, iter_num: int) -> None:
        """Set current iteration number for tracking."""
        self._current_iter = iter_num
    
    def _update_worker_stats(self, dw: "DeviceWorker", result: "DetailedCallResult") -> None:
        """Update per-worker stats for a completed result."""
        wid = f"{dw.server_id}:{dw.device}"
        if wid not in self._per_worker_stats:
            self._per_worker_stats[wid] = {
                "total_ms": 0.0, "calls": 0, "serialize_ms": 0.0,
                "network_ms": 0.0, "compute_ms": 0.0, "deserialize_ms": 0.0,
                "total_bytes": 0
            }
        ws = self._per_worker_stats[wid]
        ws["total_ms"] += result.total_ms
        ws["calls"] += 1
        ws["serialize_ms"] += result.serialize_ms
        ws["network_ms"] += result.network_ms
        ws["compute_ms"] += result.compute_ms
        ws["deserialize_ms"] += result.deserialize_ms
        ws["total_bytes"] += result.payload_bytes

    # ---- attribute forwarding for controller.setup() mutations ----
    def __getattr__(self, name: str) -> Any:  # pragma: no cover (delegation)
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Safely check if _base exists and has the attribute
            base = getattr(self, "_base", None)
            if base is not None:
                return getattr(base, name)
            raise

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover (delegation)
        if name in {"_base", "algorithm", "dataset", "_pool"} or name.startswith("_"):
            return super().__setattr__(name, value)
        
        base = getattr(self, "_base", None)
        if base is not None and hasattr(base, name):
            setattr(base, name, value)
            return
        
        super().__setattr__(name, value)

    def forward(self, population: Any) -> Any:
        """Evaluate fitness using GPU-level distribution across all servers."""
        torch, _nn = _require_torch()
        if not isinstance(population, torch.Tensor):
            return self._base(population)

        n = int(population.shape[0])
        if n <= 0:
            return self._base(population)

        # Build DeviceWorkers from all servers' locked devices
        device_workers = self._build_device_workers()
        
        # Log worker distribution on first call
        if not getattr(self, "_logged", False) and device_workers:
            worker_desc = [f"{dw.server_id}:{dw.device}" for dw in device_workers]
            print(f"[INFO] Distributed fitness: {len(device_workers)} workers: {', '.join(worker_desc)}")
            self._logged = True
        
        # If no device workers, fall back to local
        if not device_workers:
            if self._current_iter == 0:
                print("[WARN] No device workers available, falling back to local computation")
            return self._base(population)
        
        # Limit workers to population size
        if len(device_workers) > n:
            device_workers = device_workers[:n]

        # Split population by worker weights (equal by default)
        weights = [max(1e-6, dw.weight) for dw in device_workers]
        total = sum(weights) or 1.0
        sizes = [max(1, int(n * (w / total))) for w in weights]
        
        # Distribute remaining items
        remainder = n - sum(sizes)
        if remainder > 0:
            for i in range(remainder):
                sizes[i % len(sizes)] += 1
        
        # Trim if overshot
        overshoot = sum(sizes) - n
        if overshoot > 0:
            for i in range(overshoot):
                idx = len(sizes) - 1 - (i % len(sizes))
                if sizes[idx] > 1:
                    sizes[idx] -= 1
        
        # Final guard
        if sum(sizes) != n:
            sizes = [n] + [0] * (len(sizes) - 1)

        chunks = [c for c in torch.split(population, sizes, dim=0) if int(c.shape[0]) > 0]
        fits: List[torch.Tensor] = [None] * len(chunks)  # Pre-allocate to preserve order
        iter_calls: List[DetailedCallResult] = []
        
        # Pre-build context ONCE (not per-worker)
        current_context = {}
        base = self._base
        if hasattr(base, "get_distributed_context"):
            try:
                ctx = base.get_distributed_context()
                if isinstance(ctx, dict):
                    current_context.update(ctx)
            except Exception:
                pass
        if hasattr(base, "genes_index") and "genes_index" not in current_context:
            gi = getattr(base, "genes_index", None)
            if gi is not None:
                current_context["genes_index"] = gi
        
        # Parallel dispatch using ThreadPoolExecutor
        # Now that server uses run_in_executor, we can dispatch ALL GPUs in parallel
        from concurrent.futures import ThreadPoolExecutor, Future
        
        def _remote_call(idx: int, dw: "DeviceWorker", pop_cpu: torch.Tensor) -> Tuple[int, "DetailedCallResult"]:
            temp_worker = Worker(server_id=dw.server_id, base_url=dw.base_url)
            result = self._pool.remote_fitness(
                temp_worker,
                algorithm=self.algorithm,
                dataset=self.dataset,
                population_cpu=pop_cpu,
                device=dw.device,
                extra_context=current_context,
            )
            return idx, result
        
        futures: List[Tuple[Future, int, "DeviceWorker", torch.Tensor]] = []
        # Track wall clock time for parallel dispatch
        t_wall_start = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=len(device_workers)) as executor:
            for idx, (dw, pop_chunk) in enumerate(zip(device_workers[:len(chunks)], chunks)):
                try:
                    pop_cpu = pop_chunk.detach().to("cpu")
                    
                    if dw.is_local:
                        # Local computation (synchronous, fast)
                        t0 = time.perf_counter()
                        fit_local = self._base(pop_chunk)
                        compute_ms = (time.perf_counter() - t0) * 1000.0
                        
                        result = DetailedCallResult(
                            fitness=fit_local.detach().to("cpu") if isinstance(fit_local, torch.Tensor) else fit_local,
                            worker_id=f"local:{dw.device}",
                            serialize_ms=0,
                            network_ms=0,
                            compute_ms=compute_ms,
                            deserialize_ms=0,
                            total_ms=compute_ms,
                            payload_bytes=0,
                            chunk_size=int(pop_chunk.shape[0]),
                        )
                        # Process local result immediately
                        self._detailed_calls.append(result)
                        iter_calls.append(result)
                        self._comm_total_ms += result.total_ms
                        self._comm_calls += 1
                        self._update_worker_stats(dw, result)
                        
                        fit_cpu = result.fitness
                        if not isinstance(fit_cpu, torch.Tensor):
                            fit_cpu = torch.tensor(fit_cpu)
                        fits[idx] = fit_cpu.to(population.device)
                    else:
                        # Remote computation - dispatch in parallel
                        future = executor.submit(_remote_call, idx, dw, pop_cpu)
                        futures.append((future, idx, dw, pop_chunk))
                except Exception as e:
                    print(f"[WARN] Device {dw.server_id}:{dw.device} prep failed: {e}, falling back to local")
                    fits[idx] = self._base(pop_chunk)
            
            # Wait for all remote futures
            for future, stored_idx, dw, pop_chunk in futures:
                try:
                    idx, result = future.result()
                    self._detailed_calls.append(result)
                    iter_calls.append(result)
                    self._comm_total_ms += result.total_ms
                    self._comm_calls += 1
                    self._update_worker_stats(dw, result)
                    
                    fit_cpu = result.fitness
                    if not isinstance(fit_cpu, torch.Tensor):
                        fit_cpu = torch.tensor(fit_cpu)
                    fits[idx] = fit_cpu.to(population.device)
                except Exception as e:
                    print(f"[WARN] Device {dw.server_id}:{dw.device} failed: {e}, falling back to local")
                    fits[stored_idx] = self._base(pop_chunk)
        
        # Update wall clock time
        iter_wall_ms = (time.perf_counter() - t_wall_start) * 1000.0
        self._comm_wall_clock_ms += iter_wall_ms

        # Record per-iteration stats
        if iter_calls:
            iter_cumulative = sum(c.total_ms for c in iter_calls)
            self._per_iter_stats.append({
                "iter": self._current_iter,
                "total_ms": iter_cumulative,  # Cumulative
                "wall_ms": iter_wall_ms,      # Wall clock
                "workers": [c.worker_id for c in iter_calls],
                "calls": len(iter_calls),
            })

        return torch.cat(fits, dim=0)

    def comm_stats(self) -> Dict[str, Any]:
        """Return legacy-compatible stats."""
        avg_ms = (self._comm_total_ms / self._comm_calls) if self._comm_calls else 0.0
        efficiency = (self._comm_total_ms / self._comm_wall_clock_ms) if self._comm_wall_clock_ms > 0 else 1.0
        return {
            "type": "http",
            "avg_ms": float(avg_ms),
            "total_ms": float(self._comm_total_ms),  # Cumulative
            "wall_clock_ms": float(self._comm_wall_clock_ms),  # Actual elapsed
            "parallel_efficiency": float(efficiency),  # Speedup from parallelism
            "calls": int(self._comm_calls),
        }

    def detailed_stats(self) -> Dict[str, Any]:
        """Return comprehensive communication statistics with full breakdown."""
        if not self._detailed_calls:
            return {"type": "http_detailed", "total_comm_ms": 0.0}
        
        # Aggregate phase totals
        total_serialize = sum(c.serialize_ms for c in self._detailed_calls)
        total_network = sum(c.network_ms for c in self._detailed_calls)
        total_compute = sum(c.compute_ms for c in self._detailed_calls)
        total_deserialize = sum(c.deserialize_ms for c in self._detailed_calls)
        total_bytes = sum(c.payload_bytes for c in self._detailed_calls)
        total_comm = sum(c.total_ms for c in self._detailed_calls)
        
        # Calculate averages
        n = len(self._detailed_calls)
        avg_total = total_comm / n if n else 0.0
        
        # Per-worker summary
        per_worker = {}
        for wid, ws in self._per_worker_stats.items():
            calls = ws["calls"]
            per_worker[wid] = {
                "calls": calls,
                "total_ms": ws["total_ms"],
                "avg_ms": ws["total_ms"] / calls if calls else 0.0,
                "serialize_ms": ws["serialize_ms"],
                "network_ms": ws["network_ms"],
                "compute_ms": ws["compute_ms"],
                "deserialize_ms": ws["deserialize_ms"],
                "total_bytes": ws["total_bytes"],
            }
        
        # Wall clock and parallel efficiency
        wall_clock_ms = self._comm_wall_clock_ms
        efficiency = (total_comm / wall_clock_ms) if wall_clock_ms > 0 else 1.0
        
        return {
            "type": "http_detailed",
            "total_comm_ms": total_comm,  # Cumulative
            "wall_clock_ms": wall_clock_ms,  # Actual elapsed
            "parallel_efficiency": efficiency,  # Speedup from parallelism
            "total_serialize_ms": total_serialize,
            "total_network_ms": total_network,
            "total_compute_ms": total_compute,
            "total_deserialize_ms": total_deserialize,
            "total_bytes": total_bytes,
            "calls": n,
            "avg_ms": avg_total,
            "per_worker": per_worker,
            "per_iteration": self._per_iter_stats,
        }

