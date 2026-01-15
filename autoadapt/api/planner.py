from __future__ import annotations
from typing import Callable, Optional, Tuple, Dict, Any, List

import time
import os
from .schemas import Plan
from ..route.perf_profiler import PerformanceProfiler
from ..route.strategy_router import StrategyRouter
from ..exec.executor import apply_plan_env


def _gpu_filter_from_snapshot(
    snapshot: Optional[Dict[str, Any]],
    gpu_busy_threshold: float,
    min_gpu_free_mb: int,
) -> tuple[Optional[List[int]], Optional[List[int]]]:
    if not isinstance(snapshot, dict):
        return None, None
    gpus = snapshot.get("gpus") or []
    if not isinstance(gpus, list):
        return None, None
    allowed: List[int] = []
    excluded: List[int] = []
    for g in gpus:
        if not isinstance(g, dict):
            continue
        gid = g.get("id")
        try:
            gid_int = int(gid)
        except Exception:
            continue
        util = g.get("gpu_util_percent")
        if util is None:
            load = g.get("load")
            if load is not None:
                try:
                    util = float(load) * 100.0
                except Exception:
                    util = None
        if util is not None:
            try:
                util = float(util)
            except Exception:
                util = None
        free_mb = g.get("free_mb")
        if free_mb is not None:
            try:
                free_mb = float(free_mb)
            except Exception:
                free_mb = None
        busy = False
        if util is not None and util >= gpu_busy_threshold:
            busy = True
        if free_mb is not None and free_mb < min_gpu_free_mb:
            busy = True
        if busy:
            excluded.append(gid_int)
        else:
            allowed.append(gid_int)
    if not gpus:
        return None, None
    return allowed, excluded


def _synthetic_executor(plan: Plan, iters: int) -> float:
    try:
        import torch
    except Exception:
        return float("inf")

    def _sync(dev: str) -> None:
        if dev.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _prepare(device: str):
        n = 1024 if device == "cpu" else 2048
        a = torch.randn((n, n), device=device)
        b = torch.randn((n, n), device=device)
        return a, b

    def _run_once(a, b) -> None:
        y = a @ b
        y = y.relu_()
        _ = float(y[0, 0].item())

    if plan.backend == "cpu":
        a, b = _prepare("cpu")
        start = time.perf_counter()
        for _ in range(iters):
            _run_once(a, b)
        return (time.perf_counter() - start) * 1000.0

    if not torch.cuda.is_available():
        return float("inf")

    if plan.backend == "cuda":
        dev_id = None
        if isinstance(plan.devices, list) and plan.devices:
            try:
                dev_id = int(plan.devices[0])
            except Exception:
                dev_id = None
        if dev_id is None:
            dev_id = 0
        device = f"cuda:{dev_id}"
        a, b = _prepare(device)
        _sync(device)
        start = time.perf_counter()
        for _ in range(iters):
            _run_once(a, b)
        _sync(device)
        return (time.perf_counter() - start) * 1000.0

    if plan.backend == "multi-gpu":
        devices = []
        if isinstance(plan.devices, list) and plan.devices:
            for d in plan.devices:
                try:
                    devices.append(int(d))
                except Exception:
                    continue
        if not devices:
            dev_count = int(plan.world_size or 2)
            devices = list(range(max(2, dev_count)))
        if torch.cuda.is_available():
            try:
                max_idx = torch.cuda.device_count() - 1
                devices = [d for d in devices if d <= max_idx]
            except Exception:
                pass
        if len(devices) < 2:
            return float("inf")
        
        # Optimization: Use a larger matrix for multi-GPU synthetic test to dilute overhead,
        # or use sync points properly. In GAPA, multi-GPU is typically process-parallel,
        # but here we use threads for a quick check. GIL is a bottleneck for tiny tasks.
        # We'll increase the iterations per thread slightly to ensure CUDA kernels overlap.
        # GIL is a major bottleneck for multi-threading small tasks.
        # We increase matrix size and iterations to ensure GPU computation outweighs Python overhead.
        def _prepare(d):
            # Use 1024x1024 to ensure measurable GPU time
            return torch.randn(1024, 1024, device=d), torch.randn(1024, 1024, device=d)

        mats = []
        for dev_id in devices:
            dev = f"cuda:{dev_id}"
            mats.append((dev, *_prepare(dev)))
        
        import threading
        # Ensure all GPUs are ready
        for dev, _, _ in mats:
            _sync(dev)
        
        start = time.perf_counter()
        # Larger iters to dilute thread startup costs
        iters_heavy = iters * 2 
        
        def _thread_worker(dev, a, b, i_count):
            for _ in range(i_count):
                _run_once(a, b)
            _sync(dev)

        threads = []
        for dev, a, b in mats:
            t = threading.Thread(target=_thread_worker, args=(dev, a, b, iters_heavy), daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
            
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        # Normalize back to requested iters
        return elapsed_ms * (iters / iters_heavy)
    
    # CPU case
    if plan.backend == "cpu":
        # CPU benchmark also uses a larger matrix for stability
        a = torch.randn(512, 512, device="cpu")
        b = torch.randn(512, 512, device="cpu")
        start = time.perf_counter()
        for _ in range(iters):
            torch.mm(a, b)
        return (time.perf_counter() - start) * 1000.0

    return float("inf")


def _tpe_select(
    plans: List[Plan],
    executor: Callable[[Plan, int], float],
    warmup_iters: int,
    trials: int,
    gamma: float = 0.25,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> Plan:
    if not plans:
        raise ValueError("no candidate plans")
    trials = max(1, int(trials))
    warmup_iters = max(1, int(warmup_iters))
    gamma = min(max(float(gamma), 0.1), 0.8)

    ids = [f"{p.backend}:{','.join(str(d) for d in p.devices)}" for p in plans]
    priors = {}
    for pid, plan in zip(ids, plans):
        # We use a blend of estimated time and priority
        t = max(plan.estimated_time_ms, 1e-6)
        priors[pid] = 1.0 / t
    
    prior_sum = sum(priors.values()) or 1.0
    for k in priors:
        priors[k] = priors[k] / prior_sum

    observed: Dict[str, List[float]] = {}
    
    # Heuristic: Always test the static "best" plan first to establish a strong baseline.
    static_best_idx = 0
    min_est = float('inf')
    for i, p in enumerate(plans):
        if p.estimated_time_ms < min_est:
            min_est = p.estimated_time_ms
            static_best_idx = i

    def _pick_next() -> Plan:
        tested_count = len(observed)
        # Forced exploration of static top-2 if we have enough trials
        if tested_count < min(trials, 2):
            sorted_indices = sorted(range(len(plans)), key=lambda i: plans[i].estimated_time_ms)
            for idx in sorted_indices:
                if ids[idx] not in observed:
                    return plans[idx]

        tested = set(observed.keys())
        untested = [p for p, pid in zip(plans, ids) if pid not in tested]
        
        # Before we have enough data, act greedily on priors/estimates
        if len(observed) < max(2, int(trials * gamma)):
            if untested:
                untested.sort(key=lambda p: p.estimated_time_ms)
                return untested[0]
            return plans[0]

        all_obs = [(pid, min(vals)) for pid, vals in observed.items() if vals]
        all_obs.sort(key=lambda kv: kv[1])
        
        cutoff_idx = max(1, int(len(all_obs) * gamma))
        good_set = {pid for pid, _ in all_obs[:cutoff_idx]}
        bad_set = {pid for pid, _ in all_obs[cutoff_idx:]}
        
        scores = []
        for plan, pid in zip(plans, ids):
            if pid in tested:
                # We can re-test to refine, but usually prioritize untested
                continue
            l = (1 if pid in good_set else 0) + priors[pid]
            g = (1 if pid in bad_set else 0) + priors[pid]
            scores.append((l / g, plan))
            
        if scores:
            scores.sort(key=lambda x: x[0], reverse=True)
            return scores[0][1]
        
        # If all tested, pick best observed
        best_pid = all_obs[0][0]
        return next(p for p, pid in zip(plans, ids) if pid == best_pid)

    best_plan = plans[static_best_idx]
    best_ms = float("inf")
    
    if progress_cb:
        try:
            progress_cb(0, trials, "running")
        except Exception:
            pass
            
    for i in range(trials):
        plan = _pick_next()
        try:
            ms = float(executor(plan, warmup_iters))
        except Exception:
            ms = float("inf")
            
        pid = f"{plan.backend}:{','.join(str(d) for d in plan.devices)}"
        observed.setdefault(pid, []).append(ms)
        
        if ms < best_ms:
            best_ms = ms
            best_plan = plan
            
        if progress_cb:
            try:
                progress_cb(i + 1, trials, "running")
            except Exception:
                pass

    # Final "Static Anchor" Logic:
    # We want to catch cases where TPE measurements are noisy or skewed (like GIL issues),
    # leading it to pick a plan that human intuition and static profiling know is worse.
    static_best_plan = plans[static_best_idx]
    
    # If the TPE winner's static estimate is > 50% worse than the global static best,
    # and the measured time for the winner isn't MUCH better than the static best's measured time.
    if best_plan != static_best_plan:
        static_best_id = ids[static_best_idx]
        static_best_meas = min(observed.get(static_best_id, [float('inf')]))
        
        # Heuristic: If TPE winner is statically predicted to be much slower than the best,
        # but "measured" faster by a thin margin or due to noisy static-best measurements.
        # We only override static best if the measurement is "convincing".
        ratio_stat = best_plan.estimated_time_ms / max(1e-6, static_best_plan.estimated_time_ms)
        
        # If static best is predicted to be 1.3x faster (e.g. 5.6ms vs 7.3ms),
        # but measured slightly slower, we fallback to static.
        if ratio_stat > 1.25 and best_ms >= static_best_plan.estimated_time_ms * 0.9:
            best_plan = static_best_plan
            best_ms = static_best_meas if static_best_meas != float('inf') else static_best_plan.estimated_time_ms
            best_plan.reason += " | fallback_to_static(TPE_noise_guard)"

    best_plan.estimated_time_ms = max(1e-3, min(best_plan.estimated_time_ms, best_ms))
    best_plan.reason += f" | tpe_trials={trials} warmup={warmup_iters}"
    if progress_cb:
        try:
            progress_cb(trials, trials, "done")
        except Exception:
            pass
    return best_plan


def StrategyPlan(
        fitness: Optional[Callable[..., None]] = None,
        warmup: int = 0,
        objective: str = 'time',
        multi_gpu: bool = True,
        fitness_args: Optional[Tuple[Any, ...]] = None,
        fitness_kwargs: Optional[Dict[str, Any]] = None,
        resource_snapshot: Optional[Dict[str, Any]] = None,
        gpu_busy_threshold: Optional[float] = None,
        min_gpu_free_mb: Optional[int] = None,
        tpe_trials: Optional[int] = None,
        tpe_warmup: Optional[int] = None,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> Plan:
    """生成当前环境的资源执行方案。

    - 若未提供 ``fitness``，仅基于静态性能画像选择方案；
    - 若提供 ``fitness`` 且 ``warmup`` > 0，则在候选方案上执行指定次数的
      ``fitness`` 调用以做热身评估，并返回实测耗时最短的方案。
      可以通过 ``fitness_args`` 与 ``fitness_kwargs`` 为 ``fitness`` 传参。
    """
    # Extract GPU utilization mapping from snapshot
    util_map = {}
    if resource_snapshot and "gpus" in resource_snapshot:
        for g in resource_snapshot["gpus"]:
            gid = g.get("id")
            util = g.get("gpu_util_percent")
            if gid is not None and util is not None:
                try:
                    util_map[int(gid)] = float(util)
                except Exception:
                    pass

    prof = PerformanceProfiler(quick=True).profile(gpu_utils=util_map)
    if gpu_busy_threshold is None:
        gpu_busy_threshold = float(os.getenv("GAPA_STRATEGY_GPU_BUSY", "60") or 60)
    if min_gpu_free_mb is None:
        min_gpu_free_mb = int(os.getenv("GAPA_STRATEGY_MIN_FREE_MB", "1024") or 1024)
    avail_gpus, excluded_gpus = _gpu_filter_from_snapshot(
        resource_snapshot, float(gpu_busy_threshold), int(min_gpu_free_mb)
    )
    router = StrategyRouter(prof, available_gpus=avail_gpus)

    class _SyntheticWL:
        """Synthetic workload used when the caller does not supply one.

        The previous implementation used a 1-edge dummy graph which made the
        router heavily favour the CPU due to fixed kernel launch overheads. In
        practice, graph workloads are orders of magnitude larger, so we model a
        moderate-size graph here to let the performance model reflect the
        relative throughput of CPU vs GPU correctly.
        """

        # A modest graph: 50k nodes, 5M edges and 50 steps is sufficient to
        # reveal GPU advantages without being unrealistically large.  The values
        # only affect the static estimation and incur no runtime cost.
        n_nodes = 50_000
        n_edges = 5_000_000
        steps = 50
        batch_individuals = 1

    wl = _SyntheticWL()

    tpe_trials = int(tpe_trials or os.getenv("GAPA_TPE_TRIALS", "6") or 6)
    tpe_gamma = float(os.getenv("GAPA_TPE_GAMMA", "0.25") or 0.25)
    tpe_warmup = int(tpe_warmup or os.getenv("GAPA_TPE_WARMUP_ITERS", "1") or 1)
    if warmup > 0:
        tpe_warmup = int(warmup)

    if fitness is not None:
        args = fitness_args or ()
        kwargs = fitness_kwargs or {}

        def executor(p: Plan, iters: int) -> float:
            apply_plan_env(p)
            start = time.perf_counter()
            for _ in range(iters):
                fitness(*args, **kwargs)
            return (time.perf_counter() - start) * 1000.0
    else:
        def executor(p: Plan, iters: int) -> float:
            apply_plan_env(p)
            return _synthetic_executor(p, iters)

    candidates = router.candidate_plans(wl, objective=objective, multi_gpu=multi_gpu)

    if os.getenv("GAPA_TPE_DISABLE", "0") == "1":
        # Skip TPE and use static best
        static_best = min(candidates, key=lambda p: p.estimated_time_ms)
        static_best.reason += " | TPE=disabled"
        return static_best

    plan = _tpe_select(candidates, executor, tpe_warmup, tpe_trials, gamma=tpe_gamma, progress_cb=progress_cb)
    
    # Compare TPE result with static best to avoid TPE noise/overhead
    static_best = router.route(wl, objective=objective, multi_gpu=multi_gpu, power_cap_w=None)
    
    # Use TPE if it's meaningfully faster (15% threshold), otherwise use static
    # This prevents TPE from picking slightly-better-measured plans that have worse static profiles
    tpe_faster_ratio = static_best.estimated_time_ms / max(1e-6, plan.estimated_time_ms)
    
    if tpe_faster_ratio >= 1.15:  # TPE is at least 15% faster
        final_plan = plan
        final_plan.reason += " | tpe_winner"
    else:
        final_plan = static_best
        final_plan.reason += f" | static_winner(tpe_ratio={tpe_faster_ratio:.2f})"

    if avail_gpus is not None:
        try:
            final_plan.notes = (
                (final_plan.notes + " " if final_plan.notes else "")
                + f"gpu_filter=allowed:{avail_gpus} excluded:{excluded_gpus or []}"
            )
            final_plan.reason = (final_plan.reason + f" | 过滤繁忙GPU:{excluded_gpus or []}").strip()
        except Exception:
            pass
    return final_plan


def StrategyCompare(
    objective: str = "time",
    multi_gpu: bool = True,
    warmup_iters: int = 0,
    resource_snapshot: Optional[Dict[str, Any]] = None,
    gpu_busy_threshold: Optional[float] = None,
    min_gpu_free_mb: Optional[int] = None,
) -> Dict[str, Any]:
    """Return best plan and key candidates for explaining the decision in UI."""
    # Extract GPU utilization mapping from snapshot
    util_map = {}
    if resource_snapshot and "gpus" in resource_snapshot:
        for g in resource_snapshot["gpus"]:
            gid = g.get("id")
            util = g.get("gpu_util_percent")
            if gid is not None and util is not None:
                try:
                    util_map[int(gid)] = float(util)
                except Exception:
                    pass

    prof = PerformanceProfiler(quick=True).profile(gpu_utils=util_map)
    if gpu_busy_threshold is None:
        gpu_busy_threshold = float(os.getenv("GAPA_STRATEGY_GPU_BUSY", "60") or 60)
    if min_gpu_free_mb is None:
        min_gpu_free_mb = int(os.getenv("GAPA_STRATEGY_MIN_FREE_MB", "1024") or 1024)
    avail_gpus, excluded_gpus = _gpu_filter_from_snapshot(
        resource_snapshot, float(gpu_busy_threshold), int(min_gpu_free_mb)
    )
    router = StrategyRouter(prof, available_gpus=avail_gpus)

    class _SyntheticWL:
        n_nodes = 50_000
        n_edges = 5_000_000
        steps = 50
        batch_individuals = 1

    wl = _SyntheticWL()
    candidates = router._candidates(wl, objective, multi_gpu, power_cap_w=None)  # type: ignore[attr-defined]
    items = []
    for tag, plan in candidates:
        items.append({"tag": tag, "plan": plan.to_dict()})
    # Use public route() for best selection (same behavior, includes calibration handling)
    best_plan = router.route(wl, objective=objective, multi_gpu=multi_gpu, power_cap_w=None)

    measured: Dict[str, Any] = {}
    warmup_iters = int(warmup_iters or 0)
    if warmup_iters > 0:
        measured = _warmup_benchmark(candidates, warmup_iters)

    return {
        "best": best_plan.to_dict(),
        "candidates": items,
        "profile": {"has_cuda": getattr(prof.device, "has_cuda", False), "gpus": prof.device.gpus},
        "gpu_filter": {"allowed": avail_gpus, "excluded": excluded_gpus} if avail_gpus is not None else None,
        "warmup": measured,
    }


def _warmup_benchmark(candidates: List[tuple], iters: int) -> Dict[str, Any]:
    """Measure real latency curves for CPU vs GPU candidates (best-effort).

    Returns:
      {
        "series": [{"label": "CPU", "ms": [...]}, {"label": "GPU(0)", "ms":[...]}, {"label":"Multi-GPU(3)","ms":[...]}],
        "avg_ms": {"CPU": 12.3, "GPU(0)": 3.4}
      }
    """
    try:
        import torch
    except Exception:
        return {"error": "torch not available"}

    def _sync(device: str) -> None:
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    def _prepare_mats(device: str):
        # Use GEMM for robust cross-platform timing; sparse CUDA support can vary by build.
        n = 1024 if device == "cpu" else 2048
        a = torch.randn((n, n), device=device)
        b = torch.randn((n, n), device=device)
        return a, b

    def _run_once(a, b) -> None:
        y = a @ b
        y = y.relu_()
        _ = float(y[0, 0].item())

    def measure_cpu() -> Optional[List[float]]:
        device = "cpu"
        a, b = _prepare_mats(device)
        ms: List[float] = []
        for _ in range(iters):
            start = time.perf_counter()
            _run_once(a, b)
            ms.append((time.perf_counter() - start) * 1000.0)
        return ms

    def measure_single_gpu(label: str) -> Optional[List[float]]:
        if not torch.cuda.is_available():
            return None
        device = "cuda:0"
        a, b = _prepare_mats(device)
        ms: List[float] = []
        for _ in range(iters):
            _sync(device)
            start = time.perf_counter()
            _run_once(a, b)
            _sync(device)
            ms.append((time.perf_counter() - start) * 1000.0)
        try:
            del a, b
            torch.cuda.empty_cache()
        except Exception:
            pass
        return ms

    def measure_multi_gpu(num_devices: int) -> Optional[List[float]]:
        if not torch.cuda.is_available():
            return None
        num_devices = max(2, int(num_devices))
        mats = []
        for i in range(num_devices):
            dev = f"cuda:{i}"
            mats.append((dev, *_prepare_mats(dev)))

        import threading

        ms: List[float] = []
        for _ in range(iters):
            for dev, *_ in mats:
                _sync(dev)
            start = time.perf_counter()

            threads = []
            for dev, a, b in mats:
                t = threading.Thread(target=_run_once, args=(a, b), daemon=True)
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

            for dev, *_ in mats:
                _sync(dev)
            ms.append((time.perf_counter() - start) * 1000.0)
        try:
            for _dev, a, b in mats:
                del a, b
            torch.cuda.empty_cache()
        except Exception:
            pass
        return ms

    series: List[Dict[str, Any]] = []
    avg_ms: Dict[str, float] = {}
    errors: List[str] = []

    # Pick CPU + best single GPU candidates only (clear CPU vs GPU comparison).
    cpu_plan = next((p for _tag, p in candidates if isinstance(p, Plan) and p.backend == "cpu"), None)
    gpu_plan = next((p for _tag, p in candidates if isinstance(p, Plan) and p.backend == "cuda"), None)
    mgpu_plan = next((p for _tag, p in candidates if isinstance(p, Plan) and p.backend == "multi-gpu"), None)

    if cpu_plan is not None:
        try:
            ms = measure_cpu()
            if ms:
                series.append({"label": "CPU", "ms": ms})
                avg_ms["CPU"] = sum(ms) / max(1, len(ms))
        except Exception as exc:
            errors.append(f"cpu: {exc}")

    if gpu_plan is not None:
        idx = (gpu_plan.devices[0] if gpu_plan.devices else 0)
        label = f"GPU({idx})"
        try:
            apply_plan_env(gpu_plan)
            ms = measure_single_gpu(label)
            if ms:
                series.append({"label": label, "ms": ms})
                avg_ms[label] = sum(ms) / max(1, len(ms))
        except Exception as exc:
            errors.append(f"gpu: {exc}")

    if mgpu_plan is not None:
        try:
            apply_plan_env(mgpu_plan)
            n = len(mgpu_plan.devices) if mgpu_plan.devices else int(mgpu_plan.world_size or 2)
            ms = measure_multi_gpu(n)
            label = f"Multi-GPU({n})"
            if ms:
                series.append({"label": label, "ms": ms})
                avg_ms[label] = sum(ms) / max(1, len(ms))
        except Exception as exc:
            errors.append(f"multi-gpu: {exc}")

    if not series:
        return {"error": "no measurable candidates", "details": errors}

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return {"series": series, "avg_ms": avg_ms, "iters": iters, "errors": errors}


def DistributedStrategyPlan(
    *,
    server_resources: Dict[str, Any],
    server_plans: Optional[Dict[str, Plan]] = None,
    per_server_gpus: int = 1,
    min_gpu_free_mb: int = 1024,
    gpu_busy_threshold: float = 85.0,
) -> Dict[str, Any]:
    """Heuristic distributed plan across servers (merge single-server plans/resources).

    Returns:
      {
        "backend": "distributed",
        "servers": {sid: {"backend": ..., "devices": [...], "reason": "..."}},
        "devices_by_server": {sid: [...]},
        "notes": "...",
      }
    """
    per_server_gpus = max(1, int(per_server_gpus or 1))
    min_gpu_free_mb = max(0, int(min_gpu_free_mb or 0))
    gpu_busy_threshold = float(gpu_busy_threshold or 85.0)

    def _rank_gpus(snap: Dict[str, Any]) -> list[int]:
        gpus = snap.get("gpus") or []
        scored = []
        for g in gpus:
            try:
                gid = int(g.get("id"))
            except Exception:
                continue
            free_mb = g.get("free_mb")
            util = g.get("gpu_util_percent")
            if free_mb is None:
                free_mb = 0
            if util is None:
                util = 0
            score = float(free_mb) - (float(util) * 10.0)
            scored.append((gid, float(free_mb), float(util), score))
        if not scored:
            return []
        # prefer low util and high free_mb
        scored.sort(key=lambda x: (x[3], x[1]), reverse=True)
        picks = []
        for gid, free_mb, util, _ in scored:
            if free_mb >= min_gpu_free_mb and util <= gpu_busy_threshold:
                picks.append(gid)
        return picks

    servers_out: Dict[str, Any] = {}
    devices_by_server: Dict[str, Any] = {}
    for sid, snap in server_resources.items():
        plan = server_plans.get(sid) if server_plans else None
        if plan and plan.backend in ("cuda", "multi-gpu"):
            devices = list(plan.devices or [])
            backend = plan.backend
            reason = "strategy_plan"
        else:
            ranked = _rank_gpus(snap or {})
            devices = ranked[:per_server_gpus]
            backend = "cuda" if len(devices) == 1 else ("multi-gpu" if len(devices) > 1 else "cpu")
            reason = "resource_snapshot"
        servers_out[sid] = {"backend": backend, "devices": devices, "reason": reason}
        if devices:
            devices_by_server[sid] = devices

    return {
        "backend": "distributed",
        "servers": servers_out,
        "devices_by_server": devices_by_server,
        "notes": "heuristic: per-server plan merged; not a global optimizer",
    }
