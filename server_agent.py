#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
server_agent.py
---------------
一个独立的 FastAPI HTTP 服务，用于：
1) 实时监控本机资源（CPU/内存/GPU）
2) 接收 GA 分析任务并后台异步执行，提供日志与结果轮询接口

部署：uvicorn server_agent:app --host 0.0.0.0 --port 7777
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
import traceback
import uuid
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from server.agent_monitor import resources_payload
from server.agent_state import TaskState, start_consumer
from server.fitness_protocol import dumps as fitness_dumps, loads as fitness_loads
from server.fitness_worker import compute_fitness_batch
from server.ga_worker import ga_worker, select_run_mode
from server.resource_lock import LOCK_MANAGER



app = FastAPI(title="GAPA Server Agent", version="0.1.0")

def check_dependencies():
    """Check for essential packages and print warnings if missing."""
    missing = []
    try:
        import psutil
    except ImportError:
        missing.append("psutil (REQUIRED for CPU/Memory monitoring)")
    
    try:
        import pynvml
    except ImportError:
        missing.append("pynvml (REQUIRED for GPU monitoring)")

    try:
        import torch
    except ImportError:
        missing.append("torch (REQUIRED for GA execution)")

    if missing:
        print("\n" + "!" * 60)
        print("  WARNING: Missing Dependencies Detected!")
        for item in missing:
            print(f"  - {item}")
        print("\n  Please install missing packages via:")
        print("  pip install " + " ".join([m.split()[0] for m in missing]))
        print("!" * 60 + "\n")
    else:
        print("\n[INFO] All critical dependencies (psutil, pynvml, torch) are installed.\n")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/resources")
def api_resources() -> Dict[str, Any]:
    return resources_payload()


TASK = TaskState()
STRATEGY_PROGRESS: Dict[str, Dict[str, Any]] = {}


def _summarize_warmup_result(result: dict | None) -> dict:
    if not isinstance(result, dict):
        return {"summary": {}, "per_iter_ms": []}
    timing = result.get("timing") or {}
    summary = {
        "iter_seconds": timing.get("iter_seconds"),
        "iter_avg_ms": timing.get("iter_avg_ms"),
        "throughput_ips": timing.get("throughput_ips"),
        "iterations": (result.get("hyperparams") or {}).get("iterations"),
        "pop_size": (result.get("hyperparams") or {}).get("pop_size"),
        "mode": (result.get("selected") or {}).get("mode"),
        "devices": (result.get("selected") or {}).get("devices"),
        "remote_servers": (result.get("selected") or {}).get("remote_servers"),
    }
    points = result.get("points") or []
    per_iter_ms = []
    last = None
    for item in points:
        elapsed = item.get("elapsed_s")
        if elapsed is None:
            continue
        if last is not None:
            per_iter_ms.append((float(elapsed) - float(last)) * 1000.0)
        last = float(elapsed)
    return {"summary": summary, "per_iter_ms": per_iter_ms}


def _run_ga_warmup_local(payload: Dict[str, Any]) -> Dict[str, Any] | tuple[Dict[str, Any], int]:
    algorithm = str(payload.get("algorithm") or "")
    dataset = str(payload.get("dataset") or "")
    if not algorithm or not dataset:
        return {"error": "algorithm and dataset are required"}, 400
    iterations = int(payload.get("iterations") or payload.get("warmup_iters") or 2)
    crossover_rate = float(payload.get("crossover_rate") or payload.get("pc") or 0.8)
    mutate_rate = float(payload.get("mutate_rate") or payload.get("pm") or 0.2)
    selected = select_run_mode(payload.get("mode"), payload.get("devices"))
    remote_servers = payload.get("remote_servers") or payload.get("allowed_server_ids")
    if remote_servers is not None:
        selected["remote_servers"] = remote_servers
    timeout_s = float(payload.get("timeout_s", 180) or 180)

    with TASK.lock:
        if TASK.state == "running":
            return {"error": "Agent is busy running a GA task"}, 409

    task_id = str(uuid.uuid4())
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(
        target=_ga_entry,
        args=(task_id, algorithm, dataset, iterations, crossover_rate, mutate_rate, selected, q),
    )
    proc.start()

    logs = []
    result = None
    state = "running"
    error = None
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            evt = q.get(timeout=0.2)
        except Exception:
            if not proc.is_alive():
                break
            continue
        if not isinstance(evt, dict):
            continue
        etype = evt.get("type")
        if etype == "log":
            logs.append(evt.get("line"))
        elif etype == "result":
            result = evt.get("result")
        elif etype == "state":
            state = evt.get("state") or state
            error = evt.get("error") or error
        if state in ("completed", "error") and result is not None:
            break

    if proc.is_alive():
        try:
            proc.terminate()
            proc.join(timeout=1.0)
        except Exception:
            pass
        state = "timeout"
        error = error or "warmup timeout"
    else:
        try:
            proc.join(timeout=0.2)
        except Exception:
            pass

    summary = _summarize_warmup_result(result)
    return {
        "task_id": task_id,
        "state": state,
        "summary": summary["summary"],
        "per_iter_ms": summary["per_iter_ms"],
        "comm": (result or {}).get("comm"),
        "logs": logs[-200:],
        "error": error,
    }
def _ga_entry(
    task_id: str,
    algorithm: str,
    dataset: str,
    iterations: int,
    crossover_rate: float,
    mutate_rate: float,
    selected: Dict[str, Any],
    q: Any,
) -> None:
    if os.name == "posix":
        try:
            os.setsid()
        except Exception:
            pass
    ga_worker(task_id, algorithm, dataset, iterations, crossover_rate, mutate_rate, selected, q)


@app.post("/api/analysis/start")
def api_analysis_start(payload: Dict[str, Any]) -> Dict[str, Any]:
    algorithm = str(payload.get("algorithm") or "ga")
    dataset = str(payload.get("dataset") or "")
    iterations = int(payload.get("iterations") or payload.get("max_generation") or 20)
    crossover_rate = float(payload.get("crossover_rate") or payload.get("pc") or 0.8)
    mutate_rate = float(payload.get("mutate_rate") or payload.get("pm") or 0.2)
    selected = select_run_mode(payload.get("mode"), payload.get("devices"))
    release_lock_on_finish = bool(payload.get("release_lock_on_finish", True))

    with TASK.lock:
        if TASK.state == "running":
            raise HTTPException(status_code=409, detail="A task is already running")
        task_id = str(uuid.uuid4())
        TASK.reset_for_new_task(task_id)
        TASK.release_lock_on_finish = release_lock_on_finish
        TASK.append_log(f"[INFO] 运行模式: {selected.get('mode')} devices={selected.get('devices')}")

        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        proc = ctx.Process(
            target=_ga_entry,
            args=(task_id, algorithm, dataset, iterations, crossover_rate, mutate_rate, selected, q),
        )
        TASK.queue = q
        TASK.process = proc
        proc.start()
        start_consumer(TASK)

    return {"task_id": task_id, "status": "started"}


@app.post("/api/ga_warmup")
def api_ga_warmup(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run a short real GA warmup to measure actual latency/throughput."""
    result = _run_ga_warmup_local(payload)
    if isinstance(result, tuple):
        body, code = result
        raise HTTPException(status_code=code, detail=body.get("error") or "warmup failed")
    return result


@app.get("/api/analysis/status")
def api_analysis_status() -> Dict[str, Any]:
    with TASK.lock:
        return {
            "task_id": TASK.task_id,
            "state": TASK.state,
            "progress": TASK.progress,
            "logs": list(TASK.logs),
            "result": TASK.result,
            "error": TASK.error,
        }

@app.post("/api/analysis/stop")
def api_analysis_stop() -> Dict[str, Any]:
    with TASK.lock:
        proc = TASK.process
        if TASK.state != "running" or not proc or not proc.is_alive():
            TASK.state = "idle" if TASK.state != "error" else TASK.state
            return {"status": "idle", "task_id": TASK.task_id}

        pid = proc.pid
        TASK.append_log("[WARN] stop requested, terminating task...")

    # terminate outside lock
    try:
        def kill_process_tree(pid: int, sig: int = signal.SIGTERM):
            try:
                import psutil
                parent = psutil.Process(pid)
                children = parent.children(recursive=True)
                for child in children:
                    try:
                        child.send_signal(sig)
                    except psutil.NoSuchProcess:
                        pass
                parent.send_signal(sig)
            except Exception:
                # Fallback if psutil fails or pid is gone
                if os.name == "posix":
                    try:
                        os.killpg(pid, sig)
                    except Exception:
                        try:
                            os.kill(pid, sig)
                        except Exception:
                            pass
                else:
                    try:
                        proc.terminate() if sig == signal.SIGTERM else proc.kill()
                    except Exception:
                        pass

        if pid:
            kill_process_tree(pid, signal.SIGTERM)
        
        proc.join(timeout=2.0)
        if proc.is_alive():
            print(f"[WARN] Process {pid} still alive after SIGTERM, sending SIGKILL...")
            if pid:
                kill_process_tree(pid, signal.SIGKILL)
            proc.join(timeout=1.0)
    finally:
        with TASK.lock:
            TASK.state = "idle"
            TASK.progress = 0
            TASK.error = "stopped"
            TASK.append_log("[INFO] task stopped.")

    return {"status": "stopped", "task_id": TASK.task_id}

@app.post("/api/fitness/batch")
async def api_fitness_batch(req: Request) -> Response:
    """Compute fitness for a population chunk (used by MNM distributed fitness mode).
    
    The caller can specify a device in the request. If not specified, uses the first
    locked GPU or falls back to default.
    """
    with TASK.lock:
        # Keep it simple: avoid fighting for GPU/CPU when a full GA task is running.
        if TASK.state == "running":
            raise HTTPException(status_code=409, detail="Agent is busy running a GA task")

    raw = await req.body()
    try:
        msg = fitness_loads(raw)
        algorithm = str(msg.get("algorithm") or "")
        dataset = str(msg.get("dataset") or "")
        population = msg.get("population")
        
        # Check if caller specified a device
        requested_device = msg.get("device")
        
        if requested_device:
            # Use the caller-specified device
            device = str(requested_device)
        else:
            # Fallback: use first locked device or default
            device = None
            try:
                from server.resource_lock import LOCK_MANAGER
                lock_status = LOCK_MANAGER.status()
                if lock_status.get("active") and lock_status.get("devices"):
                    device = f"cuda:{lock_status['devices'][0]}"
            except Exception:
                pass
            if not device:
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        t0 = time.perf_counter()
        
        # Extract extra context (e.g. genes_index for SixDST)
        extra_context = {}
        if "genes_index" in msg:
            extra_context["genes_index"] = msg["genes_index"]
        
        # Run blocking GPU computation in thread pool to allow concurrent requests
        import asyncio
        from functools import partial
        loop = asyncio.get_event_loop()
        compute_func = partial(
            compute_fitness_batch,
            algorithm=algorithm,
            dataset=dataset,
            population_cpu=population,
            device=device,
            extra_context=extra_context,
        )
        fitness, meta = await loop.run_in_executor(None, compute_func)
        compute_ms = (time.perf_counter() - t0) * 1000.0
        
        meta["device"] = device
        
        # Include compute_ms in response for timing breakdown
        out = fitness_dumps({
            "fitness": fitness,
            "meta": meta,
            "compute_ms": compute_ms,
        })
        return Response(content=out, media_type="application/octet-stream")
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[ERROR] fitness_batch failed: {exc}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/resource_lock")
def api_resource_lock(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Lock local resources by allocating warmup + memory on best device."""
    with TASK.lock:
        if TASK.state == "running":
            raise HTTPException(status_code=409, detail="Agent is busy running a GA task")
    duration_s = float(payload.get("duration_s", 600) or 600)
    warmup_iters = int(payload.get("warmup_iters", 2) or 2)
    mem_mb = int(payload.get("mem_mb", 1024) or 1024)
    strict_idle = bool(payload.get("strict_idle", False))
    devices = payload.get("devices")
    if devices is not None and not isinstance(devices, list):
        devices = [devices]
    try:
        info = LOCK_MANAGER.lock(duration_s=duration_s, warmup_iters=warmup_iters, mem_mb=mem_mb, devices=devices, strict_idle=strict_idle)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return info


@app.post("/api/resource_lock/release")
def api_resource_lock_release() -> Dict[str, Any]:
    try:
        info = LOCK_MANAGER.release(reason="manual")
        try:
            from server.fitness_worker import clear_contexts

            clear_contexts()
        except Exception:
            pass
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return info


@app.get("/api/resource_lock/status")
def api_resource_lock_status() -> Dict[str, Any]:
    try:
        return LOCK_MANAGER.status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/strategy_plan")
def api_strategy_plan(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compute StrategyPlan on this server (static profiling)."""
    try:
        from autoadapt import StrategyPlan as _StrategyPlan  # type: ignore
    except Exception:
        _StrategyPlan = None
    if _StrategyPlan is None:
        raise HTTPException(status_code=500, detail="StrategyPlan not available on this host")
    warmup = int(payload.get("warmup", 0) or 0)
    objective = str(payload.get("objective") or "time")
    multi_gpu = bool(payload.get("multi_gpu", True))
    gpu_busy_threshold = payload.get("gpu_busy_threshold")
    min_gpu_free_mb = payload.get("min_gpu_free_mb")
    tpe_trials = payload.get("tpe_trials")
    tpe_warmup = payload.get("tpe_warmup")
    progress_id = payload.get("progress_id")
    algorithm = payload.get("algorithm")
    snap = resources_payload()
    if progress_id:
        total_trials = int(tpe_trials or os.getenv("GAPA_TPE_TRIALS", "6") or 6)
        STRATEGY_PROGRESS[progress_id] = {"current": 0, "total": total_trials, "status": "running"}
    def _progress_cb(cur: int, total: int, status: str) -> None:
        if not progress_id:
            return
        STRATEGY_PROGRESS[progress_id] = {"current": int(cur), "total": int(total), "status": str(status)}
    plan = _StrategyPlan(
        fitness=None,
        warmup=warmup,
        objective=objective,
        multi_gpu=multi_gpu,
        resource_snapshot=snap,
        gpu_busy_threshold=gpu_busy_threshold,
        min_gpu_free_mb=min_gpu_free_mb,
        tpe_trials=tpe_trials,
        tpe_warmup=tpe_warmup,
        progress_cb=_progress_cb if progress_id else None,
    )
    if algorithm:
        try:
            plan.notes = (plan.notes + " " if plan.notes else "") + f"algorithm={algorithm}"
        except Exception:
            pass
    return plan.to_dict()


@app.get("/api/strategy_plan/progress")
def api_strategy_plan_progress(progress_id: str) -> Dict[str, Any]:
    if not progress_id:
        raise HTTPException(status_code=400, detail="progress_id required")
    return STRATEGY_PROGRESS.get(progress_id, {"status": "unknown", "current": 0, "total": 0})


@app.post("/api/strategy_compare")
def api_strategy_compare(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return candidate plan comparison for UI explanation."""
    try:
        from autoadapt import StrategyCompare as _StrategyCompare  # type: ignore
    except Exception:
        _StrategyCompare = None
    if _StrategyCompare is None:
        raise HTTPException(status_code=500, detail="StrategyCompare not available on this host")
    objective = str(payload.get("objective") or "time")
    multi_gpu = bool(payload.get("multi_gpu", True))
    warmup_iters = int(payload.get("warmup_iters", 0) or 0)
    gpu_busy_threshold = payload.get("gpu_busy_threshold")
    min_gpu_free_mb = payload.get("min_gpu_free_mb")
    snap = resources_payload()
    return _StrategyCompare(
        objective=objective,
        multi_gpu=multi_gpu,
        warmup_iters=warmup_iters,
        resource_snapshot=snap,
        gpu_busy_threshold=gpu_busy_threshold,
        min_gpu_free_mb=min_gpu_free_mb,
    )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("GAPA_AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("GAPA_AGENT_PORT", "4467"))
    check_dependencies()
    uvicorn.run("server_agent:app", host=host, port=port)
