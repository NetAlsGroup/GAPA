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
from typing import Any, Dict, List
import threading

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from server.agent_monitor import resources_payload
from server.agent_state import TaskState, start_consumer
from server.fitness_protocol import dumps as fitness_dumps, loads as fitness_loads
from server.fitness_worker import compute_fitness_batch
from server.ga_worker import ga_worker, select_run_mode
from server.mode_runtime import build_mode_decision, choose_mode, detect_capability, transport_contract
from server.resource_lock import LOCK_MANAGER
from server.task_queue import TaskQueueManager
from server.db_manager import db_manager
from server.api_schemas import (
    CURRENT_SCHEMA_VERSION,
    DEFAULT_SCHEMA_VERSION,
    TERMINAL_TASK_STATES,
    build_resume_metadata,
    is_terminal_task_state,
    normalize_mode_decision,
    resolve_schema_version,
)
from server.shared_runtime import (
    run_ga_warmup_local as run_ga_warmup_local_shared,
    summarize_warmup_result,
    terminate_process_tree,
)



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
QUEUE_MANAGER = TaskQueueManager(
    max_total=int(os.getenv("GAPA_QUEUE_MAX_TOTAL", "16") or 16),
    max_per_owner=int(os.getenv("GAPA_QUEUE_MAX_PER_OWNER", "2") or 2),
    storage=db_manager,
    terminal_filter=lambda ids: db_manager.get_terminal_task_ids(ids),
)
QUEUE_LOCK = threading.Lock()


def _summarize_warmup_result(result: dict | None) -> dict:
    return summarize_warmup_result(result)


def _run_ga_warmup_local(payload: Dict[str, Any]) -> Dict[str, Any] | tuple[Dict[str, Any], int]:
    return run_ga_warmup_local_shared(
        payload=payload,
        task=TASK,
        task_lock=TASK.lock,
        busy_message="Agent is busy running a GA task",
        ga_entry=_ga_entry,
        select_run_mode=select_run_mode,
        require_algorithm_dataset=True,
    )
def _ga_entry(
    task_id: str,
    algorithm: str,
    dataset: str,
    iterations: int,
    crossover_rate: float,
    mutate_rate: float,
    selected: Dict[str, Any],
    q: Any,
    resume_id: str | None = None,
) -> None:
    if os.name == "posix":
        try:
            os.setsid()
        except Exception:
            pass
    resume_state = None
    if resume_id:
        try:
            resume_state = db_manager.get_ga_state(resume_id)
        except Exception:
            resume_state = None
    ga_worker(task_id, algorithm, dataset, iterations, crossover_rate, mutate_rate, selected, q, resume_state=resume_state)


def _resolve_selected(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve run mode/devices with lightweight adaptive pre-check to reduce OOM risk."""
    requested_mode = str(payload.get("mode") or "AUTO").upper()
    requested_devices = payload.get("devices")
    auto_select = bool(payload.get("auto_select_devices", True))
    use_strategy_plan = bool(payload.get("use_strategy_plan", True))
    note = ""
    try:
        capability = detect_capability(target="local", resource_snapshot=resources_payload(), remote_reachable=True)
    except Exception:
        capability = detect_capability(target="local", resource_snapshot={}, remote_reachable=True)
    allow_mnm = bool(payload.get("remote_servers") or payload.get("allowed_server_ids"))

    def _normalize_devices(val: Any) -> List[int]:
        out: List[int] = []
        if isinstance(val, list):
            src = val
        elif val is None:
            src = []
        else:
            src = [val]
        for x in src:
            try:
                out.append(int(x))
            except Exception:
                continue
        # unique preserve order
        uniq: List[int] = []
        for d in out:
            if d not in uniq:
                uniq.append(d)
        return uniq

    req_devs = _normalize_devices(requested_devices)

    def _finalize_selected(selected: Dict[str, Any]) -> Dict[str, Any]:
        current_mode = str(selected.get("mode") or requested_mode).upper()
        normalized_mode, normalized_degraded, normalized_reason = choose_mode(
            current_mode,
            capability,
            allow_mnm=allow_mnm,
        )
        if current_mode != normalized_mode:
            selected = select_run_mode(normalized_mode, selected.get("devices"))
            if allow_mnm:
                remote_servers = payload.get("remote_servers") or payload.get("allowed_server_ids")
                if remote_servers is not None:
                    selected["remote_servers"] = remote_servers
        existing_note = str(selected.get("selection_note") or "").strip()
        if normalized_degraded and normalized_reason:
            selected["selection_note"] = (
                f"{existing_note}; {normalized_reason}" if existing_note else normalized_reason
            )
        return selected

    if req_devs:
        selected = select_run_mode(requested_mode, req_devs)
        selected["selection_note"] = "manual devices"
        return _finalize_selected(selected)

    # No manual devices: StrategyPlan first, then lock + resource snapshot fallback.
    if auto_select and requested_mode in ("M", "SM"):
        locked_devices: List[int] = []
        try:
            lock_status = LOCK_MANAGER.status()
            if lock_status.get("active"):
                locked_devices = _normalize_devices(lock_status.get("devices"))
        except Exception:
            locked_devices = []

        # 1) StrategyPlan as primary selector for M/SM.
        try:
            from gapa.autoadapt import StrategyPlan as _StrategyPlan  # type: ignore
        except Exception:
            _StrategyPlan = None
        if use_strategy_plan and _StrategyPlan is not None:
            try:
                snap = resources_payload()
                plan = _StrategyPlan(
                    fitness=None,
                    warmup=int(payload.get("plan_warmup", 0) or 0),
                    objective=str(payload.get("plan_objective") or "time"),
                    multi_gpu=(requested_mode == "M"),
                    resource_snapshot=snap,
                    gpu_busy_threshold=payload.get("gpu_busy_threshold"),
                    min_gpu_free_mb=payload.get("min_gpu_free_mb"),
                    tpe_trials=payload.get("tpe_trials"),
                    tpe_warmup=payload.get("tpe_warmup"),
                )
                plan_backend = str(getattr(plan, "backend", "") or "").lower()
                plan_devices = _normalize_devices(getattr(plan, "devices", []) or [])

                # Respect active lock: plan result can only use locked devices.
                if locked_devices:
                    plan_devices = [d for d in plan_devices if d in locked_devices]

                if plan_backend == "multi-gpu" and requested_mode == "M":
                    if len(plan_devices) >= 2:
                        selected = select_run_mode("M", plan_devices)
                        selected["selection_note"] = f"strategy_plan backend=multi-gpu devices={plan_devices}"
                        return _finalize_selected(selected)
                    if len(plan_devices) == 1:
                        selected = select_run_mode("SM", plan_devices)
                        selected["selection_note"] = f"strategy_plan fallback M->SM devices={plan_devices}"
                        return _finalize_selected(selected)
                    selected = select_run_mode("S", [])
                    selected["selection_note"] = "strategy_plan fallback M->S (no usable planned devices)"
                    return _finalize_selected(selected)

                if plan_backend == "cuda":
                    if len(plan_devices) >= 1:
                        dev = [plan_devices[0]]
                        if requested_mode == "M":
                            selected = select_run_mode("SM", dev)
                            selected["selection_note"] = f"strategy_plan fallback M->SM backend=cuda devices={dev}"
                        else:
                            selected = select_run_mode("SM", dev)
                            selected["selection_note"] = f"strategy_plan backend=cuda device={dev[0]}"
                        return _finalize_selected(selected)
                    selected = select_run_mode("S", [])
                    selected["selection_note"] = "strategy_plan fallback to S (cuda backend without usable device)"
                    return _finalize_selected(selected)

                if plan_backend == "cpu":
                    selected = select_run_mode("S", [])
                    selected["selection_note"] = "strategy_plan chose cpu; fallback to S"
                    return _finalize_selected(selected)
            except Exception as exc:
                note = f"strategy_plan failed: {exc}"

        # 2) Fallback heuristics when StrategyPlan disabled/unavailable/failed.
        min_free_mb = int(os.getenv("GAPA_M_DEVICE_MIN_FREE_MB", "2500") or 2500)
        max_util = float(os.getenv("GAPA_M_DEVICE_MAX_UTIL", "90") or 90)
        max_devices = int(os.getenv("GAPA_M_MAX_DEVICES", "8") or 8)
        try:
            snap = resources_payload()
            gpus = snap.get("gpus") or []
        except Exception:
            gpus = []

        candidates: List[tuple[int, float, float]] = []
        for g in gpus if isinstance(gpus, list) else []:
            if not isinstance(g, dict):
                continue
            try:
                gid = int(g.get("id"))
            except Exception:
                continue
            if locked_devices and gid not in locked_devices:
                continue
            free_mb = g.get("free_mb")
            util = g.get("gpu_util_percent")
            free_val = float(free_mb) if isinstance(free_mb, (int, float)) else 0.0
            util_val = float(util) if isinstance(util, (int, float)) else 0.0
            if free_val >= min_free_mb and util_val <= max_util:
                candidates.append((gid, free_val, util_val))

        candidates.sort(key=lambda x: (x[1], -x[2]), reverse=True)
        picked = [gid for gid, _, _ in candidates[:max_devices]]

        if requested_mode == "M":
            if len(picked) >= 2:
                selected = select_run_mode("M", picked)
                note = f"{note + '; ' if note else ''}adaptive devices={picked} (min_free_mb={min_free_mb}, max_util={max_util})"
            elif len(picked) == 1:
                selected = select_run_mode("SM", picked)
                note = f"{note + '; ' if note else ''}adaptive fallback M->SM (only one healthy GPU: {picked})"
            else:
                selected = select_run_mode("S", [])
                note = f"{note + '; ' if note else ''}adaptive fallback M->S (no healthy GPU candidates)"
        else:  # requested_mode == "SM"
            if len(picked) >= 1:
                selected = select_run_mode("SM", [picked[0]])
                note = f"{note + '; ' if note else ''}adaptive device={picked[0]} (min_free_mb={min_free_mb}, max_util={max_util})"
            else:
                selected = select_run_mode("S", [])
                note = f"{note + '; ' if note else ''}adaptive fallback SM->S (no healthy GPU candidates)"
        selected["selection_note"] = note
        return _finalize_selected(selected)

    # Legacy/default path
    selected = select_run_mode(requested_mode, requested_devices)
    selected["selection_note"] = "default selection"
    return _finalize_selected(selected)


def _start_task_locked(
    *,
    task_id: str,
    algorithm: str,
    dataset: str,
    iterations: int,
    crossover_rate: float,
    mutate_rate: float,
    selected: Dict[str, Any],
    mode_decision: Dict[str, Any],
    schema_version: str,
    run_id: str,
    resume_metadata: Dict[str, Any],
    release_lock_on_finish: bool,
    resume_id: str = "",
    queued_owner: str = "",
) -> None:
    TASK.reset_for_new_task(
        task_id,
        mode_decision=mode_decision,
        run_id=run_id,
        schema_version=schema_version,
        resume_metadata=resume_metadata,
    )
    TASK.release_lock_on_finish = release_lock_on_finish
    mode = selected.get("mode")
    devices = selected.get("devices")
    selection_note = selected.get("selection_note")
    owner_txt = f" owner={queued_owner}" if queued_owner else ""
    TASK.append_log(f"[INFO] 运行模式: {mode} devices={devices}{owner_txt}")
    if mode_decision:
        TASK.append_log(
            f"[INFO] 模式决策: requested={mode_decision.get('requested_mode')} "
            f"selected={mode_decision.get('selected_mode')} degraded={mode_decision.get('degraded')} "
            f"reason={mode_decision.get('reason') or '-'} code={mode_decision.get('code') or '-'}"
        )
    if selection_note:
        TASK.append_log(f"[INFO] 设备选择: {selection_note}")

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(
        target=_ga_entry,
        args=(task_id, algorithm, dataset, iterations, crossover_rate, mutate_rate, selected, q, resume_id),
    )
    TASK.queue = q
    TASK.process = proc
    proc.start()
    start_consumer(TASK, on_finish=_try_dispatch_next_task)


def _try_dispatch_next_task() -> None:
    with TASK.lock:
        if TASK.state == "running":
            return
        with QUEUE_LOCK:
            nxt = QUEUE_MANAGER.pop_next()
        if nxt is None:
            return
        payload = nxt.payload
        schema_version = resolve_schema_version(payload)
        run_id = str(payload.get("run_id") or nxt.task_id)
        resume_id = str(payload.get("resume_id") or payload.get("checkpoint_ref") or "")
        selected = _resolve_selected(payload)
        capability = detect_capability(target="local", resource_snapshot=resources_payload(), remote_reachable=True)
        requested_mode = payload.get("mode")
        mode_decision = build_mode_decision(
            requested_mode=requested_mode,
            selected_mode=selected.get("mode"),
            devices=selected.get("devices"),
            target="local",
            capability=capability,
            reason=str(selected.get("selection_note") or ""),
            use_strategy_plan=payload.get("use_strategy_plan"),
        )
        mode_decision = normalize_mode_decision(mode_decision)
        resume_metadata = build_resume_metadata(
            run_id=run_id,
            task_id=nxt.task_id,
            schema_version=schema_version,
            mode_plan_snapshot=mode_decision,
            checkpoint_ref=resume_id,
        )
        selected["run_id"] = run_id
        selected["schema_version"] = schema_version
        selected["mode_decision"] = mode_decision
        _start_task_locked(
            task_id=nxt.task_id,
            algorithm=str(payload.get("algorithm") or "ga"),
            dataset=str(payload.get("dataset") or ""),
            iterations=int(payload.get("iterations") or payload.get("max_generation") or 20),
            crossover_rate=float(payload.get("crossover_rate") or payload.get("pc") or 0.8),
            mutate_rate=float(payload.get("mutate_rate") or payload.get("pm") or 0.2),
            selected=selected,
            mode_decision=mode_decision,
            schema_version=schema_version,
            run_id=run_id,
            resume_metadata=resume_metadata,
            resume_id=resume_id,
            release_lock_on_finish=bool(payload.get("release_lock_on_finish", True)),
            queued_owner=nxt.owner,
        )


@app.post("/api/analysis/start")
def api_analysis_start(payload: Dict[str, Any]) -> Dict[str, Any]:
    schema_version = resolve_schema_version(payload)
    algorithm = str(payload.get("algorithm") or "ga")
    dataset = str(payload.get("dataset") or "")
    iterations = int(payload.get("iterations") or payload.get("max_generation") or 20)
    crossover_rate = float(payload.get("crossover_rate") or payload.get("pc") or 0.8)
    mutate_rate = float(payload.get("mutate_rate") or payload.get("pm") or 0.2)
    run_id = str(payload.get("run_id") or uuid.uuid4())
    retry_last = bool(payload.get("retry_last", False))
    resume_id = str(payload.get("resume_id") or payload.get("checkpoint_ref") or "")
    selected = _resolve_selected(payload)
    capability = detect_capability(target="local", resource_snapshot=resources_payload(), remote_reachable=True)
    mode_decision = build_mode_decision(
        requested_mode=payload.get("mode"),
        selected_mode=selected.get("mode"),
        devices=selected.get("devices"),
        target="local",
        capability=capability,
        reason=str(selected.get("selection_note") or ""),
        use_strategy_plan=payload.get("use_strategy_plan"),
    )
    mode_decision = normalize_mode_decision(mode_decision)
    release_lock_on_finish = bool(payload.get("release_lock_on_finish", True))
    queue_if_busy = bool(payload.get("queue_if_busy", False))
    owner = str(payload.get("owner") or "anonymous")
    priority = int(payload.get("priority") or 0)

    with TASK.lock:
        if retry_last and not resume_id:
            resume_id = str(TASK.last_checkpoint_ref or "")
        if TASK.state == "running":
            if not queue_if_busy:
                raise HTTPException(status_code=409, detail={"code": "TASK_BUSY", "message": "A task is already running"})
            task_id = str(uuid.uuid4())
            payload["schema_version"] = schema_version
            payload["run_id"] = run_id
            payload["checkpoint_ref"] = resume_id
            with QUEUE_LOCK:
                ok, info = QUEUE_MANAGER.enqueue(task_id=task_id, payload=payload, owner=owner, priority=priority)
            if not ok:
                raise HTTPException(status_code=429, detail={"code": "QUEUE_LIMIT", "message": info})
            return {
                "schema_version": schema_version,
                "schema_current": CURRENT_SCHEMA_VERSION,
                "task_id": task_id,
                "run_id": run_id,
                "status": "queued",
                "mode_decision": mode_decision,
                "resume_metadata": build_resume_metadata(
                    run_id=run_id,
                    task_id=task_id,
                    schema_version=schema_version,
                    mode_plan_snapshot=mode_decision,
                    checkpoint_ref=resume_id,
                ),
                "transport": transport_contract(),
                **info,
            }
        task_id = str(uuid.uuid4())
        resume_metadata = build_resume_metadata(
            run_id=run_id,
            task_id=task_id,
            schema_version=schema_version,
            mode_plan_snapshot=mode_decision,
            checkpoint_ref=resume_id,
        )
        selected["run_id"] = run_id
        selected["schema_version"] = schema_version
        selected["mode_decision"] = mode_decision
        _start_task_locked(
            task_id=task_id,
            algorithm=algorithm,
            dataset=dataset,
            iterations=iterations,
            crossover_rate=crossover_rate,
            mutate_rate=mutate_rate,
            selected=selected,
            mode_decision=mode_decision,
            schema_version=schema_version,
            run_id=run_id,
            resume_metadata=resume_metadata,
            resume_id=resume_id,
            release_lock_on_finish=release_lock_on_finish,
            queued_owner=owner,
        )

    return {
        "schema_version": schema_version,
        "schema_current": CURRENT_SCHEMA_VERSION,
        "task_id": task_id,
        "run_id": run_id,
        "status": "started",
        "owner": owner,
        "mode_decision": mode_decision,
        "resume_metadata": resume_metadata,
        "transport": transport_contract(),
    }


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
        with QUEUE_LOCK:
            queue_size = QUEUE_MANAGER.size()
        return {
            "schema_version": TASK.schema_version or DEFAULT_SCHEMA_VERSION,
            "schema_current": CURRENT_SCHEMA_VERSION,
            "task_id": TASK.task_id,
            "run_id": TASK.run_id,
            "state": TASK.state,
            "progress": TASK.progress,
            "logs": list(TASK.logs),
            "result": TASK.result,
            "error": TASK.error,
            "queued": queue_size,
            "mode_decision": normalize_mode_decision(TASK.mode_decision),
            "resume_metadata": TASK.resume_metadata,
            "lifecycle": {
                "terminal_states": list(TERMINAL_TASK_STATES),
                "is_terminal": is_terminal_task_state(TASK.state),
            },
        }


@app.get("/api/analysis/queue")
def api_analysis_queue() -> Dict[str, Any]:
    with QUEUE_LOCK:
        items = QUEUE_MANAGER.list_items()
    return {
        "schema_version": TASK.schema_version or DEFAULT_SCHEMA_VERSION,
        "schema_current": CURRENT_SCHEMA_VERSION,
        "size": len(items),
        "items": items,
    }

@app.post("/api/analysis/stop")
def api_analysis_stop() -> Dict[str, Any]:
    with TASK.lock:
        proc = TASK.process
        if TASK.state != "running" or not proc or not proc.is_alive():
            return {
                "schema_version": TASK.schema_version or DEFAULT_SCHEMA_VERSION,
                "schema_current": CURRENT_SCHEMA_VERSION,
                "status": TASK.state if TASK.state != "idle" else "idle",
                "task_id": TASK.task_id,
                "run_id": TASK.run_id,
                "mode_decision": normalize_mode_decision(TASK.mode_decision),
                "resume_metadata": TASK.resume_metadata,
            }

        pid = proc.pid
        TASK.append_log("[WARN] stop requested, terminating task...")

    # terminate outside lock
    try:
        if pid:
            terminate_process_tree(proc, pid, signal.SIGTERM)
        
        proc.join(timeout=2.0)
        if proc.is_alive():
            print(f"[WARN] Process {pid} still alive after SIGTERM, sending SIGKILL...")
            if pid:
                terminate_process_tree(proc, pid, signal.SIGKILL)
            proc.join(timeout=1.0)
    finally:
        with TASK.lock:
            TASK.state = "cancelled"
            TASK.progress = 0
            TASK.error = "cancelled by user"
            if TASK.task_id:
                TASK.last_checkpoint_ref = TASK.task_id
            TASK.append_log("[INFO] task stopped.")
        try:
            db_manager.save_task_terminal(task_id=str(TASK.task_id or ""), state="cancelled")
        except Exception:
            pass
    _try_dispatch_next_task()

    return {
        "schema_version": TASK.schema_version or DEFAULT_SCHEMA_VERSION,
        "schema_current": CURRENT_SCHEMA_VERSION,
        "status": "cancelled",
        "task_id": TASK.task_id,
        "run_id": TASK.run_id,
        "mode_decision": normalize_mode_decision(TASK.mode_decision),
        "resume_metadata": TASK.resume_metadata,
    }

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
    owner = str(payload.get("owner") or "")
    devices = payload.get("devices")
    if devices is not None and not isinstance(devices, list):
        devices = [devices]
    try:
        info = LOCK_MANAGER.lock(
            duration_s=duration_s,
            warmup_iters=warmup_iters,
            mem_mb=mem_mb,
            devices=devices,
            strict_idle=strict_idle,
            owner=owner,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return info


@app.post("/api/resource_lock/renew")
def api_resource_lock_renew(payload: Dict[str, Any]) -> Dict[str, Any]:
    duration_s = payload.get("duration_s")
    lock_id = payload.get("lock_id")
    owner = payload.get("owner")
    try:
        info = LOCK_MANAGER.renew(duration_s=duration_s, lock_id=lock_id, owner=owner)
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
        from gapa.autoadapt import StrategyPlan as _StrategyPlan  # type: ignore
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
        from gapa.autoadapt import StrategyCompare as _StrategyCompare  # type: ignore
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
