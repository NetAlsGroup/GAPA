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
import uuid
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from server.agent_monitor import resources_payload
from server.agent_state import TaskState, start_consumer
from server.ga_worker import ga_worker, select_run_mode


app = FastAPI(title="GAPA Server Agent", version="0.1.0")
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

    with TASK.lock:
        if TASK.state == "running":
            raise HTTPException(status_code=409, detail="A task is already running")
        task_id = str(uuid.uuid4())
        TASK.reset_for_new_task(task_id)
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
        # Best-effort: terminate child processes as well (torch/mp.spawn).
        def kill_children(sig: int) -> None:
            try:
                import psutil  # type: ignore
            except Exception:
                return
            try:
                parent = psutil.Process(pid) if pid else None
                if not parent:
                    return
                for ch in parent.children(recursive=True):
                    try:
                        ch.send_signal(sig)
                    except Exception:
                        pass
            except Exception:
                pass

        if pid and os.name == "posix":
            try:
                os.killpg(pid, signal.SIGTERM)
            except Exception:
                try:
                    os.kill(pid, signal.SIGTERM)
                except Exception:
                    pass
            kill_children(signal.SIGTERM)
        else:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                kill_children(signal.SIGTERM)
            except Exception:
                pass

        proc.join(timeout=3.0)
        if proc.is_alive():
            if pid and os.name == "posix":
                try:
                    os.killpg(pid, signal.SIGKILL)
                except Exception:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except Exception:
                        pass
                kill_children(signal.SIGKILL)
            else:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    kill_children(signal.SIGKILL)
                except Exception:
                    pass
            proc.join(timeout=1.0)
    finally:
        with TASK.lock:
            TASK.state = "idle"
            TASK.progress = 0
            TASK.error = "stopped"
            TASK.append_log("[INFO] task stopped.")

    return {"status": "stopped", "task_id": TASK.task_id}


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
    algorithm = payload.get("algorithm")
    plan = _StrategyPlan(fitness=None, warmup=warmup, objective=objective, multi_gpu=multi_gpu)
    if algorithm:
        try:
            plan.notes = (plan.notes + " " if plan.notes else "") + f"algorithm={algorithm}"
        except Exception:
            pass
    return plan.to_dict()


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
    return _StrategyCompare(objective=objective, multi_gpu=multi_gpu, warmup_iters=warmup_iters)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("GAPA_AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("GAPA_AGENT_PORT", "7777"))
    uvicorn.run("server_agent:app", host=host, port=port)
