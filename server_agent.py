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

import socket
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from autoadapt import StrategyPlan as _StrategyPlan  # optional on remote host
except Exception:
    _StrategyPlan = None

import psutil
import pynvml


app = FastAPI(title="GAPA Server Agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------- Monitor helpers --------------------
def _cpu_snapshot() -> Dict[str, Any]:
    if not psutil:
        return {"usage_percent": None, "cores": None}
    try:
        return {
            "usage_percent": psutil.cpu_percent(interval=0.1),
            "cores": psutil.cpu_count(logical=True),
        }
    except Exception:
        return {"usage_percent": None, "cores": psutil.cpu_count(logical=True) if psutil else None}


def _memory_snapshot() -> Dict[str, Any]:
    if not psutil:
        return {"total_mb": None, "used_mb": None, "percent": None}
    try:
        mem = psutil.virtual_memory()
        return {
            "total_mb": round(mem.total / (1024**2)),
            "used_mb": round(mem.used / (1024**2)),
            "percent": mem.percent,
        }
    except Exception:
        return {"total_mb": None, "used_mb": None, "percent": None}


def _gpu_snapshot() -> List[Dict[str, Any]]:
    gpus: List[Dict[str, Any]] = []

    try:
        pynvml.nvmlInit()

        # 关键补丁！消费级卡不加这句功率永远是 None
        # 开启后即使程序退出功率监控也持续有效（对性能无影响）
        try:
            pynvml.nvmlDeviceSetPersistenceMode(
                pynvml.nvmlDeviceGetHandleByIndex(0), 1
            )
        except:
            pass

        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mb = round(mem_info.total / 1024 ** 2)
            used_mb = round(mem_info.used / 1024 ** 2)
            free_mb = round(mem_info.free / 1024 ** 2)

            # 利用率：安全获取，失败返回 0 而不是 None
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu  # 0–100
                mem_util = util.memory
            except pynvml.NVMLError:
                gpu_util = mem_util = 0

            # 功率：最容易失败的，单独 try
            power_w = None
            try:
                # 返回值是毫瓦，转成瓦
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_w = round(power_mw / 1000.0, 2)
            except pynvml.NVMLError as e:
                # NVMLError_NotSupported 消费卡未开启持久模式
                # NVMLError_Unknown   其他未知错误
                power_w = None

            gpus.append({
                "id": i,
                "name": name,
                "total_mb": total_mb,
                "used_mb": used_mb,
                "free_mb": free_mb,
                "gpu_util_percent": float(gpu_util),  # 现在正常显示
                "mem_util_percent": float(mem_util),
                "power_w": power_w,  # 开启持久模式后正常显示
                "temperature_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            })

    except Exception as e:
        print(f"NVML init failed: {e}")
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

    return gpus


@app.get("/api/resources")
def api_resources() -> Dict[str, Any]:
    return {
        "status": "online",
        "hostname": socket.gethostname(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cpu": _cpu_snapshot(),
        "memory": _memory_snapshot(),
        "gpus": _gpu_snapshot(),
    }


# -------------------- GA Task Manager --------------------
class TaskState:
    def __init__(self):
        self.lock = threading.Lock()
        self.task_id: Optional[str] = None
        self.state: str = "idle"  # idle | running | completed | error
        self.progress: int = 0
        self.logs: List[str] = []
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.thread: Optional[threading.Thread] = None

    def reset_for_new_task(self, task_id: str) -> None:
        self.task_id = task_id
        self.state = "running"
        self.progress = 0
        self.logs = []
        self.result = None
        self.error = None

    def append_log(self, line: str) -> None:
        self.logs.append(line)
        if len(self.logs) > 500:
            self.logs = self.logs[-500:]

    def set_progress(self, val: int) -> None:
        self.progress = max(0, min(100, int(val)))


TASK = TaskState()


def run_genetic_algorithm(algorithm: str, warmup_steps: int, task_id: str) -> None:
    """Mock GA workload with warmup and convergence curve output."""
    try:
        with TASK.lock:
            if TASK.task_id != task_id:
                return
            TASK.append_log(f"[INFO] 任务 {task_id} 启动，算法={algorithm}")
            TASK.set_progress(1)

        time.sleep(0.3)
        with TASK.lock:
            TASK.append_log("[INFO] 初始化种群...")
            TASK.set_progress(5)

        time.sleep(0.4)

        # Warmup phase
        warmup_steps = max(0, int(warmup_steps))
        for i in range(warmup_steps):
            time.sleep(0.25)
            with TASK.lock:
                TASK.append_log(f"[INFO] Warmup 轮次 {i+1}/{warmup_steps} 完成...")
                TASK.set_progress(5 + int((i + 1) / max(1, warmup_steps) * 25))

        # Main GA analysis (mock)
        steps = 8
        for s in range(steps):
            time.sleep(0.35)
            with TASK.lock:
                TASK.append_log(f"[INFO] 进化迭代 {s+1}/{steps} ...")
                TASK.set_progress(30 + int((s + 1) / steps * 60))

        # Mock convergence data
        convergence = []
        base = 1.0
        for t in range(20):
            base *= 0.92
            convergence.append(round(base + (0.02 * (t % 3) / 3.0), 5))

        with TASK.lock:
            TASK.append_log("[INFO] 分析完成。")
            TASK.set_progress(100)
            TASK.state = "completed"
            TASK.result = {
                "algorithm": algorithm,
                "convergence": convergence,
                "best_score": min(convergence) if convergence else None,
            }
    except Exception as exc:
        with TASK.lock:
            TASK.state = "error"
            TASK.error = str(exc)
            TASK.append_log(f"[ERROR] {exc}")


@app.post("/api/analysis/start")
def api_analysis_start(payload: Dict[str, Any]) -> Dict[str, Any]:
    algorithm = str(payload.get("algorithm") or "ga")
    warmup_steps = int(payload.get("warmup_steps") or 0)

    with TASK.lock:
        if TASK.state == "running":
            raise HTTPException(status_code=409, detail="A task is already running")
        task_id = str(uuid.uuid4())
        TASK.reset_for_new_task(task_id)

        thread = threading.Thread(
            target=run_genetic_algorithm, args=(algorithm, warmup_steps, task_id), daemon=True
        )
        TASK.thread = thread
        thread.start()

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


@app.post("/api/strategy_plan")
def api_strategy_plan(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compute StrategyPlan on this server (static profiling)."""
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server_agent:app", host="0.0.0.0", port=7777, reload=False)
