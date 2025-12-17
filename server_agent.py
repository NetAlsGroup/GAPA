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
import os
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from autoadapt import StrategyCompare as _StrategyCompare  # optional on remote host
    from autoadapt import StrategyPlan as _StrategyPlan  # optional on remote host
except Exception:
    _StrategyPlan = None
    _StrategyCompare = None

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
        self.process: Optional[mp.Process] = None
        self.queue: Any = None
        self.consumer: Optional[threading.Thread] = None

    def reset_for_new_task(self, task_id: str) -> None:
        self.task_id = task_id
        self.state = "running"
        self.progress = 0
        self.logs = []
        self.result = None
        self.error = None
        self.process = None
        self.queue = None
        self.consumer = None

    def append_log(self, line: str) -> None:
        self.logs.append(line)
        if len(self.logs) > 500:
            self.logs = self.logs[-500:]

    def set_progress(self, val: int) -> None:
        self.progress = max(0, min(100, int(val)))


TASK = TaskState()


def _resource_point() -> Dict[str, Any]:
    """A lightweight per-iteration snapshot for monitoring charts."""
    snap = api_resources()
    # Reduce payload size: keep only key fields
    gpus = []
    for g in snap.get("gpus", []) or []:
        gpus.append(
            {
                "id": g.get("id"),
                "used_mb": g.get("used_mb"),
                "total_mb": g.get("total_mb"),
                "power_w": g.get("power_w"),
                "gpu_util_percent": g.get("gpu_util_percent"),
                "mem_util_percent": g.get("mem_util_percent"),
            }
        )
    return {
        "timestamp": snap.get("timestamp"),
        "cpu_usage_percent": (snap.get("cpu") or {}).get("usage_percent"),
        "memory_percent": (snap.get("memory") or {}).get("percent"),
        "gpus": gpus,
    }


def run_genetic_algorithm(*_args: Any, **_kwargs: Any) -> None:
    """Deprecated: GA now runs in a subprocess via `_ga_worker` for strict device control."""
    raise RuntimeError("run_genetic_algorithm is deprecated; use subprocess-based execution")


def _select_run_mode(mode: str | None, devices: Any) -> Dict[str, Any]:
    """Select requested mode/devices without mutating environment.

    The actual CUDA visibility is applied inside the GA subprocess before torch import/initialization.
    """
    mode = (mode or "AUTO").upper()
    selected: List[int] = []
    if isinstance(devices, list):
        for d in devices:
            try:
                selected.append(int(d))
            except Exception:
                continue
    elif devices is not None:
        try:
            selected = [int(devices)]
        except Exception:
            selected = []

    if mode == "CPU":
        return {"mode": mode, "devices": [], "cuda_visible_devices": None}

    if mode in ("S", "SM"):
        if selected:
            return {"mode": mode, "devices": [selected[0]], "cuda_visible_devices": str(selected[0])}
        return {"mode": mode, "devices": [], "cuda_visible_devices": None}

    if mode in ("M", "MNM"):
        if selected:
            return {
                "mode": mode,
                "devices": selected,
                "cuda_visible_devices": ",".join(str(x) for x in selected),
            }
        return {"mode": mode, "devices": [], "cuda_visible_devices": None}

    # AUTO: no override here; caller/UI can provide devices based on evaluation
    return {
        "mode": mode,
        "devices": selected,
        "cuda_visible_devices": ",".join(str(x) for x in selected) if selected else None,
    }


def _ga_worker(
    task_id: str,
    algorithm: str,
    dataset: str,
    iterations: int,
    crossover_rate: float,
    mutate_rate: float,
    selected: Dict[str, Any],
    q: Any,
) -> None:
    """Run GA in a subprocess and emit events to parent via queue."""
    try:
        # Apply CUDA visibility here for strict device control.
        cvd = selected.get("cuda_visible_devices")
        if selected.get("mode") == "CPU":
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        elif cvd:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cvd)

        # Import monitoring deps in child (keeps parent light and avoids CUDA init issues).
        try:
            import psutil as _psutil  # type: ignore
        except Exception:
            _psutil = None
        try:
            import pynvml as _pynvml  # type: ignore
        except Exception:
            _pynvml = None

        dataset_dir = Path(os.getenv("GAPA_DATASET_DIR", str(Path(__file__).resolve().parent / "dataset")))
        repo_root = Path(__file__).resolve().parent

        def _find_dataset_file(name: str) -> Optional[Path]:
            if not name:
                return None
            candidates = []
            # Common patterns in this repo
            candidates.append(dataset_dir / f"{name}.txt")
            candidates.append(dataset_dir / f"{name.lower()}.txt")
            candidates.append(dataset_dir / name / f"{name}.txt")
            candidates.append(dataset_dir / name.lower() / f"{name.lower()}.txt")
            # Try normalized hyphen/case variants
            norm = name.replace("_", "-")
            candidates.append(dataset_dir / f"{norm}.txt")
            candidates.append(dataset_dir / f"{norm.lower()}.txt")
            candidates.append(dataset_dir / norm / f"{norm}.txt")
            candidates.append(dataset_dir / norm.lower() / f"{norm.lower()}.txt")
            for p in candidates:
                if p.exists():
                    return p
            # Fallback: scan dataset_dir (shallow) for case-insensitive match of "<name>.txt"
            target = f"{name}".lower()
            try:
                for p in dataset_dir.glob("**/*.txt"):
                    if p.name.lower() in (f"{target}.txt", f"{norm.lower()}.txt"):
                        return p
            except Exception:
                pass
            return None

        selected_phys = selected.get("devices") or []
        try:
            selected_phys = [int(x) for x in selected_phys]
        except Exception:
            selected_phys = []

        def snapshot() -> Dict[str, Any]:
            cpu_usage = None
            mem_percent = None
            if _psutil:
                try:
                    cpu_usage = _psutil.cpu_percent(interval=0.0)
                    mem_percent = _psutil.virtual_memory().percent
                except Exception:
                    pass
            gpus: List[Dict[str, Any]] = []
            if _pynvml:
                try:
                    _pynvml.nvmlInit()
                    if selected_phys:
                        indices = selected_phys
                    else:
                        indices = list(range(_pynvml.nvmlDeviceGetCount()))
                    for i in indices:
                        h = _pynvml.nvmlDeviceGetHandleByIndex(i)
                        mem = _pynvml.nvmlDeviceGetMemoryInfo(h)
                        try:
                            util = _pynvml.nvmlDeviceGetUtilizationRates(h)
                            gpu_util = float(util.gpu)
                            mem_util = float(util.memory)
                        except Exception:
                            gpu_util = mem_util = None
                        try:
                            power_w = _pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                        except Exception:
                            power_w = None
                        gpus.append(
                            {
                                "id": i,
                                "used_mb": round(mem.used / (1024**2)),
                                "total_mb": round(mem.total / (1024**2)),
                                "power_w": power_w,
                                "gpu_util_percent": gpu_util,
                                "mem_util_percent": mem_util,
                            }
                        )
                except Exception:
                    gpus = []
                finally:
                    try:
                        _pynvml.nvmlShutdown()
                    except Exception:
                        pass
            return {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "cpu_usage_percent": cpu_usage,
                "memory_percent": mem_percent,
                "gpus": gpus,
            }

        def emit(evt: Dict[str, Any]) -> None:
            evt["task_id"] = task_id
            q.put(evt)

        emit(
            {
                "type": "log",
                "line": f"[INFO] 任务 {task_id} 启动，算法={algorithm} dataset={dataset} pc={crossover_rate:.3f} pm={mutate_rate:.3f} iters={iterations}",
            }
        )
        emit({"type": "log", "line": f"[INFO] 运行模式: {selected.get('mode')} devices={selected.get('devices')}"})
        emit({"type": "progress", "value": 1})

        # ---- Real algorithm execution (currently: SixDST) ----
        if algorithm != "SixDST":
            raise RuntimeError(f"Real GA runner not implemented for algorithm={algorithm}")

        # Ensure repo root is importable when running as a service
        try:
            import sys

            sys.path.insert(0, str(repo_root))
        except Exception:
            pass

        import networkx as nx  # type: ignore
        import torch  # type: ignore
        from gapa.utils.DataLoader import Loader  # type: ignore
        from gapa.algorithm.CND.SixDST import SixDST, SixDSTController, SixDSTEvaluator  # type: ignore

        # Map UI mode to algorithm mode
        ui_mode = (selected.get("mode") or "AUTO").upper()
        if ui_mode == "CPU":
            algo_mode = "s"
            device = "cpu"
        elif ui_mode == "SM":
            algo_mode = "sm"
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif ui_mode == "S":
            algo_mode = "s"
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif ui_mode == "M":
            algo_mode = "m"
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif ui_mode == "MNM":
            algo_mode = "mnm"
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            algo_mode = "s"
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        world_size = int(torch.cuda.device_count()) if device.startswith("cuda") else 1
        if algo_mode in ("m", "mnm") and world_size < 2:
            emit({"type": "log", "line": f"[WARN] mode={algo_mode} requires >=2 GPUs; fallback to S"})
            algo_mode = "s"
            world_size = 1

        ds_file = _find_dataset_file(dataset)
        if ds_file is None:
            raise FileNotFoundError(f"dataset file not found for '{dataset}' under {dataset_dir}")

        emit({"type": "log", "line": f"[INFO] Load dataset file: {ds_file}"})
        G = nx.read_adjlist(str(ds_file), nodetype=int)
        nodelist = list(G.nodes())
        A_cpu = torch.tensor(nx.to_numpy_array(G, nodelist=nodelist), dtype=torch.float32)
        A = A_cpu.to(device) if device != "cpu" else A_cpu

        data_loader = Loader(dataset=dataset, device=device)
        data_loader.G = G
        data_loader.A = A
        data_loader.nodes_num = int(A.shape[0])
        data_loader.nodes = torch.tensor(nodelist, device=device)
        data_loader.selected_genes_num = int(0.4 * data_loader.nodes_num)
        data_loader.k = int(0.1 * data_loader.nodes_num)
        data_loader.mode = algo_mode
        data_loader.world_size = world_size

        pop_size = int(os.getenv("GAPA_GA_POP_SIZE", "100"))
        result: Dict[str, Any] = {
            "algorithm": algorithm,
            "dataset": dataset,
            "convergence": [],
            "metrics": [],
            "hyperparams": {
                "iterations": int(iterations),
                "crossover_rate": float(crossover_rate),
                "mutate_rate": float(mutate_rate),
                "pop_size": pop_size,
            },
            "selected": {"mode": ui_mode, "devices": selected.get("devices")},
            "exec": {"algo_mode": algo_mode, "world_size": world_size, "device": device},
        }

        res_lock = threading.Lock()

        def on_iter(gen: int, max_gen: int, best_fit: float) -> None:
            emit({"type": "progress", "value": int(gen / max(1, max_gen) * 100)})
            emit({"type": "log", "line": f"[INFO] iter {gen}/{max_gen} best_fitness={best_fit}"})
            with res_lock:
                result["convergence"].append(best_fit)
                result["metrics"].append({"stage": "iter", "iter": gen, "best_fitness": best_fit, **snapshot()})

        obs_path = None
        tail_stop = False

        def tail_jsonl(path: Path) -> None:
            nonlocal tail_stop
            last_pos = 0
            while not tail_stop:
                try:
                    if not path.exists():
                        time.sleep(0.2)
                        continue
                    with path.open("r", encoding="utf-8") as f:
                        f.seek(last_pos)
                        for line in f:
                            last_pos = f.tell()
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                import json

                                obj = json.loads(line)
                                gen = int(obj.get("generation") or 0)
                                mg = int(obj.get("max_generation") or 1)
                                bf = float(obj.get("best_fitness") or 0.0)
                                on_iter(gen, mg, bf)
                            except Exception:
                                continue
                except Exception:
                    pass
                time.sleep(0.2)

        results_dir = Path(os.getenv("GAPA_RESULTS_DIR", str(repo_root / "results")))
        results_dir.mkdir(parents=True, exist_ok=True)
        controller = SixDSTController(
            path=str(results_dir) + "/",
            pattern="write",
            data_loader=data_loader,
            loops=1,
            crossover_rate=float(crossover_rate),
            mutate_rate=float(mutate_rate),
            pop_size=pop_size,
            device=device,
        )
        evaluator = SixDSTEvaluator(pop_size=pop_size, adj=data_loader.A, device=device)

        if algo_mode in ("m", "mnm"):
            obs_path = Path(str(results_dir / f"obs_{task_id}.jsonl"))
            try:
                obs_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
            except Exception:
                pass
            controller.observer = {"type": "jsonl", "path": str(obs_path)}
            t_tail = threading.Thread(target=tail_jsonl, args=(obs_path,), daemon=True)
            t_tail.start()
        else:
            controller.observer = on_iter

        emit({"type": "log", "line": f"[INFO] Start SixDST: mode={algo_mode} device={device} world_size={world_size}"})
        result["metrics"].append({"stage": "init", **snapshot()})
        SixDST(mode=algo_mode, max_generation=int(iterations), data_loader=data_loader, controller=controller, evaluator=evaluator, world_size=world_size, verbose=False)

        # Stop tailer and finalize
        tail_stop = True
        conv = result.get("convergence") or []
        result["best_score"] = min(conv) if conv else None
        emit({"type": "log", "line": "[INFO] 分析完成。"})
        emit({"type": "result", "result": result})
        emit({"type": "state", "state": "completed"})
    except Exception as exc:
        try:
            q.put({"type": "log", "line": f"[ERROR] {exc}", "task_id": task_id})
            q.put({"type": "state", "state": "error", "error": str(exc), "task_id": task_id})
        except Exception:
            pass


def _start_consumer(task: TaskState) -> None:
    """Drain child events into TASK state in a background thread."""
    def loop() -> None:
        q = task.queue
        proc = task.process
        while True:
            alive = proc.is_alive() if proc else False
            try:
                evt = q.get(timeout=0.3)
            except Exception:
                evt = None
            if evt:
                with task.lock:
                    if evt.get("type") == "log":
                        task.append_log(str(evt.get("line")))
                    elif evt.get("type") == "progress":
                        task.set_progress(int(evt.get("value") or 0))
                    elif evt.get("type") == "result":
                        task.result = evt.get("result")
                    elif evt.get("type") == "state":
                        task.state = evt.get("state") or task.state
                        task.error = evt.get("error")
            if not alive and (q.empty() if hasattr(q, "empty") else True):
                break
        # Ensure terminal state
        with task.lock:
            if task.state == "running":
                task.state = "completed" if task.result else "idle"

    t = threading.Thread(target=loop, daemon=True)
    task.consumer = t
    t.start()


@app.post("/api/analysis/start")
def api_analysis_start(payload: Dict[str, Any]) -> Dict[str, Any]:
    algorithm = str(payload.get("algorithm") or "ga")
    dataset = str(payload.get("dataset") or "")
    iterations = int(payload.get("iterations") or payload.get("max_generation") or 20)
    crossover_rate = float(payload.get("crossover_rate") or payload.get("pc") or 0.8)
    mutate_rate = float(payload.get("mutate_rate") or payload.get("pm") or 0.2)
    mode = payload.get("mode")
    devices = payload.get("devices")
    selected = _select_run_mode(mode, devices)

    with TASK.lock:
        if TASK.state == "running":
            raise HTTPException(status_code=409, detail="A task is already running")
        task_id = str(uuid.uuid4())
        TASK.reset_for_new_task(task_id)
        TASK.append_log(f"[INFO] 运行模式: {selected.get('mode')} devices={selected.get('devices')}")

        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        proc = ctx.Process(
            target=_ga_worker,
            args=(task_id, algorithm, dataset, iterations, crossover_rate, mutate_rate, selected, q),
            daemon=True,
        )
        TASK.queue = q
        TASK.process = proc
        proc.start()
        _start_consumer(TASK)

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


@app.post("/api/strategy_compare")
def api_strategy_compare(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return candidate plan comparison for UI explanation."""
    if _StrategyCompare is None:
        raise HTTPException(status_code=500, detail="StrategyCompare not available on this host")
    objective = str(payload.get("objective") or "time")
    multi_gpu = bool(payload.get("multi_gpu", True))
    warmup_iters = int(payload.get("warmup_iters", 0) or 0)
    return _StrategyCompare(objective=objective, multi_gpu=multi_gpu, warmup_iters=warmup_iters)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server_agent:app", host="0.0.0.0", port=7777)
