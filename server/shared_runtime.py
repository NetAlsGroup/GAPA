from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
import uuid
from typing import Any, Callable, Dict, Optional, Tuple


def summarize_warmup_result(result: Dict[str, Any] | None) -> Dict[str, Any]:
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


def terminate_process_tree(proc: Any, pid: int | None, sig: int = signal.SIGTERM) -> None:
    if not pid:
        try:
            proc.terminate() if sig == signal.SIGTERM else proc.kill()
        except Exception:
            pass
        return
    try:
        import psutil  # type: ignore

        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.send_signal(sig)
            except psutil.NoSuchProcess:
                pass
        parent.send_signal(sig)
        return
    except Exception:
        pass

    if os.name == "posix":
        try:
            os.killpg(pid, sig)
            return
        except Exception:
            pass
        try:
            os.kill(pid, sig)
            return
        except Exception:
            pass
    else:
        try:
            proc.terminate() if sig == signal.SIGTERM else proc.kill()
        except Exception:
            pass


def run_ga_warmup_local(
    *,
    payload: Dict[str, Any],
    task: Any,
    task_lock: Any,
    busy_message: str,
    ga_entry: Callable[..., Any],
    select_run_mode: Callable[[Any, Any], Dict[str, Any]],
    require_algorithm_dataset: bool = False,
) -> Dict[str, Any] | Tuple[Dict[str, Any], int]:
    algorithm = str(payload.get("algorithm") or "")
    dataset = str(payload.get("dataset") or "")
    if require_algorithm_dataset and (not algorithm or not dataset):
        return {"error": "algorithm and dataset are required"}, 400

    iterations = int(payload.get("iterations") or payload.get("warmup_iters") or 2)
    crossover_rate = float(payload.get("crossover_rate") or payload.get("pc") or 0.8)
    mutate_rate = float(payload.get("mutate_rate") or payload.get("pm") or 0.2)
    selected = select_run_mode(payload.get("mode"), payload.get("devices"))
    remote_servers = payload.get("remote_servers") or payload.get("allowed_server_ids")
    if remote_servers is not None:
        selected["remote_servers"] = remote_servers
    timeout_s = float(payload.get("timeout_s", 180) or 180)

    with task_lock:
        if str(getattr(task, "state", "")) == "running":
            return {"error": busy_message}, 409

    task_id = str(uuid.uuid4())
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(
        target=ga_entry,
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

    summary = summarize_warmup_result(result)
    return {
        "task_id": task_id,
        "state": state,
        "summary": summary["summary"],
        "per_iter_ms": summary["per_iter_ms"],
        "comm": (result or {}).get("comm"),
        "logs": logs[-200:],
        "error": error,
    }
