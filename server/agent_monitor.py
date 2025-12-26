from __future__ import annotations

import socket
import time
from typing import Any, Dict, List

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

try:
    import pynvml  # type: ignore
except Exception:
    pynvml = None


def cpu_snapshot() -> Dict[str, Any]:
    if not psutil:
        return {"usage_percent": None, "cores": None}
    try:
        return {
            "usage_percent": psutil.cpu_percent(interval=0.1),
            "cores": psutil.cpu_count(logical=True),
        }
    except Exception:
        return {"usage_percent": None, "cores": psutil.cpu_count(logical=True) if psutil else None}


def memory_snapshot() -> Dict[str, Any]:
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


def gpu_snapshot() -> List[Dict[str, Any]]:
    if not pynvml:
        return []

    gpus: List[Dict[str, Any]] = []
    try:
        pynvml.nvmlInit()
        try:
            pynvml.nvmlDeviceSetPersistenceMode(pynvml.nvmlDeviceGetHandleByIndex(0), 1)
        except Exception:
            pass

        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mb = round(mem_info.total / 1024**2)
            used_mb = round(mem_info.used / 1024**2)
            free_mb = round(mem_info.free / 1024**2)

            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = float(util.gpu)
                mem_util = float(util.memory)
            except Exception:
                gpu_util = mem_util = 0.0

            power_w = None
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_w = round(power_mw / 1000.0, 2)
            except Exception:
                power_w = None

            temperature_c = None
            try:
                temperature_c = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temperature_c = None

            gpus.append(
                {
                    "id": i,
                    "name": name,
                    "total_mb": total_mb,
                    "used_mb": used_mb,
                    "free_mb": free_mb,
                    "gpu_util_percent": gpu_util,
                    "mem_util_percent": mem_util,
                    "power_w": power_w,
                    "temperature_c": temperature_c,
                }
            )
    except Exception:
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return gpus


def resources_payload() -> Dict[str, Any]:
    return {
        "status": "online",
        "hostname": socket.gethostname(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cpu": cpu_snapshot(),
        "memory": memory_snapshot(),
        "gpus": gpu_snapshot(),
    }


def resource_point(payload: Dict[str, Any]) -> Dict[str, Any]:
    gpus = []
    for g in payload.get("gpus", []) or []:
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
        "timestamp": payload.get("timestamp"),
        "cpu_usage_percent": (payload.get("cpu") or {}).get("usage_percent"),
        "memory_percent": (payload.get("memory") or {}).get("percent"),
        "gpus": gpus,
    }

