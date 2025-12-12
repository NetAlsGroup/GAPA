from __future__ import annotations

import json
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
    import torch.cuda as cuda
except ImportError:
    torch = cuda = None

try:
    import pynvml  # type: ignore

    _HAS_NVML = True
except Exception:
    pynvml = None
    _HAS_NVML = False

# Paths and shared executor for resource collection
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_ROOT = BASE_DIR / "web"
SERVERS_FILE = Path(os.getenv("GAPA_SERVERS_FILE", BASE_DIR / "servers.json"))
_executor = ThreadPoolExecutor(max_workers=10)


def _ensure_server_file() -> None:
    """Create a minimal server config file when it is missing."""
    if SERVERS_FILE.exists():
        return
    default_data = {
        "servers": [
            {
                "id": "local",
                "name": "本机",
                "base_url": "",
            }
        ]
    }
    SERVERS_FILE.write_text(json.dumps(default_data, indent=4, ensure_ascii=False), encoding="utf-8")
    print("已自动生成 servers.json")


def _normalize_server_entry(entry: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Return (id, entry) pair with a best-effort base_url for remote servers."""
    server_id = entry.get("id") or entry.get("name") or "remote"
    base_url = entry.get("base_url")
    if not base_url:
        host = entry.get("ip") or entry.get("host")
        port = entry.get("port")
        protocol = entry.get("protocol") or "http"
        if host:
            base_url = f"{protocol}://{host}{f':{port}' if port else ''}"
        entry["base_url"] = base_url or ""
    entry.setdefault("name", server_id)
    entry["id"] = server_id
    return server_id, entry


def load_server_list() -> List[Dict[str, str]]:
    """Safely read the server list and ensure the local host is first."""
    _ensure_server_file()
    try:
        config = json.loads(SERVERS_FILE.read_text(encoding="utf-8"))
        servers = config.get("servers", [])
    except Exception as exc:  # pragma: no cover - defensive path
        print(f"加载服务器列表失败：{exc}，回退到仅本机模式")
        return [{"id": "local", "name": "本机", "base_url": ""}]

    seen = set()
    unique_servers: List[Dict[str, Any]] = []
    for entry in servers:
        if not isinstance(entry, dict):
            continue
        server_id, normalized = _normalize_server_entry(entry)
        if server_id in seen:
            continue
        seen.add(server_id)
        unique_servers.append(normalized)

    has_local = any(s.get("id") == "local" for s in unique_servers)
    if not has_local:
        unique_servers.insert(0, {"id": "local", "name": "本机", "base_url": ""})
    else:
        local_index = next(i for i, s in enumerate(unique_servers) if s.get("id") == "local")
        local_server = unique_servers.pop(local_index)
        local_server["name"] = "本机"
        unique_servers.insert(0, local_server)

    return unique_servers


async def fetch_resource_from(url: str) -> Dict[str, Any]:
    """Async resource fetch helper kept for compatibility."""
    try:
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=6)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            api_url = url.rstrip("/") + "/api/resources"
            async with session.get(api_url) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception:
        pass
    return {"error": "连接超时或拒绝访问", "hostname": url or "未知"}


def _cpu_snapshot() -> Dict[str, Any]:
    if not psutil:
        return {"usage_percent": None, "load_avg": None}
    return {
        "usage_percent": psutil.cpu_percent(interval=0.1),
        "load_avg": os.getloadavg() if hasattr(os, "getloadavg") else None,
    }


def _memory_snapshot() -> Dict[str, Any]:
    if not psutil:
        return {"total_mb": None, "used_mb": None, "percent": None}
    mem = psutil.virtual_memory()
    return {
        "total_mb": round(mem.total / (1024 ** 2)),
        "used_mb": round(mem.used / (1024 ** 2)),
        "percent": mem.percent,
    }


def _gpu_snapshot() -> List[Dict[str, Any]]:
    """Collect GPU information (memory + optional power) without failing on GPU-less machines."""
    gpus: List[Dict[str, Any]] = []

    # NVML preferred for memory/utilization/power
    if _HAS_NVML:
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h).decode("utf-8")
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    gpu_util_percent = float(util.gpu)
                    mem_util_percent = float(util.memory)
                except Exception:
                    gpu_util_percent = mem_util_percent = None
                try:
                    power_w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                except Exception:
                    power_w = None
                gpus.append(
                    {
                        "id": i,
                        "name": name,
                        "total_mb": round(mem.total / (1024**2)),
                        "free_mb": round(mem.free / (1024**2)),
                        "used_mb": round(mem.used / (1024**2)),
                        "load": (gpu_util_percent / 100.0) if gpu_util_percent is not None else None,
                        "gpu_util_percent": gpu_util_percent,
                        "mem_util_percent": mem_util_percent,
                        "power_w": power_w,
                    }
                )
        except Exception:
            gpus = []
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    # Torch fallback for memory only (no NVML on this host)
    if not gpus and torch and cuda and cuda.is_available():
        for i in range(cuda.device_count()):
            try:
                props = cuda.get_device_properties(i)
                try:
                    free_bytes, total_bytes = cuda.mem_get_info(i)
                    free_mb = round(free_bytes / (1024 ** 2))
                    total_mb = round(total_bytes / (1024 ** 2))
                    used_mb = total_mb - free_mb
                except Exception:
                    total_mb = round(props.total_memory / (1024 ** 2))
                    free_mb = used_mb = None

                gpus.append(
                    {
                        "id": i,
                        "name": props.name,
                        "total_mb": total_mb,
                        "free_mb": free_mb,
                        "used_mb": used_mb,
                        "multi_processor_count": props.multi_processor_count,
                        "load": None,
                        "gpu_util_percent": None,
                        "mem_util_percent": None,
                        "power_w": None,
                    }
                )
            except Exception:
                continue

    return gpus


def current_resource_snapshot() -> Dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cpu": _cpu_snapshot(),
        "memory": _memory_snapshot(),
        "gpus": _gpu_snapshot(),
        "python": sys.version.split()[0],
    }


def _local_base_url(remote_addr: Optional[str]) -> str:
    if remote_addr and remote_addr != "127.0.0.1":
        return f"http://{remote_addr}:7777"
    return "http://127.0.0.1:7777"


def get_all_resources(remote_addr: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Concurrently collect resources from all configured servers."""
    servers = load_server_list()
    results: Dict[str, Dict[str, Any]] = {}

    def fetch_one(server: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        base_url = server.get("base_url") or _local_base_url(remote_addr)
        server_id = server.get("id") or server.get("name") or base_url
        data = current_resource_snapshot() if server_id == "local" else None
        if data is None:
            try:
                import requests

                resp = requests.get(base_url.rstrip("/") + "/api/resources", timeout=5)
                data = resp.json() if resp.ok else {"error": f"HTTP {resp.status_code}"}
            except Exception as exc:  # pragma: no cover - network issues
                data = {"error": str(exc)}
        # Normalize timestamp field from agent to UI expected time
        if isinstance(data, dict) and "time" not in data and "timestamp" in data:
            data["time"] = data.get("timestamp")
        data["name"] = server.get("name", server_id)
        return server_id, data

    futures = [_executor.submit(fetch_one, server) for server in servers]
    for future in as_completed(futures, timeout=10):
        sid, data = future.result()
        results[sid] = data

    return results


def load_server_config(mask_password: bool = False) -> List[Dict[str, Any]]:
    """Load remote server config from JSON file; optionally mask passwords."""
    if not SERVERS_FILE.exists():
        return []
    try:
        raw = json.loads(SERVERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(raw, dict):
        raw = raw.get("servers", [])
    if not isinstance(raw, list):
        return []

    servers: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        host = item.get("ip") or item.get("host")
        port = item.get("port")
        name = item.get("name") or host or "remote"
        protocol = item.get("protocol") or "http"
        server_type = item.get("type") or item.get("os") or "linux"
        server_id = item.get("id") or (f"{name}:{port}" if port else name)

        servers.append(
            {
                "id": server_id,
                "name": name,
                "ip": host,
                "port": port,
                "username": item.get("username"),
                "password": "***" if mask_password and item.get("password") else item.get("password"),
                "type": server_type,
                "protocol": protocol,
            }
        )
    return servers
