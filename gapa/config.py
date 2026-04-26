from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dotenv_path() -> Path:
    return _project_root() / ".env"


def _servers_json_path() -> Path:
    return _project_root() / "servers.json"


@lru_cache(maxsize=1)
def _load_dotenv_values() -> Dict[str, str]:
    path = _dotenv_path()
    if not path.exists():
        return {}
    values: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        values[key] = value
    return values


def _setting(name: str) -> str | None:
    value = os.getenv(name)
    if value is not None:
        return value
    return _load_dotenv_values().get(name)


def _env_bool(name: str, default: bool) -> bool:
    value = _setting(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = _setting(name)
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except Exception:
        return default


def get_app_host() -> str:
    return str(_setting("GAPA_APP_HOST") or "127.0.0.1").strip() or "127.0.0.1"


def get_app_port() -> int:
    return _env_int("GAPA_APP_PORT", 5000)


def get_server_agent_host() -> str:
    return str(_setting("GAPA_SERVER_AGENT_HOST") or "0.0.0.0").strip() or "0.0.0.0"


def get_server_agent_port() -> int:
    return _env_int("GAPA_SERVER_AGENT_PORT", 7777)


def get_results_dir(base_dir: Path | None = None) -> Path:
    root = base_dir or _project_root()
    raw = _setting("GAPA_RESULTS_DIR")
    if not raw:
        return (root / "results").resolve()
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def get_dataset_dir(base_dir: Path | None = None) -> Path:
    root = base_dir or _project_root()
    raw = _setting("GAPA_DATASET_DIR")
    if raw:
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = root / path
        return path.resolve()

    candidates = [
        root / "datasets",
        root / "dataset",
        root / "gapa" / "datasets",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return (root / "datasets").resolve()


def get_resource_filters() -> Dict[str, Any]:
    return {
        "min_gpu_count": _env_int("GAPA_RESOURCE_MIN_GPU_COUNT", 0),
        "require_cuda": _env_bool("GAPA_RESOURCE_REQUIRE_CUDA", False),
        "require_mps": _env_bool("GAPA_RESOURCE_REQUIRE_MPS", False),
    }


def get_app_base_url() -> str:
    return f"http://{get_app_host()}:{get_app_port()}"


def build_remote_server_entries() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in _load_servers_json_entries():
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        base_url = str(entry.get("base_url") or "").strip()
        host = str(entry.get("ip") or entry.get("host") or "").strip()
        port = entry.get("port")
        protocol = str(entry.get("protocol") or "http").strip() or "http"
        if not base_url and host:
            base_url = f"{protocol}://{host}{f':{port}' if port else ''}"
        if not base_url:
            continue
        if base_url in seen:
            continue
        seen.add(base_url)
        entry["base_url"] = base_url.rstrip("/")
        entry.setdefault("protocol", protocol)
        entry.setdefault("host", host)
        entry.setdefault("ip", host)
        entry.setdefault("type", "remote")
        if port is not None:
            entry["port"] = port
        entry.setdefault("id", f"{host}-{port}" if host and port else (host or entry["base_url"]))
        entry.setdefault("name", host or entry["id"])
        entries.append(entry)
    return entries


def _load_servers_json_entries() -> List[Dict[str, Any]]:
    path = _servers_json_path()
    if not path.exists():
        return []
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(parsed, dict):
        servers = parsed.get("servers")
    else:
        servers = parsed
    if not isinstance(servers, list):
        return []
    return [item for item in servers if isinstance(item, dict)]
