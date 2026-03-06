from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dotenv_path() -> Path:
    return _project_root() / ".env"


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


def _env_json_list(name: str, default: List[str] | None = None) -> List[str]:
    value = _setting(name)
    if value is None or not str(value).strip():
        return list(default or [])
    try:
        parsed = json.loads(value)
    except Exception:
        return list(default or [])
    if not isinstance(parsed, list):
        return list(default or [])
    return [str(item).strip() for item in parsed if str(item).strip()]


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
    return Path(_setting("GAPA_RESULTS_DIR") or str(root / "results")).expanduser().resolve()


def get_remote_servers() -> List[str]:
    return _env_json_list("GAPA_REMOTE_SERVERS", [])


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
    for index, url in enumerate(get_remote_servers(), start=1):
        base_url = str(url).rstrip("/")
        if not base_url or base_url in seen:
            continue
        seen.add(base_url)
        parsed = urlparse(base_url)
        host = parsed.hostname or ""
        port = parsed.port
        protocol = parsed.scheme or "http"
        server_id = f"{host}-{port}" if host and port else (host or f"remote-{index}")
        entries.append(
            {
                "id": server_id,
                "name": host or f"remote-{index}",
                "host": host,
                "ip": host,
                "port": port,
                "protocol": protocol,
                "base_url": base_url,
                "type": "remote",
            }
        )
    return entries
