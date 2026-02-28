from __future__ import annotations

import platform
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

ORDERED_MODES = ("S", "SM", "M", "MNM")


def normalize_mode(mode: Any) -> str:
    raw = str(mode or "AUTO").strip().upper()
    if raw in ("CPU", "AUTO"):
        return "S"
    if raw in ORDERED_MODES:
        return raw
    return "S"


def normalize_devices(devices: Any) -> List[int]:
    if isinstance(devices, list):
        src = devices
    elif devices is None:
        src = []
    else:
        src = [devices]
    out: List[int] = []
    for item in src:
        try:
            out.append(int(item))
        except Exception:
            continue
    uniq: List[int] = []
    for d in out:
        if d not in uniq:
            uniq.append(d)
    return uniq


def detect_capability(
    *,
    target: str,
    resource_snapshot: Optional[Dict[str, Any]] = None,
    remote_reachable: bool = True,
) -> Dict[str, Any]:
    snap = resource_snapshot or {}
    gpus = snap.get("gpus") if isinstance(snap, dict) else None
    gpus = gpus if isinstance(gpus, list) else []
    gpu_count = len(gpus)
    backend = "cpu"
    if gpu_count >= 2:
        backend = "multi-gpu"
    elif gpu_count == 1:
        backend = "cuda"
    os_name = platform.system().lower()
    return {
        "os": os_name,
        "target": target,
        "reachable": bool(remote_reachable),
        "gpu": {"count": gpu_count, "backend": backend},
        "process_model": {
            "spawn_safe": True,
            "fork_safe": os_name != "windows",
        },
    }


def _supported_modes(capability: Dict[str, Any], *, allow_mnm: bool) -> List[str]:
    gpu_count = int((((capability or {}).get("gpu") or {}).get("count") or 0))
    supported = ["S"]
    if gpu_count >= 1:
        supported.append("SM")
    if gpu_count >= 2:
        supported.append("M")
    if allow_mnm:
        supported.append("MNM")
    return supported


def choose_mode(
    requested_mode: Any,
    capability: Dict[str, Any],
    *,
    allow_mnm: bool,
) -> Tuple[str, bool, str]:
    requested = normalize_mode(requested_mode)
    supported = _supported_modes(capability, allow_mnm=allow_mnm)
    if requested in supported:
        return requested, False, ""
    fallback_chain = {
        "MNM": ["M", "SM", "S"],
        "M": ["SM", "S"],
        "SM": ["S"],
        "S": ["S"],
    }
    for candidate in fallback_chain.get(requested, ["S"]):
        if candidate in supported:
            return candidate, True, f"unsupported_mode({requested})->{candidate}"
    return "S", True, f"unsupported_mode({requested})->S"


def build_mode_decision(
    *,
    requested_mode: Any,
    selected_mode: Any,
    devices: Any,
    target: str,
    capability: Dict[str, Any],
    reason: str = "",
    use_strategy_plan: Optional[bool] = None,
) -> Dict[str, Any]:
    requested = normalize_mode(requested_mode)
    selected = normalize_mode(selected_mode)
    devs = normalize_devices(devices)
    degraded = requested != selected
    final_reason = reason or (f"unsupported_mode({requested})->{selected}" if degraded else "")
    payload: Dict[str, Any] = {
        "requested_mode": requested,
        "selected_mode": selected,
        "degraded": degraded,
        "reason": final_reason,
        "code": "MODE_DEGRADED" if degraded else "",
        "target": target,
        "devices": devs,
        "capability": capability,
    }
    if use_strategy_plan is not None:
        payload["use_strategy_plan"] = bool(use_strategy_plan)
    return payload


def transport_contract() -> Dict[str, Any]:
    return {
        "timeouts": {"connect_ms": 3000, "read_ms_default": 10000},
        "retry_policy": {
            "analysis_start": {"max_attempts": 2, "backoff_ms": 200},
            "analysis_status": {"max_attempts": 3, "backoff_ms": 200},
            "analysis_stop": {"max_attempts": 2, "backoff_ms": 200},
            "fitness_batch": {"max_attempts": 2, "backoff_ms": 120},
            "resource_lock": {"max_attempts": 3, "backoff_ms": 250},
            "resource_lock_status": {"max_attempts": 3, "backoff_ms": 250},
        },
        "error_codes": [
            "INVALID_REQUEST",
            "UNKNOWN_SERVER",
            "TASK_BUSY",
            "QUEUE_LIMIT",
            "TIMEOUT",
            "UNREACHABLE",
            "INVALID_RESPONSE",
            "RESOURCE_BUSY",
            "PROXY_FAILED",
            "REMOTE_FAILED",
            "INTERNAL_ERROR",
        ],
    }


def classify_transport_error(
    *,
    status_code: Optional[int] = None,
    exc: Optional[BaseException] = None,
    response_body: Optional[Any] = None,
) -> str:
    if status_code is not None:
        if status_code in (408, 504):
            return "TIMEOUT"
        if status_code == 429:
            return "RESOURCE_BUSY"
        if status_code in (502, 503):
            return "UNREACHABLE"
        if status_code >= 500:
            return "REMOTE_FAILED"
        if status_code >= 400:
            return "INVALID_REQUEST"
        if response_body is not None and not isinstance(response_body, (dict, list, str, int, float, bool, type(None))):
            return "INVALID_RESPONSE"
        return "REMOTE_FAILED"
    if exc is not None:
        msg = str(exc).lower()
        if "timed out" in msg or "timeout" in msg:
            return "TIMEOUT"
        if "connection" in msg or "name or service not known" in msg or "refused" in msg:
            return "UNREACHABLE"
    return "PROXY_FAILED"


def _safe_sleep(ms: float) -> None:
    if ms <= 0:
        return
    time.sleep(ms / 1000.0)


def request_with_retry(
    *,
    session: Any,
    method: str,
    url: str,
    timeout: Tuple[float, float],
    op: str,
    json_payload: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
    allow_status_retry: Optional[List[int]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    if requests is None:  # pragma: no cover
        raise RuntimeError("requests is required")
    contract = transport_contract()
    policy = (contract.get("retry_policy") or {}).get(op) or {"max_attempts": 1, "backoff_ms": 0}
    max_attempts = max(1, int(policy.get("max_attempts") or 1))
    backoff_ms = float(policy.get("backoff_ms") or 0.0)
    retry_status = set(allow_status_retry or [408, 429, 500, 502, 503, 504])

    attempts = 0
    retries = 0
    total_backoff = 0.0
    last_code = ""
    t0 = time.perf_counter()
    last_exc: Optional[BaseException] = None
    last_resp = None
    while attempts < max_attempts:
        attempts += 1
        try:
            resp = session.request(
                method=method.upper(),
                url=url,
                timeout=timeout,
                json=json_payload,
                params=params,
                data=data,
                headers=headers,
            )
            last_resp = resp
            if resp.status_code in retry_status and attempts < max_attempts:
                retries += 1
                last_code = classify_transport_error(status_code=resp.status_code)
                wait_ms = backoff_ms * (2 ** (attempts - 1))
                total_backoff += wait_ms
                _safe_sleep(wait_ms)
                continue
            duration_ms = (time.perf_counter() - t0) * 1000.0
            return resp, {
                "attempts": attempts,
                "retries": retries,
                "backoff_ms": round(total_backoff, 3),
                "duration_ms": round(duration_ms, 3),
                "error_code": classify_transport_error(status_code=resp.status_code) if not resp.ok else "",
            }
        except Exception as exc:
            last_exc = exc
            last_code = classify_transport_error(exc=exc)
            if attempts >= max_attempts:
                break
            retries += 1
            wait_ms = backoff_ms * (2 ** (attempts - 1))
            total_backoff += wait_ms
            _safe_sleep(wait_ms)

    duration_ms = (time.perf_counter() - t0) * 1000.0
    if last_resp is not None:
        return last_resp, {
            "attempts": attempts,
            "retries": retries,
            "backoff_ms": round(total_backoff, 3),
            "duration_ms": round(duration_ms, 3),
            "error_code": last_code or "REMOTE_FAILED",
        }
    raise RuntimeError(
        f"{op} failed after {attempts} attempts: {last_code or 'PROXY_FAILED'}: {last_exc}"
    )
