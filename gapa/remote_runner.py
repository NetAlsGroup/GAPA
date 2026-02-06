from __future__ import annotations

from typing import Any, Dict, List

try:
    import requests as _requests  # type: ignore
except Exception:  # pragma: no cover
    _requests = None

if _requests is None:  # pragma: no cover - test environments without requests
    class _RequestsStub:
        post = None
        get = None
    requests = _RequestsStub()
    _HAS_REQUESTS = False
else:
    requests = _requests
    _HAS_REQUESTS = True


def resolve_algorithm_id(algorithm: Any) -> str:
    for attr in ("algo_id", "algorithm_id", "name", "id"):
        val = getattr(algorithm, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return algorithm.__class__.__name__


def extract_iteration_progress(logs: Any) -> Dict[str, int] | None:
    if not isinstance(logs, list):
        return None
    import re
    pattern = re.compile(r"iter\s+(\d+)\s*/\s*(\d+)")
    last = None
    for line in logs:
        if not isinstance(line, str):
            continue
        m = pattern.search(line)
        if m:
            try:
                last = {"current": int(m.group(1)), "total": int(m.group(2))}
            except Exception:
                last = None
    return last


def _match_server(server: Dict[str, Any], query: str) -> bool:
    sid = str(server.get("id") or "")
    name = str(server.get("name") or "")
    q = str(query or "")
    if not q:
        return False
    if q == sid or q == name:
        return True
    if q.lower() == sid.lower() or q.lower() == name.lower():
        return True
    if q.isdigit() and sid.lower() == f"server {q}".lower():
        return True
    return False


def select_online_server(monitor: Any, server_query: str) -> Dict[str, Any]:
    servers = monitor.server()
    if isinstance(servers, dict) and servers.get("error"):
        return {"error": "server list error", "detail": servers}
    if not isinstance(servers, list):
        return {"error": "invalid server list", "detail": servers}
    for s in servers:
        if isinstance(s, dict) and _match_server(s, server_query):
            if s.get("status") != "Activate":
                return {"error": "server offline", **s}
            return s
    return {"error": "server not found", "input": server_query, "servers": servers}


def start_remote_run(
    server: Dict[str, Any],
    *,
    algorithm: str,
    dataset: str,
    iterations: int,
    mode: str,
    crossover_rate: float,
    mutate_rate: float,
    timeout_s: float = 20.0,
) -> Dict[str, Any]:
    base_url = (server.get("base_url") or "").rstrip("/")
    if not base_url:
        return {"error": "missing base_url", "server": server}
    url = base_url + "/api/analysis/start"
    payload = {
        "algorithm": algorithm,
        "dataset": dataset,
        "iterations": int(iterations),
        "crossover_rate": float(crossover_rate),
        "mutate_rate": float(mutate_rate),
        "mode": str(mode or "").upper(),
    }
    if getattr(requests, "post", None) is None:
        return {"error": "requests not available", "url": url}
    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
    except Exception as exc:
        return {"error": str(exc), "url": url}
    if not resp.ok:
        try:
            body = resp.json()
        except Exception:
            body = {"raw": getattr(resp, "text", "")}
        return {"error": f"HTTP {resp.status_code}", "body": body, "url": url}
    try:
        return resp.json()
    except Exception as exc:
        return {"error": f"invalid json: {exc}", "url": url}


def poll_remote_status(
    server: Dict[str, Any],
    *,
    max_polls: int = 120,
    interval_s: float = 1.0,
    progress_cb: Any = None,
) -> Dict[str, Any]:
    base_url = (server.get("base_url") or "").rstrip("/")
    if not base_url:
        return {"error": "missing base_url", "server": server}
    if getattr(requests, "get", None) is None:
        return {"error": "requests not available", "url": base_url}
    url = base_url + "/api/analysis/status"
    last = {"state": "unknown"}
    for _ in range(max_polls):
        try:
            resp = requests.get(url, timeout=10)
        except Exception as exc:
            return {"error": str(exc), "url": url}
        if not resp.ok:
            try:
                body = resp.json()
            except Exception:
                body = {"raw": getattr(resp, "text", "")}
            return {"error": f"HTTP {resp.status_code}", "body": body, "url": url}
        try:
            data = resp.json()
        except Exception as exc:
            return {"error": f"invalid json: {exc}", "url": url}
        last = data if isinstance(data, dict) else {"state": "unknown", "raw": data}
        if progress_cb:
            try:
                info = extract_iteration_progress(last.get("logs"))
                progress_cb(info, last)
            except Exception:
                pass
        if last.get("state") in ("completed", "error", "idle"):
            return last
        if interval_s:
            import time
            time.sleep(interval_s)
    return last


def run_remote_task(
    monitor: Any,
    server_query: str,
    *,
    algorithm: str,
    dataset: str,
    iterations: int,
    mode: str,
    crossover_rate: float,
    mutate_rate: float,
    max_polls: int = 600,
    interval_s: float = 1.0,
) -> Dict[str, Any]:
    server = select_online_server(monitor, server_query)
    if server.get("error"):
        return server
    started = start_remote_run(
        server,
        algorithm=algorithm,
        dataset=dataset,
        iterations=iterations,
        mode=mode,
        crossover_rate=crossover_rate,
        mutate_rate=mutate_rate,
    )
    if started.get("error"):
        return started
    last_seen = {"current": None, "total": None}
    def _progress_cb(info, _data):
        if not info:
            return
        cur = info.get("current")
        total = info.get("total")
        if cur == last_seen["current"] and total == last_seen["total"]:
            return
        last_seen["current"] = cur
        last_seen["total"] = total
        print(f"[INFO] iter {cur}/{total}")

    final = poll_remote_status(server, max_polls=max_polls, interval_s=interval_s, progress_cb=_progress_cb)
    best = None
    if isinstance(final.get("result"), dict):
        result = final["result"]
        best = result.get("best_score")
        if not isinstance(best, (int, float)):
            objectives = result.get("objectives") or {}
            primary = objectives.get("primary")
            best_metrics = result.get("best_metrics") or {}
            if primary and isinstance(best_metrics, dict) and primary in best_metrics:
                best = best_metrics.get(primary)
            if best is None:
                curves = result.get("curves") or {}
                if primary and isinstance(curves, dict):
                    series = curves.get(primary)
                    if isinstance(series, list) and series:
                        best = series[-1]
    if best is not None and hasattr(monitor, "_best_fitness"):
        try:
            monitor._best_fitness = float(best)
        except Exception:
            pass
    if isinstance(final.get("result"), dict) and hasattr(monitor, "_remote_result"):
        try:
            monitor._remote_result = final["result"]
        except Exception:
            pass
    return final
