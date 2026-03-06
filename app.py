# filename: app.py
from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
from pathlib import Path
import uuid
from datetime import datetime

from flask import Flask, jsonify, request, send_from_directory

from autoadapt import StrategyPlan, DistributedStrategyPlan

from server import (
    STATIC_ROOT,
    JobStore,
    current_resource_snapshot,
    get_all_resources,
    load_server_config,
    load_server_list,
    db_manager,
    state_manager,
)
from server.resource_lock import LOCK_MANAGER
from server.agent_state import TaskState, start_consumer
from server.ga_worker import ga_worker, select_run_mode
from server.mode_runtime import (
    build_mode_decision,
    choose_mode,
    classify_transport_error,
    detect_capability,
    request_with_retry,
    transport_contract,
)
from server.api_schemas import (
    CURRENT_SCHEMA_VERSION,
    DEFAULT_SCHEMA_VERSION,
    ErrorResponse,
    HTTPStatus,
    TERMINAL_TASK_STATES,
    build_resume_metadata,
    is_terminal_task_state,
    make_error_response,
    make_paginated_response,
    normalize_task_state,
    normalize_mode_decision,
    resolve_schema_version,
)
from server.shared_runtime import (
    run_ga_warmup_local as run_ga_warmup_local_shared,
    summarize_warmup_result,
    terminate_process_tree,
)

import threading
import time

HISTORY_FILE = (Path(__file__).resolve().parent / "results" / "history.json").resolve()
HISTORY_LOCK = threading.Lock()
STRATEGY_PROGRESS: dict[str, dict[str, object]] = {}
STRATEGY_PROGRESS_LOCK = threading.Lock()
TRANSPORT_METRICS: dict[str, dict[str, float | int]] = {}
TRANSPORT_METRICS_LOCK = threading.Lock()
DEGRADE_REASON_COUNTS: dict[str, int] = {}
DEGRADE_REASON_LOCK = threading.Lock()


def _load_history() -> list[dict]:
    return db_manager.get_history()


def _save_history(items: list[dict]) -> None:
    # This is now handled by db_manager internally or via add_history
    # Providing a compatibility stub for any leftover calls
    for item in items:
        db_manager.add_history(item)


def _set_strategy_progress(progress_id: str, *, current: int, total: int, status: str, server_id: str | None = None) -> None:
    if not progress_id:
        return
    with STRATEGY_PROGRESS_LOCK:
        STRATEGY_PROGRESS[progress_id] = {
            "current": int(current),
            "total": int(total),
            "status": str(status),
            "server_id": server_id,
            "updated_at": time.time(),
        }


def _get_strategy_progress(progress_id: str) -> dict[str, object] | None:
    with STRATEGY_PROGRESS_LOCK:
        return STRATEGY_PROGRESS.get(progress_id)


def _record_transport_metric(op: str, meta: dict | None, ok: bool) -> None:
    if not meta:
        return
    with TRANSPORT_METRICS_LOCK:
        row = TRANSPORT_METRICS.setdefault(
            op,
            {
                "calls": 0,
                "failures": 0,
                "retries": 0,
                "backoff_ms": 0.0,
                "recovery_ms_sum": 0.0,
            },
        )
        row["calls"] = int(row.get("calls", 0)) + 1
        if not ok:
            row["failures"] = int(row.get("failures", 0)) + 1
        row["retries"] = int(row.get("retries", 0)) + int(meta.get("retries") or 0)
        row["backoff_ms"] = float(row.get("backoff_ms", 0.0)) + float(meta.get("backoff_ms") or 0.0)
        if int(meta.get("retries") or 0) > 0 and ok:
            row["recovery_ms_sum"] = float(row.get("recovery_ms_sum", 0.0)) + float(meta.get("duration_ms") or 0.0)


def _transport_metric_summary() -> dict[str, dict[str, float | int]]:
    with TRANSPORT_METRICS_LOCK:
        out: dict[str, dict[str, float | int]] = {}
        for op, row in TRANSPORT_METRICS.items():
            calls = int(row.get("calls", 0))
            failures = int(row.get("failures", 0))
            retries = int(row.get("retries", 0))
            recovered = max(0, calls - failures)
            recovery_avg_ms = float(row.get("recovery_ms_sum", 0.0)) / max(1, recovered)
            out[op] = {
                "calls": calls,
                "failures": failures,
                "failure_rate": round((failures / max(1, calls)), 4),
                "retries": retries,
                "avg_retries_per_call": round(retries / max(1, calls), 4),
                "avg_recovery_ms": round(recovery_avg_ms, 3),
            }
        return out


def _record_degrade_reason(mode_decision: dict | None) -> None:
    if not isinstance(mode_decision, dict):
        return
    reason = str(mode_decision.get("reason") or "").strip()
    if not reason:
        return
    with DEGRADE_REASON_LOCK:
        DEGRADE_REASON_COUNTS[reason] = int(DEGRADE_REASON_COUNTS.get(reason, 0)) + 1


def _response_with_schema(body: dict, schema_version: str) -> dict:
    out = dict(body or {})
    out.setdefault("schema_version", schema_version or DEFAULT_SCHEMA_VERSION)
    out.setdefault("schema_current", CURRENT_SCHEMA_VERSION)
    return out


def _resolve_server_base_url(server_id: str | None) -> str | None:
    if not server_id or server_id == "local":
        return None
    servers = load_server_list()
    target = next((s for s in servers if s.get("id") == server_id), None)
    if target is None and isinstance(server_id, str) and "-" in server_id:
        try:
            ip_part, port_part = server_id.rsplit("-", 1)
            port_val = int(port_part)
            target = next(
                (
                    s
                    for s in servers
                    if (s.get("ip") == ip_part or s.get("host") == ip_part)
                    and int(s.get("port") or 7777) == port_val
                ),
                None,
            )
        except Exception:
            target = None
    return target.get("base_url") if target else None


def _summarize_warmup_result(result: dict | None) -> dict:
    return summarize_warmup_result(result)


def _ga_entry(
    task_id: str,
    algorithm: str,
    dataset: str,
    iterations: int,
    pc: float,
    pm: float,
    selected: dict,
    q: mp.Queue,
    resume_id: str | None = None,
):
    """Function to run ga_worker in a subprocess; handles result persistence."""
    try:
        resume_state = None
        if resume_id:
            resume_state = db_manager.get_ga_state(resume_id)

        ga_worker(
            task_id,
            algorithm,
            dataset,
            iterations,
            pc,
            pm,
            selected,
            q,
            resume_state=resume_state,
        )
    except Exception as exc:
        q.put({"type": "state", "state": "error", "error": str(exc)})


def _run_ga_warmup_local(payload: dict) -> dict:
    return run_ga_warmup_local_shared(
        payload=payload,
        task=LOCAL_TASK,
        task_lock=LOCAL_TASK.lock,
        busy_message="A task is already running",
        ga_entry=_ga_entry,
        select_run_mode=select_run_mode,
        require_algorithm_dataset=False,
    )


LOCAL_TASK = TaskState()

# 使用 /static 避免与 /api/* 路由冲突
app = Flask(__name__, static_folder=str(STATIC_ROOT), static_url_path="/static")
store = JobStore()
ALGO_MANIFEST_PATH = Path(__file__).resolve().parent / "gapa" / "algorithm" / "manifest.json"
DATASETS_MANIFEST_PATH = Path(__file__).resolve().parent / "datasets" / "registry.json"


@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,DELETE,OPTIONS"
    return resp


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "dashboard.html")


# ============================================================================
# Global Error Handler
# ============================================================================

@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler for standardized error responses."""
    from werkzeug.exceptions import HTTPException
    
    if isinstance(e, HTTPException):
        return jsonify(make_error_response(
            e.name.replace(" ", ""),
            e.description or str(e),
            request.path
        )), e.code
    
    # Log unexpected errors
    import traceback
    traceback.print_exc()
    
    return jsonify(make_error_response(
        "InternalServerError",
        str(e),
        request.path
    )), HTTPStatus.INTERNAL_SERVER_ERROR


@app.errorhandler(404)
def handle_not_found(e):
    return jsonify(make_error_response(
        "NotFound",
        f"Resource not found: {request.path}",
        request.path
    )), HTTPStatus.NOT_FOUND


# ============================================================================
# V1 API - Core Resource Endpoints
# ============================================================================

@app.route("/api/v1/resources", methods=["GET"])
def api_v1_resources():
    """Get current resource snapshot for local server."""
    try:
        return jsonify(current_resource_snapshot())
    except Exception as exc:
        return jsonify(make_error_response(
            "ResourceError",
            f"Failed to fetch resources: {exc}",
            request.path
        )), HTTPStatus.INTERNAL_SERVER_ERROR


@app.route("/api/v1/resources/all", methods=["GET"])
def api_v1_all_resources():
    """Get resource snapshots from all registered servers."""
    return jsonify(state_manager.get_snapshots())


@app.route("/api/v1/servers", methods=["GET"])
def api_v1_servers():
    """List all registered compute servers."""
    return jsonify(load_server_config(mask_password=True))


@app.route("/api/v1/algorithms", methods=["GET"])
def api_v1_algorithms():
    """List available GA algorithms."""
    entries = []
    try:
        from server.algorithm_registry import load_algorithm_entries

        entries = load_algorithm_entries() or []
    except Exception:
        entries = []
    manifest = []
    if ALGO_MANIFEST_PATH.exists():
        try:
            raw = json.loads(ALGO_MANIFEST_PATH.read_text(encoding="utf-8"))
            manifest = raw if isinstance(raw, list) else []
        except Exception as exc:
            return jsonify(make_error_response(
                "ManifestError",
                f"Failed to load algorithms: {exc}",
                request.path
            )), HTTPStatus.INTERNAL_SERVER_ERROR
    if not manifest:
        return jsonify(entries)
    if not entries:
        return jsonify(manifest)
    merged = []
    seen = set()
    for item in manifest + entries:
        if not isinstance(item, dict):
            continue
        algo_id = item.get("id") or item.get("name")
        if algo_id in seen:
            continue
        seen.add(algo_id)
        merged.append(item)
    return jsonify(merged)


@app.route("/api/v1/datasets", methods=["GET"])
def api_v1_datasets():
    """List available datasets."""
    if not DATASETS_MANIFEST_PATH.exists():
        return jsonify({})
    try:
        raw = json.loads(DATASETS_MANIFEST_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return jsonify({})
        algorithms = raw.get("algorithms", {})
        by_algorithm = {}
        if isinstance(algorithms, dict):
            for name, meta in algorithms.items():
                if isinstance(meta, dict):
                    by_algorithm[name] = list(meta.get("datasets", []))
        return jsonify(
            {
                "datasets": raw.get("datasets", {}),
                "algorithms": algorithms,
                "by_algorithm": by_algorithm,
            }
        )
    except Exception as exc:
        return jsonify(make_error_response(
            "ManifestError",
            f"Failed to load datasets: {exc}",
            request.path
        )), HTTPStatus.INTERNAL_SERVER_ERROR


# ============================================================================
# Legacy API Routes (for backward compatibility)
# ============================================================================

@app.route("/api/all_resources")
def api_all_resources():
    """Legacy: Returns snapshots from the background state_manager."""
    return api_v1_all_resources()


@app.route("/api/resources")
def api_resources():
    """Legacy: Get local resource snapshot."""
    return api_v1_resources()


@app.route("/api/servers", methods=["GET"])
def api_servers():
    """Legacy: List servers."""
    return api_v1_servers()


@app.route("/api/algorithms", methods=["GET"])
def api_algorithms():
    """Legacy: List algorithms."""
    return api_v1_algorithms()


@app.route("/api/datasets", methods=["GET"])
def api_datasets():
    """Legacy: List datasets."""
    return api_v1_datasets()


@app.route("/api/strategy_plan", methods=["POST"])
def api_strategy_plan():
    """Generate an adaptive resource plan via StrategyPlan (static profiling)."""
    payload = request.get_json(silent=True) or {}
    server_id = payload.get("server_id") or payload.get("server")
    warmup = int(payload.get("warmup", 0) or 0)
    objective = str(payload.get("objective") or "time")
    multi_gpu = bool(payload.get("multi_gpu", True))
    gpu_busy_threshold = payload.get("gpu_busy_threshold")
    min_gpu_free_mb = payload.get("min_gpu_free_mb")
    tpe_trials = payload.get("tpe_trials")
    tpe_warmup = payload.get("tpe_warmup")
    progress_id = payload.get("progress_id")
    algo = payload.get("algorithm")
    try:
        # If a remote server is selected, proxy to its own agent to compute plan locally.
        if server_id and server_id != "local":
            base_url = _resolve_server_base_url(server_id)
            if not base_url:
                servers = load_server_list()
                return (
                    jsonify(
                        {
                            "error": "unknown server_id or missing base_url",
                            "server_id": server_id,
                            "known_servers": [
                                {"id": s.get("id"), "name": s.get("name"), "base_url": s.get("base_url")}
                                for s in servers
                            ],
                        }
                    ),
                    404,
                )

            try:
                import requests

                session = requests.Session()
                session.trust_env = False  # avoid routing LAN traffic via HTTP(S)_PROXY
                timeout_s = float(payload.get("timeout_s", 600) or 600)
                resp = session.post(
                    base_url.rstrip("/") + "/api/strategy_plan",
                    json={
                        "algorithm": algo,
                        "warmup": warmup,
                        "objective": objective,
                        "multi_gpu": multi_gpu,
                        "gpu_busy_threshold": gpu_busy_threshold,
                        "min_gpu_free_mb": min_gpu_free_mb,
                        "tpe_trials": tpe_trials,
                        "tpe_warmup": tpe_warmup,
                        "progress_id": progress_id,
                    },
                    timeout=(3.0, timeout_s),
                )
                if resp.ok:
                    return jsonify(resp.json())
                # Do not silently fall back to local plan; remote is authoritative here.
                try:
                    body = resp.json()
                except Exception:
                    body = {"raw": (resp.text or "")[:2000]}
                return (
                    jsonify(
                        {
                            "error": "remote strategy_plan failed",
                            "server_id": server_id,
                            "base_url": base_url,
                            "status_code": resp.status_code,
                            "body": body,
                        }
                    ),
                    502,
                )
            except Exception as exc:
                return jsonify({"error": f"proxy failed: {exc}", "server_id": server_id, "base_url": base_url}), 502

        snap = current_resource_snapshot()
        if progress_id:
            total_trials = int(tpe_trials or os.getenv("GAPA_TPE_TRIALS", "6") or 6)
            _set_strategy_progress(progress_id, current=0, total=total_trials, status="running", server_id=str(server_id or "local"))
        def _progress_cb(cur: int, total: int, status: str) -> None:
            _set_strategy_progress(progress_id, current=cur, total=total, status=status, server_id=str(server_id or "local"))
        plan = StrategyPlan(
            fitness=None,
            warmup=warmup,
            objective=objective,
            multi_gpu=multi_gpu,
            resource_snapshot=snap,
            gpu_busy_threshold=gpu_busy_threshold,
            min_gpu_free_mb=min_gpu_free_mb,
            tpe_trials=tpe_trials,
            tpe_warmup=tpe_warmup,
            progress_cb=_progress_cb if progress_id else None,
        )
        if algo:
            try:
                plan.notes = (plan.notes + " " if plan.notes else "") + f"algorithm={algo}"
            except Exception:
                pass
        if progress_id:
            total_trials = int(tpe_trials or os.getenv("GAPA_TPE_TRIALS", "6") or 6)
            _set_strategy_progress(progress_id, current=total_trials, total=total_trials, status="done", server_id=str(server_id or "local"))
        return jsonify(plan.to_dict())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/strategy_compare", methods=["POST"])
def api_strategy_compare():
    """Proxy StrategyCompare to remote server agent for UI explanation."""
    payload = request.get_json(silent=True) or {}
    server_id = payload.get("server_id") or payload.get("server")
    objective = str(payload.get("objective") or "time")
    multi_gpu = bool(payload.get("multi_gpu", True))
    warmup_iters = int(payload.get("warmup_iters", 0) or 0)
    gpu_busy_threshold = payload.get("gpu_busy_threshold")
    min_gpu_free_mb = payload.get("min_gpu_free_mb")
    try:
        if server_id and server_id != "local":
            base_url = _resolve_server_base_url(server_id)
            if not base_url:
                return jsonify({"error": "unknown server_id or missing base_url", "server_id": server_id}), 404

            import requests

            session = requests.Session()
            session.trust_env = False
            timeout_s = float(payload.get("timeout_s", 30) or 30)
            resp = session.post(
                base_url.rstrip("/") + "/api/strategy_compare",
                json={
                    "objective": objective,
                    "multi_gpu": multi_gpu,
                    "warmup_iters": warmup_iters,
                    "gpu_busy_threshold": gpu_busy_threshold,
                    "min_gpu_free_mb": min_gpu_free_mb,
                },
                timeout=(3.0, timeout_s),
            )
            if resp.ok:
                return jsonify(resp.json())
            try:
                body = resp.json()
            except Exception:
                body = {"raw": (resp.text or "")[:2000]}
            return (
                jsonify(
                    {
                        "error": "remote strategy_compare failed",
                        "server_id": server_id,
                        "base_url": base_url,
                        "status_code": resp.status_code,
                        "body": body,
                    }
                ),
                502,
            )

        # Local fallback
        from autoadapt import StrategyCompare

        snap = current_resource_snapshot()
        return jsonify(
            StrategyCompare(
                objective=objective,
                multi_gpu=multi_gpu,
                warmup_iters=warmup_iters,
                resource_snapshot=snap,
                gpu_busy_threshold=gpu_busy_threshold,
                min_gpu_free_mb=min_gpu_free_mb,
            )
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/strategy_plan/progress", methods=["GET"])
def api_strategy_plan_progress():
    progress_id = request.args.get("progress_id") or request.args.get("id") or ""
    server_id = request.args.get("server_id") or request.args.get("server")
    if not progress_id:
        return jsonify({"error": "progress_id required"}), 400
    if server_id and server_id != "local":
        base_url = _resolve_server_base_url(server_id)
        if not base_url:
            return jsonify({"error": "unknown server_id or missing base_url", "server_id": server_id}), 404
        try:
            import requests

            session = requests.Session()
            session.trust_env = False
            resp = session.get(
                base_url.rstrip("/") + "/api/strategy_plan/progress",
                params={"progress_id": progress_id},
                timeout=(3.0, 10.0),
            )
            if resp.ok:
                return jsonify(resp.json())
            return jsonify({"error": f"HTTP {resp.status_code}"}), resp.status_code
        except Exception as exc:
            return jsonify({"error": f"proxy failed: {exc}", "server_id": server_id, "base_url": base_url}), 502
    data = _get_strategy_progress(progress_id)
    return jsonify(data or {"status": "unknown", "current": 0, "total": 0})


@app.route("/api/ga_warmup", methods=["POST"])
def api_ga_warmup():
    """Run a short real GA warmup to measure actual latency/throughput."""
    payload = request.get_json(silent=True) or {}
    server_id = payload.get("server_id") or payload.get("server") or "local"
    algorithm = payload.get("algorithm")
    dataset = payload.get("dataset")
    if not algorithm or not dataset:
        return jsonify({"error": "algorithm and dataset are required"}), 400

    if server_id and server_id != "local":
        base_url = _resolve_server_base_url(server_id)
        if not base_url:
            return jsonify({"error": "unknown server_id or missing base_url", "server_id": server_id}), 404
        try:
            import requests

            session = requests.Session()
            session.trust_env = False
            timeout_s = float(payload.get("timeout_s", 180) or 180)
            body = dict(payload)
            body.pop("server_id", None)
            body.pop("server", None)
            resp = session.post(base_url.rstrip("/") + "/api/ga_warmup", json=body, timeout=(3.0, timeout_s))
            if resp.ok:
                return jsonify(resp.json())
            try:
                err = resp.json()
            except Exception:
                err = {"raw": (resp.text or "")[:2000]}
            return (
                jsonify(
                    {
                        "error": "remote ga_warmup failed",
                        "server_id": server_id,
                        "base_url": base_url,
                        "status_code": resp.status_code,
                        "body": err,
                    }
                ),
                502,
            )
        except Exception as exc:
            return jsonify({"error": f"proxy failed: {exc}", "server_id": server_id, "base_url": base_url}), 502

    result = _run_ga_warmup_local(payload)
    if isinstance(result, tuple):
        payload, code = result
        return jsonify(payload), code
    return jsonify(result)


@app.route("/api/distributed_strategy_plan", methods=["POST"])
def api_distributed_strategy_plan():
    """Distributed plan by merging per-server StrategyPlan + resource snapshots."""
    payload = request.get_json(silent=True) or {}
    server_ids = payload.get("servers") or payload.get("server_ids") or []
    if not server_ids:
        server_ids = [s.get("id") for s in load_server_list()]
    per_server_gpus = int(payload.get("per_server_gpus", 1) or 1)
    min_gpu_free_mb = int(payload.get("min_gpu_free_mb", 1024) or 1024)
    gpu_busy_threshold = float(payload.get("gpu_busy_threshold", 85.0) or 85.0)

    server_resources: dict = {}
    server_plans: dict = {}

    for sid in server_ids:
        if sid == "local":
            server_resources[sid] = current_resource_snapshot()
            try:
                plan = StrategyPlan(
                    fitness=None,
                    warmup=0,
                    objective="time",
                    multi_gpu=True,
                    resource_snapshot=server_resources[sid],
                )
                server_plans[sid] = plan
            except Exception:
                server_plans[sid] = None
            continue
        base_url = _resolve_server_base_url(sid)
        if not base_url:
            server_resources[sid] = {"error": "missing base_url"}
            continue
        try:
            import requests

            session = requests.Session()
            session.trust_env = False
            r_resp = session.get(base_url.rstrip("/") + "/api/resources", timeout=(3.0, 8.0))
            server_resources[sid] = r_resp.json() if r_resp.ok else {"error": f"HTTP {r_resp.status_code}"}
            p_resp = session.post(
                base_url.rstrip("/") + "/api/strategy_plan",
                json={
                    "warmup": 0,
                    "objective": "time",
                    "multi_gpu": True,
                    "gpu_busy_threshold": gpu_busy_threshold,
                    "min_gpu_free_mb": min_gpu_free_mb,
                },
                timeout=(3.0, 20.0),
            )
            if p_resp.ok:
                try:
                    from autoadapt.api.schemas import Plan

                    pdata = p_resp.json()
                    if isinstance(pdata, dict) and "backend" in pdata:
                        server_plans[sid] = Plan(**{k: pdata.get(k) for k in Plan.__dataclass_fields__.keys()})
                except Exception:
                    server_plans[sid] = None
        except Exception as exc:
            server_resources[sid] = {"error": str(exc)}

    plan = DistributedStrategyPlan(
        server_resources=server_resources,
        server_plans=server_plans,
        per_server_gpus=per_server_gpus,
        min_gpu_free_mb=min_gpu_free_mb,
        gpu_busy_threshold=gpu_busy_threshold,
    )
    plan["server_resources"] = server_resources
    plan["server_plans"] = {k: (v.to_dict() if v else None) for k, v in server_plans.items()}
    return jsonify(plan)


@app.route("/api/actions/<action>", methods=["POST"])
def api_actions(action: str):
    payload = request.get_json(force=True, silent=True) or {}
    store.log(action, payload)
    store.update_state(action)
    return jsonify({"status": "ok"})


@app.route("/api/logs")
def api_logs():
    return jsonify(store.logs)


@app.route("/api/history", methods=["GET", "POST", "DELETE"])
def api_history():
    """Legacy history endpoint - redirects to V1 behavior."""
    return api_v1_history()


# ============================================================================
# V1 API - Standardized RESTful Endpoints
# ============================================================================

@app.route("/api/v1/history", methods=["GET"])
def api_v1_history_list():
    """List history entries with pagination.
    
    Query params:
        page: Page number (1-indexed, default: 1)
        page_size: Items per page (default: 20, max: 100)
    
    Returns:
        Paginated response with items, total, page, page_size, pages
    """
    try:
        page = max(1, int(request.args.get("page", 1)))
        page_size = min(100, max(1, int(request.args.get("page_size", 20))))
    except ValueError:
        return jsonify(make_error_response(
            "ValidationError",
            "Invalid pagination parameters",
            request.path
        )), HTTPStatus.BAD_REQUEST

    total = db_manager.count_history()
    offset = (page - 1) * page_size
    order = request.args.get("order", "DESC")
    items = db_manager.get_history(limit=page_size, offset=offset, order=order)
    
    return jsonify(make_paginated_response(items, total, page, page_size))


@app.route("/api/v1/history", methods=["POST"])
def api_v1_history_create():
    """Create new history entry.
    
    Request body:
        Single history item dict, or {"items": [...]} for batch create
    
    Returns:
        201 Created with created item(s)
    """
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify(make_error_response(
            "ValidationError",
            "Request body must be a JSON object",
            request.path
        )), HTTPStatus.BAD_REQUEST
    
    batch = payload.get("items")
    if isinstance(batch, list) and batch:
        for item in batch:
            if isinstance(item, dict):
                db_manager.add_history(item)
        return jsonify({"status": "created", "count": len(batch)}), HTTPStatus.CREATED
    else:
        db_manager.add_history(payload)
        return jsonify({"status": "created", "item": payload}), HTTPStatus.CREATED


@app.route("/api/v1/history/<history_id>", methods=["DELETE"])
def api_v1_history_delete(history_id: str):
    """Delete a specific history entry.
    
    Returns:
        204 No Content on success
    """
    db_manager.delete_history([history_id])
    return "", HTTPStatus.NO_CONTENT


@app.route("/api/v1/history", methods=["DELETE"])
def api_v1_history_batch_delete():
    """Batch delete history entries.
    
    Request body:
        {"ids": ["id1", "id2", ...]} - list of IDs to delete
    
    Returns:
        200 OK with count of deleted items
    """
    payload = request.get_json(silent=True) or {}
    ids = payload.get("ids") or []
    
    if isinstance(ids, str):
        ids = [ids]
    if not isinstance(ids, list):
        return jsonify(make_error_response(
            "ValidationError",
            "ids must be a list of strings",
            request.path
        )), HTTPStatus.BAD_REQUEST
    
    if ids:
        db_manager.delete_history(ids)
    
    return jsonify({"status": "deleted", "count": len(ids)})


def api_v1_history():
    """Internal handler for legacy /api/history compatibility."""
    if request.method == "GET":
        return api_v1_history_list()
    if request.method == "POST":
        return api_v1_history_create()
    # DELETE
    return api_v1_history_batch_delete()


@app.route("/api/state")
def api_state():
    legacy = dict(store.state or {})
    canonical = normalize_task_state(legacy.get("status"))
    legacy.setdefault("status", legacy.get("status") or "空闲")
    legacy["state"] = canonical
    legacy["terminal_states"] = list(TERMINAL_TASK_STATES)
    legacy["is_terminal"] = is_terminal_task_state(canonical)
    return jsonify(legacy)


@app.route("/api/observer", methods=["GET", "POST"])
def api_observer():
    if request.method == "POST":
        data = request.get_json() or {}
        store.add_observer(data)
        return jsonify({"status": "ok"})
    return jsonify(store.observer)


@app.route("/api/analysis/start", methods=["POST"])
def api_analysis_start():
    """Proxy analysis start to remote server_agent; includes server_id."""
    payload = request.get_json(silent=True) or {}
    schema_version = resolve_schema_version(payload)
    server_id = payload.get("server_id") or payload.get("server")
    base_url = _resolve_server_base_url(server_id)
    if base_url is None and server_id and server_id != "local":
        return jsonify({"error": "unknown server_id or missing base_url", "code": "UNKNOWN_SERVER", "server_id": server_id}), 404

    try:
        import requests

        session = requests.Session()
        session.trust_env = False
        timeout_s = float(payload.get("timeout_s", 20) or 20)
        if base_url:
            resp, tmeta = request_with_retry(
                session=session,
                method="POST",
                url=base_url.rstrip("/") + "/api/analysis/start",
                json_payload={
                    "server_id": "local",  # Force local on remote agent to prevent proxy loop
                    "algorithm": payload.get("algorithm"),
                    "dataset": payload.get("dataset"),
                    "iterations": payload.get("iterations"),
                    "crossover_rate": payload.get("crossover_rate"),
                    "mutate_rate": payload.get("mutate_rate"),
                    "mode": payload.get("mode"),
                    "devices": payload.get("devices"),
                    "use_strategy_plan": payload.get("use_strategy_plan"),
                    "remote_servers": payload.get("remote_servers") or payload.get("allowed_server_ids"),
                    "release_lock_on_finish": payload.get("release_lock_on_finish", True),
                    "queue_if_busy": payload.get("queue_if_busy", False),
                    "owner": payload.get("owner"),
                    "priority": payload.get("priority"),
                    "resume_id": payload.get("resume_id"),
                    "checkpoint_ref": payload.get("checkpoint_ref"),
                    "retry_last": payload.get("retry_last"),
                    "run_id": payload.get("run_id"),
                    "schema_version": schema_version,
                },
                timeout=(3.0, timeout_s),
                op="analysis_start",
            )
            _record_transport_metric("analysis_start", tmeta, bool(resp.ok))
        else:
            # Local execution with real GA worker
            import uuid

            algorithm = str(payload.get("algorithm") or "ga")
            dataset = str(payload.get("dataset") or "")
            iterations = int(payload.get("iterations") or payload.get("max_generation") or 20)
            pc = float(payload.get("crossover_rate") or payload.get("pc") or 0.8)
            pm = float(payload.get("mutate_rate") or payload.get("pm") or 0.2)
            run_id = str(payload.get("run_id") or uuid.uuid4())
            retry_last = bool(payload.get("retry_last", False))
            resume_id = payload.get("resume_id") or payload.get("checkpoint_ref")
            selected = select_run_mode(payload.get("mode"), payload.get("devices"))
            remote_servers = payload.get("remote_servers") or payload.get("allowed_server_ids")
            if remote_servers is not None:
                selected["remote_servers"] = remote_servers
            release_lock_on_finish = bool(payload.get("release_lock_on_finish", True))
            capability = detect_capability(target="local", resource_snapshot=current_resource_snapshot(), remote_reachable=True)
            mode_reason = ""
            allow_mnm = bool(remote_servers)

            # Sanitize selected mode against capability/downgrade chain before lock policy.
            resolved_mode, resolved_degraded, resolved_reason = choose_mode(
                selected.get("mode"),
                capability,
                allow_mnm=allow_mnm,
            )
            if str(selected.get("mode") or "").upper() != resolved_mode:
                selected = select_run_mode(resolved_mode, selected.get("devices"))
                if remote_servers is not None:
                    selected["remote_servers"] = remote_servers
            if resolved_degraded and resolved_reason:
                mode_reason = resolved_reason

            # If no resource lock is active, force CPU execution.
            try:
                lock_active = bool(LOCK_MANAGER.status().get("active"))
            except Exception:
                lock_active = False
            if not lock_active and (selected.get("mode") or "").upper() not in ("MNM",):
                selected = select_run_mode("CPU", None)
                mode_reason = (f"{mode_reason} | " if mode_reason else "") + "no_active_lock->CPU"

            mode_decision = build_mode_decision(
                requested_mode=payload.get("mode"),
                selected_mode=selected.get("mode"),
                devices=selected.get("devices"),
                target="local",
                capability=capability,
                reason=mode_reason,
                use_strategy_plan=payload.get("use_strategy_plan"),
            )
            mode_decision = normalize_mode_decision(mode_decision)
            _record_degrade_reason(mode_decision)

            with LOCAL_TASK.lock:
                if LOCAL_TASK.state == "running":
                    return jsonify({"error": "A task is already running", "code": "TASK_BUSY"}), 409
                task_id = str(uuid.uuid4())
                if retry_last and not resume_id:
                    resume_id = LOCAL_TASK.last_checkpoint_ref
                checkpoint_ref = str(resume_id or "")
                resume_metadata = build_resume_metadata(
                    run_id=run_id,
                    task_id=task_id,
                    schema_version=schema_version,
                    mode_plan_snapshot=mode_decision,
                    checkpoint_ref=checkpoint_ref,
                )
                
                LOCAL_TASK.reset_for_new_task(
                    task_id,
                    mode_decision=mode_decision,
                    run_id=run_id,
                    schema_version=schema_version,
                    resume_metadata=resume_metadata,
                )
                LOCAL_TASK.release_lock_on_finish = release_lock_on_finish
                LOCAL_TASK.last_checkpoint_ref = checkpoint_ref or LOCAL_TASK.last_checkpoint_ref
                LOCAL_TASK.append_log(f"[INFO] 运行模式: {selected.get('mode')} devices={selected.get('devices')}")
                LOCAL_TASK.append_log(
                    f"[INFO] 模式决策: requested={mode_decision.get('requested_mode')} "
                    f"selected={mode_decision.get('selected_mode')} degraded={mode_decision.get('degraded')} "
                    f"reason={mode_decision.get('reason') or '-'} code={mode_decision.get('code') or '-'}"
                )
                if resume_id:
                    LOCAL_TASK.append_log(f"[INFO] 正在尝试从状态 {resume_id} 继续迭代...")

                ctx = mp.get_context("spawn")
                q = ctx.Queue()
                selected["run_id"] = run_id
                selected["schema_version"] = schema_version
                selected["mode_decision"] = mode_decision
                proc = ctx.Process(
                    target=_ga_entry,
                    args=(task_id, algorithm, dataset, iterations, pc, pm, selected, q, resume_id),
                )
                LOCAL_TASK.queue = q
                LOCAL_TASK.process = proc
                proc.start()
                start_consumer(LOCAL_TASK)

            return jsonify(
                _response_with_schema(
                    {
                        "task_id": task_id,
                        "run_id": run_id,
                        "status": "started",
                        "mode_decision": mode_decision,
                        "resume_metadata": resume_metadata,
                        "transport": transport_contract(),
                    },
                    schema_version,
                )
            )

        if resp.ok:
            body = resp.json()
            if isinstance(body, dict):
                body.setdefault("transport", transport_contract())
                body.setdefault("transport_meta", tmeta if base_url else {})
                body.setdefault("transport_metrics", _transport_metric_summary())
                if body.get("mode_decision") is not None:
                    body["mode_decision"] = normalize_mode_decision(body.get("mode_decision"))
                    _record_degrade_reason(body.get("mode_decision"))
            return jsonify(_response_with_schema(body if isinstance(body, dict) else {}, schema_version))
        try:
            body = resp.json()
        except Exception:
            body = {"raw": (resp.text or "")[:2000]}
        return jsonify(
            _response_with_schema(
                {
                "error": "remote analysis start failed",
                "code": classify_transport_error(status_code=resp.status_code, response_body=body),
                "status_code": resp.status_code,
                "body": body,
                "transport_meta": tmeta if base_url else {},
                "transport_metrics": _transport_metric_summary(),
                },
                schema_version,
            )
        ), 502
    except Exception as exc:
        return jsonify({"error": f"proxy failed: {exc}", "code": "PROXY_FAILED", "server_id": server_id, "base_url": base_url}), 502


@app.route("/api/analysis/queue", methods=["GET"])
def api_analysis_queue():
    """Proxy queue status from remote server_agent; includes server_id query param."""
    server_id = request.args.get("server_id") or request.args.get("server")
    schema_version = resolve_schema_version({"schema_version": request.args.get("schema_version")})
    base_url = _resolve_server_base_url(server_id)
    if base_url is None and server_id and server_id != "local":
        return jsonify({"error": "unknown server_id or missing base_url", "code": "UNKNOWN_SERVER", "server_id": server_id}), 404

    try:
        import requests

        session = requests.Session()
        session.trust_env = False
        timeout_s = float(request.args.get("timeout_s") or 10)
        if base_url:
            proxy_url = base_url.rstrip("/") + "/api/analysis/queue?server_id=local"
            resp, tmeta = request_with_retry(
                session=session,
                method="GET",
                url=proxy_url,
                timeout=(3.0, timeout_s),
                op="analysis_status",
            )
            _record_transport_metric("analysis_queue", tmeta, bool(resp.ok))
            if resp.ok:
                body = resp.json()
                if isinstance(body, dict):
                    body.setdefault("transport_meta", tmeta)
                    body.setdefault("transport_metrics", _transport_metric_summary())
                return jsonify(_response_with_schema(body if isinstance(body, dict) else {}, schema_version))
            try:
                body = resp.json()
            except Exception:
                body = {"raw": (resp.text or "")[:2000]}
            return jsonify(
                _response_with_schema(
                    {
                    "error": "remote analysis queue failed",
                    "code": classify_transport_error(status_code=resp.status_code, response_body=body),
                    "status_code": resp.status_code,
                    "body": body,
                    "transport_meta": tmeta,
                    "transport_metrics": _transport_metric_summary(),
                    },
                    schema_version,
                )
            ), 502
        return jsonify(_response_with_schema({"size": 0, "items": []}, schema_version))
    except Exception as exc:
        return jsonify({"error": f"proxy failed: {exc}", "code": "PROXY_FAILED", "server_id": server_id, "base_url": base_url}), 502


@app.route("/api/analysis/status", methods=["GET"])
def api_analysis_status():
    """Proxy analysis status from remote server_agent; includes server_id query param."""
    server_id = request.args.get("server_id") or request.args.get("server")
    schema_version = resolve_schema_version({"schema_version": request.args.get("schema_version")})
    base_url = _resolve_server_base_url(server_id)
    if base_url is None and server_id and server_id != "local":
        return jsonify({"error": "unknown server_id or missing base_url", "code": "UNKNOWN_SERVER", "server_id": server_id}), 404

    try:
        import requests

        session = requests.Session()
        session.trust_env = False
        timeout_s = float(request.args.get("timeout_s") or 10)
        if base_url:
            # Append server_id=local to prevent proxy loop on the remote side
            proxy_url = base_url.rstrip("/") + "/api/analysis/status?server_id=local"
            resp, tmeta = request_with_retry(
                session=session,
                method="GET",
                url=proxy_url,
                timeout=(3.0, timeout_s),
                op="analysis_status",
            )
            _record_transport_metric("analysis_status", tmeta, bool(resp.ok))
        else:
            with LOCAL_TASK.lock:
                return jsonify(
                    _response_with_schema(
                        {
                        "task_id": LOCAL_TASK.task_id,
                        "run_id": LOCAL_TASK.run_id,
                        "state": LOCAL_TASK.state,
                        "progress": LOCAL_TASK.progress,
                        "logs": list(LOCAL_TASK.logs),
                        "result": LOCAL_TASK.result,
                        "error": LOCAL_TASK.error,
                        "mode_decision": normalize_mode_decision(LOCAL_TASK.mode_decision),
                        "resume_metadata": LOCAL_TASK.resume_metadata,
                        "lifecycle": {
                            "terminal_states": list(TERMINAL_TASK_STATES),
                            "is_terminal": is_terminal_task_state(LOCAL_TASK.state),
                        },
                        "transport_metrics": _transport_metric_summary(),
                        },
                        schema_version,
                    )
                )

        if resp.ok:
            body = resp.json()
            if isinstance(body, dict):
                body.setdefault("transport_meta", tmeta if base_url else {})
                body.setdefault("transport_metrics", _transport_metric_summary())
            if isinstance(body, dict) and body.get("mode_decision") is not None:
                body["mode_decision"] = normalize_mode_decision(body.get("mode_decision"))
            return jsonify(_response_with_schema(body if isinstance(body, dict) else {}, schema_version))
        try:
            body = resp.json()
        except Exception:
            body = {"raw": (resp.text or "")[:2000]}
        return jsonify(
            _response_with_schema(
                {
                "error": "remote analysis status failed",
                "code": classify_transport_error(status_code=resp.status_code, response_body=body),
                "status_code": resp.status_code,
                "body": body,
                "transport_meta": tmeta if base_url else {},
                "transport_metrics": _transport_metric_summary(),
                },
                schema_version,
            )
        ), 502
    except Exception as exc:
        return jsonify({"error": f"proxy failed: {exc}", "code": "PROXY_FAILED", "server_id": server_id, "base_url": base_url}), 502


@app.route("/api/analysis/stop", methods=["POST"])
def api_analysis_stop():
    """Proxy analysis stop to remote server_agent; includes server_id."""
    payload = request.get_json(silent=True) or {}
    schema_version = resolve_schema_version(payload)
    server_id = payload.get("server_id") or payload.get("server")
    base_url = _resolve_server_base_url(server_id)
    if base_url is None and server_id and server_id != "local":
        return jsonify({"error": "unknown server_id or missing base_url", "code": "UNKNOWN_SERVER", "server_id": server_id}), 404

    try:
        import requests

        session = requests.Session()
        session.trust_env = False
        timeout_s = float(payload.get("timeout_s", 10) or 10)
        if base_url:
            resp, tmeta = request_with_retry(
                session=session,
                method="POST",
                url=base_url.rstrip("/") + "/api/analysis/stop",
                json_payload={"server_id": "local"},  # Prevent proxy loop
                timeout=(3.0, timeout_s),
                op="analysis_stop",
            )
            _record_transport_metric("analysis_stop", tmeta, bool(resp.ok))
            if resp.ok:
                body = resp.json()
                if isinstance(body, dict):
                    body.setdefault("transport_meta", tmeta)
                    body.setdefault("transport_metrics", _transport_metric_summary())
                    if body.get("mode_decision") is not None:
                        body["mode_decision"] = normalize_mode_decision(body.get("mode_decision"))
                return jsonify(_response_with_schema(body if isinstance(body, dict) else {}, schema_version))
            try:
                body = resp.json()
            except Exception:
                body = {"raw": (resp.text or "")[:2000]}
            return jsonify(
                _response_with_schema(
                    {
                    "error": "remote analysis stop failed",
                    "code": classify_transport_error(status_code=resp.status_code, response_body=body),
                    "status_code": resp.status_code,
                    "body": body,
                    "transport_meta": tmeta,
                    "transport_metrics": _transport_metric_summary(),
                    },
                    schema_version,
                )
            ), 502

        # Local fallback stop
        with LOCAL_TASK.lock:
            proc = LOCAL_TASK.process
            if LOCAL_TASK.state != "running" or not proc or not proc.is_alive():
                return jsonify(
                    _response_with_schema(
                        {
                        "status": LOCAL_TASK.state if LOCAL_TASK.state != "idle" else "idle",
                        "task_id": LOCAL_TASK.task_id,
                        "run_id": LOCAL_TASK.run_id,
                        "mode_decision": normalize_mode_decision(LOCAL_TASK.mode_decision),
                        "resume_metadata": LOCAL_TASK.resume_metadata,
                        },
                        schema_version,
                    )
                )
            pid = proc.pid
            LOCAL_TASK.append_log("[WARN] stop requested, terminating task...")

        # terminate outside lock
        try:
            if pid:
                terminate_process_tree(proc, pid, signal.SIGTERM)
            
            proc.join(timeout=2.0)
            if proc.is_alive():
                print(f"[WARN] Process {pid} still alive after SIGTERM, sending SIGKILL...")
                if pid:
                    terminate_process_tree(proc, pid, signal.SIGKILL)
                proc.join(timeout=1.0)
        finally:
            with LOCAL_TASK.lock:
                LOCAL_TASK.state = "cancelled"
                LOCAL_TASK.progress = 0
                LOCAL_TASK.error = "cancelled by user"
                if LOCAL_TASK.task_id:
                    LOCAL_TASK.last_checkpoint_ref = LOCAL_TASK.task_id
                LOCAL_TASK.append_log("[INFO] task stopped.")
            try:
                db_manager.save_task_terminal(task_id=str(LOCAL_TASK.task_id or ""), state="cancelled")
            except Exception:
                pass

        return jsonify(
            _response_with_schema(
                {
                    "status": "cancelled",
                    "task_id": LOCAL_TASK.task_id,
                    "run_id": LOCAL_TASK.run_id,
                    "mode_decision": normalize_mode_decision(LOCAL_TASK.mode_decision),
                    "resume_metadata": LOCAL_TASK.resume_metadata,
                },
                schema_version,
            )
        )
    except Exception as exc:
        return jsonify({"error": f"proxy failed: {exc}", "code": "PROXY_FAILED", "server_id": server_id, "base_url": base_url}), 502


@app.route("/api/transport/metrics", methods=["GET"])
def api_transport_metrics():
    schema_version = resolve_schema_version({"schema_version": request.args.get("schema_version")})
    with DEGRADE_REASON_LOCK:
        degrade_reasons = dict(DEGRADE_REASON_COUNTS)
    return jsonify(
        _response_with_schema(
            {
            "transport_metrics": _transport_metric_summary(),
            "degrade_reasons": degrade_reasons,
            "transport_contract": transport_contract(),
            },
            schema_version,
        )
    )


def _proxy_resource_lock(base_url: str, endpoint: str, payload: dict | None = None):
    import requests

    session = requests.Session()
    session.trust_env = False
    resp, tmeta = request_with_retry(
        session=session,
        method="POST",
        url=base_url.rstrip("/") + endpoint,
        json_payload=payload,
        timeout=(3.0, 20.0),
        op="resource_lock",
    )
    _record_transport_metric("resource_lock", tmeta, bool(resp.ok))
    return resp, tmeta


@app.route("/api/resource_lock", methods=["POST"])
def api_resource_lock():
    payload = request.get_json(silent=True) or {}
    scope = payload.get("scope") or payload.get("server_id") or payload.get("server") or "local"
    duration_s = float(payload.get("duration_s", 600) or 600)
    warmup_iters = int(payload.get("warmup_iters", 2) or 2)
    mem_mb = int(payload.get("mem_mb", 1024) or 1024)
    strict_idle = bool(payload.get("strict_idle", False))
    owner = str(payload.get("owner") or "")
    devices = payload.get("devices")
    devices_by_server = payload.get("devices_by_server") or {}

    print(f"[LOCK] Request: scope={scope}, duration={duration_s}s, warmup={warmup_iters}, mem={mem_mb}MB, devices={devices}, devices_by_server={devices_by_server}")

    servers = load_server_list()
    if scope in ("all", "*"):
        if isinstance(devices_by_server, dict) and devices_by_server:
            targets = [s for s in servers if s.get("id") in devices_by_server]
        else:
            targets = servers
    else:
        targets = [s for s in servers if s.get("id") == scope]
    if not targets:
        print(f"[LOCK] Error: Unknown server scope={scope}")
        return jsonify({"error": "unknown server", "scope": scope}), 404

    results = {}
    for s in targets:
        sid = s.get("id")
        print(f"[LOCK] Processing server: {sid}")
        if sid == "local":
            try:
                devs = devices_by_server.get(sid) if isinstance(devices_by_server, dict) else None
                lock_params = {
                    "duration_s": duration_s,
                    "warmup_iters": warmup_iters,
                    "mem_mb": mem_mb,
                    "devices": devs if devs is not None else devices,
                    "strict_idle": strict_idle,
                    "owner": owner,
                }
                print(f"[LOCK] Calling LOCK_MANAGER.lock with params: {lock_params}")
                results[sid] = LOCK_MANAGER.lock(**lock_params)
                print(f"[LOCK] Success: {results[sid]}")
            except Exception as exc:
                print(f"[LOCK] Error on local: {exc}")
                import traceback
                traceback.print_exc()
                results[sid] = {"error": str(exc), "traceback": traceback.format_exc()}
            continue
        base_url = _resolve_server_base_url(sid)
        if not base_url:
            results[sid] = {"error": "missing base_url"}
            continue
        try:
            resp, tmeta = _proxy_resource_lock(
                base_url,
                "/api/resource_lock",
                {
                    "duration_s": duration_s,
                    "warmup_iters": warmup_iters,
                    "mem_mb": mem_mb,
                    "devices": (devices_by_server.get(sid) if isinstance(devices_by_server, dict) else None) or devices,
                    "strict_idle": strict_idle,
                    "owner": owner,
                },
            )
            results[sid] = (
                resp.json()
                if resp.ok
                else {
                    "error": f"HTTP {resp.status_code}",
                    "code": classify_transport_error(status_code=resp.status_code),
                    "transport_meta": tmeta,
                    "body": resp.text[:500],
                }
            )
        except Exception as exc:
            print(f"[LOCK] Error on remote {sid}: {exc}")
            results[sid] = {"error": str(exc), "code": classify_transport_error(exc=exc)}
    print(f"[LOCK] Final results: {results}")
    return jsonify({"scope": scope, "results": results, "transport_metrics": _transport_metric_summary()})


@app.route("/api/resource_lock/renew", methods=["POST"])
def api_resource_lock_renew():
    payload = request.get_json(silent=True) or {}
    scope = payload.get("scope") or payload.get("server_id") or payload.get("server") or "local"
    duration_s = payload.get("duration_s")
    lock_id = payload.get("lock_id")
    owner = payload.get("owner")
    servers = load_server_list()
    targets = servers if scope in ("all", "*") else [s for s in servers if s.get("id") == scope]
    if not targets:
        return jsonify({"error": "unknown server", "scope": scope}), 404

    results = {}
    release_results = {}
    for s in targets:
        sid = s.get("id")
        if sid == "local":
            try:
                results[sid] = LOCK_MANAGER.renew(duration_s=duration_s, lock_id=lock_id, owner=owner)
            except Exception as exc:
                results[sid] = {"error": str(exc)}
            continue
        base_url = _resolve_server_base_url(sid)
        if not base_url:
            results[sid] = {"error": "missing base_url"}
            continue
        try:
            resp, tmeta = _proxy_resource_lock(
                base_url,
                "/api/resource_lock/renew",
                {
                    "duration_s": duration_s,
                    "lock_id": lock_id,
                    "owner": owner,
                },
            )
            results[sid] = (
                resp.json()
                if resp.ok
                else {
                    "error": f"HTTP {resp.status_code}",
                    "code": classify_transport_error(status_code=resp.status_code),
                    "transport_meta": tmeta,
                    "body": resp.text[:500],
                }
            )
        except Exception as exc:
            results[sid] = {"error": str(exc), "code": classify_transport_error(exc=exc)}

    def _is_renew_failed(item: dict | None) -> bool:
        if not isinstance(item, dict):
            return True
        if item.get("error"):
            return True
        note = str(item.get("note") or "").strip().lower()
        if note in ("lock_id_mismatch", "owner_mismatch", "not_active"):
            return True
        return False

    failed_servers = [sid for sid, item in results.items() if _is_renew_failed(item)]
    if failed_servers:
        for s in targets:
            sid = s.get("id")
            if sid not in failed_servers:
                continue
            if sid == "local":
                try:
                    release_results[sid] = LOCK_MANAGER.release(reason="renew_failed")
                except Exception as exc:
                    release_results[sid] = {"error": str(exc)}
                continue
            base_url = _resolve_server_base_url(sid)
            if not base_url:
                release_results[sid] = {"error": "missing base_url"}
                continue
            try:
                resp, tmeta = _proxy_resource_lock(base_url, "/api/resource_lock/release", None)
                release_results[sid] = (
                    resp.json()
                    if resp.ok
                    else {
                        "error": f"HTTP {resp.status_code}",
                        "code": classify_transport_error(status_code=resp.status_code),
                        "transport_meta": tmeta,
                        "body": resp.text[:500],
                    }
                )
            except Exception as exc:
                release_results[sid] = {"error": str(exc), "code": classify_transport_error(exc=exc)}
        return jsonify(
            {
                "scope": scope,
                "results": results,
                "release_results": release_results,
                "code": "MNM_LOCK_RENEW_FAILED",
                "reason": f"renew_failed_on={','.join(failed_servers)}",
                "transport_metrics": _transport_metric_summary(),
            }
        )
    return jsonify(
        {
            "scope": scope,
            "results": results,
            "code": "",
            "reason": "",
            "transport_metrics": _transport_metric_summary(),
        }
    )


@app.route("/api/resource_lock/release", methods=["POST"])
def api_resource_lock_release():
    payload = request.get_json(silent=True) or {}
    scope = payload.get("scope") or payload.get("server_id") or payload.get("server") or "local"
    print(f"[LOCK_RELEASE] Request: scope={scope}")
    servers = load_server_list()
    targets = servers if scope in ("all", "*") else [s for s in servers if s.get("id") == scope]
    if not targets:
        return jsonify({"error": "unknown server", "scope": scope}), 404

    results = {}
    for s in targets:
        sid = s.get("id")
        if sid == "local":
            try:
                print(f"[LOCK_RELEASE] Calling LOCK_MANAGER.release()")
                results[sid] = LOCK_MANAGER.release(reason="manual")
                print(f"[LOCK_RELEASE] Success: {results[sid]}")
                try:
                    from server.fitness_worker import clear_contexts

                    clear_contexts()
                except Exception:
                    pass
            except Exception as exc:
                print(f"[LOCK_RELEASE] Error on local: {exc}")
                import traceback
                traceback.print_exc()
                results[sid] = {"error": str(exc), "traceback": traceback.format_exc()}
            continue
        base_url = _resolve_server_base_url(sid)
        if not base_url:
            results[sid] = {"error": "missing base_url"}
            continue
        try:
            resp, tmeta = _proxy_resource_lock(base_url, "/api/resource_lock/release", None)
            results[sid] = (
                resp.json()
                if resp.ok
                else {
                    "error": f"HTTP {resp.status_code}",
                    "code": classify_transport_error(status_code=resp.status_code),
                    "transport_meta": tmeta,
                    "body": resp.text[:500],
                }
            )
        except Exception as exc:
            print(f"[LOCK_RELEASE] Error on remote {sid}: {exc}")
            results[sid] = {"error": str(exc), "code": classify_transport_error(exc=exc)}
    print(f"[LOCK_RELEASE] Final results: {results}")
    return jsonify({"scope": scope, "results": results, "transport_metrics": _transport_metric_summary()})


@app.route("/api/resource_lock/status", methods=["GET"])
def api_resource_lock_status():
    scope = request.args.get("scope") or request.args.get("server_id") or request.args.get("server") or "local"
    realtime = request.args.get("realtime", "false").lower() == "true"
    
    if realtime:
        # Force refresh for valid scope
        if scope == "all":
            # For "all", we refresh everything in parallel (which is what refresh_all does)
            state_manager.refresh_all()
        else:
            state_manager.refresh_one(scope)

    snapshots = state_manager.get_snapshots()
    
    if scope == "all":
        # Merge all "lock" info from snapshots
        all_results = {}
        for sid, snap in snapshots.items():
            if not snap.get("online"):
                all_results[sid] = {"error": snap.get("error") or "Server Offline", "active": False}
                continue
            lock_payload = snap.get("lock") or {}
            if isinstance(lock_payload, dict) and isinstance(lock_payload.get("results"), dict):
                all_results.update(lock_payload.get("results") or {})
            else:
                if isinstance(lock_payload, dict) and any(k in lock_payload for k in ("active", "backend", "devices")):
                    all_results[sid] = lock_payload
                elif isinstance(lock_payload, dict) and "error_lock" in lock_payload:
                    all_results[sid] = {"error": f"HTTP {lock_payload.get('error_lock')}", "active": False}
                elif isinstance(lock_payload, dict) and "error" in lock_payload:
                    all_results[sid] = {"error": lock_payload.get("error"), "active": False}
        return jsonify({"scope": "all", "results": all_results, "transport_metrics": _transport_metric_summary()})
    
    # Single server scope
    snap = snapshots.get(scope)
    if not snap:
        return jsonify({"error": "unknown server", "scope": scope}), 404
    
    if not snap.get("online"):
        return jsonify(
            {
                "scope": scope,
                "results": {scope: {"error": snap.get("error") or "Server Offline", "active": False}},
                "transport_metrics": _transport_metric_summary(),
            }
        )
    
    lock_payload = snap.get("lock") or {}
    if isinstance(lock_payload, dict) and isinstance(lock_payload.get("results"), dict):
        if isinstance(lock_payload, dict):
            lock_payload.setdefault("transport_metrics", _transport_metric_summary())
        return jsonify(lock_payload)
    if isinstance(lock_payload, dict) and any(k in lock_payload for k in ("active", "backend", "devices", "note")):
        return jsonify({"scope": scope, "results": {scope: lock_payload}, "transport_metrics": _transport_metric_summary()})
    if isinstance(lock_payload, dict) and "error_lock" in lock_payload:
        return jsonify(
            {
                "scope": scope,
                "results": {scope: {"error": f"HTTP {lock_payload.get('error_lock')}", "active": False}},
                "transport_metrics": _transport_metric_summary(),
            }
        )
    if isinstance(lock_payload, dict) and "error" in lock_payload:
        return jsonify(
            {
                "scope": scope,
                "results": {scope: {"error": lock_payload.get("error"), "active": False}},
                "transport_metrics": _transport_metric_summary(),
            }
        )
    return jsonify({"scope": scope, "results": {}, "transport_metrics": _transport_metric_summary()})

def check_dependencies():
    """Check for essential packages and print warnings if missing."""
    missing = []
    try:
        import psutil
    except ImportError:
        missing.append("psutil (REQUIRED for CPU/Memory monitoring)")
    
    try:
        import pynvml
    except ImportError:
        missing.append("pynvml (REQUIRED for GPU monitoring)")

    try:
        import torch
    except ImportError:
        missing.append("torch (REQUIRED for GA execution)")

    if missing:
        print("\n" + "!" * 60)
        print("  WARNING: Missing Dependencies Detected!")
        for item in missing:
            print(f"  - {item}")
        print("\n  Please install missing packages via:")
        print("  pip install " + " ".join([m.split()[0] for m in missing]))
        print("!" * 60 + "\n")
    else:
        print("\n[INFO] All critical dependencies (psutil, pynvml, torch) are installed.\n")

if __name__ == "__main__":
    print("=" * 50)
    print("GAPA Console 启动成功！")
    check_dependencies()
    print("本地访问 → http://localhost:4467")
    print("局域网访问 → http://本机IP:4467")
    print("=" * 50)
    app.run(host="0.0.0.0", port=4467, threaded=True)
