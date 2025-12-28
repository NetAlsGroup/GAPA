# filename: app.py
from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
from pathlib import Path
import uuid

from flask import Flask, jsonify, request, send_from_directory

from autoadapt import StrategyPlan, DistributedStrategyPlan

from server import (
    STATIC_ROOT,
    JobStore,
    current_resource_snapshot,
    get_all_resources,
    load_server_config,
    load_server_list,
)
from server.resource_lock import LOCK_MANAGER
from server.agent_state import TaskState, start_consumer
from server.ga_worker import ga_worker, select_run_mode

import threading
import time

HISTORY_FILE = (Path(__file__).resolve().parent / "results" / "history.json").resolve()
HISTORY_LOCK = threading.Lock()
STRATEGY_PROGRESS: dict[str, dict[str, object]] = {}
STRATEGY_PROGRESS_LOCK = threading.Lock()


def _load_history() -> list[dict]:
    try:
        if not HISTORY_FILE.exists():
            return []
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_history(items: list[dict]) -> None:
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


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


def _ga_entry(
    task_id: str,
    algorithm: str,
    dataset: str,
    iterations: int,
    crossover_rate: float,
    mutate_rate: float,
    selected: dict,
    q: any,
) -> None:
    if os.name == "posix":
        try:
            os.setsid()
        except Exception:
            pass
    ga_worker(task_id, algorithm, dataset, iterations, crossover_rate, mutate_rate, selected, q)


def _run_ga_warmup_local(payload: dict) -> dict:
    import uuid

    algorithm = str(payload.get("algorithm") or "")
    dataset = str(payload.get("dataset") or "")
    iterations = int(payload.get("iterations") or payload.get("warmup_iters") or 2)
    pc = float(payload.get("crossover_rate") or payload.get("pc") or 0.8)
    pm = float(payload.get("mutate_rate") or payload.get("pm") or 0.2)
    selected = select_run_mode(payload.get("mode"), payload.get("devices"))
    remote_servers = payload.get("remote_servers") or payload.get("allowed_server_ids")
    if remote_servers is not None:
        selected["remote_servers"] = remote_servers
    timeout_s = float(payload.get("timeout_s", 180) or 180)

    with LOCAL_TASK.lock:
        if LOCAL_TASK.state == "running":
            return {"error": "A task is already running"}, 409

    task_id = str(uuid.uuid4())
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(
        target=_ga_entry,
        args=(task_id, algorithm, dataset, iterations, pc, pm, selected, q),
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

    summary = _summarize_warmup_result(result)
    return {
        "task_id": task_id,
        "state": state,
        "summary": summary["summary"],
        "per_iter_ms": summary["per_iter_ms"],
        "comm": (result or {}).get("comm"),
        "logs": logs[-200:],
        "error": error,
    }


LOCAL_TASK = TaskState()

# 使用 /static 避免与 /api/* 路由冲突
app = Flask(__name__, static_folder=str(STATIC_ROOT), static_url_path="/static")
store = JobStore()
ALGO_MANIFEST_PATH = Path(__file__).resolve().parent / "gapa" / "algorithm" / "manifest.json"
DATASETS_MANIFEST_PATH = Path(__file__).resolve().parent / "datasets.json"


@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "dashboard.html")


@app.route("/api/all_resources")
def api_all_resources():
    """前端每2秒调用此接口获取所有机器最新状态"""
    data = get_all_resources(request.remote_addr)
    return jsonify(data)


@app.route("/api/resources")
def api_resources():
    return jsonify(current_resource_snapshot())


@app.route("/api/servers", methods=["GET"])
def api_servers():
    return jsonify(load_server_config(mask_password=True))


@app.route("/api/algorithms", methods=["GET"])
def api_algorithms():
    """Return GA algorithm manifest for the UI."""
    if not ALGO_MANIFEST_PATH.exists():
        return jsonify([])
    try:
        raw = json.loads(ALGO_MANIFEST_PATH.read_text(encoding="utf-8"))
        return jsonify(raw if isinstance(raw, list) else [])
    except Exception:
        return jsonify([])


@app.route("/api/datasets", methods=["GET"])
def api_datasets():
    """Return datasets manifest for the UI."""
    if not DATASETS_MANIFEST_PATH.exists():
        return jsonify({})
    try:
        raw = json.loads(DATASETS_MANIFEST_PATH.read_text(encoding="utf-8"))
        return jsonify(raw if isinstance(raw, dict) else {})
    except Exception:
        return jsonify({})


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
    if request.method == "GET":
        with HISTORY_LOCK:
            return jsonify(_load_history())

    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"error": "invalid payload"}), 400
        with HISTORY_LOCK:
            items = _load_history()
            batch = payload.get("items")
            if isinstance(batch, list) and batch:
                normalized = []
                for item in batch:
                    if not isinstance(item, dict):
                        continue
                    item.setdefault("id", str(uuid.uuid4()))
                    normalized.append(item)
                if normalized:
                    items = normalized + items
            else:
                payload.setdefault("id", str(uuid.uuid4()))
                items.insert(0, payload)
            items = items[:200]
            _save_history(items)
        return jsonify({"items": items, "id": payload.get("id")})

    payload = request.get_json(silent=True) or {}
    ids = payload.get("ids") or []
    if payload.get("all"):
        with HISTORY_LOCK:
            _save_history([])
        return jsonify({"items": []})
    if isinstance(ids, str):
        ids = [ids]
    if not isinstance(ids, list) or not ids:
        return jsonify({"error": "ids required"}), 400
    with HISTORY_LOCK:
        items = [it for it in _load_history() if it.get("id") not in set(ids)]
        _save_history(items)
    return jsonify({"items": items})


@app.route("/api/state")
def api_state():
    return jsonify(store.state)


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
    server_id = payload.get("server_id") or payload.get("server")
    base_url = _resolve_server_base_url(server_id)
    if base_url is None and server_id and server_id != "local":
        return jsonify({"error": "unknown server_id or missing base_url", "server_id": server_id}), 404

    try:
        import requests

        session = requests.Session()
        session.trust_env = False
        timeout_s = float(payload.get("timeout_s", 20) or 20)
        if base_url:
            resp = session.post(
                base_url.rstrip("/") + "/api/analysis/start",
                json={
                    "algorithm": payload.get("algorithm"),
                    "dataset": payload.get("dataset"),
                    "iterations": payload.get("iterations"),
                    "crossover_rate": payload.get("crossover_rate"),
                    "mutate_rate": payload.get("mutate_rate"),
                    "mode": payload.get("mode"),
                    "devices": payload.get("devices"),
                    "release_lock_on_finish": payload.get("release_lock_on_finish", True),
                },
                timeout=(3.0, timeout_s),
            )
        else:
            # Local execution with real GA worker
            import uuid

            algorithm = str(payload.get("algorithm") or "ga")
            dataset = str(payload.get("dataset") or "")
            iterations = int(payload.get("iterations") or payload.get("max_generation") or 20)
            pc = float(payload.get("crossover_rate") or payload.get("pc") or 0.8)
            pm = float(payload.get("mutate_rate") or payload.get("pm") or 0.2)
            selected = select_run_mode(payload.get("mode"), payload.get("devices"))
            remote_servers = payload.get("remote_servers") or payload.get("allowed_server_ids")
            if remote_servers is not None:
                selected["remote_servers"] = remote_servers
            release_lock_on_finish = bool(payload.get("release_lock_on_finish", True))

            # If no resource lock is active, force CPU execution.
            try:
                lock_active = bool(LOCK_MANAGER.status().get("active"))
            except Exception:
                lock_active = False
            if not lock_active and (selected.get("mode") or "").upper() not in ("MNM",):
                selected = select_run_mode("CPU", None)

            with LOCAL_TASK.lock:
                if LOCAL_TASK.state == "running":
                    return jsonify({"error": "A task is already running"}), 409
                task_id = str(uuid.uuid4())
                LOCAL_TASK.reset_for_new_task(task_id)
                LOCAL_TASK.release_lock_on_finish = release_lock_on_finish
                LOCAL_TASK.append_log(f"[INFO] 运行模式: {selected.get('mode')} devices={selected.get('devices')}")

                ctx = mp.get_context("spawn")
                q = ctx.Queue()
                proc = ctx.Process(
                    target=_ga_entry,
                    args=(task_id, algorithm, dataset, iterations, pc, pm, selected, q),
                )
                LOCAL_TASK.queue = q
                LOCAL_TASK.process = proc
                proc.start()
                start_consumer(LOCAL_TASK)

            return jsonify({"task_id": task_id, "status": "started"})

        if resp.ok:
            return jsonify(resp.json())
        try:
            body = resp.json()
        except Exception:
            body = {"raw": (resp.text or "")[:2000]}
        return jsonify({"error": "remote analysis start failed", "status_code": resp.status_code, "body": body}), 502
    except Exception as exc:
        return jsonify({"error": f"proxy failed: {exc}", "server_id": server_id, "base_url": base_url}), 502


@app.route("/api/analysis/status", methods=["GET"])
def api_analysis_status():
    """Proxy analysis status from remote server_agent; includes server_id query param."""
    server_id = request.args.get("server_id") or request.args.get("server")
    base_url = _resolve_server_base_url(server_id)
    if base_url is None and server_id and server_id != "local":
        return jsonify({"error": "unknown server_id or missing base_url", "server_id": server_id}), 404

    try:
        import requests

        session = requests.Session()
        session.trust_env = False
        timeout_s = float(request.args.get("timeout_s") or 10)
        if base_url:
            resp = session.get(base_url.rstrip("/") + "/api/analysis/status", timeout=(3.0, timeout_s))
        else:
            with LOCAL_TASK.lock:
                return jsonify(
                    {
                        "task_id": LOCAL_TASK.task_id,
                        "state": LOCAL_TASK.state,
                        "progress": LOCAL_TASK.progress,
                        "logs": list(LOCAL_TASK.logs),
                        "result": LOCAL_TASK.result,
                        "error": LOCAL_TASK.error,
                    }
                )

        if resp.ok:
            return jsonify(resp.json())
        try:
            body = resp.json()
        except Exception:
            body = {"raw": (resp.text or "")[:2000]}
        return jsonify({"error": "remote analysis status failed", "status_code": resp.status_code, "body": body}), 502
    except Exception as exc:
        return jsonify({"error": f"proxy failed: {exc}", "server_id": server_id, "base_url": base_url}), 502


@app.route("/api/analysis/stop", methods=["POST"])
def api_analysis_stop():
    """Proxy analysis stop to remote server_agent; includes server_id."""
    payload = request.get_json(silent=True) or {}
    server_id = payload.get("server_id") or payload.get("server")
    base_url = _resolve_server_base_url(server_id)
    if base_url is None and server_id and server_id != "local":
        return jsonify({"error": "unknown server_id or missing base_url", "server_id": server_id}), 404

    try:
        import requests

        session = requests.Session()
        session.trust_env = False
        timeout_s = float(payload.get("timeout_s", 10) or 10)
        if base_url:
            resp = session.post(base_url.rstrip("/") + "/api/analysis/stop", timeout=(3.0, timeout_s))
            if resp.ok:
                return jsonify(resp.json())
            try:
                body = resp.json()
            except Exception:
                body = {"raw": (resp.text or "")[:2000]}
            return jsonify({"error": "remote analysis stop failed", "status_code": resp.status_code, "body": body}), 502

        # Local fallback stop
        with LOCAL_TASK.lock:
            proc = LOCAL_TASK.process
            if LOCAL_TASK.state != "running" or not proc or not proc.is_alive():
                LOCAL_TASK.state = "idle" if LOCAL_TASK.state != "error" else LOCAL_TASK.state
                return jsonify({"status": "idle", "task_id": LOCAL_TASK.task_id})
            pid = proc.pid
            LOCAL_TASK.append_log("[WARN] stop requested, terminating task...")

        # terminate outside lock
        try:
            def kill_children(sig: int) -> None:
                try:
                    import psutil  # type: ignore
                except Exception:
                    return
                try:
                    parent = psutil.Process(pid) if pid else None
                    if not parent:
                        return
                    for ch in parent.children(recursive=True):
                        try:
                            ch.send_signal(sig)
                        except Exception:
                            pass
                except Exception:
                    pass

            if pid and os.name == "posix":
                try:
                    os.killpg(pid, signal.SIGTERM)
                except Exception:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except Exception:
                        pass
                kill_children(signal.SIGTERM)
            else:
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    kill_children(signal.SIGTERM)
                except Exception:
                    pass

            proc.join(timeout=3.0)
            if proc.is_alive():
                if pid and os.name == "posix":
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except Exception:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except Exception:
                            pass
                    kill_children(signal.SIGKILL)
                else:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    try:
                        kill_children(signal.SIGKILL)
                    except Exception:
                        pass
                proc.join(timeout=1.0)
        finally:
            with LOCAL_TASK.lock:
                LOCAL_TASK.state = "idle"
                LOCAL_TASK.progress = 0
                LOCAL_TASK.error = "stopped"
                LOCAL_TASK.append_log("[INFO] task stopped.")

        return jsonify({"status": "stopped", "task_id": LOCAL_TASK.task_id})
    except Exception as exc:
        return jsonify({"error": f"proxy failed: {exc}", "server_id": server_id, "base_url": base_url}), 502


def _proxy_resource_lock(base_url: str, endpoint: str, payload: dict | None = None):
    import requests

    session = requests.Session()
    session.trust_env = False
    if payload is None:
        resp = session.post(base_url.rstrip("/") + endpoint, timeout=(3.0, 20.0))
    else:
        resp = session.post(base_url.rstrip("/") + endpoint, json=payload, timeout=(3.0, 20.0))
    return resp


@app.route("/api/resource_lock", methods=["POST"])
def api_resource_lock():
    payload = request.get_json(silent=True) or {}
    scope = payload.get("scope") or payload.get("server_id") or payload.get("server") or "local"
    duration_s = float(payload.get("duration_s", 600) or 600)
    warmup_iters = int(payload.get("warmup_iters", 2) or 2)
    mem_mb = int(payload.get("mem_mb", 1024) or 1024)
    strict_idle = bool(payload.get("strict_idle", False))
    devices = payload.get("devices")
    devices_by_server = payload.get("devices_by_server") or {}

    servers = load_server_list()
    if scope in ("all", "*"):
        if isinstance(devices_by_server, dict) and devices_by_server:
            targets = [s for s in servers if s.get("id") in devices_by_server]
        else:
            targets = servers
    else:
        targets = [s for s in servers if s.get("id") == scope]
    if not targets:
        return jsonify({"error": "unknown server", "scope": scope}), 404

    results = {}
    for s in targets:
        sid = s.get("id")
        if sid == "local":
            try:
                devs = devices_by_server.get(sid) if isinstance(devices_by_server, dict) else None
                results[sid] = LOCK_MANAGER.lock(
                    duration_s=duration_s,
                    warmup_iters=warmup_iters,
                    mem_mb=mem_mb,
                    devices=devs if devs is not None else devices,
                    strict_idle=strict_idle,
                )
            except Exception as exc:
                results[sid] = {"error": str(exc)}
            continue
        base_url = _resolve_server_base_url(sid)
        if not base_url:
            results[sid] = {"error": "missing base_url"}
            continue
        try:
            resp = _proxy_resource_lock(
                base_url,
                "/api/resource_lock",
                {
                    "duration_s": duration_s,
                    "warmup_iters": warmup_iters,
                    "mem_mb": mem_mb,
                    "devices": (devices_by_server.get(sid) if isinstance(devices_by_server, dict) else None) or devices,
                    "strict_idle": strict_idle,
                },
            )
            results[sid] = resp.json() if resp.ok else {"error": f"HTTP {resp.status_code}"}
        except Exception as exc:
            results[sid] = {"error": str(exc)}
    return jsonify({"scope": scope, "results": results})


@app.route("/api/resource_lock/release", methods=["POST"])
def api_resource_lock_release():
    payload = request.get_json(silent=True) or {}
    scope = payload.get("scope") or payload.get("server_id") or payload.get("server") or "local"
    servers = load_server_list()
    targets = servers if scope in ("all", "*") else [s for s in servers if s.get("id") == scope]
    if not targets:
        return jsonify({"error": "unknown server", "scope": scope}), 404

    results = {}
    for s in targets:
        sid = s.get("id")
        if sid == "local":
            try:
                results[sid] = LOCK_MANAGER.release(reason="manual")
                try:
                    from server.fitness_worker import clear_contexts

                    clear_contexts()
                except Exception:
                    pass
            except Exception as exc:
                results[sid] = {"error": str(exc)}
            continue
        base_url = _resolve_server_base_url(sid)
        if not base_url:
            results[sid] = {"error": "missing base_url"}
            continue
        try:
            resp = _proxy_resource_lock(base_url, "/api/resource_lock/release", None)
            results[sid] = resp.json() if resp.ok else {"error": f"HTTP {resp.status_code}"}
        except Exception as exc:
            results[sid] = {"error": str(exc)}
    return jsonify({"scope": scope, "results": results})


@app.route("/api/resource_lock/status", methods=["GET"])
def api_resource_lock_status():
    scope = request.args.get("scope") or request.args.get("server_id") or request.args.get("server") or "local"
    servers = load_server_list()
    targets = servers if scope in ("all", "*") else [s for s in servers if s.get("id") == scope]
    if not targets:
        return jsonify({"error": "unknown server", "scope": scope}), 404

    results = {}
    for s in targets:
        sid = s.get("id")
        if sid == "local":
            try:
                results[sid] = LOCK_MANAGER.status()
            except Exception as exc:
                results[sid] = {"error": str(exc)}
            continue
        base_url = _resolve_server_base_url(sid)
        if not base_url:
            results[sid] = {"error": "missing base_url"}
            continue
        try:
            import requests

            session = requests.Session()
            session.trust_env = False
            resp = session.get(base_url.rstrip("/") + "/api/resource_lock/status", timeout=(3.0, 10.0))
            results[sid] = resp.json() if resp.ok else {"error": f"HTTP {resp.status_code}"}
        except Exception as exc:
            results[sid] = {"error": str(exc)}
    return jsonify({"scope": scope, "results": results})

if __name__ == "__main__":
    print("=" * 50)
    print("GAPA Console 启动成功！")
    print("本地访问 → http://localhost:4467")
    print("局域网访问 → http://本机IP:4467")
    print("=" * 50)
    app.run(host="0.0.0.0", port=4467, threaded=True)
