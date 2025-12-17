# filename: app.py
from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from autoadapt import StrategyPlan

from server import (
    STATIC_ROOT,
    JobStore,
    current_resource_snapshot,
    get_all_resources,
    load_server_config,
    load_server_list,
)

import threading
import time


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


class _LocalTaskState:
    def __init__(self):
        self.lock = threading.Lock()
        self.task_id: str | None = None
        self.state: str = "idle"
        self.progress: int = 0
        self.logs: list[str] = []
        self.result: dict | None = None
        self.error: str | None = None

    def reset(self, task_id: str) -> None:
        self.task_id = task_id
        self.state = "running"
        self.progress = 0
        self.logs = []
        self.result = None
        self.error = None

    def log(self, line: str) -> None:
        self.logs.append(line)
        if len(self.logs) > 500:
            self.logs = self.logs[-500:]


LOCAL_TASK = _LocalTaskState()


def _local_resource_point() -> dict:
    snap = current_resource_snapshot()
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
        "timestamp": snap.get("time"),
        "cpu_usage_percent": (snap.get("cpu") or {}).get("usage_percent"),
        "memory_percent": (snap.get("memory") or {}).get("percent"),
        "gpus": gpus,
    }


def _run_local_ga(task_id: str, algorithm: str, iterations: int, pc: float, pm: float) -> None:
    try:
        with LOCAL_TASK.lock:
            if LOCAL_TASK.task_id != task_id:
                return
            LOCAL_TASK.log(f"[INFO] 任务 {task_id} 启动，算法={algorithm} pc={pc:.3f} pm={pm:.3f} iters={iterations}")
            LOCAL_TASK.progress = 1
            LOCAL_TASK.result = {"algorithm": algorithm, "convergence": [], "metrics": [], "hyperparams": {"iterations": iterations, "crossover_rate": pc, "mutate_rate": pm}}

        time.sleep(0.2)
        with LOCAL_TASK.lock:
            LOCAL_TASK.log("[INFO] 初始化种群...")
            LOCAL_TASK.progress = 5
            LOCAL_TASK.result["metrics"].append({"stage": "init", **_local_resource_point()})

        base = 1.0
        steps = max(1, int(iterations or 1))
        for i in range(steps):
            time.sleep(0.25)
            with LOCAL_TASK.lock:
                metric = round(0.5 + 0.5 * (1.0 - (i + 1) / steps), 4)
                LOCAL_TASK.log(f"[INFO] 进化迭代 {i+1}/{steps} metric={metric} ...")
                LOCAL_TASK.progress = 5 + int((i + 1) / steps * 95)
                base *= 0.88
                LOCAL_TASK.result["convergence"].append(round(base + (0.01 * ((i + 1) % 3) / 3.0), 6))
                LOCAL_TASK.result["metrics"].append({"stage": "iter", "iter": i + 1, **_local_resource_point()})

        with LOCAL_TASK.lock:
            LOCAL_TASK.log("[INFO] 分析完成。")
            LOCAL_TASK.progress = 100
            LOCAL_TASK.state = "completed"
            conv = LOCAL_TASK.result.get("convergence") or []
            LOCAL_TASK.result["best_score"] = min(conv) if conv else None
    except Exception as exc:
        with LOCAL_TASK.lock:
            LOCAL_TASK.state = "error"
            LOCAL_TASK.error = str(exc)
            LOCAL_TASK.log(f"[ERROR] {exc}")

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
                timeout_s = float(payload.get("timeout_s", 120) or 120)
                resp = session.post(
                    base_url.rstrip("/") + "/api/strategy_plan",
                    json={
                        "algorithm": algo,
                        "warmup": warmup,
                        "objective": objective,
                        "multi_gpu": multi_gpu,
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

        plan = StrategyPlan(fitness=None, warmup=warmup, objective=objective, multi_gpu=multi_gpu)
        if algo:
            try:
                plan.notes = (plan.notes + " " if plan.notes else "") + f"algorithm={algo}"
            except Exception:
                pass
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
                json={"objective": objective, "multi_gpu": multi_gpu, "warmup_iters": warmup_iters},
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

        return jsonify(StrategyCompare(objective=objective, multi_gpu=multi_gpu, warmup_iters=warmup_iters))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/actions/<action>", methods=["POST"])
def api_actions(action: str):
    payload = request.get_json(force=True, silent=True) or {}
    store.log(action, payload)
    store.update_state(action)
    return jsonify({"status": "ok"})


@app.route("/api/logs")
def api_logs():
    return jsonify(store.logs)


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
                },
                timeout=(3.0, timeout_s),
            )
        else:
            # Local execution in console process
            import uuid

            algorithm = str(payload.get("algorithm") or "ga")
            iterations = int(payload.get("iterations") or 20)
            pc = float(payload.get("crossover_rate") or 0.8)
            pm = float(payload.get("mutate_rate") or 0.2)
            with LOCAL_TASK.lock:
                if LOCAL_TASK.state == "running":
                    return jsonify({"error": "A task is already running"}), 409
                task_id = str(uuid.uuid4())
                LOCAL_TASK.reset(task_id)
                t = threading.Thread(target=_run_local_ga, args=(task_id, algorithm, iterations, pc, pm), daemon=True)
                t.start()
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


if __name__ == "__main__":
    print("=" * 50)
    print("GAPA Console 启动成功！")
    print("本地访问 → http://localhost:7777")
    print("局域网访问 → http://本机IP:7777")
    print("=" * 50)
    app.run(host="0.0.0.0", port=7777, threaded=True)
