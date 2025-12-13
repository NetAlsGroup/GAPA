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

# 使用 /static 避免与 /api/* 路由冲突
app = Flask(__name__, static_folder=str(STATIC_ROOT), static_url_path="/static")
store = JobStore()
ALGO_MANIFEST_PATH = Path(__file__).resolve().parent / "gapa" / "algorithm" / "manifest.json"


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
            servers = load_server_list()
            target = next((s for s in servers if s.get("id") == server_id), None)
            # Fallback match: UI may use "ip-port" when config has no explicit id.
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
            base_url = target.get("base_url") if target else None
            if not base_url:
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


if __name__ == "__main__":
    print("=" * 50)
    print("GAPA Console 启动成功！")
    print("本地访问 → http://localhost:7777")
    print("局域网访问 → http://本机IP:7777")
    print("=" * 50)
    app.run(host="0.0.0.0", port=7777, threaded=True)
