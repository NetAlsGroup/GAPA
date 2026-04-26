from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

from gapa.config import get_app_base_url, get_results_dir


def _results_root() -> Path:
    return get_results_dir(Path(__file__).resolve().parents[1])


def _load_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _load_run_report_rows(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    jsonl_path = path or (_results_root() / "run_reports.jsonl")
    if not jsonl_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            if "algorithm_id" in obj and "dataset" in obj:
                rows.append(obj)
    except Exception:
        return []
    return rows


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _benchmark_key(algorithm_id: str, dataset: str, mode: str, remote: bool) -> str:
    return f"{algorithm_id}|{dataset}|{mode}|remote={int(bool(remote))}"


def _aggregate_run_trends(rows: List[Dict[str, Any]], *, last_n: int = 20) -> Dict[str, Any]:
    if last_n <= 0:
        last_n = 20
    groups: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        algo = str(row.get("algorithm_id") or "")
        ds = str(row.get("dataset") or "")
        mode = str(row.get("resolved_mode") or row.get("mode") or "")
        remote = bool(row.get("remote"))
        key = _benchmark_key(algo, ds, mode, remote)
        if key not in groups:
            groups[key] = {
                "key": key,
                "algorithm_id": algo,
                "dataset": ds,
                "mode": mode,
                "remote": remote,
                "rows": [],
            }
        groups[key]["rows"].append(row)

    out_groups: List[Dict[str, Any]] = []
    for g in groups.values():
        hist = g["rows"][-last_n:]
        avg_list = [_to_float(x.get("iter_avg_ms")) for x in hist]
        sec_list = [_to_float(x.get("iter_seconds")) for x in hist]
        thr_list = [_to_float(x.get("throughput_ips")) for x in hist]
        fit_list = [_to_float(x.get("best_fitness")) for x in hist]
        avg_list = [x for x in avg_list if x is not None]
        sec_list = [x for x in sec_list if x is not None]
        thr_list = [x for x in thr_list if x is not None]
        fit_list = [x for x in fit_list if x is not None]
        regressions = 0
        for x in hist:
            bench = x.get("benchmark")
            if isinstance(bench, dict) and bench.get("regressed"):
                regressions += 1

        def _mean(vals: List[float]) -> Optional[float]:
            return (sum(vals) / len(vals)) if vals else None

        out_groups.append(
            {
                "key": g["key"],
                "algorithm_id": g["algorithm_id"],
                "dataset": g["dataset"],
                "mode": g["mode"],
                "remote": g["remote"],
                "runs": len(hist),
                "regressions": regressions,
                "iter_avg_ms_mean": _mean(avg_list),
                "iter_seconds_mean": _mean(sec_list),
                "throughput_ips_mean": _mean(thr_list),
                "best_fitness_mean": _mean(fit_list),
                "last_run": hist[-1] if hist else None,
            }
        )
    out_groups.sort(key=lambda x: (-int(x.get("runs") or 0), str(x.get("key") or "")))
    return {"groups": out_groups, "total_groups": len(out_groups), "total_rows": len(rows), "last_n": last_n}


def _resolve_default_api_base() -> str:
    env_base = get_app_base_url()
    return str(env_base).rstrip("/")


class ResourceManager:
    """Independent public resource and remote-runtime access surface."""

    def __init__(self, api_base: Optional[str] = None, timeout_s: float = 10.0):
        self.api_base = api_base
        self.timeout_s = float(timeout_s)

    def _use_proxy(self) -> bool:
        return bool(self.api_base)

    def _resolve_api_base(self) -> str:
        if self.api_base:
            return self.api_base.rstrip("/")
        return _resolve_default_api_base()

    def _is_direct_remote_scope(self, scope: Optional[str]) -> bool:
        if self._use_proxy():
            return False
        token = str(scope or "").strip()
        if not token or token in {"all", "local"}:
            return False
        servers = self.server()
        if not isinstance(servers, list):
            return False
        for item in servers:
            if not isinstance(item, dict):
                continue
            sid = str(item.get("id") or "")
            name = str(item.get("name") or "")
            if token in (sid, name) or token.lower() in (sid.lower(), name.lower()):
                return sid != "local"
        return False

    def _local_snapshots(self) -> Dict[str, Any]:
        from server.resource_service import get_all_resources

        return get_all_resources(None)

    def _local_server_entries(self) -> List[Dict[str, Any]]:
        from server.resource_service import load_server_list

        return load_server_list()

    def _local_strategy_plan(
        self,
        *,
        server_id: Optional[str] = None,
        algorithm: Optional[str] = None,
        dataset: Optional[str] = None,
        mode: Optional[str] = None,
        warmup: int = 0,
        objective: str = "time",
        multi_gpu: bool = True,
        gpu_busy_threshold: Optional[float] = None,
        min_gpu_free_mb: Optional[int] = None,
        tpe_trials: Optional[int] = None,
        tpe_warmup: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        from gapa.autoadapt import StrategyPlan

        target_server = str(server_id or "local")
        if target_server != "local":
            payload: Dict[str, Any] = {
                "algorithm": algorithm,
                "dataset": dataset,
                "mode": mode,
                "warmup": int(warmup),
                "objective": objective,
                "multi_gpu": bool(multi_gpu),
                "gpu_busy_threshold": gpu_busy_threshold,
                "min_gpu_free_mb": min_gpu_free_mb,
                "tpe_trials": tpe_trials,
                "tpe_warmup": tpe_warmup,
            }
            return self._post_server_direct(
                target_server,
                "/api/strategy_plan",
                payload,
                timeout_s=max(float(timeout_s or self.timeout_s), 30.0),
            )

        snap = self._local_snapshots().get("local")
        if not isinstance(snap, dict):
            return {"error": "local resource snapshot unavailable", "detail": snap}
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
        )
        out = plan.to_dict() if hasattr(plan, "to_dict") else dict(plan)
        if algorithm:
            out["algorithm"] = algorithm
        if dataset:
            out["dataset"] = dataset
        if mode:
            out["mode"] = mode
        out["server_id"] = "local"
        return out

    def _local_distributed_strategy_plan(
        self,
        *,
        server_ids: Optional[List[str]] = None,
        servers: Optional[List[str]] = None,
        algorithm: Optional[str] = None,
        dataset: Optional[str] = None,
        mode: Optional[str] = None,
        per_server_gpus: int = 1,
        min_gpu_free_mb: int = 1024,
        gpu_busy_threshold: float = 85.0,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        from gapa.autoadapt import DistributedStrategyPlan

        requested = list(server_ids or servers or [])
        snapshots = self._local_snapshots()
        if not requested:
            requested = list(snapshots.keys())

        server_resources: Dict[str, Any] = {}
        server_plans: Dict[str, Any] = {}
        for sid in requested:
            sid = str(sid)
            snap = snapshots.get(sid)
            if not isinstance(snap, dict):
                server_resources[sid] = {"error": "snapshot unavailable"}
                continue
            server_resources[sid] = snap
            if sid == "local":
                local_plan = self._local_strategy_plan(
                    server_id="local",
                    algorithm=algorithm,
                    dataset=dataset,
                    mode=mode,
                    warmup=0,
                    objective="time",
                    multi_gpu=True,
                    gpu_busy_threshold=gpu_busy_threshold,
                    min_gpu_free_mb=min_gpu_free_mb,
                )
                if isinstance(local_plan, dict) and "backend" in local_plan:
                    server_plans[sid] = local_plan
                continue
            remote_plan = self._post_server_direct(
                sid,
                "/api/strategy_plan",
                {
                    "algorithm": algorithm,
                    "dataset": dataset,
                    "mode": mode,
                    "warmup": 0,
                    "objective": "time",
                    "multi_gpu": True,
                    "gpu_busy_threshold": gpu_busy_threshold,
                    "min_gpu_free_mb": min_gpu_free_mb,
                },
                timeout_s=max(float(timeout_s or self.timeout_s), 30.0),
            )
            if isinstance(remote_plan, dict) and "backend" in remote_plan:
                server_plans[sid] = remote_plan

        def _plan_obj(raw: Any) -> Any:
            if not isinstance(raw, dict) or "backend" not in raw:
                return None
            try:
                from gapa.autoadapt.api.schemas import Plan

                keys = getattr(Plan, "__dataclass_fields__", {}).keys()
                return Plan(**{k: raw.get(k) for k in keys})
            except Exception:
                return None

        plan = DistributedStrategyPlan(
            server_resources=server_resources,
            server_plans={sid: _plan_obj(raw) for sid, raw in server_plans.items()},
            per_server_gpus=per_server_gpus,
            min_gpu_free_mb=min_gpu_free_mb,
            gpu_busy_threshold=gpu_busy_threshold,
        )
        plan["server_resources"] = server_resources
        plan["server_plans"] = server_plans
        if algorithm:
            plan["algorithm"] = algorithm
        if dataset:
            plan["dataset"] = dataset
        if mode:
            plan["mode"] = mode
        return plan

    def _http_get_json(self, url: str) -> Dict[str, Any]:
        if requests is None:
            return {"error": "requests not available"}
        try:
            session = requests.Session()
            session.trust_env = False
            resp = session.get(url, timeout=self.timeout_s)
        except Exception as exc:
            return {"error": str(exc)}
        if not getattr(resp, "ok", False):
            try:
                body = resp.json()
            except Exception:
                body = {"raw": getattr(resp, "text", "")}
            return {"error": f"HTTP {resp.status_code}", "body": body}
        try:
            return resp.json()
        except Exception as exc:
            return {"error": f"invalid json: {exc}"}

    def _http_get_json_with_params(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not params:
            return self._http_get_json(url)
        query = urlencode({k: v for k, v in params.items() if v is not None})
        target = f"{url}?{query}" if query else url
        return self._http_get_json(target)

    def _http_post_json(self, url: str, payload: Dict[str, Any], timeout_s: Optional[float] = None) -> Dict[str, Any]:
        if requests is None:
            return {"error": "requests not available"}
        try:
            session = requests.Session()
            session.trust_env = False
            read_timeout = max(5.0, float(timeout_s if timeout_s is not None else self.timeout_s))
            resp = session.post(url, json=payload, timeout=(3.0, read_timeout))
        except Exception as exc:
            return {"error": str(exc)}
        try:
            body: Any = resp.json()
        except Exception:
            body = {"raw": (getattr(resp, "text", "") or "")[:500]}
        if not getattr(resp, "ok", False):
            return {"error": f"HTTP {resp.status_code}", "body": body}
        return body if isinstance(body, dict) else {"data": body}

    def _post_server_direct(
        self,
        server_id: str,
        endpoint: str,
        payload: Dict[str, Any],
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        if requests is None:
            return {"error": "requests not available"}
        servers = self.server()
        if not isinstance(servers, list):
            return {"error": "server list unavailable", "detail": servers}
        target = None
        for s in servers:
            sid = str(s.get("id") or "")
            name = str(s.get("name") or "")
            if server_id in (sid, name) or server_id.lower() in (sid.lower(), name.lower()):
                target = s
                break
        if target is None:
            return {"error": "server not found", "server_id": server_id}
        base_url = str(target.get("base_url") or "").rstrip("/")
        if not base_url:
            return {"error": "missing base_url", "server_id": server_id, "server": target}
        try:
            session = requests.Session()
            session.trust_env = False
            read_timeout = max(5.0, float(timeout_s if timeout_s is not None else self.timeout_s))
            resp = session.post(base_url + endpoint, json=payload, timeout=(3.0, read_timeout))
        except Exception as exc:
            return {"error": str(exc), "server_id": server_id, "url": base_url + endpoint}
        try:
            body = resp.json()
        except Exception:
            body = {"raw": (getattr(resp, "text", "") or "")[:500]}
        if not getattr(resp, "ok", False):
            return {"error": f"HTTP {resp.status_code}", "body": body, "server_id": server_id, "url": base_url + endpoint}
        return body if isinstance(body, dict) else {"data": body}

    def _normalize_server_id(self, raw: str, servers: List[Dict[str, Any]]) -> Optional[str]:
        token = str(raw or "").strip()
        if not token:
            return None
        token_lower = token.lower()
        aliases = [token, f"Server {token}", f"server {token}"]
        alias_lower = [a.lower() for a in aliases]
        for item in servers:
            sid = str(item.get("id") or "")
            name = str(item.get("name") or "")
            sid_lower = sid.lower()
            name_lower = name.lower()
            if token in (sid, name) or token_lower in (sid_lower, name_lower):
                return sid
            if sid in aliases or sid_lower in alias_lower:
                return sid
            if name in aliases or name_lower in alias_lower:
                return sid
        return None

    def _fetch_snapshots(self) -> Dict[str, Any]:
        base = self._resolve_api_base()
        data = self._http_get_json(f"{base}/api/v1/resources/all")
        if isinstance(data, dict) and "error" not in data:
            return data
        legacy = self._http_get_json(f"{base}/api/all_resources")
        if isinstance(legacy, dict) and "error" not in legacy:
            return legacy
        return data if isinstance(data, dict) else {"error": "invalid snapshots"}

    def server(self) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        if not self._use_proxy():
            servers = self._local_server_entries()
            snapshots = self._local_snapshots()
            results: List[Dict[str, Any]] = []
            for item in servers:
                if not isinstance(item, dict):
                    continue
                sid = item.get("id") or item.get("name") or item.get("host")
                snap = snapshots.get(sid) if isinstance(snapshots, dict) else None
                online = isinstance(snap, dict) and "error" not in snap
                base_url = item.get("base_url")
                results.append(
                    {
                        "id": sid,
                        "name": item.get("name") or sid,
                        "base_url": base_url,
                        "status": "Activate" if online else "Deactivate",
                        "online": bool(online),
                    }
                )
            return results
        base = self._resolve_api_base()
        servers = self._http_get_json(f"{base}/api/servers")
        if not isinstance(servers, list):
            return {"error": "invalid servers response", "body": servers}
        snapshots = self._fetch_snapshots()
        if isinstance(snapshots, dict) and "error" in snapshots:
            snapshots = {}
        results: List[Dict[str, Any]] = []
        for item in servers:
            if not isinstance(item, dict):
                continue
            sid = item.get("id") or item.get("name") or item.get("host")
            name = item.get("name") or item.get("id") or sid
            base_url = item.get("base_url")
            if not base_url:
                host = item.get("ip") or item.get("host")
                port = item.get("port")
                protocol = item.get("protocol") or "http"
                if host:
                    base_url = f"{protocol}://{host}{f':{port}' if port else ''}"
            snap = snapshots.get(sid) if isinstance(snapshots, dict) else None
            online = bool(snap and snap.get("online"))
            status = "Activate" if online else "Deactivate"
            results.append(
                {
                    "id": sid,
                    "name": name,
                    "base_url": base_url,
                    "status": status,
                    "online": online,
                }
            )
        return results

    def server_resource(self, server_id_or_name: str) -> Dict[str, Any]:
        if not self._use_proxy():
            servers = self.server()
            if isinstance(servers, dict) and "error" in servers:
                return servers
            target_id = None
            lookup = str(server_id_or_name or "")
            for item in servers:
                sid = item.get("id")
                name = item.get("name")
                if lookup == sid or lookup == name:
                    target_id = sid
                    break
            if target_id is None:
                for item in servers:
                    sid = item.get("id")
                    name = item.get("name")
                    if name and name.lower() == lookup.lower():
                        target_id = sid
                        break
            if not target_id:
                return {"error": "server not found", "input": server_id_or_name}
            snapshots = self._local_snapshots()
            snap = snapshots.get(target_id) if isinstance(snapshots, dict) else None
            if isinstance(snap, dict):
                return snap
            return {"error": "server not found in snapshots", "server_id": target_id}
        servers = self.server()
        if isinstance(servers, dict) and "error" in servers:
            return servers
        target_id = None
        lookup = str(server_id_or_name or "")
        for item in servers:
            sid = item.get("id")
            name = item.get("name")
            if lookup == sid or lookup == name:
                target_id = sid
                break
        if target_id is None:
            for item in servers:
                sid = item.get("id")
                name = item.get("name")
                if name and name.lower() == lookup.lower():
                    target_id = sid
                    break
        if not target_id:
            return {"error": "server not found", "input": server_id_or_name}
        snapshots = self._fetch_snapshots()
        if isinstance(snapshots, dict) and "error" in snapshots:
            return snapshots
        if isinstance(snapshots, dict) and target_id in snapshots:
            return snapshots[target_id]
        return {"error": "server not found in snapshots", "server_id": target_id}

    def resources(self, all_servers: bool = True) -> Dict[str, Any]:
        if not self._use_proxy():
            snapshots = self._local_snapshots()
            if all_servers:
                return snapshots
            local = snapshots.get("local")
            return local if isinstance(local, dict) else {"error": "local resource snapshot unavailable", "detail": local}
        base = self._resolve_api_base()
        if all_servers:
            data = self._http_get_json(f"{base}/api/v1/resources/all")
            if isinstance(data, dict) and "error" not in data:
                return data
            return self._http_get_json(f"{base}/api/all_resources")
        return self._http_get_json(f"{base}/api/resources")

    def lock_status(self, scope: str = "all", realtime: bool = True) -> Dict[str, Any]:
        if self._is_direct_remote_scope(scope):
            server = self._normalize_server_id(scope, self.server() if isinstance(self.server(), list) else [])
            if not server:
                return {"error": "server not found", "scope": scope}
            if requests is None:
                return {"error": "requests not available"}
            servers = self.server()
            if not isinstance(servers, list):
                return {"error": "server list unavailable", "detail": servers}
            target = next((s for s in servers if str(s.get("id")) == server), None)
            if not isinstance(target, dict) or not target.get("base_url"):
                return {"error": "missing base_url", "scope": scope}
            base = str(target.get("base_url")).rstrip("/")
            return self._http_get_json_with_params(
                f"{base}/api/resource_lock/status",
                {"realtime": str(bool(realtime)).lower()},
            )
        base = self._resolve_api_base()
        return self._http_get_json_with_params(
            f"{base}/api/resource_lock/status",
            {"scope": scope, "realtime": str(bool(realtime)).lower()},
        )

    def lock_resource(
        self,
        scope: str = "all",
        duration_s: float = 600.0,
        warmup_iters: int = 2,
        mem_mb: int = 1024,
        strict_idle: bool = False,
        devices: Optional[List[Any]] = None,
        devices_by_server: Optional[Dict[str, List[Any]]] = None,
    ) -> Dict[str, Any]:
        if self._is_direct_remote_scope(scope):
            server = self._normalize_server_id(scope, self.server() if isinstance(self.server(), list) else [])
            if not server:
                return {"error": "server not found", "scope": scope}
            payload: Dict[str, Any] = {
                "duration_s": float(duration_s),
                "warmup_iters": int(warmup_iters),
                "mem_mb": int(mem_mb),
                "strict_idle": bool(strict_idle),
            }
            if devices is not None:
                payload["devices"] = devices
            if devices_by_server is not None:
                payload["devices"] = devices_by_server.get(server)
            return self._post_server_direct(server, "/api/resource_lock", payload)
        base = self._resolve_api_base()
        payload: Dict[str, Any] = {
            "scope": scope,
            "duration_s": float(duration_s),
            "warmup_iters": int(warmup_iters),
            "mem_mb": int(mem_mb),
            "strict_idle": bool(strict_idle),
        }
        if devices is not None:
            payload["devices"] = devices
        if devices_by_server is not None:
            payload["devices_by_server"] = devices_by_server
        return self._http_post_json(f"{base}/api/resource_lock", payload)

    def unlock_resource(self, scope: str = "all") -> Dict[str, Any]:
        if self._is_direct_remote_scope(scope):
            server = self._normalize_server_id(scope, self.server() if isinstance(self.server(), list) else [])
            if not server:
                return {"error": "server not found", "scope": scope}
            return self._post_server_direct(server, "/api/resource_lock/release", {})
        base = self._resolve_api_base()
        return self._http_post_json(f"{base}/api/resource_lock/release", {"scope": scope})

    def release_resource(self, scope: str = "all") -> Dict[str, Any]:
        return self.unlock_resource(scope=scope)

    def renew_resource(
        self,
        scope: str = "all",
        duration_s: Optional[float] = None,
        lock_id: Optional[str] = None,
        owner: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self._is_direct_remote_scope(scope):
            server = self._normalize_server_id(scope, self.server() if isinstance(self.server(), list) else [])
            if not server:
                return {"error": "server not found", "scope": scope}
            payload: Dict[str, Any] = {}
            if duration_s is not None:
                payload["duration_s"] = float(duration_s)
            if lock_id:
                payload["lock_id"] = str(lock_id)
            if owner:
                payload["owner"] = str(owner)
            return self._post_server_direct(server, "/api/resource_lock/renew", payload)
        base = self._resolve_api_base()
        payload: Dict[str, Any] = {"scope": scope}
        if duration_s is not None:
            payload["duration_s"] = float(duration_s)
        if lock_id:
            payload["lock_id"] = str(lock_id)
        if owner:
            payload["owner"] = str(owner)
        return self._http_post_json(f"{base}/api/resource_lock/renew", payload)

    def strategy_plan(
        self,
        server_id: Optional[str] = None,
        algorithm: Optional[str] = None,
        dataset: Optional[str] = None,
        mode: Optional[str] = None,
        warmup: int = 0,
        objective: str = "time",
        multi_gpu: bool = True,
        gpu_busy_threshold: Optional[float] = None,
        min_gpu_free_mb: Optional[int] = None,
        tpe_trials: Optional[int] = None,
        tpe_warmup: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not self._use_proxy():
            return self._local_strategy_plan(
                server_id=server_id,
                algorithm=algorithm,
                dataset=dataset,
                mode=mode,
                warmup=warmup,
                objective=objective,
                multi_gpu=multi_gpu,
                gpu_busy_threshold=gpu_busy_threshold,
                min_gpu_free_mb=min_gpu_free_mb,
                tpe_trials=tpe_trials,
                tpe_warmup=tpe_warmup,
                timeout_s=timeout_s,
            )
        base = self._resolve_api_base()
        payload: Dict[str, Any] = {
            "server_id": server_id or "local",
            "warmup": int(warmup),
            "objective": objective,
            "multi_gpu": bool(multi_gpu),
        }
        if algorithm:
            payload["algorithm"] = algorithm
        if dataset:
            payload["dataset"] = dataset
        if mode:
            payload["mode"] = mode
        if gpu_busy_threshold is not None:
            payload["gpu_busy_threshold"] = float(gpu_busy_threshold)
        if min_gpu_free_mb is not None:
            payload["min_gpu_free_mb"] = int(min_gpu_free_mb)
        if tpe_trials is not None:
            payload["tpe_trials"] = int(tpe_trials)
        if tpe_warmup is not None:
            payload["tpe_warmup"] = int(tpe_warmup)
        return self._http_post_json(f"{base}/api/strategy_plan", payload, timeout_s=timeout_s)

    def distributed_strategy_plan(
        self,
        server_ids: Optional[List[str]] = None,
        servers: Optional[List[str]] = None,
        algorithm: Optional[str] = None,
        dataset: Optional[str] = None,
        mode: Optional[str] = None,
        per_server_gpus: int = 1,
        min_gpu_free_mb: int = 1024,
        gpu_busy_threshold: float = 85.0,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not self._use_proxy():
            return self._local_distributed_strategy_plan(
                server_ids=server_ids,
                servers=servers,
                algorithm=algorithm,
                dataset=dataset,
                mode=mode,
                per_server_gpus=per_server_gpus,
                min_gpu_free_mb=min_gpu_free_mb,
                gpu_busy_threshold=gpu_busy_threshold,
                timeout_s=timeout_s,
            )
        base = self._resolve_api_base()
        payload: Dict[str, Any] = {
            "server_ids": server_ids or servers or [],
            "per_server_gpus": int(per_server_gpus),
            "min_gpu_free_mb": int(min_gpu_free_mb),
            "gpu_busy_threshold": float(gpu_busy_threshold),
        }
        if algorithm:
            payload["algorithm"] = algorithm
        if dataset:
            payload["dataset"] = dataset
        if mode:
            payload["mode"] = mode
        return self._http_post_json(f"{base}/api/distributed_strategy_plan", payload, timeout_s=timeout_s)

    def analysis_status(self, server_id: Optional[str] = None) -> Dict[str, Any]:
        base = self._resolve_api_base()
        return self._http_get_json_with_params(
            f"{base}/api/analysis/status",
            {"server_id": server_id} if server_id else None,
        )

    def analysis_queue(self, server_id: Optional[str] = None) -> Dict[str, Any]:
        base = self._resolve_api_base()
        return self._http_get_json_with_params(
            f"{base}/api/analysis/queue",
            {"server_id": server_id} if server_id else None,
        )

    def analysis_start(
        self,
        *,
        algorithm: str,
        dataset: str,
        iterations: int = 20,
        mode: str = "S",
        crossover_rate: float = 0.8,
        mutate_rate: float = 0.2,
        pop_size: Optional[int] = None,
        server_id: Optional[str] = None,
        queue_if_busy: bool = False,
        owner: str = "",
        priority: int = 0,
        release_lock_on_finish: bool = True,
        timeout_s: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        base = self._resolve_api_base()
        payload: Dict[str, Any] = {
            "algorithm": algorithm,
            "dataset": dataset,
            "iterations": int(iterations),
            "mode": mode,
            "crossover_rate": float(crossover_rate),
            "mutate_rate": float(mutate_rate),
            "queue_if_busy": bool(queue_if_busy),
            "owner": str(owner or ""),
            "priority": int(priority),
            "release_lock_on_finish": bool(release_lock_on_finish),
        }
        if pop_size is not None:
            payload["pop_size"] = int(pop_size)
        if server_id:
            payload["server_id"] = server_id
        if timeout_s is not None:
            payload["timeout_s"] = float(timeout_s)
        if isinstance(extra, dict):
            payload.update(extra)
        return self._http_post_json(
            f"{base}/api/analysis/start",
            payload,
            timeout_s=max(float(timeout_s or self.timeout_s), 30.0),
        )

    def analysis_stop(self, server_id: Optional[str] = None) -> Dict[str, Any]:
        base = self._resolve_api_base()
        payload: Dict[str, Any] = {}
        if server_id:
            payload["server_id"] = server_id
        return self._http_post_json(f"{base}/api/analysis/stop", payload)

    def transport_metrics(self) -> Dict[str, Any]:
        base = self._resolve_api_base()
        return self._http_get_json(f"{base}/api/transport/metrics")

    def resource_rows(self, resources: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        snap = resources or self.resources(all_servers=True)
        if not isinstance(snap, dict):
            return []
        rows: List[Dict[str, Any]] = []
        for sid, val in snap.items():
            if not isinstance(val, dict):
                continue
            cpu = val.get("cpu") or {}
            memory = val.get("memory") or {}
            gpus = val.get("gpus") or []
            gpu_count = len(gpus) if isinstance(gpus, list) else 0
            gpu_util = None
            if isinstance(gpus, list) and gpus:
                util_values = [g.get("gpu_util_percent") for g in gpus if isinstance(g, dict)]
                util_values = [float(x) for x in util_values if isinstance(x, (int, float))]
                if util_values:
                    gpu_util = sum(util_values) / len(util_values)
            rows.append(
                {
                    "server_id": sid,
                    "name": val.get("name"),
                    "online": val.get("online"),
                    "time": val.get("time"),
                    "cpu_usage_percent": cpu.get("usage_percent"),
                    "memory_percent": memory.get("percent"),
                    "gpu_count": gpu_count,
                    "gpu_util_avg_percent": gpu_util,
                }
            )
        return rows

    def resource_dataframe(self, rows: Optional[List[Dict[str, Any]]] = None):
        try:
            import pandas as pd  # type: ignore
        except Exception:
            return None
        return pd.DataFrame(rows or self.resource_rows())

    def plot_resources(self, metric: str = "cpu_usage_percent", rows: Optional[List[Dict[str, Any]]] = None):
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return None, None
        data = rows or self.resource_rows()
        labels = [str(r.get("server_id")) for r in data]
        vals: List[float] = []
        for r in data:
            val = r.get(metric)
            vals.append(float(val) if isinstance(val, (int, float)) else 0.0)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(labels, vals)
        ax.set_ylabel(metric)
        ax.set_title("GAPA Resource Snapshot")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return fig, ax

    def run_trends(self, last_n: int = 20) -> Dict[str, Any]:
        rows = _load_run_report_rows()
        return _aggregate_run_trends(rows, last_n=last_n)

    def lock_mnm(
        self,
        server_inputs: Optional[List[str]] = None,
        duration_s: float = 900.0,
        warmup_iters: int = 1,
        mem_mb: int = 1024,
        owner: str = "",
        print_log: bool = False,
    ) -> Dict[str, Any]:
        server_list = self.server()
        if not isinstance(server_list, list):
            return {"error": "server list error", "detail": server_list}
        online = [s for s in server_list if s.get("online") and str(s.get("id")) != "local"]
        if server_inputs:
            targets: List[str] = []
            for raw in server_inputs:
                sid = self._normalize_server_id(str(raw), online)
                if not sid:
                    return {"error": "server offline or unknown", "server": raw}
                targets.append(sid)
        else:
            targets = [str(s.get("id")) for s in online]
        if not targets:
            return {"error": "no online remote servers for mnm lock"}

        api_base = self._resolve_api_base().rstrip("/")
        if print_log:
            print(f"[GAPA] MNM lock enabled. api_base={api_base}, targets={targets}")

        locked: List[str] = []
        details: Dict[str, Any] = {}
        for sid in targets:
            payload = {
                "scope": sid,
                "duration_s": float(duration_s),
                "warmup_iters": int(warmup_iters),
                "mem_mb": int(mem_mb),
                "owner": str(owner or ""),
            }
            body = self._http_post_json(f"{api_base}/api/resource_lock", payload)
            details[sid] = body
            result = body.get("results", {}).get(sid, {}) if isinstance(body, dict) else {}
            active = bool(isinstance(result, dict) and result.get("active"))
            if print_log:
                print(f"[GAPA] MNM lock -> {sid}: {'ok' if active else 'failed'}")
            if active:
                locked.append(sid)

        if not locked:
            return {"error": "mnm lock failed on all targets", "api_base": api_base, "targets": targets, "details": details}
        lock_ids: Dict[str, str] = {}
        for sid in locked:
            one = details.get(sid, {})
            if isinstance(one, dict):
                lock_obj = one.get("results", {}).get(sid, {})
                if isinstance(lock_obj, dict) and lock_obj.get("lock_id"):
                    lock_ids[sid] = str(lock_obj.get("lock_id"))
        return {
            "api_base": api_base,
            "targets": targets,
            "locked": locked,
            "lock_ids": lock_ids,
            "owner": str(owner or ""),
            "details": details,
        }

    def unlock_servers(self, server_ids: List[str], api_base: Optional[str] = None, print_log: bool = False) -> Dict[str, Any]:
        base = (api_base or self._resolve_api_base()).rstrip("/")
        results: Dict[str, Any] = {}
        for sid in server_ids:
            body = self._http_post_json(f"{base}/api/resource_lock/release", {"scope": sid})
            results[sid] = body
            if print_log:
                ok = bool(isinstance(body, dict) and body.get("results", {}).get(sid))
                print(f"[GAPA] MNM unlock -> {sid}: {'ok' if ok else 'failed'}")
        return {"api_base": base, "results": results}

    def renew_mnm(
        self,
        *,
        lock_info: Dict[str, Any],
        duration_s: Optional[float] = None,
        print_log: bool = False,
    ) -> Dict[str, Any]:
        api_base = str(lock_info.get("api_base") or self._resolve_api_base()).rstrip("/")
        owner = lock_info.get("owner")
        lock_ids = lock_info.get("lock_ids") or {}
        servers = lock_info.get("locked") or lock_info.get("targets") or []
        results: Dict[str, Any] = {}
        for sid in servers:
            payload: Dict[str, Any] = {"scope": sid}
            if duration_s is not None:
                payload["duration_s"] = float(duration_s)
            if isinstance(lock_ids, dict) and lock_ids.get(sid):
                payload["lock_id"] = str(lock_ids.get(sid))
            if owner:
                payload["owner"] = owner
            body = self._http_post_json(f"{api_base}/api/resource_lock/renew", payload)
            if isinstance(body, dict) and str(body.get("error") or "").startswith("HTTP 404"):
                body = self._post_server_direct(str(sid), "/api/resource_lock/renew", payload)
            results[sid] = body
            if print_log:
                node = body.get("results", {}).get(sid, {}) if isinstance(body, dict) else {}
                if isinstance(node, dict) and node:
                    ok = bool(node.get("active"))
                elif isinstance(body, dict) and body.get("active") is not None:
                    ok = bool(body.get("active"))
                else:
                    ok = False
                print(f"[GAPA] MNM renew -> {sid}: {'ok' if ok else 'failed'}")
        return {"api_base": api_base, "results": results}
