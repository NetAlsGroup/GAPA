import threading
import time
import requests
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from .resource_service import load_server_list, current_resource_snapshot
from .mode_runtime import classify_transport_error, request_with_retry

class ServerStateManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ServerStateManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, interval: int = 10):
        if self._initialized:
            return
        
        self.interval = interval
        self.snapshots: Dict[str, Dict[str, Any]] = {}
        self.status: Dict[str, str] = {}  # "online" or "offline"
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self.transport_metrics: Dict[str, Dict[str, float | int]] = {}
        self._initialized = True

    def _worker(self):
        while not self._stop_event.is_set():
            self.refresh_all()
            time.sleep(self.interval)

    def refresh_one(self, sid: str) -> Dict[str, Any]:
        """Fetch status for a single server immediately (blocking)."""
        servers = load_server_list()
        target = next((s for s in servers if s.get("id") == sid), None)
        if not target:
            return {"error": "unknown server", "online": False}
        
        base_url = target.get("base_url")
        
        if sid == "local":
            try:
                data = current_resource_snapshot()
                from .resource_lock import LOCK_MANAGER
                data["lock"] = {"results": {"local": LOCK_MANAGER.status()}}
                data["online"] = True
                with self.lock:
                    self.snapshots[sid] = data
                    self.status[sid] = "online"
                return data
            except Exception as e:
                res = {"error": str(e), "online": False}
                with self.lock:
                    self.snapshots[sid] = res
                    self.status[sid] = "offline"
                return res
        
        try:
            session = requests.Session()
            session.trust_env = False
            r_resp, r_meta = request_with_retry(
                session=session,
                method="GET",
                url=base_url.rstrip("/") + "/api/resources",
                timeout=(2.0, 5.0),
                op="resource_lock_status",
            )
            l_resp, l_meta = request_with_retry(
                session=session,
                method="GET",
                url=base_url.rstrip("/") + "/api/resource_lock/status",
                timeout=(2.0, 5.0),
                op="resource_lock_status",
            )
            
            data = r_resp.json() if r_resp.ok else {"error_res": r_resp.status_code}
            data["lock"] = l_resp.json() if l_resp.ok else {"error_lock": l_resp.status_code}
            data["online"] = True
            data["transport_meta"] = {"resources": r_meta, "lock_status": l_meta}
            self.transport_metrics[sid] = {
                "retries": int(r_meta.get("retries") or 0) + int(l_meta.get("retries") or 0),
                "calls": 2,
                "failures": int(not r_resp.ok) + int(not l_resp.ok),
            }
            
            with self.lock:
                self.snapshots[sid] = data
                self.status[sid] = "online"
            return data
        except Exception as e:
            res = {"error": str(e), "code": classify_transport_error(exc=e), "online": False}
            with self.lock:
                self.snapshots[sid] = res
                self.status[sid] = "offline"
            return res

    def refresh_all(self):
        servers = load_server_list()
        results = {}
        
        # Use our new refresh_one logic but in parallel
        # Note: refresh_one updates self.snapshots internally, but here we gather them to avoid partial updates?
        # Actually refresh_one updates atomically per server. That's fine.
        
        def _fetch(server):
            sid = server.get("id")
            # We call the internal logic of refresh_one but maybe optimize to avoiding reloading server list?
            # For simplicity, we can just call the logic directly or reuse.
            # Let's reuse internal logic to avoid server list reload for each thread if possible,
            # but load_server_list is cached typically or fast enough.
            # To be safe and reuse code, we can just call self.refresh_one(sid)
            return sid, self.refresh_one(sid)

        with ThreadPoolExecutor(max_workers=len(servers) or 1) as executor:
            futures = [executor.submit(_fetch, s) for s in servers]
            for future in as_completed(futures):
                sid, data = future.result()
                results[sid] = data
        
        # Snapshots are already updated by refresh_one calls


    def get_snapshots(self) -> Dict[str, Dict[str, Any]]:
        with self.lock:
            return self.snapshots.copy()

    def get_status(self, sid: str) -> str:
        with self.lock:
            return self.status.get(sid, "unknown")

    def stop(self):
        self._stop_event.set()

state_manager = ServerStateManager()
