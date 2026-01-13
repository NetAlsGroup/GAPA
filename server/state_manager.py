import threading
import time
import requests
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from .resource_service import load_server_list, current_resource_snapshot

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
        self._initialized = True

    def _worker(self):
        while not self._stop_event.is_set():
            self.refresh_all()
            time.sleep(self.interval)

    def refresh_all(self):
        servers = load_server_list()
        results = {}
        
        def fetch_one(server: Dict[str, Any]):
            sid = server.get("id")
            base_url = server.get("base_url")
            
            if sid == "local":
                try:
                    data = current_resource_snapshot()
                    from .resource_lock import LOCK_MANAGER
                    data["lock"] = {"results": {"local": LOCK_MANAGER.status()}}
                    data["online"] = True
                    return sid, data
                except Exception as e:
                    return sid, {"error": str(e), "online": False}
            
            try:
                session = requests.Session()
                session.trust_env = False
                # Fetch resources
                r_resp = session.get(base_url.rstrip("/") + "/api/resources", timeout=5)
                # Fetch lock status
                l_resp = session.get(base_url.rstrip("/") + "/api/resource_lock/status", timeout=5)
                
                data = r_resp.json() if r_resp.ok else {"error_res": r_resp.status_code}
                data["lock"] = l_resp.json() if l_resp.ok else {"error_lock": l_resp.status_code}
                data["online"] = True
                return sid, data
            except Exception as e:
                return sid, {"error": str(e), "online": False}

        with ThreadPoolExecutor(max_workers=len(servers) or 1) as executor:
            futures = [executor.submit(fetch_one, s) for s in servers]
            for future in as_completed(futures):
                sid, data = future.result()
                results[sid] = data

        with self.lock:
            self.snapshots = results
            for sid, data in results.items():
                self.status[sid] = "online" if data.get("online") else "offline"

    def get_snapshots(self) -> Dict[str, Dict[str, Any]]:
        with self.lock:
            return self.snapshots.copy()

    def get_status(self, sid: str) -> str:
        with self.lock:
            return self.status.get(sid, "unknown")

    def stop(self):
        self._stop_event.set()

state_manager = ServerStateManager()
