from __future__ import annotations

import time
from typing import Dict, List


class JobStore:
    def __init__(self):
        self.logs: List[Dict] = []
        self.observer: List[Dict] = []
        self.state = {"status": "空闲", "last_action": None}

    def log(self, action: str, payload: Dict) -> None:
        self.logs.append({"action": action, "payload": payload, "time": time.strftime("%H:%M:%S")})
        if len(self.logs) > 200:
            self.logs = self.logs[-200:]

    def update_state(self, action: str) -> None:
        mapping = {"deploy": "已部署", "run": "运行中", "pause": "已暂停", "stop": "已终止"}
        self.state = {"status": mapping.get(action, self.state["status"]), "last_action": action}

    def add_observer(self, data: Dict) -> None:
        self.observer.append(data)
        if len(self.observer) > 500:
            self.observer = self.observer[-500:]
