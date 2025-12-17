from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

import multiprocessing as mp


class TaskState:
    def __init__(self):
        self.lock = threading.Lock()
        self.task_id: Optional[str] = None
        self.state: str = "idle"  # idle | running | completed | error
        self.progress: int = 0
        self.logs: List[str] = []
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.process: Optional[mp.Process] = None
        self.queue: Any = None
        self.consumer: Optional[threading.Thread] = None

    def reset_for_new_task(self, task_id: str) -> None:
        self.task_id = task_id
        self.state = "running"
        self.progress = 0
        self.logs = []
        self.result = None
        self.error = None
        self.process = None
        self.queue = None
        self.consumer = None

    def append_log(self, line: str) -> None:
        self.logs.append(line)
        if len(self.logs) > 500:
            self.logs = self.logs[-500:]

    def set_progress(self, val: int) -> None:
        self.progress = max(0, min(100, int(val)))


def start_consumer(task: TaskState) -> None:
    """Drain child events into task state in a background thread."""

    def loop() -> None:
        q = task.queue
        proc = task.process
        while True:
            alive = proc.is_alive() if proc else False
            try:
                evt = q.get(timeout=0.3)
            except Exception:
                evt = None
            if evt:
                with task.lock:
                    if evt.get("type") == "log":
                        task.append_log(str(evt.get("line")))
                    elif evt.get("type") == "progress":
                        task.set_progress(int(evt.get("value") or 0))
                    elif evt.get("type") == "result":
                        task.result = evt.get("result")
                    elif evt.get("type") == "state":
                        task.state = evt.get("state") or task.state
                        task.error = evt.get("error")
            if not alive and (q.empty() if hasattr(q, "empty") else True):
                break

        with task.lock:
            if task.state == "running":
                task.state = "completed" if task.result else "idle"

    t = threading.Thread(target=loop, daemon=True)
    task.consumer = t
    t.start()

