from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import time


@dataclass
class QueuedTask:
    task_id: str
    owner: str
    priority: int
    created_at: float
    payload: Dict[str, Any]


class TaskQueueManager:
    """In-memory queue with priority and basic fairness/quota controls."""

    def __init__(self, max_total: int = 16, max_per_owner: int = 2) -> None:
        self.max_total = int(max_total)
        self.max_per_owner = int(max_per_owner)
        self._items: List[QueuedTask] = []
        self._last_dispatched_owner: Optional[str] = None

    def size(self) -> int:
        return len(self._items)

    def list_items(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for idx, item in enumerate(self._items, start=1):
            out.append(
                {
                    "task_id": item.task_id,
                    "owner": item.owner,
                    "priority": item.priority,
                    "created_at": item.created_at,
                    "position": idx,
                }
            )
        return out

    def enqueue(self, task_id: str, payload: Dict[str, Any], owner: str, priority: int) -> Tuple[bool, Dict[str, Any]]:
        owner = str(owner or "anonymous")
        priority = int(priority)
        if len(self._items) >= self.max_total:
            return False, {"error": "queue full", "error_code": "QUEUE_FULL", "max_total": self.max_total}
        owner_count = sum(1 for i in self._items if i.owner == owner)
        if owner_count >= self.max_per_owner:
            return False, {
                "error": "owner queue quota exceeded",
                "error_code": "OWNER_QUOTA_EXCEEDED",
                "owner": owner,
                "max_per_owner": self.max_per_owner,
            }
        self._items.append(
            QueuedTask(
                task_id=task_id,
                owner=owner,
                priority=priority,
                created_at=time.time(),
                payload=dict(payload),
            )
        )
        self._items.sort(key=lambda x: (-x.priority, x.created_at))
        pos = next((idx for idx, x in enumerate(self._items, start=1) if x.task_id == task_id), len(self._items))
        return True, {"task_id": task_id, "owner": owner, "priority": priority, "position": pos, "queued": len(self._items)}

    def pop_next(self) -> Optional[QueuedTask]:
        if not self._items:
            return None
        top_priority = self._items[0].priority
        candidates = [i for i in self._items if i.priority == top_priority]
        pick = candidates[0]
        if self._last_dispatched_owner is not None:
            alt = next((i for i in candidates if i.owner != self._last_dispatched_owner), None)
            if alt is not None:
                pick = alt
        self._items = [i for i in self._items if i.task_id != pick.task_id]
        self._last_dispatched_owner = pick.owner
        return pick
