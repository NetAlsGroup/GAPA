from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
import time

LOGGER = logging.getLogger(__name__)


@dataclass
class QueuedTask:
    task_id: str
    owner: str
    priority: int
    created_at: float
    payload: Dict[str, Any]
    retry_count: int = 0


class TaskQueueManager:
    """In-memory queue with priority and basic fairness/quota controls."""

    def __init__(
        self,
        max_total: int = 16,
        max_per_owner: int = 2,
        *,
        storage: Any = None,
        terminal_filter: Optional[Callable[[List[str]], List[str]]] = None,
    ) -> None:
        self.max_total = int(max_total)
        self.max_per_owner = int(max_per_owner)
        self._items: List[QueuedTask] = []
        self._last_dispatched_owner: Optional[str] = None
        self._storage = storage
        self._terminal_filter = terminal_filter
        self._recovery_metrics: Dict[str, int] = {
            "recovered": 0,
            "skipped_terminal": 0,
            "skipped_duplicate": 0,
        }
        self._persist_error_counts: Dict[str, int] = {"save": 0, "delete": 0, "recover": 0}
        self._persist_error_events: List[Dict[str, str]] = []
        self._recover_from_storage()

    def _record_persist_error(self, *, op: str, task_id: str, exc: Exception) -> None:
        if op in self._persist_error_counts:
            self._persist_error_counts[op] += 1
        event = {
            "task_id": str(task_id or ""),
            "op": str(op),
            "error_type": exc.__class__.__name__,
        }
        self._persist_error_events.append(event)
        if len(self._persist_error_events) > 32:
            self._persist_error_events = self._persist_error_events[-32:]
        LOGGER.error(
            "queue_persist_error op=%s task_id=%s error_type=%s detail=%s",
            event["op"],
            event["task_id"] or "-",
            event["error_type"],
            str(exc),
        )

    def _recover_from_storage(self) -> None:
        if self._storage is None:
            return
        try:
            records = self._storage.list_queue_tasks()
        except Exception as exc:
            self._record_persist_error(op="recover", task_id="", exc=exc)
            return
        if not isinstance(records, list):
            return
        all_ids = [str(r.get("task_id") or "") for r in records if isinstance(r, dict)]
        terminal = set(self._terminal_filter(all_ids) if self._terminal_filter else [])
        seen: set[str] = set()
        for rec in records:
            if not isinstance(rec, dict):
                continue
            task_id = str(rec.get("task_id") or "")
            if not task_id or task_id in terminal:
                if task_id in terminal:
                    self._recovery_metrics["skipped_terminal"] += 1
                try:
                    if self._storage is not None and task_id:
                        self._storage.delete_queue_task(task_id)
                except Exception as exc:
                    self._record_persist_error(op="recover", task_id=task_id, exc=exc)
                continue
            if task_id in seen:
                self._recovery_metrics["skipped_duplicate"] += 1
                try:
                    if self._storage is not None:
                        self._storage.delete_queue_task(task_id)
                except Exception as exc:
                    self._record_persist_error(op="recover", task_id=task_id, exc=exc)
                continue
            seen.add(task_id)
            self._items.append(
                QueuedTask(
                    task_id=task_id,
                    owner=str(rec.get("owner") or "anonymous"),
                    priority=int(rec.get("priority") or 0),
                    created_at=float(rec.get("created_at") or time.time()),
                    payload=dict(rec.get("payload") or {}),
                    retry_count=int(rec.get("retry_count") or 0),
                )
            )
            self._recovery_metrics["recovered"] += 1
        self._items.sort(key=lambda x: (-x.priority, x.created_at))

    def _persist_upsert(self, item: QueuedTask) -> None:
        if self._storage is None:
            return
        try:
            self._storage.save_queue_task(
                task_id=item.task_id,
                owner=item.owner,
                priority=item.priority,
                created_at=item.created_at,
                payload=item.payload,
                retry_count=item.retry_count,
            )
        except Exception as exc:
            self._record_persist_error(op="save", task_id=item.task_id, exc=exc)

    def _persist_delete(self, task_id: str) -> None:
        if self._storage is None:
            return
        try:
            self._storage.delete_queue_task(task_id)
        except Exception as exc:
            self._record_persist_error(op="delete", task_id=task_id, exc=exc)

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
                    "retry_count": item.retry_count,
                    "status": "queued",
                }
            )
        return out

    def enqueue(self, task_id: str, payload: Dict[str, Any], owner: str, priority: int) -> Tuple[bool, Dict[str, Any]]:
        owner = str(owner or "anonymous")
        priority = int(priority)
        existing = next((i for i in self._items if i.task_id == task_id), None)
        if existing is not None:
            existing.owner = owner
            existing.priority = priority
            existing.payload = dict(payload)
            self._items.sort(key=lambda x: (-x.priority, x.created_at))
            self._persist_upsert(existing)
            pos = next((idx for idx, x in enumerate(self._items, start=1) if x.task_id == task_id), len(self._items))
            return True, {
                "task_id": task_id,
                "owner": owner,
                "priority": priority,
                "position": pos,
                "queued": len(self._items),
                "status": "queued",
                "error_code": "",
                "deduped": True,
            }
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
        queued = next((i for i in self._items if i.task_id == task_id), None)
        if queued is not None:
            self._persist_upsert(queued)
        pos = next((idx for idx, x in enumerate(self._items, start=1) if x.task_id == task_id), len(self._items))
        return True, {"task_id": task_id, "owner": owner, "priority": priority, "position": pos, "queued": len(self._items), "status": "queued", "error_code": "", "deduped": False}

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
        self._persist_delete(pick.task_id)
        return pick

    def requeue(self, item: QueuedTask, *, reason: str = "") -> Tuple[bool, Dict[str, Any]]:
        item.retry_count = int(item.retry_count) + 1
        if len(self._items) >= self.max_total:
            return False, {"error": "queue full", "error_code": "QUEUE_FULL", "reason": reason}
        self._items.append(item)
        self._items.sort(key=lambda x: (-x.priority, x.created_at))
        self._persist_upsert(item)
        pos = next((idx for idx, x in enumerate(self._items, start=1) if x.task_id == item.task_id), len(self._items))
        return True, {"task_id": item.task_id, "position": pos, "retry_count": item.retry_count, "reason": reason, "status": "queued", "error_code": ""}

    def recovery_stats(self) -> Dict[str, int]:
        return dict(self._recovery_metrics)

    def persistence_observability(self) -> Dict[str, Any]:
        return {
            "error_counts": dict(self._persist_error_counts),
            "recent_errors": list(self._persist_error_events),
        }
