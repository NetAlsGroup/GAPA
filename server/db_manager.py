import sqlite3
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading
import time

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "results" / "gapa.db"
HISTORY_JSON = BASE_DIR / "results" / "history.json"

class DBManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DBManager, cls).__new__(cls)
                cls._instance._initialized = False
                cls._instance._conn = None
                cls._instance._instance_pid = None
            return cls._instance

    def __init__(self):
        self._ensure_connection()

    def _ensure_connection(self):
        """Ensure we have a valid connection for the current process."""
        current_pid = os.getpid()
        if self._conn is not None and self._instance_pid == current_pid:
            return  # Connection is valid for this process
        
        # Close old connection if it exists (shouldn't be used across processes anyway)
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Use WAL mode and timeout for better concurrent access
        self._conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=30.0, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.row_factory = sqlite3.Row
        self._instance_pid = current_pid
        
        if not self._initialized:
            self._init_db()
            self._migrate_from_json()
            self._initialized = True

    @property
    def conn(self):
        """Get connection, ensuring it's valid for current process."""
        self._ensure_connection()
        return self._conn

    def _init_db(self):
        with self.conn:
            # History table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id TEXT PRIMARY KEY,
                    algorithm TEXT,
                    dataset TEXT,
                    server_id TEXT,
                    server_label TEXT,
                    state TEXT,
                    timestamp TEXT,
                    best_score REAL,
                    comm_avg_ms REAL,
                    result_json TEXT,
                    logs_json TEXT,
                    created_at REAL
                )
            """)
            
            # Migration: add logs_json column if it doesn't exist
            try:
                cursor = self.conn.execute("PRAGMA table_info(history)")
                columns = [row[1] for row in cursor.fetchall()]
                if "logs_json" not in columns:
                    self.conn.execute("ALTER TABLE history ADD COLUMN logs_json TEXT")
            except Exception:
                pass
            
            # GA State table for resuming
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS ga_state (
                    task_id TEXT PRIMARY KEY,
                    algorithm TEXT,
                    dataset TEXT,
                    run_id TEXT,
                    schema_version TEXT,
                    state_json TEXT,
                    updated_at REAL
                )
            """)
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS queue_tasks (
                    task_id TEXT PRIMARY KEY,
                    owner TEXT,
                    priority INTEGER,
                    created_at REAL,
                    payload_json TEXT,
                    retry_count INTEGER,
                    updated_at REAL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_terminal (
                    task_id TEXT PRIMARY KEY,
                    state TEXT,
                    updated_at REAL
                )
                """
            )
            try:
                cursor = self.conn.execute("PRAGMA table_info(ga_state)")
                columns = [row[1] for row in cursor.fetchall()]
                if "run_id" not in columns:
                    self.conn.execute("ALTER TABLE ga_state ADD COLUMN run_id TEXT")
                if "schema_version" not in columns:
                    self.conn.execute("ALTER TABLE ga_state ADD COLUMN schema_version TEXT")
            except Exception:
                pass

    def _migrate_from_json(self):
        if not HISTORY_JSON.exists():
            return
        
        try:
            with open(HISTORY_JSON, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    return
                
                for item in data:
                    self.add_history(item)
            
            # Backup and remove JSON after successful migration
            # backup_path = HISTORY_JSON.with_suffix(".json.bak")
            # HISTORY_JSON.rename(backup_path) 
            # (Keeping it for now to be safe, but ideally we remove it)
        except Exception as e:
            print(f"Migration from JSON failed: {e}")

    def add_history(self, item: Dict[str, Any]):
        hid = item.get("id") or str(int(time.time() * 1000))
        result_json = json.dumps(item.get("result") or {}, ensure_ascii=False)
        logs = item.get("logs") or []
        logs_json = json.dumps(logs, ensure_ascii=False) if logs else None
        
        with self.lock_for_write():
            self.conn.execute("""
                INSERT OR REPLACE INTO history (
                    id, algorithm, dataset, server_id, server_label, 
                    state, timestamp, best_score, comm_avg_ms, result_json, logs_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                hid,
                item.get("algorithm"),
                item.get("dataset"),
                item.get("server_id"),
                item.get("server_label"),
                item.get("state"),
                item.get("timestamp"),
                item.get("best_score"),
                item.get("comm_avg_ms"),
                result_json,
                logs_json,
                time.time()
            ))

    def get_history(self, limit: int = 100, offset: int = 0, order: str = "DESC") -> List[Dict[str, Any]]:
        # Validate order to prevent SQL injection
        if order.upper() not in ("ASC", "DESC"):
            order = "DESC"
        
        cursor = self.conn.execute(
            f"SELECT * FROM history ORDER BY created_at {order}, id {order} LIMIT ? OFFSET ?",
            (limit, offset)
        )
        rows = cursor.fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["result"] = json.loads(d.pop("result_json") or "{}")
            logs_json = d.pop("logs_json", None)
            d["logs"] = json.loads(logs_json) if logs_json else []
            results.append(d)
        return results

    def count_history(self) -> int:
        """Return total count of history entries for pagination."""
        cursor = self.conn.execute("SELECT COUNT(*) as cnt FROM history")
        row = cursor.fetchone()
        return row["cnt"] if row else 0

    def delete_history(self, ids: List[str]):
        if not ids: return
        with self.lock_for_write():
            placeholders = ",".join("?" * len(ids))
            self.conn.execute(f"DELETE FROM history WHERE id IN ({placeholders})", ids)

    def save_ga_state(
        self,
        task_id: str,
        algorithm: str,
        dataset: str,
        state: Dict[str, Any],
        *,
        run_id: Optional[str] = None,
        schema_version: str = "v1",
    ):
        state_json = json.dumps(state, ensure_ascii=False)
        with self.lock_for_write():
            self.conn.execute("""
                INSERT OR REPLACE INTO ga_state (task_id, algorithm, dataset, run_id, schema_version, state_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (task_id, algorithm, dataset, run_id or task_id, schema_version or "v1", state_json, time.time()))

    def get_ga_state(self, task_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute("SELECT state_json, run_id, schema_version FROM ga_state WHERE task_id = ?", (task_id,))
        row = cursor.fetchone()
        if not row:
            return None
        state = json.loads(row["state_json"])
        if isinstance(state, dict):
            state.setdefault("run_id", row["run_id"])
            state.setdefault("schema_version", row["schema_version"] or "v1")
        return state

    def get_latest_ga_state(self, algorithm: str, dataset: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            """
            SELECT task_id, state_json, run_id, schema_version
            FROM ga_state
            WHERE algorithm = ? AND dataset = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (algorithm, dataset),
        )
        row = cursor.fetchone()
        if not row:
            return None
        state = json.loads(row["state_json"])
        if isinstance(state, dict):
            state.setdefault("task_id", row["task_id"])
            state.setdefault("run_id", row["run_id"])
            state.setdefault("schema_version", row["schema_version"] or "v1")
        return state

    def lock_for_write(self):
        # sqlite3 handle its own locking but we can use threading.Lock for extra safety in multi-threaded Flask
        return self._lock

    def save_queue_task(
        self,
        *,
        task_id: str,
        owner: str,
        priority: int,
        created_at: float,
        payload: Dict[str, Any],
        retry_count: int = 0,
    ) -> None:
        payload_json = json.dumps(payload or {}, ensure_ascii=False)
        with self.lock_for_write():
            self.conn.execute(
                """
                INSERT OR REPLACE INTO queue_tasks (
                    task_id, owner, priority, created_at, payload_json, retry_count, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(task_id),
                    str(owner or "anonymous"),
                    int(priority),
                    float(created_at),
                    payload_json,
                    int(retry_count or 0),
                    time.time(),
                ),
            )

    def delete_queue_task(self, task_id: str) -> None:
        with self.lock_for_write():
            self.conn.execute("DELETE FROM queue_tasks WHERE task_id = ?", (str(task_id),))

    def list_queue_tasks(self) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            """
            SELECT task_id, owner, priority, created_at, payload_json, retry_count
            FROM queue_tasks
            ORDER BY priority DESC, created_at ASC
            """
        )
        rows = cursor.fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["payload_json"] or "{}")
            out.append(
                {
                    "task_id": row["task_id"],
                    "owner": row["owner"],
                    "priority": int(row["priority"] or 0),
                    "created_at": float(row["created_at"] or 0.0),
                    "payload": payload if isinstance(payload, dict) else {},
                    "retry_count": int(row["retry_count"] or 0),
                }
            )
        return out

    def save_task_terminal(self, *, task_id: str, state: str) -> None:
        with self.lock_for_write():
            self.conn.execute(
                """
                INSERT OR REPLACE INTO task_terminal (task_id, state, updated_at)
                VALUES (?, ?, ?)
                """,
                (str(task_id), str(state), time.time()),
            )

    def get_terminal_task_ids(self, task_ids: Optional[List[str]] = None) -> List[str]:
        if not task_ids:
            cursor = self.conn.execute("SELECT task_id FROM task_terminal")
            return [row["task_id"] for row in cursor.fetchall()]
        placeholders = ",".join(["?"] * len(task_ids))
        cursor = self.conn.execute(
            f"SELECT task_id FROM task_terminal WHERE task_id IN ({placeholders})",
            [str(t) for t in task_ids],
        )
        return [row["task_id"] for row in cursor.fetchall()]

db_manager = DBManager()
