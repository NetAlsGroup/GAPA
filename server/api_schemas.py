"""
GAPA API Schemas - 标准化请求/响应模型

基于 REST API 设计最佳实践，提供统一的数据结构定义。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Any, Dict
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    """Task execution status enum."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class ErrorDetail:
    """Individual error detail for field-level validation errors."""
    message: str
    field: Optional[str] = None
    code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ErrorResponse:
    """Standardized error response structure.
    
    Follows REST best practices:
    - error: Error type identifier (e.g., "NotFound", "ValidationError")
    - message: Human-readable error message
    - details: Optional list of field-level errors
    - timestamp: ISO8601 timestamp of when error occurred
    - path: Request path that triggered the error
    """
    error: str
    message: str
    path: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    details: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "error": self.error,
            "message": self.message,
            "timestamp": self.timestamp,
            "path": self.path,
        }
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class PaginatedResponse:
    """Standard paginated list response.
    
    Attributes:
        items: List of items for current page
        total: Total number of items across all pages
        page: Current page number (1-indexed)
        page_size: Number of items per page
        pages: Total number of pages
    """
    items: List[Any]
    total: int
    page: int
    page_size: int

    @property
    def pages(self) -> int:
        """Calculate total pages."""
        if self.total <= 0:
            return 0
        return (self.total + self.page_size - 1) // self.page_size

    @property
    def has_next(self) -> bool:
        return self.page < self.pages

    @property
    def has_prev(self) -> bool:
        return self.page > 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": self.items,
            "total": self.total,
            "page": self.page,
            "page_size": self.page_size,
            "pages": self.pages,
            "has_next": self.has_next,
            "has_prev": self.has_prev,
        }


# HTTP Status Code Constants
class HTTPStatus:
    """Standard HTTP status codes used in GAPA API."""
    OK = 200
    CREATED = 201
    NO_CONTENT = 204
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    CONFLICT = 409
    UNPROCESSABLE_ENTITY = 422
    TOO_MANY_REQUESTS = 429
    INTERNAL_SERVER_ERROR = 500
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503


def make_error_response(
    error_type: str,
    message: str,
    path: str = "",
    details: Optional[List[ErrorDetail]] = None,
) -> Dict[str, Any]:
    """Helper to create standardized error response dict."""
    return ErrorResponse(
        error=error_type,
        message=message,
        path=path,
        details=[d.to_dict() for d in details] if details else None,
    ).to_dict()


def make_paginated_response(
    items: List[Any],
    total: int,
    page: int = 1,
    page_size: int = 20,
) -> Dict[str, Any]:
    """Helper to create standardized paginated response dict."""
    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    ).to_dict()


DEFAULT_SCHEMA_VERSION = "v1"
CURRENT_SCHEMA_VERSION = "v2"


def resolve_schema_version(payload: Optional[Dict[str, Any]] = None) -> str:
    if not isinstance(payload, dict):
        return DEFAULT_SCHEMA_VERSION
    raw = str(payload.get("schema_version") or DEFAULT_SCHEMA_VERSION).strip().lower()
    if raw in ("v1", "1", "1.0"):
        return "v1"
    if raw in ("v2", "2", "2.0"):
        return "v2"
    return DEFAULT_SCHEMA_VERSION


def mode_decision_code(reason: str, degraded: bool) -> str:
    r = str(reason or "").lower()
    if not degraded:
        return ""
    if "unsupported_mode" in r:
        return "MODE_UNSUPPORTED"
    if "fallback" in r or "->" in r:
        return "MODE_FALLBACK"
    if "busy" in r:
        return "MODE_RESOURCE_BUSY"
    return "MODE_DEGRADED"


def normalize_mode_decision(mode_decision: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not mode_decision:
        return {}
    md = dict(mode_decision or {})
    requested = str(md.get("requested_mode") or md.get("mode") or "S").upper()
    selected = str(md.get("selected_mode") or md.get("mode") or requested).upper()
    degraded = bool(md.get("degraded")) or requested != selected
    reason = str(md.get("reason") or "").strip()
    code = str(md.get("code") or mode_decision_code(reason, degraded))
    devices = md.get("devices")
    if not isinstance(devices, list):
        devices = [] if devices is None else [devices]
    out = {
        "requested_mode": requested,
        "selected_mode": selected,
        "degraded": degraded,
        "reason": reason,
        "code": code,
        "target": md.get("target") or "local",
        "devices": devices,
    }
    if md.get("capability") is not None:
        out["capability"] = md.get("capability")
    if md.get("use_strategy_plan") is not None:
        out["use_strategy_plan"] = bool(md.get("use_strategy_plan"))
    return out


def build_resume_metadata(
    *,
    run_id: str,
    task_id: str,
    schema_version: str,
    mode_plan_snapshot: Optional[Dict[str, Any]] = None,
    checkpoint_ref: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "run_id": str(run_id),
        "task_id": str(task_id),
        "schema_version": str(schema_version or DEFAULT_SCHEMA_VERSION),
        "mode_plan_snapshot": mode_plan_snapshot or {},
        "checkpoint_ref": checkpoint_ref or "",
    }
