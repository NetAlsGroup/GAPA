from __future__ import annotations

import io
from typing import Any, Dict


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"torch is required for fitness RPC: {exc}") from exc
    return torch


def dumps(obj: Dict[str, Any]) -> bytes:
    """Serialize payload for cross-server fitness RPC (torch.save)."""
    torch = _require_torch()
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def loads(payload: bytes) -> Dict[str, Any]:
    """Deserialize payload for cross-server fitness RPC (torch.load)."""
    torch = _require_torch()
    buf = io.BytesIO(payload)
    data = torch.load(buf, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError("fitness RPC payload must be a dict")
    return data

