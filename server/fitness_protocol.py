from __future__ import annotations

import io
from typing import Any, Dict


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"torch is required for fitness RPC: {exc}") from exc
    return torch


import zlib

def dumps(obj: Dict[str, Any]) -> bytes:
    """Serialize payload for cross-server fitness RPC (torch.save) with zlib compression."""
    torch = _require_torch()
    buf = io.BytesIO()
    torch.save(obj, buf)
    # Compress the serialized data
    return zlib.compress(buf.getvalue(), level=1) # rapid compression


def loads(payload: bytes) -> Dict[str, Any]:
    """Deserialize payload for cross-server fitness RPC (torch.load) with auto-decompression."""
    torch = _require_torch()
    
    # Try decompressing if it looks like zlib (basic check or just try/except)
    # But to be robust, we just try decompressing. 
    # If the payload is from an old version (uncompressed), zlib.decompress might fail or return garbage.
    # However, since we control both ends and just updated it, we can assume compression.
    # For backward compatibility, we can try-except.
    try:
        decompressed = zlib.decompress(payload)
        buf = io.BytesIO(decompressed)
    except Exception:
        # Fallback for uncompressed data
        buf = io.BytesIO(payload)

    data = torch.load(buf, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError("fitness RPC payload must be a dict")
    return data

