from __future__ import annotations

import io
import os
from typing import Any, Dict, Optional, Tuple

import zlib


_MAGIC = b"GAPA"
_VERSION = 1
_FLAG_ZLIB = 1 << 0


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"torch is required for fitness RPC: {exc}") from exc
    return torch


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except Exception:
        return float(default)


def _serialize(torch: Any, obj: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    # Legacy torch serialization is often faster for small tensor-heavy RPC payloads.
    try:
        use_legacy = bool(_env_int("GAPA_RPC_TORCH_LEGACY_SERIALIZATION", 1))
        if use_legacy:
            torch.save(obj, buf, _use_new_zipfile_serialization=False)
        else:
            torch.save(obj, buf)
    except TypeError:
        torch.save(obj, buf)
    return buf.getvalue()


def _pack(payload: bytes, *, compressed: bool) -> bytes:
    flags = _FLAG_ZLIB if compressed else 0
    return _MAGIC + bytes([_VERSION, flags]) + payload


def _unpack(payload: bytes) -> Optional[Tuple[bytes, bool]]:
    if len(payload) < 6 or payload[:4] != _MAGIC:
        return None
    version = payload[4]
    if version != _VERSION:
        raise ValueError(f"unsupported fitness RPC payload version: {version}")
    flags = payload[5]
    return payload[6:], bool(flags & _FLAG_ZLIB)


def dumps(obj: Dict[str, Any]) -> bytes:
    """Serialize payload for cross-server fitness RPC.

    Format:
    - v1 framed payload: b"GAPA" + version + flags + body
    - body optionally zlib-compressed
    """
    torch = _require_torch()
    raw = _serialize(torch, obj)
    min_bytes = max(0, _env_int("GAPA_RPC_COMPRESS_MIN_BYTES", 2048))
    min_saving = max(0.0, _env_float("GAPA_RPC_COMPRESS_MIN_SAVING", 0.05))
    zlib_level = max(0, min(9, _env_int("GAPA_RPC_ZLIB_LEVEL", 1)))

    if len(raw) < min_bytes:
        return _pack(raw, compressed=False)

    compressed = zlib.compress(raw, level=zlib_level)
    if len(raw) <= 0:
        return _pack(raw, compressed=False)
    saving_ratio = (len(raw) - len(compressed)) / float(len(raw))
    if saving_ratio >= min_saving:
        return _pack(compressed, compressed=True)
    return _pack(raw, compressed=False)


def loads(payload: bytes) -> Dict[str, Any]:
    """Deserialize payload for cross-server fitness RPC.

    Backward compatibility:
    - framed payloads (current)
    - old payloads that were always zlib-compressed
    - old payloads that were plain torch-save bytes
    """
    torch = _require_torch()

    framed = _unpack(payload)
    if framed is not None:
        body, is_compressed = framed
        if is_compressed:
            body = zlib.decompress(body)
        buf = io.BytesIO(body)
    else:
        try:
            body = zlib.decompress(payload)
            buf = io.BytesIO(body)
        except Exception:
            buf = io.BytesIO(payload)

    data = torch.load(buf, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError("fitness RPC payload must be a dict")
    return data
