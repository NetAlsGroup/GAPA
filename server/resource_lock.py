from __future__ import annotations

import threading
import time
from dataclasses import dataclass
import gc
import math
import os
from typing import Any, Dict, List, Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _require_torch():
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for resource lock")
    return torch


@dataclass
class LockInfo:
    active: bool
    backend: str
    devices: List[int]
    expires_at: float
    duration_s: float
    mem_mb: int
    warmup_iters: int
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active": self.active,
            "backend": self.backend,
            "devices": self.devices,
            "expires_at": self.expires_at,
            "duration_s": self.duration_s,
            "mem_mb": self.mem_mb,
            "warmup_iters": self.warmup_iters,
            "note": self.note,
        }


class ResourceLockManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._info = LockInfo(
            active=False,
            backend="cpu",
            devices=[],
            expires_at=0.0,
            duration_s=0.0,
            mem_mb=0,
            warmup_iters=0,
        )
        self._hold: List[Any] = []
        self._timer: Optional[threading.Thread] = None

    def _expired(self) -> bool:
        return bool(self._info.active and time.time() >= self._info.expires_at)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            if self._expired():
                self._release_locked("expired")
            return self._info.to_dict()

    def release(self, reason: str = "manual") -> Dict[str, Any]:
        with self._lock:
            self._release_locked(reason)
            return self._info.to_dict()

    def lock(
        self,
        *,
        duration_s: float = 600.0,
        warmup_iters: int = 2,
        mem_mb: int = 1024,
        devices: Optional[List[int]] = None,
        strict_idle: bool = False,
    ) -> Dict[str, Any]:
        torch = _require_torch()
        duration_s = max(10.0, float(duration_s))
        warmup_iters = max(0, int(warmup_iters))
        mem_mb = max(0, int(mem_mb))

        with self._lock:
            if self._expired():
                self._release_locked("expired")
            if self._info.active:
                return self._info.to_dict()

            if devices is not None:
                backend, devices = self._pick_backend_manual(devices, strict_idle=bool(strict_idle))
            else:
                backend, devices = self._pick_backend()
            if backend == "skip":
                self._info = LockInfo(
                    active=False,
                    backend="cuda",
                    devices=[],
                    expires_at=0.0,
                    duration_s=0.0,
                    mem_mb=mem_mb,
                    warmup_iters=warmup_iters,
                    note="no_available_gpu",
                )
                return self._info.to_dict()
            self._hold = []
            self._warmup(backend, devices, warmup_iters)
            self._reserve_memory(backend, devices, mem_mb)
            now = time.time()
            self._info = LockInfo(
                active=True,
                backend=backend,
                devices=devices,
                expires_at=now + duration_s,
                duration_s=duration_s,
                mem_mb=mem_mb,
                warmup_iters=warmup_iters,
                note="locked",
            )
            self._timer = threading.Thread(target=self._auto_release, daemon=True)
            self._timer.start()
            return self._info.to_dict()

    def _auto_release(self) -> None:
        while True:
            time.sleep(0.5)
            with self._lock:
                if not self._info.active:
                    return
                if self._expired():
                    self._release_locked("expired")
                    return

    def _release_locked(self, reason: str) -> None:
        if not self._info.active:
            self._info.note = reason
            return
        devices = list(self._info.devices or [])
        self._hold = []
        try:
            gc.collect()
        except Exception:
            pass
        if torch is not None and torch.cuda.is_available():
            try:
                if devices:
                    for d in devices:
                        try:
                            torch.cuda.set_device(int(d))
                        except Exception:
                            continue
                        try:
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                        except Exception:
                            pass
                else:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
        self._info.active = False
        self._info.note = reason
        self._info.devices = []
        self._info.backend = "cpu"
        self._info.expires_at = 0.0
        self._info.duration_s = 0.0

    def _pick_backend(self) -> tuple[str, List[int]]:
        try:
            from server import current_resource_snapshot  # type: ignore
        except Exception:
            current_resource_snapshot = None
        gpu_busy_threshold = float(os.getenv("GAPA_LOCK_GPU_BUSY", "65") or 65)
        min_gpu_free_mb = int(os.getenv("GAPA_LOCK_MIN_FREE_MB", "1024") or 1024)
        if current_resource_snapshot is not None:
            try:
                snap = current_resource_snapshot()
                gpus = snap.get("gpus") or []
                eligible = []
                for g in gpus:
                    try:
                        gid = int(g.get("id"))
                    except Exception:
                        continue
                    util = g.get("gpu_util_percent")
                    free_mb = g.get("free_mb")
                    util_val = float(util) if util is not None else 0.0
                    free_val = float(free_mb) if free_mb is not None else 0.0
                    if util_val <= gpu_busy_threshold and free_val >= min_gpu_free_mb:
                        eligible.append((gid, free_val, util_val))
                if eligible:
                    eligible.sort(key=lambda x: (x[1], -x[2]), reverse=True)
                    devices = [gid for gid, *_ in eligible]
                    backend = "cuda" if len(devices) == 1 else "multi-gpu"
                    return backend, devices
                if torch and torch.cuda.is_available():
                    return "skip", []
            except Exception:
                pass
        try:
            from autoadapt import StrategyPlan  # type: ignore
        except Exception:
            StrategyPlan = None
        if StrategyPlan is None:
            return ("cuda", [0]) if (torch and torch.cuda.is_available()) else ("cpu", [])
        plan = StrategyPlan(fitness=None, warmup=0, objective="time", multi_gpu=True)
        backend = str(getattr(plan, "backend", "cpu"))
        devices = list(getattr(plan, "devices", []) or [])
        if backend == "multi-gpu":
            if not devices:
                try:
                    devices = list(range(torch.cuda.device_count())) if torch and torch.cuda.is_available() else []
                except Exception:
                    devices = []
        elif backend == "cuda":
            if not devices:
                devices = [0]
        else:
            devices = []
        return backend, devices

    def _pick_backend_manual(self, devices: List[int], *, strict_idle: bool = False) -> tuple[str, List[int]]:
        if not devices:
            return "cpu", []
        unique = []
        for d in devices:
            try:
                idx = int(d)
            except Exception:
                continue
            if idx not in unique:
                unique.append(idx)
        if strict_idle:
            unique = self._filter_idle_devices(unique)
            if not unique:
                return "skip", []
        if len(unique) == 1:
            return "cuda", unique
        return "multi-gpu", unique

    def _filter_idle_devices(self, devices: List[int]) -> List[int]:
        try:
            from server import current_resource_snapshot  # type: ignore
        except Exception:
            current_resource_snapshot = None
        if current_resource_snapshot is None:
            return devices
        try:
            snap = current_resource_snapshot()
            gpus = snap.get("gpus") or []
        except Exception:
            return devices
        gpu_busy_threshold = float(os.getenv("GAPA_LOCK_GPU_BUSY", "85") or 85)
        min_gpu_free_mb = int(os.getenv("GAPA_LOCK_MIN_FREE_MB", "1024") or 1024)
        keep = []
        for dev in devices:
            match = next((g for g in gpus if int(g.get("id", -1)) == int(dev)), None)
            if not match:
                continue
            util = match.get("gpu_util_percent")
            free_mb = match.get("free_mb")
            util_val = float(util) if util is not None else 0.0
            free_val = float(free_mb) if free_mb is not None else 0.0
            if util_val <= gpu_busy_threshold and free_val >= min_gpu_free_mb:
                keep.append(int(dev))
        return keep

    def _warmup(self, backend: str, devices: List[int], iters: int) -> None:
        torch = _require_torch()
        if iters <= 0:
            return

        def _run(device: str) -> None:
            n = 256 if device == "cpu" else 512
            a = torch.randn((n, n), device=device)
            b = torch.randn((n, n), device=device)
            for _ in range(iters):
                y = a @ b
                y = y.relu_()
                _ = float(y[0, 0].item())
            if device.startswith("cuda"):
                torch.cuda.synchronize()

        if backend == "cpu":
            _run("cpu")
            return

        for d in devices:
            _run(f"cuda:{d}")

    def _reserve_memory(self, backend: str, devices: List[int], mem_mb: int) -> None:
        torch = _require_torch()
        if mem_mb <= 0:
            return

        if backend == "cpu":
            numel = int((mem_mb * 1024 * 1024) / 4)
            if numel <= 0:
                return
            self._hold.append(torch.empty((numel,), dtype=torch.float32, device="cpu"))
            return

        if not devices:
            return
        per_mb = int(math.ceil(mem_mb / max(1, len(devices))))
        numel = int((per_mb * 1024 * 1024) / 4)
        if numel <= 0:
            return
        for d in devices:
            self._hold.append(torch.empty((numel,), dtype=torch.float32, device=f"cuda:{d}"))


LOCK_MANAGER = ResourceLockManager()
