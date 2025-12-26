from __future__ import annotations

import json
from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, Optional

import torch.distributed as dist
try:
    import torch
except Exception:  # pragma: no cover
    torch = None
import re
import socket


@dataclass
class CommTimer:
    total_ms: float = 0.0
    calls: int = 0
    per_op_ms: Dict[str, float] = field(default_factory=dict)
    per_op_calls: Dict[str, int] = field(default_factory=dict)

    def add(self, elapsed_s: float, op: Optional[str] = None) -> None:
        ms = elapsed_s * 1000.0
        self.total_ms += ms
        self.calls += 1
        if op:
            self.per_op_ms[op] = self.per_op_ms.get(op, 0.0) + ms
            self.per_op_calls[op] = self.per_op_calls.get(op, 0) + 1

    def stats(self) -> Dict[str, float | int | Dict[str, float] | Dict[str, int]]:
        avg_ms = self.total_ms / self.calls if self.calls else 0.0
        return {
            "total_ms": float(self.total_ms),
            "calls": int(self.calls),
            "avg_ms": float(avg_ms),
            "per_op_ms": {k: float(v) for k, v in self.per_op_ms.items()},
            "per_op_calls": {k: int(v) for k, v in self.per_op_calls.items()},
        }


def timed_call(timer: CommTimer, op: str, fn, *args, **kwargs):
    start = perf_counter()
    out = fn(*args, **kwargs)
    timer.add(perf_counter() - start, op=op)
    return out


def finalize_comm_stats(timer: CommTimer, rank: int, world_size: int, comm_path: Optional[str]) -> None:
    if not comm_path:
        return
    def _short_gpu_name(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        text = name.upper()
        mem_match = re.search(r"(\d+)\s?GB", text)
        mem = mem_match.group(1) if mem_match else None

        model_match = re.search(
            r"(A100|H100|V100|P100|T4|A10|A40|L40S|L40|A6000|RTX\s?A?\d{3,4})",
            text,
        )
        if model_match:
            model = model_match.group(1).replace(" ", "")
            return f"{model}-{mem}GB" if mem else model
        return name

    gpu_id = None
    gpu_name = None
    gpu_name_short = None
    if torch is not None:
        try:
            if torch.cuda.is_available():
                gpu_id = int(torch.cuda.current_device())
                gpu_name = torch.cuda.get_device_name(gpu_id)
                gpu_name_short = _short_gpu_name(gpu_name)
        except Exception:
            gpu_id = None
            gpu_name = None
            gpu_name_short = None

    local = {
        "rank": int(rank),
        "host": socket.gethostname(),
        "gpu": gpu_id,
        "gpu_name": gpu_name,
        "gpu_name_short": gpu_name_short,
        **timer.stats(),
    }
    gathered = [None for _ in range(world_size)] if rank == 0 else None
    try:
        dist.gather_object(local, gathered, dst=0)
    except Exception:
        return
    if rank != 0 or not gathered:
        return

    per_rank_ms = {}
    per_rank_avg = {}
    per_rank_calls = {}
    per_rank_ops = {}
    per_rank_meta = {}
    for item in gathered:
        if not item:
            continue
        r = int(item.get("rank", -1))
        per_rank_ms[str(r)] = float(item.get("total_ms", 0.0))
        per_rank_avg[str(r)] = float(item.get("avg_ms", 0.0))
        per_rank_calls[str(r)] = int(item.get("calls", 0))
        per_rank_ops[str(r)] = item.get("per_op_ms", {})
        per_rank_meta[str(r)] = {
            "host": item.get("host"),
            "gpu": item.get("gpu"),
            "gpu_name": item.get("gpu_name"),
            "gpu_name_short": item.get("gpu_name_short"),
        }

    if world_size > 1:
        non_zero = [v for k, v in per_rank_avg.items() if k != "0"]
        avg_ms = sum(non_zero) / len(non_zero) if non_zero else 0.0
    else:
        avg_ms = per_rank_avg.get("0", 0.0)

    payload = {
        "type": "process",
        "world_size": int(world_size),
        "avg_ms": float(avg_ms),
        "per_rank_ms": per_rank_ms,
        "per_rank_avg_ms": per_rank_avg,
        "per_rank_calls": per_rank_calls,
        "per_rank_ops": per_rank_ops,
        "per_rank_meta": per_rank_meta,
    }
    try:
        with open(comm_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        pass
