
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any


@dataclass
class JobSpec:
    """
    统一任务输入（可由 JSON 反序列化）。
    - objective: 'time' | 'energy' | 'edp'（可扩）
    - constraints: [{"name":"latency_ms","op":"<=","value":50}, ...]
    - search_space: 任意可序列化结构（交给 GA 解释）
    - evaluator: {"kind":"sixdst", ...}
    - budget: {"generations": 200, "parallelism": 32}
    - compute: {"gpu": 1, "mem_gb": 16, "timeout_s": 3600, "power_cap_w": null}
    """
    task_id: str
    objective: str
    constraints: List[Dict[str, Any]]
    search_space: Dict[str, Any]
    evaluator: Dict[str, Any]
    budget: Dict[str, Any]
    compute: Dict[str, Any]
    seed: int = 42
    resume_from: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Plan:
    """资源路由产出的标准方案（独立于 GA）。"""
    backend: str                 # 'cpu' | 'cuda' | 'multi-gpu'
    devices: List[int]
    allocation: Dict[int, float]
    world_size: int
    estimated_time_ms: float = 0.0
    estimated_energy_j: Optional[float] = None
    reason: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
