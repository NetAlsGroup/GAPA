
# -*- coding: utf-8 -*-
"""落实资源方案（环境变量/并行度），与 GA 解耦。"""
from __future__ import annotations
import os
from ..api.schemas import Plan

def apply_plan_env(plan: Plan) -> None:
    """根据方案设置运行环境（只负责资源可见性，不起进程）"""
    if plan.backend == "cpu":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    elif plan.backend == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in plan.devices[:1])
    elif plan.backend == "multi-gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in plan.devices)
    else:
        raise ValueError(f"未知 backend: {plan.backend}")

def pick_mode(plan: Plan) -> str:
    """把方案映射到 SixDST 的 mode 字段：'s' 单进程，'m' 多进程。"""
    return "m" if plan.backend == "multi-gpu" else "s"

def world_size(plan: Plan) -> int:
    return plan.world_size if plan.backend == "multi-gpu" else 1
