# -*- coding: utf-8 -*-
"""
自适应资源规划与路由模块。
可单独调用以获取本机性能画像、生成资源方案、并在 GA 运行前自动挑选最优加速配置。
"""

from .api.schemas import JobSpec, Plan
from .api.planner import StrategyCompare, StrategyPlan, DistributedStrategyPlan
from .route.router_adapter import route_plan
from .route.profiler_adapter import get_profile
from .exec.executor import apply_plan_env, pick_mode, world_size
from .data.loader_factory import build_loader

__all__ = [
    "JobSpec",
    "Plan",
    "StrategyPlan",
    "StrategyCompare",
    "DistributedStrategyPlan",
    "route_plan",
    "get_profile",
    "apply_plan_env",
    "pick_mode",
    "world_size",
    "build_loader",
]
