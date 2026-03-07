from __future__ import annotations
from typing import Optional
from ..api.schemas import Plan
from .strategy_router import StrategyRouter
from .perf_profiler import PerformanceProfiler
def route_plan(workload, objective='time', multi_gpu=True, power_cap_w: Optional[float]=None, warmup=False, warmup_iters=2):
    prof = PerformanceProfiler(quick=True).profile()
    router = StrategyRouter(prof)
    if hasattr(router, 'route') and not warmup:
        plan = router.route(workload, objective=objective,
                            multi_gpu=multi_gpu, power_cap_w=power_cap_w)
    else:
        def _noop(p, iters=warmup_iters):
            return getattr(p, 'estimated_time_ms', 1.0)

        plan = router.choose_and_warmup(workload, executor=_noop,
                                        objective=objective,
                                        multi_gpu=multi_gpu,
                                        power_cap_w=power_cap_w,
                                        warmup_iters=warmup_iters)

    if isinstance(plan, Plan):
        return plan
    if isinstance(plan, dict):
        return Plan(**plan)
    return Plan(
        backend=getattr(plan, 'backend'),
        devices=list(getattr(plan, 'devices')),
        allocation=getattr(plan, 'allocation', {}),
        world_size=int(getattr(plan, 'world_size', 1)),
        estimated_time_ms=float(getattr(plan, 'estimated_time_ms', 0.0)),
        estimated_energy_j=getattr(plan, 'est_energy_j', None),
        reason=getattr(plan, 'reason', ''),
        notes=getattr(plan, 'notes', '')
    )
