from __future__ import annotations
from typing import Callable, Optional, Tuple, Dict, Any

import time
from .schemas import Plan
from ..route.perf_profiler import PerformanceProfiler
from ..route.strategy_router import StrategyRouter
from ..exec.executor import apply_plan_env


def StrategyPlan(
        fitness: Optional[Callable[..., None]] = None,
        warmup: int = 0,
        objective: str = 'time',
        multi_gpu: bool = True,
        fitness_args: Optional[Tuple[Any, ...]] = None,
        fitness_kwargs: Optional[Dict[str, Any]] = None,
) -> Plan:
    """生成当前环境的资源执行方案。

    - 若未提供 ``fitness``，仅基于静态性能画像选择方案；
    - 若提供 ``fitness`` 且 ``warmup`` > 0，则在候选方案上执行指定次数的
      ``fitness`` 调用以做热身评估，并返回实测耗时最短的方案。
      可以通过 ``fitness_args`` 与 ``fitness_kwargs`` 为 ``fitness`` 传参。
    """
    prof = PerformanceProfiler(quick=True).profile()
    router = StrategyRouter(prof)

    class _SyntheticWL:
        """Synthetic workload used when the caller does not supply one.

        The previous implementation used a 1-edge dummy graph which made the
        router heavily favour the CPU due to fixed kernel launch overheads. In
        practice, graph workloads are orders of magnitude larger, so we model a
        moderate-size graph here to let the performance model reflect the
        relative throughput of CPU vs GPU correctly.
        """

        # A modest graph: 50k nodes, 5M edges and 50 steps is sufficient to
        # reveal GPU advantages without being unrealistically large.  The values
        # only affect the static estimation and incur no runtime cost.
        n_nodes = 50_000
        n_edges = 5_000_000
        steps = 50
        batch_individuals = 1

    wl = _SyntheticWL()

    if fitness is not None and warmup > 0:
        args = fitness_args or ()
        kwargs = fitness_kwargs or {}

        def executor(p: Plan, iters: int) -> float:
            apply_plan_env(p)
            start = time.perf_counter()
            for _ in range(iters):
                fitness(*args, **kwargs)
            return (time.perf_counter() - start) * 1000.0

        plan = router.choose_and_warmup(wl, executor,
                                        objective=objective,
                                        multi_gpu=multi_gpu,
                                        warmup_iters=warmup)
    else:
        plan = router.route(wl, objective=objective, multi_gpu=multi_gpu)

    return plan
