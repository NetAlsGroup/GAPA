#!/usr/bin/env python3
"""
Run built-in GAPA algorithms from a single script.

Examples:
  python examples/run_builtin_algorithms.py --method CutOff --dataset Circuit --mode s
  python examples/run_builtin_algorithms.py --method TDE --generations 20 --pop-size 40
  python examples/run_builtin_algorithms.py --method CGN --dataset karate --mode s
  python examples/run_builtin_algorithms.py --all --generations 10
  python examples/run_builtin_algorithms.py --method SixDST --mode m --use-strategy-plan
  python examples/run_builtin_algorithms.py --method SixDST --mode m --use-strategy-plan --plan-server "Server 6"
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path
from time import perf_counter
from time import sleep
from typing import Any, Callable, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gapa import DataLoader  # noqa: E402
from gapa.workflow import Monitor  # noqa: E402
from gapa.remote_runner import run_remote_task  # noqa: E402
from gapa.utils.init_device import init_device  # noqa: E402
from gapa.utils.functions import set_seed  # noqa: E402

from gapa.algorithm.CND.Cutoff import CutoffEvaluator, CutoffController, Cutoff  # noqa: E402
from gapa.algorithm.CND.TDE import TDEEvaluator, TDEController, TDE  # noqa: E402
from gapa.algorithm.CND.SixDST import SixDSTEvaluator, SixDSTController, SixDST  # noqa: E402

from gapa.algorithm.CDA.CGN import CGNEvaluator, CGNController, CGN  # noqa: E402
from gapa.algorithm.CDA.QAttack import QAttackEvaluator, QAttackController, QAttack  # noqa: E402
from gapa.algorithm.CDA.EDA import EDAEvaluator, EDAController, EDA  # noqa: E402


METHODS = ["CutOff", "TDE", "SixDST", "CGN", "QAttack", "CDA-EDA"]
DEFAULT_DATASET = {
    "CutOff": "Circuit",
    "TDE": "Circuit",
    "SixDST": "Circuit",
    "CGN": "karate",
    "QAttack": "karate",
    "CDA-EDA": "karate",
}


def _result_path(method: str, mode: str) -> str:
    p = PROJECT_ROOT / "results" / "tests" / method / mode.upper()
    p.mkdir(parents=True, exist_ok=True)
    return str(p) + "/"


def _run_cutoff(args, data_loader, device, world_size) -> None:
    evaluator = CutoffEvaluator(
        pop_size=args.pop_size,
        graph=data_loader.G,
        nodes=data_loader.nodes,
        device=device,
    )
    controller = CutoffController(
        path=_result_path("CutOff", args.mode),
        pattern="write",
        cutoff_tag=args.cutoff_tag,
        data_loader=data_loader,
        loops=1,
        crossover_rate=args.pc,
        mutate_rate=args.pm,
        pop_size=args.pop_size,
        device=device,
    )
    Cutoff(
        mode=args.mode,
        max_generation=args.generations,
        data_loader=data_loader,
        controller=controller,
        evaluator=evaluator,
        world_size=world_size,
        verbose=True,
    )


def _run_tde(args, data_loader, device, world_size) -> None:
    evaluator = TDEEvaluator(
        pop_size=args.pop_size,
        graph=data_loader.G,
        budget=data_loader.k,
        device=device,
    )
    controller = TDEController(
        path=_result_path("TDE", args.mode),
        pattern="write",
        data_loader=data_loader,
        loops=1,
        crossover_rate=args.pc,
        mutate_rate=args.pm,
        pop_size=args.pop_size,
        device=device,
    )
    TDE(
        mode=args.mode,
        max_generation=args.generations,
        data_loader=data_loader,
        controller=controller,
        evaluator=evaluator,
        world_size=world_size,
        verbose=True,
    )


def _run_sixdst(args, data_loader, device, world_size) -> None:
    evaluator = SixDSTEvaluator(
        pop_size=args.pop_size,
        adj=data_loader.A,
        device=device,
    )
    controller = SixDSTController(
        path=_result_path("SixDST", args.mode),
        pattern="write",
        cutoff_tag=args.cutoff_tag,
        data_loader=data_loader,
        loops=1,
        crossover_rate=args.pc,
        mutate_rate=args.pm,
        pop_size=args.pop_size,
        device=device,
    )
    SixDST(
        mode=args.mode,
        max_generation=args.generations,
        data_loader=data_loader,
        controller=controller,
        evaluator=evaluator,
        world_size=world_size,
        verbose=True,
    )


def _run_cgn(args, data_loader, device, world_size) -> None:
    evaluator = CGNEvaluator(
        pop_size=args.pop_size,
        graph=data_loader.G,
        device=device,
    )
    controller = CGNController(
        path=_result_path("CGN", args.mode),
        pattern="write",
        data_loader=data_loader,
        loops=1,
        crossover_rate=args.pc,
        mutate_rate=args.pm,
        pop_size=args.pop_size,
        device=device,
    )
    CGN(
        mode=args.mode,
        max_generation=args.generations,
        data_loader=data_loader,
        controller=controller,
        evaluator=evaluator,
        world_size=world_size,
        verbose=True,
    )


def _run_qattack(args, data_loader, device, world_size) -> None:
    evaluator = QAttackEvaluator(
        pop_size=args.pop_size,
        graph=data_loader.G,
        device=device,
    )
    controller = QAttackController(
        path=_result_path("QAttack", args.mode),
        pattern="write",
        data_loader=data_loader,
        loops=1,
        crossover_rate=args.pc,
        mutate_rate=args.pm,
        pop_size=args.pop_size,
        device=device,
    )
    QAttack(
        mode=args.mode,
        max_generation=args.generations,
        data_loader=data_loader,
        controller=controller,
        evaluator=evaluator,
        world_size=world_size,
        verbose=True,
    )


def _run_cda_eda(args, data_loader, device, world_size) -> None:
    evaluator = EDAEvaluator(
        pop_size=args.pop_size,
        graph=data_loader.G,
        adj=data_loader.A,
        nodes_num=data_loader.nodes_num,
        device=device,
    )
    controller = EDAController(
        path=_result_path("CDA-EDA", args.mode),
        pattern="write",
        data_loader=data_loader,
        loops=1,
        crossover_rate=args.pc,
        mutate_rate=args.pm,
        pop_size=args.pop_size,
        device=device,
    )
    EDA(
        mode=args.mode,
        max_generation=args.generations,
        data_loader=data_loader,
        controller=controller,
        evaluator=evaluator,
        world_size=world_size,
        verbose=True,
    )


RUNNERS: Dict[str, Callable] = {
    "CutOff": _run_cutoff,
    "TDE": _run_tde,
    "SixDST": _run_sixdst,
    "CGN": _run_cgn,
    "QAttack": _run_qattack,
    "CDA-EDA": _run_cda_eda,
}


def _parse_methods(args) -> List[str]:
    if args.all:
        return METHODS
    if args.method:
        method = args.method.strip()
        if method.lower() == "eda":
            method = "CDA-EDA"
        if method not in RUNNERS:
            raise ValueError(f"Unsupported method: {method}. choose from {METHODS}")
        return [method]
    raise ValueError("Please specify --method or use --all")


def _choose_dataset(method: str, dataset_arg: str | None) -> str:
    return dataset_arg if dataset_arg else DEFAULT_DATASET[method]


def _run_one(method: str, args, device, world_size) -> Tuple[str, bool, str]:
    dataset = _choose_dataset(method, args.dataset)
    print(f"\n=== Running {method} | dataset={dataset} | mode={args.mode} ===")
    try:
        if args.remote_server and args.mode in ("s", "m"):
            monitor = Monitor(api_base=args.api_base if args.api_base else None, timeout_s=float(args.api_timeout_s))
            result = run_remote_task(
                monitor,
                args.remote_server,
                algorithm=method,
                dataset=dataset,
                iterations=args.generations,
                mode=args.mode,
                crossover_rate=args.pc,
                mutate_rate=args.pm,
                use_strategy_plan=args.use_strategy_plan,
                max_polls=int(args.remote_max_polls),
                interval_s=float(args.remote_poll_s),
            )
            # Fallback path: app-level analysis_start/status does not require
            # querying /api/servers and is more robust when server list API is unavailable.
            if isinstance(result, dict) and result.get("error"):
                if "server list error" in str(result.get("error")):
                    start_resp = monitor.analysis_start(
                        algorithm=method,
                        dataset=dataset,
                        iterations=args.generations,
                        mode=args.mode.upper(),
                        crossover_rate=args.pc,
                        mutate_rate=args.pm,
                        server_id=args.remote_server,
                        extra={
                            "use_strategy_plan": bool(args.use_strategy_plan),
                            "timeout_s": float(args.remote_start_timeout_s),
                        },
                    )
                    if isinstance(start_resp, dict) and start_resp.get("error"):
                        return method, False, f"remote failed: {start_resp.get('error')}"
                    final = None
                    for _ in range(args.remote_max_polls):
                        st = monitor.analysis_status(server_id=args.remote_server)
                        if isinstance(st, dict) and st.get("error"):
                            return method, False, f"remote status failed: {st.get('error')}"
                        state = str((st or {}).get("state") or "")
                        if state in ("completed", "error", "idle"):
                            final = st
                            break
                        sleep(max(0.1, float(args.remote_poll_s)))
                    if not isinstance(final, dict):
                        return method, False, "remote failed: polling timeout"
                    if str(final.get("state")) == "error":
                        return method, False, f"remote failed: {final.get('error')}"
                    return method, True, "remote ok (fallback path)"
                return method, False, f"remote failed: {result.get('error')}"
            best = None
            try:
                best = float(monitor.best_fitness)
            except Exception:
                best = None
            return method, True, f"remote ok (best={best})"

        data_loader = DataLoader.load(dataset, device=str(device))
        t0 = perf_counter()
        RUNNERS[method](args, data_loader, device, world_size)
        dt = perf_counter() - t0
        return method, True, f"ok ({dt:.2f}s)"
    except Exception as exc:
        return method, False, str(exc)


def _pick_plan(args) -> Optional[Dict[str, Any]]:
    if not args.use_strategy_plan:
        return None
    multi_gpu = str(args.mode).lower() == "m"
    if args.plan_server:
        if not args.remote_server and platform.system() == "Darwin":
            raise RuntimeError(
                "PLAN_SERVER is set but REMOTE_SERVER is not set on macOS. "
                "This path triggers local planning/execution. "
                "Use --remote-server <server> (or set REMOTE_SERVER in batch script)."
            )
        monitor = Monitor(api_base=args.api_base if args.api_base else None, timeout_s=float(args.api_timeout_s))
        plan = monitor.strategy_plan(
            server_id=args.plan_server,
            algorithm=args.method if args.method else None,
            multi_gpu=multi_gpu,
        )
        if isinstance(plan, dict) and plan.get("error"):
            raise RuntimeError(f"remote strategy_plan failed: {plan}")
        if not isinstance(plan, dict):
            raise RuntimeError(f"invalid remote strategy_plan response: {plan}")
        return plan

    from autoadapt import StrategyPlan  # type: ignore
    from server.agent_monitor import resources_payload  # type: ignore

    plan_obj = StrategyPlan(
        fitness=None,
        warmup=0,
        objective=args.plan_objective,
        multi_gpu=multi_gpu,
        resource_snapshot=resources_payload(),
    )
    return plan_obj.to_dict() if hasattr(plan_obj, "to_dict") else None


def _resolve_runtime(args) -> Tuple[str, str, int, Optional[Dict[str, Any]]]:
    mode = str(args.mode).lower()
    if not args.use_strategy_plan:
        device, world_size = init_device(world_size=args.world_size)
        world_size = int(world_size) if isinstance(world_size, (int, float)) else 0
        if world_size <= 0:
            world_size = max(1, int(args.world_size))
        return mode, str(device), world_size, None

    plan = _pick_plan(args)
    if not isinstance(plan, dict):
        raise RuntimeError("StrategyPlan returned empty/invalid result")
    backend = str(plan.get("backend") or "").lower()
    devices = plan.get("devices") if isinstance(plan.get("devices"), list) else []
    devices = [int(x) for x in devices if isinstance(x, (int, float))]

    # Map StrategyPlan -> runtime.
    if backend in ("cpu", ""):
        resolved_mode = "s"
        device = "cpu"
        world_size = 1
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    elif backend == "cuda":
        resolved_mode = "s" if mode == "m" else mode
        if devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[0])
        device = "cuda:0"
        world_size = 1
    elif backend == "multi-gpu":
        if devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices)
        resolved_mode = "m" if mode == "m" else "s"
        world_size = max(2, len(devices)) if mode == "m" else 1
        device = "cuda:0"
    else:
        resolved_mode = "s"
        device = "cpu"
        world_size = 1

    return resolved_mode, device, world_size, plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Run built-in GAPA algorithms quickly")
    parser.add_argument("--method", type=str, default=None, help=f"One method from: {METHODS}")
    parser.add_argument("--all", action="store_true", help="Run all built-in methods")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    parser.add_argument("--mode", type=str, default="s", choices=["s", "sm", "m", "mnm"], help="Execution mode")
    parser.add_argument("--generations", type=int, default=10, help="Generation count")
    parser.add_argument("--pop-size", type=int, default=40, help="Population size")
    parser.add_argument("--pc", type=float, default=0.6, help="Crossover rate")
    parser.add_argument("--pm", type=float, default=0.2, help="Mutation rate")
    parser.add_argument("--seed", type=int, default=1024, help="Random seed")
    parser.add_argument("--world-size", type=int, default=2, help="Target world size for m/mnm")
    parser.add_argument("--remote-server", type=str, default=None, help="Run s/m remotely via app+agent server id/name")
    parser.add_argument("--api-timeout-s", type=float, default=600.0, help="HTTP timeout for app API calls")
    parser.add_argument("--remote-start-timeout-s", type=float, default=600.0, help="App proxy timeout for remote start")
    parser.add_argument("--remote-poll-s", type=float, default=1.0, help="Remote status polling interval (seconds)")
    parser.add_argument("--remote-max-polls", type=int, default=600, help="Remote polling max iterations")
    parser.add_argument("--use-strategy-plan", action="store_true", help="Enable StrategyPlan-based resource selection")
    parser.add_argument("--plan-server", type=str, default=None, help="Use /api/strategy_plan from remote server id/name")
    parser.add_argument("--api-base", type=str, default=None, help="API base for remote strategy plan (optional)")
    parser.add_argument("--plan-objective", type=str, default="time", choices=["time", "energy", "edp"], help="StrategyPlan objective")
    parser.add_argument(
        "--cutoff-tag",
        type=str,
        default="popGreedy_cutoff_",
        choices=["no_cutoff_", "random_cutoff_", "greedy_cutoff_", "popGreedy_cutoff_", "popGA_cutoff_"],
        help="Cutoff strategy for CND cutoff-based methods",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    if args.remote_server and args.mode in ("s", "m"):
        resolved_mode, device, world_size, plan = args.mode, "remote", 1, None
    else:
        resolved_mode, device, world_size, plan = _resolve_runtime(args)
    if resolved_mode != args.mode:
        print(f"[Runner] mode adjusted by runtime planner: {args.mode} -> {resolved_mode}")
        args.mode = resolved_mode

    if args.mode == "m":
        system = platform.system()
        if system == "Darwin" and not args.remote_server:
            raise RuntimeError("M mode is not supported on macOS for this legacy algorithm runner")
        if world_size < 2 and not args.remote_server:
            raise RuntimeError(f"M mode requires world_size >= 2, got {world_size}")

    print(f"[Runner] device={device} world_size={world_size}")
    if isinstance(plan, dict):
        print(f"[Runner] strategy_plan backend={plan.get('backend')} devices={plan.get('devices')} reason={plan.get('reason')}")

    methods = _parse_methods(args)
    results = [_run_one(m, args, device, world_size) for m in methods]

    print("\n=== Summary ===")
    has_error = False
    for method, ok, msg in results:
        status = "OK" if ok else "FAILED"
        print(f"- {method}: {status} | {msg}")
        if not ok:
            has_error = True
    if has_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
