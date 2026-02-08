#!/usr/bin/env python3
"""
SixDST Runner - Unified Example
================================
Demonstrates the unified GAPA workflow interface.
Uses the same core engine (Start, CustomController) as the frontend.

Usage:
    python run_sixdst.py --dataset ForestFire_n500
    python run_sixdst.py --dataset ForestFire_n500 --mode m
    python run_sixdst.py --dataset ForestFire_n500 --mode mnm --auto-select

    python run_sixdst.py --mode m --server 6 --use-strategy-plan
    python run_sixdst.py --mode m --server 6 --no-strategy-plan
"""

import argparse
import os
import sys
from typing import List

# Auto-detect GAPA path (allows running without pip install -e .)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# GAPA imports
from gapa.workflow import Workflow, load_dataset, Monitor
from sixdst_custom import SixDSTAlgorithm


def main():
    parser = argparse.ArgumentParser(description="SixDST Algorithm Runner (Unified)")
    parser.add_argument("--dataset", default="ForestFire_n500", help="Dataset name")
    parser.add_argument("--mode", default="s", choices=["s", "sm", "m", "mnm"],
                        help="Execution mode (default: s)")
    parser.add_argument("--generations", type=int, default=50, help="Max generations")
    parser.add_argument("--pop_size", type=int, default=80, help="Population size")
    parser.add_argument("--auto-select", action="store_true", help="Auto-select servers (MNM)")
    parser.add_argument("--servers", nargs="+", help="Server IDs (MNM)")
    parser.add_argument("--server", default=None, help="Remote server ID/name for s/sm/m (e.g., 163 or Server 163)")
    parser.add_argument("--mnm-lock-duration", type=float, default=900.0, help="MNM lock duration (seconds)")
    parser.add_argument("--mnm-lock-warmup", type=int, default=1, help="MNM lock warmup iterations")
    parser.add_argument("--mnm-lock-mem-mb", type=int, default=1024, help="MNM lock memory hint (MB)")
    parser.add_argument("--no-mnm-unlock", action="store_true", help="Do not release MNM lock after run")
    parser.add_argument(
        "--use-strategy-plan",
        dest="use_strategy_plan",
        action="store_true",
        help="Enable remote StrategyPlan device selection for s/sm/m (default: server default)",
    )
    parser.add_argument(
        "--no-strategy-plan",
        dest="use_strategy_plan",
        action="store_false",
        help="Disable remote StrategyPlan device selection for s/sm/m",
    )
    parser.set_defaults(use_strategy_plan=None)
    args = parser.parse_args()

    monitor = Monitor()
    lock_api_base = ""
    locked_servers: List[str] = []

    # 1. Load data
    data = load_dataset(args.dataset)
    
    # 2. Create algorithm
    algo = SixDSTAlgorithm(pop_size=args.pop_size)
    
    # 3. Create workflow with monitor
    # print(monitor.server())
    # print(monitor.server_resource("6"))
    # exit(0)

    workflow = Workflow(
        algo, data,
        monitor=monitor,
        mode=args.mode,
        auto_select=args.auto_select,
        # auto_select=False,
        servers=args.servers if str(args.mode).lower() == "mnm" else None,
        remote_server=args.server if str(args.mode).lower() in ("s", "sm", "m") and args.server else None,
        remote_use_strategy_plan=args.use_strategy_plan if str(args.mode).lower() in ("s", "sm", "m") else None,
    )
    
    # 4. Run
    try:
        if str(args.mode).lower() in ("s", "sm", "m") and args.server:
            print(f"[GAPA] Remote StrategyPlan: {args.use_strategy_plan if args.use_strategy_plan is not None else 'server-default'}")
        if str(args.mode).lower() == "mnm":
            lock_info = monitor.lock_mnm(
                server_inputs=args.servers,
                duration_s=args.mnm_lock_duration,
                warmup_iters=args.mnm_lock_warmup,
                mem_mb=args.mnm_lock_mem_mb,
                owner="run_sixdst",
                print_log=True,
            )
            if lock_info.get("error"):
                raise RuntimeError(f"MNM lock failed: {lock_info}")
            lock_api_base = str(lock_info.get("api_base") or "")
            locked_servers = list(lock_info.get("locked") or [])
        workflow.run(args.generations)
    finally:
        if str(args.mode).lower() == "mnm" and not args.no_mnm_unlock:
            monitor.unlock_servers(locked_servers, api_base=lock_api_base, print_log=True)

    print(f"\n[Result] Best fitness: {monitor.best_fitness:.4f}")
    print(monitor.export_all(pretty=True))

    # workflow.init_step()
    # for i in range(10):
    #     result = workflow.step()
    #     print(result)
    # print(f"\n[Result] Best fitness: {monitor.best_fitness:.4f}")

    # workflow.init_step()
    # workflow.run_steps(10)
    # print(f"\n[Result] Best fitness: {monitor.best_fitness:.4f}")
    # workflow.run_steps(10)
    # print(f"\n[Result] Best fitness: {monitor.best_fitness:.4f}")


if __name__ == "__main__":
    main()
