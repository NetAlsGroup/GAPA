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
"""

import argparse
import os
import sys

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
    parser.add_argument("--mode", default="m", choices=["s", "sm", "m", "mnm"],
                        help="Execution mode (default: s)")
    parser.add_argument("--generations", type=int, default=50, help="Max generations")
    parser.add_argument("--pop_size", type=int, default=80, help="Population size")
    parser.add_argument("--auto-select", action="store_true", help="Auto-select servers (MNM)")
    parser.add_argument("--servers", nargs="+", help="Server IDs (MNM)")
    parser.add_argument("--server", default="", help="Remote server ID or name (e.g., 163 or Server 163)")
    args = parser.parse_args()

    monitor = Monitor()

    # 1. Load data
    data = load_dataset(args.dataset)
    
    # 2. Create algorithm
    algo = SixDSTAlgorithm(pop_size=args.pop_size)
    
    # 3. Create workflow with monitor
    # print(monitor.server())
    # print(monitor.server_resource("163"))

    workflow = Workflow(
        algo, data,
        monitor=monitor,
        mode=args.mode,
        # auto_select=args.auto_select,
        auto_select=False,
        servers=args.servers,
        remote_server=args.server,
    )
    
    # 4. Run
    workflow.run(args.generations)

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
