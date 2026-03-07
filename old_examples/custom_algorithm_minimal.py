#!/usr/bin/env python3
"""
Minimal custom algorithm example.

Shows the smallest script path for plugging a user-defined algorithm wrapper
into the unified `Workflow` API without depending on repo datasets.
"""

from __future__ import annotations

import argparse
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from gapa.demo import build_demo_data
from gapa.workflow import Monitor, Workflow
from sixdst_custom import SixDSTAlgorithm


def main(argv=None):
    parser = argparse.ArgumentParser(description="Minimal custom algorithm example.")
    parser.add_argument("--graph", default="karate", choices=["karate", "barbell", "watts_strogatz"])
    parser.add_argument("--mode", default="s", choices=["s", "sm", "m", "m_cpu"])
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--pop-size", type=int, default=16)
    args = parser.parse_args(argv)

    data = build_demo_data(graph_name=args.graph)
    algo = SixDSTAlgorithm(pop_size=args.pop_size)
    monitor = Monitor()
    workflow = Workflow(algo, data, monitor=monitor, mode=args.mode, verbose=True)
    workflow.run(args.generations)

    print(f"[GAPA] Custom example graph: {args.graph}")
    print(f"[GAPA] Best fitness: {monitor.best_fitness}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
