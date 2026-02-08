#!/usr/bin/env python3
"""
Minimal example: run a short workflow, export report, then show trend summary.

Usage:
    python examples/run_report_export.py --dataset ForestFire_n500 --generations 10
"""

import argparse
import json
import os
import sys
from pathlib import Path

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from gapa.workflow import Workflow, load_dataset, Monitor
from sixdst_custom import SixDSTAlgorithm


def main() -> None:
    parser = argparse.ArgumentParser(description="Report export example")
    parser.add_argument("--dataset", default="ForestFire_n500")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--mode", default="s", choices=["s", "sm", "m", "mnm"])
    args = parser.parse_args()

    monitor = Monitor()
    data = load_dataset(args.dataset)
    algo = SixDSTAlgorithm(pop_size=40)
    workflow = Workflow(algo, data, monitor=monitor, mode=args.mode, verbose=True)
    workflow.run(args.generations)

    out_dir = Path(os.getenv("GAPA_RESULTS_DIR", str(Path(_project_root) / "results"))) / "manual_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "latest_report.json"
    saved = monitor.save_report(str(report_path), pretty=False)

    print("\n=== saved report ===")
    print(saved)
    print("\n=== export_all ===")
    print(json.dumps(monitor.export_all(pretty=False), ensure_ascii=False, indent=2))
    print("\n=== run_trends ===")
    print(json.dumps(monitor.run_trends(last_n=20), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
