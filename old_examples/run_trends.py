#!/usr/bin/env python3
"""
Minimal example: show aggregated run trends from results/run_reports.jsonl.

Usage:
    python examples/run_trends.py
    python examples/run_trends.py --last-n 30
"""

import argparse
import json

from gapa.workflow import Monitor


def main() -> None:
    parser = argparse.ArgumentParser(description="Show GAPA run trend summary")
    parser.add_argument("--last-n", type=int, default=20, help="Recent runs per group")
    args = parser.parse_args()

    monitor = Monitor()
    trends = monitor.run_trends(last_n=args.last_n)
    print(json.dumps(trends, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
