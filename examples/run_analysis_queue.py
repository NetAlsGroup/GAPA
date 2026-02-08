#!/usr/bin/env python3
"""
Minimal example: submit analysis task with queue support and poll queue/status.

Usage:
    python examples/run_analysis_queue.py --server-id "Server 6" --queue-if-busy
"""

import argparse
import json
import time

from gapa.workflow import Monitor


def _p(data):
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Queue-aware analysis start example")
    parser.add_argument("--server-id", default=None, help="Target server id/name")
    parser.add_argument("--algorithm", default="SixDST")
    parser.add_argument("--dataset", default="ForestFire_n500")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--mode", default="M")
    parser.add_argument("--queue-if-busy", action="store_true")
    parser.add_argument("--owner", default="example_queue")
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=10)
    args = parser.parse_args()

    monitor = Monitor()

    started = monitor.analysis_start(
        algorithm=args.algorithm,
        dataset=args.dataset,
        iterations=args.iterations,
        mode=args.mode,
        server_id=args.server_id,
        queue_if_busy=args.queue_if_busy,
        owner=args.owner,
        priority=args.priority,
    )
    print("=== analysis_start ===")
    _p(started)

    for i in range(max(1, args.poll_seconds)):
        status = monitor.analysis_status(server_id=args.server_id)
        queue = monitor.analysis_queue(server_id=args.server_id)
        print(f"\n=== poll {i+1} ===")
        print("status:")
        _p(status)
        print("queue:")
        _p(queue)
        state = str(status.get("state") or "")
        if state in ("completed", "error", "idle"):
            break
        time.sleep(1)


if __name__ == "__main__":
    main()
