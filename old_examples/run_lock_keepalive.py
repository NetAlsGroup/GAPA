#!/usr/bin/env python3
"""
Minimal example: lock MNM resources, renew periodically, then unlock.

Usage:
    python examples/run_lock_keepalive.py --servers 6 --duration 120 --cycles 3
"""

import argparse
import json
import time

from gapa.workflow import Monitor


def _p(data):
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="MNM lock keepalive example")
    parser.add_argument("--servers", nargs="+", required=True, help="Server ids/names")
    parser.add_argument("--duration", type=float, default=120.0, help="Lock/renew duration seconds")
    parser.add_argument("--cycles", type=int, default=3, help="Renew cycles")
    parser.add_argument("--sleep", type=float, default=10.0, help="Seconds between renew calls")
    parser.add_argument("--owner", default="example_keepalive")
    args = parser.parse_args()

    monitor = Monitor()
    lock_info = monitor.lock_mnm(
        server_inputs=args.servers,
        duration_s=args.duration,
        owner=args.owner,
        print_log=True,
    )
    print("=== lock_mnm ===")
    _p(lock_info)
    if lock_info.get("error"):
        return

    try:
        for i in range(max(0, args.cycles)):
            time.sleep(max(0.0, args.sleep))
            renewed = monitor.renew_mnm(lock_info=lock_info, duration_s=args.duration, print_log=True)
            print(f"\n=== renew cycle {i+1} ===")
            _p(renewed)
    finally:
        unlocked = monitor.unlock_servers(
            lock_info.get("locked") or [],
            api_base=lock_info.get("api_base"),
            print_log=True,
        )
        print("\n=== unlock_servers ===")
        _p(unlocked)


if __name__ == "__main__":
    main()
