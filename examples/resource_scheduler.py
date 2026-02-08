#!/usr/bin/env python3
"""
Minimal resource/scheduler example for scripts and Jupyter users.

CLI:
    python examples/resource_scheduler.py servers
    python examples/resource_scheduler.py resources
    python examples/resource_scheduler.py lock --scope "Server 6" --duration-s 900
    python examples/resource_scheduler.py unlock --scope "Server 6"
    python examples/resource_scheduler.py plan --server-id "Server 6" --algorithm SixDST

Jupyter:
    from gapa.workflow import Monitor
    monitor = Monitor()
    monitor.server()
    monitor.resources(all_servers=True)
    monitor.lock_resource(scope="Server 6", duration_s=900)
    monitor.strategy_plan(server_id="Server 6", algorithm="SixDST", multi_gpu=True)
"""

import argparse
import json

from gapa.workflow import Monitor


def _print(data):
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal resource scheduler example")
    parser.add_argument("--api-base", default=None, help="Optional API base URL")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout")
    sub = parser.add_subparsers(dest="cmd", required=False)

    sub.add_parser("servers", help="List configured servers")
    sub.add_parser("resources", help="Show all resource snapshots")

    p_lock = sub.add_parser("lock", help="Lock resources")
    p_lock.add_argument("--scope", default="all")
    p_lock.add_argument("--duration-s", type=float, default=600.0)
    p_lock.add_argument("--warmup-iters", type=int, default=2)
    p_lock.add_argument("--mem-mb", type=int, default=1024)

    p_unlock = sub.add_parser("unlock", help="Unlock resources")
    p_unlock.add_argument("--scope", default="all")

    p_lstat = sub.add_parser("lock-status", help="Show lock status")
    p_lstat.add_argument("--scope", default="all")
    p_lstat.add_argument("--realtime", action="store_true")

    p_plan = sub.add_parser("plan", help="Run strategy plan")
    p_plan.add_argument("--server-id", default="local")
    p_plan.add_argument("--algorithm", default=None)
    p_plan.add_argument("--objective", default="time")
    p_plan.add_argument("--warmup", type=int, default=0)
    p_plan.add_argument("--multi-gpu", action="store_true")

    p_dplan = sub.add_parser("distributed-plan", help="Run distributed strategy plan")
    p_dplan.add_argument("--servers", nargs="*", default=None)
    p_dplan.add_argument("--per-server-gpus", type=int, default=1)

    args = parser.parse_args()
    monitor = Monitor(api_base=args.api_base, timeout_s=args.timeout)

    if not args.cmd:
        print("[resource_scheduler] Demo mode (no sub-command)")
        print("\n=== server() ===")
        _print(monitor.server())
        print("\n=== resources(all_servers=True) ===")
        _print(monitor.resources(all_servers=True))
        print("\n=== lock_resource(scope='Server 6', duration_s=900) ===")
        _print(monitor.lock_resource(scope="Server 6", duration_s=900))
        print("\n=== strategy_plan(server_id='Server 6', algorithm='SixDST', multi_gpu=True) ===")
        _print(monitor.strategy_plan(server_id="Server 6", algorithm="SixDST", multi_gpu=True))
        return

    if args.cmd == "servers":
        _print(monitor.server())
    elif args.cmd == "resources":
        _print(monitor.resources(all_servers=True))
    elif args.cmd == "lock":
        _print(
            monitor.lock_resource(
                scope=args.scope,
                duration_s=args.duration_s,
                warmup_iters=args.warmup_iters,
                mem_mb=args.mem_mb,
            )
        )
    elif args.cmd == "unlock":
        _print(monitor.unlock_resource(scope=args.scope))
    elif args.cmd == "lock-status":
        _print(monitor.lock_status(scope=args.scope, realtime=args.realtime))
    elif args.cmd == "plan":
        _print(
            monitor.strategy_plan(
                server_id=args.server_id,
                algorithm=args.algorithm,
                objective=args.objective,
                warmup=args.warmup,
                multi_gpu=args.multi_gpu,
            )
        )
    elif args.cmd == "distributed-plan":
        _print(
            monitor.distributed_strategy_plan(
                servers=args.servers,
                per_server_gpus=args.per_server_gpus,
            )
        )


if __name__ == "__main__":
    main()
