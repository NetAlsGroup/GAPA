from pathlib import Path
import argparse
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gapa import DataLoader, Monitor, ResourceManager, Workflow
from gapa.algorithms import CDAEDAAlgorithm, CGNAlgorithm, QAttackAlgorithm
from gapa.algorithms import TDEAlgorithm, SixDSTAlgorithm, CutOffAlgorithm
from gapa.algorithms import GANIAlgorithm, NCAGAAlgorithm
from gapa.algorithms import LPAEDAAlgorithm, LPAGAAlgorithm
from examples._cli_utils import (
    add_common_algorithm_args,
    build_algorithm_kwargs,
    constructor_defaults,
    parse_int_list,
    parse_kv_items,
)


DATASET = "karate"
STEPS = 100
REMOTE_SERVER_ID = None
REMOTE_DEVICES = [0, 1]
REMOTE_USE_STRATEGY_PLAN = False
ALGORITHM = "CDAEDA"
DEFAULT_POP_SIZE = 100

ALGORITHM_REGISTRY = {
    "SixDST": SixDSTAlgorithm,
    "CutOff": CutOffAlgorithm,
    "TDE": TDEAlgorithm,
    "CGN": CGNAlgorithm,
    "QAttack": QAttackAlgorithm,
    "CDAEDA": CDAEDAAlgorithm,
    "LPAGA": LPAGAAlgorithm,
    "LPAEDA": LPAEDAAlgorithm,
    "NCAGA": NCAGAAlgorithm,
    "GANI": GANIAlgorithm,
}


def _to_seconds(value) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value) / 1000.0, 3)
    except Exception:
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a remote M-mode GAPA workflow experiment.")
    add_common_algorithm_args(parser)
    parser.add_argument("--remote-server-id")
    parser.add_argument("--remote-devices", help="Comma-separated GPU ids, e.g. 0,1")
    parser.add_argument(
        "--remote-use-strategy-plan",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable remote strategy plan.",
    )
    return parser


def _resolve_run_config() -> tuple[str, int, str | None, list[int], bool, object]:
    parser = _build_parser()
    args = parser.parse_args()

    dataset = args.dataset or DATASET
    steps = int(args.steps or STEPS)
    remote_server_id = args.remote_server_id or REMOTE_SERVER_ID
    remote_devices = parse_int_list(args.remote_devices) if args.remote_devices else list(REMOTE_DEVICES)
    remote_use_strategy_plan = (
        REMOTE_USE_STRATEGY_PLAN if args.remote_use_strategy_plan is None else bool(args.remote_use_strategy_plan)
    )

    algorithm_name = args.algorithm or ALGORITHM
    algo_cls = ALGORITHM_REGISTRY.get(algorithm_name)
    if algo_cls is None:
        supported = ", ".join(sorted(ALGORITHM_REGISTRY))
        raise ValueError(f"unsupported algorithm: {algorithm_name}. supported: {supported}")

    base_kwargs = constructor_defaults(algo_cls)
    if "pop_size" in base_kwargs:
        base_kwargs["pop_size"] = DEFAULT_POP_SIZE
    extra_kwargs = parse_kv_items(args.kw)
    algorithm_kwargs = build_algorithm_kwargs(
        algo_cls,
        base_kwargs,
        pop_size=args.pop_size,
        pc=args.pc,
        pm=args.pm,
        attack_rate=args.attack_rate,
        homophily_ratio=args.homophily_ratio,
        extra_kwargs=extra_kwargs,
    )
    print(f"[REMOTE-M] algorithm={algorithm_name} kwargs={algorithm_kwargs}")
    return dataset, steps, remote_server_id, remote_devices, remote_use_strategy_plan, algo_cls(**algorithm_kwargs)


def _select_remote_server_id(configured_server_id: str | None) -> str | None:
    manager = ResourceManager()
    servers = manager.server()
    remote_servers = [item for item in servers if item.get("id") != "local"] if isinstance(servers, list) else []
    if not remote_servers:
        return None
    if configured_server_id:
        return configured_server_id
    return str(remote_servers[0]["id"])


def _print_result_summary(result: dict) -> None:
    summary = {
        "best_fitness": result.get("best_fitness"),
        "iterations": result.get("iterations"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "metrics": result.get("metrics"),
        "report_path": result.get("report_path"),
    }
    print(summary)


def _print_comm_summary(result: dict) -> None:
    comm = result.get("comm") if isinstance(result, dict) else None
    if not isinstance(comm, dict) or not comm:
        print("[REMOTE-M] comm summary unavailable")
        return

    print(
        "[REMOTE-M] comm summary:",
        {
            "type": comm.get("type"),
            "avg_s": _to_seconds(comm.get("avg_ms")),
            "total_s": _to_seconds(comm.get("total_ms") or comm.get("total_comm_ms")),
            "calls": comm.get("calls"),
            "wall_clock_s": _to_seconds(comm.get("wall_clock_ms")),
        },
    )

    per_rank = comm.get("per_rank_avg_ms")
    if isinstance(per_rank, dict) and per_rank:
        compact = {key: _to_seconds(value) for key, value in per_rank.items()}
        print("[REMOTE-M] per-rank avg s:", compact)

    detailed = comm.get("detailed")
    if isinstance(detailed, dict):
        transport = detailed.get("transport")
        if isinstance(transport, dict) and transport:
            print("[REMOTE-M] transport metrics:", transport)


def _comm_wall_seconds(result: dict) -> float:
    comm = result.get("comm") if isinstance(result, dict) else None
    if not isinstance(comm, dict):
        return 0.0
    wall_ms = comm.get("wall_clock_ms")
    if isinstance(wall_ms, (int, float)):
        return max(0.0, float(wall_ms) / 1000.0)
    return 0.0


def _print_key_metrics(result: dict, total_s: float) -> None:
    reported_total = result.get("elapsed_seconds")
    if isinstance(reported_total, (int, float)) and float(reported_total) > 0:
        total_s = float(reported_total)
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    metric_items = [(k, v) for k, v in metrics.items() if isinstance(v, (int, float))]
    primary_name, primary_val = (metric_items[0] if len(metric_items) >= 1 else ("metric1", None))
    secondary_name, secondary_val = (metric_items[1] if len(metric_items) >= 2 else ("metric2", None))
    comm_s = min(max(0.0, _comm_wall_seconds(result)), total_s)
    algo_s = max(0.0, total_s - comm_s)
    print(
        f"[REMOTE-M] key_metrics "
        f"total_s={total_s:.3f} "
        f"{primary_name}={primary_val} "
        f"{secondary_name}={secondary_val} "
        f"algo_s={algo_s:.3f} "
        f"comm_s={comm_s:.3f}"
    )


def main() -> None:
    dataset, steps, configured_server_id, remote_devices, remote_use_strategy_plan, algorithm = _resolve_run_config()
    remote_server_id = _select_remote_server_id(configured_server_id)
    if not remote_server_id:
        print({"error": "no remote server configured in servers.json"})
        return

    data = DataLoader.load(dataset)
    monitor = Monitor()
    workflow = Workflow(
        algorithm=algorithm,
        data_loader=data,
        monitor=monitor,
        mode="m",
        remote_server=remote_server_id,
        remote_devices=remote_devices,
        remote_use_strategy_plan=remote_use_strategy_plan,
        verbose=True,
    )
    t0 = perf_counter()
    workflow.run(steps=steps)
    total_s = perf_counter() - t0
    result = monitor.result()
    _print_key_metrics(result, total_s)
    _print_result_summary(result)
    _print_comm_summary(result)


if __name__ == "__main__":
    main()
