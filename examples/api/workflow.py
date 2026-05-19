from pathlib import Path
import argparse
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gapa import DataLoader, Monitor, Workflow
from gapa.algorithms import CDAEDAAlgorithm, CGNAlgorithm, QAttackAlgorithm
from gapa.algorithms import TDEAlgorithm, SixDSTAlgorithm, CutOffAlgorithm
from gapa.algorithms import GANIAlgorithm, NCAGAAlgorithm
from gapa.algorithms import LPAEDAAlgorithm, LPAGAAlgorithm
from examples._cli_utils import (
    add_common_algorithm_args,
    build_algorithm_kwargs,
    constructor_defaults,
    parse_kv_items,
)


DATASET = "yeast1"
STEPS = 200
MODE = "m"  # s / sm / m
ALGORITHM = "SixDST"
DEFAULT_POP_SIZE = 80

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


def _mode_label(requested_mode: str, resolved_mode: str | None = None) -> str:
    requested = str(requested_mode or "").upper()
    resolved = str(resolved_mode or requested_mode or "").upper()
    if resolved and requested and resolved != requested:
        return f"{requested}->{resolved}"
    return requested or resolved or "?"


def _format_value(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _print_kv_block(prefix: str, title: str, rows: list[tuple[str, object]]) -> None:
    width = max((len(key) for key, _ in rows), default=0)
    print(f"[{prefix}] {title}")
    for key, value in rows:
        print(f"[{prefix}]   {key:<{width}} : {_format_value(value)}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local GAPA workflow experiment.")
    add_common_algorithm_args(parser)
    parser.add_argument("--mode", choices=["s", "sm", "m"])
    return parser


def _resolve_run_config() -> tuple[str, int, str, str, dict]:
    parser = _build_parser()
    args = parser.parse_args()

    dataset = args.dataset or DATASET
    steps = int(args.steps or STEPS)
    mode = args.mode or MODE
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
    print(f"[LOCAL-{mode.upper()}] algorithm={algorithm_name} kwargs={algorithm_kwargs}")
    return dataset, steps, mode, algorithm_name, algo_cls(**algorithm_kwargs)


def _print_result_summary(result: dict, prefix: str) -> None:
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    metric_rows = [(key, value) for key, value in metrics.items() if isinstance(value, (int, float))]
    _print_kv_block(
        prefix,
        "result summary",
        [
            ("best_fitness", result.get("best_fitness")),
            ("iterations", result.get("iterations")),
            ("elapsed_seconds", result.get("elapsed_seconds")),
            ("report_path", result.get("report_path")),
        ],
    )
    if metric_rows:
        _print_kv_block(prefix, "metric summary", metric_rows)


def _print_comm_summary(result: dict, requested_mode: str, resolved_mode: str) -> None:
    label = _mode_label(requested_mode, resolved_mode)
    comm = result.get("comm") if isinstance(result, dict) else None
    if not isinstance(comm, dict) or not comm:
        if requested_mode != resolved_mode:
            print(
                f"[LOCAL-{label}] comm summary unavailable "
                f"(requested mode '{requested_mode}' resolved to '{resolved_mode}')"
            )
        else:
            print(f"[LOCAL-{label}] comm summary unavailable")
        return

    _print_kv_block(
        f"LOCAL-{label}",
        "comm summary",
        [
            ("type", comm.get("type")),
            ("avg_s", _to_seconds(comm.get("avg_ms"))),
            ("total_s", _to_seconds(comm.get("total_ms") or comm.get("total_comm_ms"))),
            ("calls", comm.get("calls")),
            ("wall_clock_s", _to_seconds(comm.get("wall_clock_ms"))),
        ],
    )

    per_rank = comm.get("per_rank_avg_ms")
    if isinstance(per_rank, dict) and per_rank:
        compact = {key: _to_seconds(value) for key, value in per_rank.items()}
        print(f"[LOCAL-{label}] per-rank avg s:", compact)

    detailed = comm.get("detailed")
    if isinstance(detailed, dict):
        transport = detailed.get("transport")
        if isinstance(transport, dict) and transport:
            print(f"[LOCAL-{label}] transport metrics:", transport)


def _comm_wall_seconds(result: dict) -> float:
    comm = result.get("comm") if isinstance(result, dict) else None
    if not isinstance(comm, dict):
        return 0.0
    wall_ms = comm.get("wall_clock_ms")
    if isinstance(wall_ms, (int, float)):
        return max(0.0, float(wall_ms) / 1000.0)
    per_rank_ms = comm.get("per_rank_ms")
    if isinstance(per_rank_ms, dict) and per_rank_ms:
        vals = [float(v) for v in per_rank_ms.values() if isinstance(v, (int, float))]
        if vals:
            return max(0.0, max(vals) / 1000.0)
    return 0.0


def _print_key_metrics(result: dict, total_s: float, requested_mode: str, resolved_mode: str) -> None:
    reported_total = result.get("elapsed_seconds")
    if isinstance(reported_total, (int, float)) and float(reported_total) > 0:
        total_s = float(reported_total)
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    metric_items = [(k, v) for k, v in metrics.items() if isinstance(v, (int, float))]
    primary_name, primary_val = (metric_items[0] if len(metric_items) >= 1 else ("metric1", None))
    secondary_name, secondary_val = (metric_items[1] if len(metric_items) >= 2 else ("metric2", None))
    comm_s = min(max(0.0, _comm_wall_seconds(result)), total_s)
    algo_s = max(0.0, total_s - comm_s)
    label = _mode_label(requested_mode, resolved_mode)
    timing = result.get("timing") if isinstance(result.get("timing"), dict) else {}
    avg_ms = timing.get("iter_avg_ms")
    throughput = timing.get("throughput_ips")
    parts = [
        f"[LOCAL-{label}] key_metrics",
        f"total_s={total_s:.3f}",
        f"{primary_name}={primary_val}",
        f"{secondary_name}={secondary_val}",
        f"algo_s={algo_s:.3f}",
        f"comm_s={comm_s:.3f}",
    ]
    if isinstance(avg_ms, (int, float)):
        parts.append(f"avg_iter_ms={avg_ms:.3f}")
    if isinstance(throughput, (int, float)):
        parts.append(f"throughput_ips={throughput:.3f}")
    print(" ".join(parts))


def _print_runtime_summary(result: dict, total_s: float, requested_mode: str, resolved_mode: str) -> None:
    reported_total = result.get("elapsed_seconds")
    if isinstance(reported_total, (int, float)) and float(reported_total) > 0:
        total_s = float(reported_total)
    timing = result.get("timing") if isinstance(result.get("timing"), dict) else {}
    comm_s = min(max(0.0, _comm_wall_seconds(result)), total_s)
    algo_s = max(0.0, total_s - comm_s)
    label = _mode_label(requested_mode, resolved_mode)
    _print_kv_block(
        f"LOCAL-{label}",
        "runtime summary",
        [
            ("requested_mode", requested_mode),
            ("resolved_mode", resolved_mode),
            ("total_s", total_s),
            ("algo_s", algo_s),
            ("comm_s", comm_s),
            ("avg_iter_ms", timing.get("iter_avg_ms")),
            ("throughput_ips", timing.get("throughput_ips")),
        ],
    )


def main() -> None:
    dataset, steps, mode, _algorithm_name, algorithm = _resolve_run_config()
    data = DataLoader.load(dataset)
    monitor = Monitor()
    workflow = Workflow(algorithm, data, monitor=monitor, mode=mode, verbose=False)
    t0 = perf_counter()
    workflow.run(steps=steps)
    total_s = perf_counter() - t0
    result = monitor.result()
    requested_mode = str(result.get("requested_mode") or mode)
    resolved_mode = str(result.get("resolved_mode") or workflow.mode)
    _print_key_metrics(result, total_s, requested_mode, resolved_mode)
    _print_runtime_summary(result, total_s, requested_mode, resolved_mode)
    _print_result_summary(result, f"LOCAL-{_mode_label(requested_mode, resolved_mode)}")
    _print_comm_summary(result, requested_mode, resolved_mode)


if __name__ == "__main__":
    main()
