from pathlib import Path
import argparse
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests

from gapa import DataLoader, Monitor, Workflow
from gapa.algorithms import CDAEDAAlgorithm, CGNAlgorithm, QAttackAlgorithm
from gapa.algorithms import TDEAlgorithm, SixDSTAlgorithm, CutOffAlgorithm
from gapa.algorithms import GANIAlgorithm, NCAGAAlgorithm
from gapa.algorithms import LPAEDAAlgorithm, LPAGAAlgorithm
from gapa.config import build_remote_server_entries
from examples._cli_utils import (
    add_common_algorithm_args,
    build_algorithm_kwargs,
    constructor_defaults,
    parse_int_list,
    parse_kv_items,
    parse_server_device_map,
    parse_string_list,
)


DATASET = "yeast1"
STEPS = 200
ALGORITHM = "SixDST"
DEFAULT_POP_SIZE = 80
# SERVER_IDS: list[str] = ["srv_004", "srv_005"]
SERVER_IDS: list[str] = []
DEFAULT_LOCK_DEVICES = [0]
# SERVER_LOCK_DEVICES: dict[str, list[int]] = {
#     "srv_004": [0, 1],
#     "srv_005": [0, 1],
# }
SERVER_LOCK_DEVICES: dict[str, list[int]] = {}

LOCK_DURATION_S = 1800
LOCK_WARMUP_ITERS = 1
LOCK_MEM_MB = 0

"""
python examples/advanced/mnm_workflow.py \
  --algorithm TDE \
  --dataset yeast1 \
  --steps 500 \
  --server-ids Node2,Node1 \
  --lock-devices Node2=0 \
  --lock-devices Node1=0 \
  --pop-size 10 \
  --pc 0.5 \
  --pm 0.3
"""

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


def _default_server_ids() -> list[str]:
    return [sid for sid in _server_base_urls() if sid]


def _resolve_lock_devices(server_ids: list[str], configured: dict[str, list[int]]) -> dict[str, list[int]]:
    return {sid: list(configured.get(sid) or DEFAULT_LOCK_DEVICES) for sid in server_ids}


def _to_seconds(value) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value) / 1000.0, 3)
    except Exception:
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an MNM GAPA workflow experiment.")
    add_common_algorithm_args(parser)
    parser.add_argument("--server-ids", help="Comma-separated server ids from servers.json")
    parser.add_argument(
        "--lock-devices",
        action="append",
        default=[],
        metavar="SERVER=0,1",
        help="Per-server GPU ids to lock. Can be repeated.",
    )
    parser.add_argument("--lock-duration-s", type=float)
    parser.add_argument("--lock-warmup-iters", type=int)
    parser.add_argument("--lock-mem-mb", type=int)
    return parser


def _resolve_run_config() -> tuple[str, int, list[str], dict[str, list[int]], float, int, int, object]:
    parser = _build_parser()
    args = parser.parse_args()

    dataset = args.dataset or DATASET
    steps = int(args.steps or STEPS)
    server_ids = parse_string_list(args.server_ids) if args.server_ids else list(SERVER_IDS or _default_server_ids())
    if not server_ids:
        raise RuntimeError("no remote servers configured; set servers.json or pass --server-ids")
    lock_devices = dict(SERVER_LOCK_DEVICES)
    cli_lock_devices = parse_server_device_map(args.lock_devices)
    if cli_lock_devices:
        lock_devices.update(cli_lock_devices)
    lock_devices = _resolve_lock_devices(server_ids, lock_devices)
    lock_duration_s = float(args.lock_duration_s or LOCK_DURATION_S)
    lock_warmup_iters = int(args.lock_warmup_iters or LOCK_WARMUP_ITERS)
    lock_mem_mb = int(args.lock_mem_mb or LOCK_MEM_MB)

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
    print(f"[MNM] algorithm={algorithm_name} kwargs={algorithm_kwargs}")
    return (
        dataset,
        steps,
        server_ids,
        lock_devices,
        lock_duration_s,
        lock_warmup_iters,
        lock_mem_mb,
        algo_cls(**algorithm_kwargs),
    )


def _server_base_urls() -> dict[str, str]:
    entries = build_remote_server_entries()
    mapping: dict[str, str] = {}
    for item in entries:
        sid = str(item.get("id") or "")
        base_url = str(item.get("base_url") or "").rstrip("/")
        if sid and base_url:
            mapping[sid] = base_url
    return mapping


def _lock_servers(
    server_ids: list[str],
    server_lock_devices: dict[str, list[int]],
    duration_s: float,
    warmup_iters: int,
    mem_mb: int,
) -> list[str]:
    base_urls = _server_base_urls()
    session = requests.Session()
    session.trust_env = False
    locked: list[str] = []
    try:
        for sid in server_ids:
            base_url = base_urls.get(sid)
            if not base_url:
                raise RuntimeError(f"missing base_url for server id: {sid}")
            devices = server_lock_devices.get(sid)
            if not devices:
                raise RuntimeError(f"missing lock devices for server id: {sid}")
            payload = {
                "duration_s": float(duration_s),
                "warmup_iters": int(warmup_iters),
                "mem_mb": int(mem_mb),
                "devices": list(devices),
                "owner": "mnm-workflow",
            }
            resp = session.post(base_url + "/api/resource_lock", json=payload, timeout=(3.0, 20.0))
            resp.raise_for_status()
            body = resp.json()
            if not bool(body.get("active")):
                raise RuntimeError(f"lock failed on {sid}: {body}")
            print(f"[MNM] locked {sid}: backend={body.get('backend')} devices={body.get('devices')}")
            locked.append(sid)
        return locked
    except Exception:
        _release_servers(locked)
        raise


def _release_servers(server_ids: list[str]) -> None:
    if not server_ids:
        return
    base_urls = _server_base_urls()
    session = requests.Session()
    session.trust_env = False
    for sid in server_ids:
        base_url = base_urls.get(sid)
        if not base_url:
            continue
        try:
            resp = session.post(base_url + "/api/resource_lock/release", timeout=(3.0, 10.0))
            resp.raise_for_status()
            print(f"[MNM] released {sid}")
        except Exception as exc:
            print(f"[MNM] release failed on {sid}: {exc}")


def _print_comm_summary(result: dict) -> None:
    comm = result.get("comm") if isinstance(result, dict) else None
    if not isinstance(comm, dict) or not comm:
        print("[MNM] comm summary unavailable")
        return

    print(
        "[MNM] comm summary:",
        {
            "type": comm.get("type"),
            "avg_s": _to_seconds(comm.get("avg_ms")),
            "total_s": _to_seconds(comm.get("total_ms") or comm.get("total_comm_ms")),
            "calls": comm.get("calls"),
            "wall_clock_s": _to_seconds(comm.get("wall_clock_ms")),
        },
    )

    detailed = comm.get("detailed")
    if isinstance(detailed, dict):
        transport = detailed.get("transport")
        if isinstance(transport, dict) and transport:
            print("[MNM] transport metrics:", transport)
        per_worker = detailed.get("per_worker")
        if isinstance(per_worker, dict) and per_worker:
            compact = {
                key: {
                    "calls": value.get("calls"),
                    "avg_s": _to_seconds(value.get("avg_ms")),
                    "network_s": _to_seconds(value.get("network_ms")),
                    "compute_s": _to_seconds(value.get("compute_ms")),
                    "copy_to_device_s": _to_seconds(value.get("copy_to_device_ms")),
                    "forward_s": _to_seconds(value.get("forward_ms")),
                }
                for key, value in per_worker.items()
                if isinstance(value, dict)
            }
            print("[MNM] per-worker stats:", compact)

        total_compute = detailed.get("total_compute_ms")
        total_copy_to_device = detailed.get("total_copy_to_device_ms")
        total_forward = detailed.get("total_forward_ms")
        total_network = detailed.get("total_network_ms")
        total_serialize = detailed.get("total_serialize_ms")
        total_deserialize = detailed.get("total_deserialize_ms")
        wall_clock = detailed.get("wall_clock_ms")
        wall_compute = detailed.get("total_wall_compute_ms")
        wall_overhead = detailed.get("total_wall_overhead_ms")
        if any(
            v is not None
            for v in (
                total_compute,
                total_copy_to_device,
                total_forward,
                total_network,
                total_serialize,
                total_deserialize,
            )
        ):
            per_worker_forward = []
            per_worker_compute = []
            per_worker_network = []
            if isinstance(per_worker, dict):
                for value in per_worker.values():
                    if isinstance(value, dict):
                        if value.get("forward_ms") is not None:
                            per_worker_forward.append(float(value["forward_ms"]))
                        if value.get("compute_ms") is not None:
                            per_worker_compute.append(float(value["compute_ms"]))
                        if value.get("network_ms") is not None:
                            per_worker_network.append(float(value["network_ms"]))
            print(
                "[MNM] breakdown cumulative:",
                {
                    "compute_s": _to_seconds(total_compute),
                    "copy_to_device_s": _to_seconds(total_copy_to_device),
                    "forward_s": _to_seconds(total_forward),
                    "network_s": _to_seconds(total_network),
                    "serialize_s": _to_seconds(total_serialize),
                    "deserialize_s": _to_seconds(total_deserialize),
                },
            )
            print(
                "[MNM] breakdown wall-clock:",
                {
                    "wall_clock_s": _to_seconds(wall_clock),
                    "wall_compute_s": _to_seconds(wall_compute),
                    "wall_comm_s": _to_seconds(wall_overhead),
                    "max_worker_compute_s": _to_seconds(max(per_worker_compute) if per_worker_compute else None),
                    "max_worker_forward_s": _to_seconds(max(per_worker_forward) if per_worker_forward else None),
                    "max_worker_network_s": _to_seconds(max(per_worker_network) if per_worker_network else None),
                },
            )


def _print_result_summary(result: dict) -> None:
    summary = {
        "best_fitness": result.get("best_fitness"),
        "iterations": result.get("iterations"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "metrics": result.get("metrics"),
        "report_path": result.get("report_path"),
    }
    print(summary)


def _comm_wall_seconds(result: dict) -> float:
    comm = result.get("comm") if isinstance(result, dict) else None
    if not isinstance(comm, dict):
        return 0.0
    detailed = comm.get("detailed")
    if isinstance(detailed, dict):
        wall_overhead_ms = detailed.get("total_wall_overhead_ms")
        if isinstance(wall_overhead_ms, (int, float)):
            return max(0.0, float(wall_overhead_ms) / 1000.0)
    wall_ms = comm.get("wall_clock_ms")
    if isinstance(wall_ms, (int, float)):
        return max(0.0, float(wall_ms) / 1000.0)
    return 0.0


def _print_key_metrics(result: dict, total_s: float) -> None:
    reported_total = result.get("elapsed_seconds")
    if isinstance(reported_total, (int, float)):
        reported_total = float(reported_total)
        if 0.0 < reported_total <= total_s:
            total_s = reported_total
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    metric_items = [(k, v) for k, v in metrics.items() if isinstance(v, (int, float))]
    primary_name, primary_val = (metric_items[0] if len(metric_items) >= 1 else ("metric1", None))
    secondary_name, secondary_val = (metric_items[1] if len(metric_items) >= 2 else ("metric2", None))
    comm_s = min(max(0.0, _comm_wall_seconds(result)), total_s)
    algo_s = max(0.0, total_s - comm_s)
    print(
        f"[MNM] key_metrics "
        f"total_s={total_s:.3f} "
        f"{primary_name}={primary_val} "
        f"{secondary_name}={secondary_val} "
        f"algo_s={algo_s:.3f} "
        f"comm_s={comm_s:.3f}"
    )


def main() -> None:
    (
        dataset,
        steps,
        server_ids,
        server_lock_devices,
        lock_duration_s,
        lock_warmup_iters,
        lock_mem_mb,
        algorithm,
    ) = _resolve_run_config()
    data = DataLoader.load(dataset)
    monitor = Monitor()
    locked = _lock_servers(server_ids, server_lock_devices, lock_duration_s, lock_warmup_iters, lock_mem_mb)
    try:
        workflow = Workflow(
            algorithm=algorithm,
            data_loader=data,
            monitor=monitor,
            mode="mnm",
            servers=server_ids,
            fallback_policy="strict",
            verbose=True,
        )
        t0 = perf_counter()
        workflow.run(steps=steps)
        total_s = perf_counter() - t0
        result = monitor.result()
        _print_key_metrics(result, total_s)
        _print_result_summary(result)
        _print_comm_summary(result)
    finally:
        _release_servers(locked)


if __name__ == "__main__":
    main()
