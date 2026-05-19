import argparse
import inspect
import json
from typing import Any


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return json.loads(value)
    except Exception:
        return value


def parse_kv_items(items: list[str] | None) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"invalid --kw item: {item!r}, expected key=value")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"invalid --kw item: {item!r}, empty key")
        parsed[key] = _parse_scalar(raw.strip())
    return parsed


def parse_int_list(value: str) -> list[int]:
    value = value.strip()
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_string_list(value: str) -> list[str]:
    value = value.strip()
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_server_device_map(items: list[str] | None) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"invalid --lock-devices item: {item!r}, expected server=0,1")
        server_id, raw_devices = item.split("=", 1)
        server_id = server_id.strip()
        if not server_id:
            raise ValueError(f"invalid --lock-devices item: {item!r}, empty server id")
        mapping[server_id] = parse_int_list(raw_devices)
    return mapping


def constructor_defaults(algo_cls: type) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    sig = inspect.signature(algo_cls.__init__)
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.default is inspect._empty:
            continue
        defaults[name] = param.default
    return defaults


def accepted_constructor_params(algo_cls: type) -> set[str]:
    sig = inspect.signature(algo_cls.__init__)
    return {name for name in sig.parameters if name != "self"}


def build_algorithm_kwargs(
    algo_cls: type,
    base_kwargs: dict[str, Any],
    *,
    pop_size: int | None,
    pc: float | None,
    pm: float | None,
    attack_rate: float | None,
    homophily_ratio: float | None,
    extra_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    accepted = accepted_constructor_params(algo_cls)
    kwargs = {key: value for key, value in base_kwargs.items() if key in accepted}

    if pop_size is not None and "pop_size" in accepted:
        kwargs["pop_size"] = int(pop_size)
    if pc is not None and "crossover_rate" in accepted:
        kwargs["crossover_rate"] = float(pc)
    if pm is not None and "mutate_rate" in accepted:
        kwargs["mutate_rate"] = float(pm)
    if attack_rate is not None and "attack_rate" in accepted:
        kwargs["attack_rate"] = float(attack_rate)
    if homophily_ratio is not None and "homophily_ratio" in accepted:
        kwargs["homophily_ratio"] = float(homophily_ratio)

    for key, value in (extra_kwargs or {}).items():
        if key not in accepted:
            supported = ", ".join(sorted(accepted))
            raise ValueError(f"{algo_cls.__name__} does not accept {key!r}. supported: {supported}")
        kwargs[key] = value
    return kwargs


def add_common_algorithm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--algorithm")
    parser.add_argument("--pop-size", type=int)
    parser.add_argument("--pc", type=float, help="Maps to crossover_rate when supported by the algorithm.")
    parser.add_argument("--pm", type=float, help="Maps to mutate_rate when supported by the algorithm.")
    parser.add_argument("--attack-rate", type=float)
    parser.add_argument("--homophily-ratio", type=float)
    parser.add_argument(
        "--kw",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra algorithm constructor kwargs. Can be repeated.",
    )
