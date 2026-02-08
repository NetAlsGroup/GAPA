from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_CAPABILITIES: Dict[str, Any] = {
    "supported_modes": ["s", "sm", "m", "mnm"],
    "supports_distributed_fitness": True,
    "supports_remote": True,
    "fitness_direction": "min",
}


def _default_algo_file() -> Path:
    return Path(__file__).resolve().parents[1] / "algorithms.json"


def _load_entries(path: Optional[str] = None) -> List[Dict[str, Any]]:
    algo_path = Path(path) if path else _default_algo_file()
    if not algo_path.exists():
        return []
    try:
        raw = json.loads(algo_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(raw, dict):
        raw = raw.get("algorithms", [])
    if not isinstance(raw, list):
        return []
    entries = [e for e in raw if isinstance(e, dict)]
    return entries


def load_algorithm_entries(path: Optional[str] = None) -> List[Dict[str, Any]]:
    return _load_entries(path)


def _normalized_capabilities(entry: Dict[str, Any]) -> Dict[str, Any]:
    caps = dict(DEFAULT_CAPABILITIES)
    raw_caps = entry.get("capabilities")
    if isinstance(raw_caps, dict):
        caps.update(raw_caps)
    # Backward-compatible top-level fields
    if isinstance(entry.get("supported_modes"), list):
        caps["supported_modes"] = entry.get("supported_modes")
    if entry.get("supports_distributed_fitness") is not None:
        caps["supports_distributed_fitness"] = bool(entry.get("supports_distributed_fitness"))
    if entry.get("supports_remote") is not None:
        caps["supports_remote"] = bool(entry.get("supports_remote"))
    if isinstance(entry.get("fitness_direction"), str):
        caps["fitness_direction"] = entry.get("fitness_direction")
    # Normalize supported_modes as lower-case unique list
    modes = caps.get("supported_modes")
    if isinstance(modes, list):
        normalized: List[str] = []
        for m in modes:
            if not isinstance(m, str):
                continue
            lm = m.strip().lower()
            if lm and lm not in normalized:
                normalized.append(lm)
        caps["supported_modes"] = normalized
    else:
        caps["supported_modes"] = list(DEFAULT_CAPABILITIES["supported_modes"])
    return caps


def _aliases_for(entry: Dict[str, Any]) -> List[str]:
    aliases: List[str] = []
    for key in ("id", "name"):
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            aliases.append(val.strip())
    extra = entry.get("aliases")
    if isinstance(extra, list):
        for a in extra:
            if isinstance(a, str) and a.strip():
                aliases.append(a.strip())
    return aliases


def resolve_algorithm_id(name: str, path: Optional[str] = None) -> str:
    algo = (name or "").strip()
    if not algo:
        return name
    entries = _load_entries(path)
    for entry in entries:
        algo_id = entry.get("id")
        if not isinstance(algo_id, str):
            continue
        for alias in _aliases_for(entry):
            if alias.lower() == algo.lower():
                return algo_id
    return name


def get_algorithm_spec(name: str, path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    algo = (name or "").strip()
    if not algo:
        return None
    entries = _load_entries(path)
    for entry in entries:
        for alias in _aliases_for(entry):
            if alias.lower() == algo.lower():
                out = dict(entry)
                out["capabilities"] = _normalized_capabilities(entry)
                return out
    return None


def get_algorithm_capabilities(name: str, path: Optional[str] = None) -> Dict[str, Any]:
    spec = get_algorithm_spec(name, path)
    if not spec:
        return dict(DEFAULT_CAPABILITIES)
    caps = spec.get("capabilities")
    return caps if isinstance(caps, dict) else dict(DEFAULT_CAPABILITIES)


def load_algorithm_registry(path: Optional[str] = None) -> Dict[str, Any]:
    entries = _load_entries(path)
    registry: Dict[str, Any] = {}
    for entry in entries:
        algo_id = entry.get("id")
        if not isinstance(algo_id, str) or not algo_id.strip():
            continue
        entry_path = entry.get("entry")
        if not isinstance(entry_path, str) or ":" not in entry_path:
            continue
        module_path, symbol = entry_path.split(":", 1)
        try:
            mod = import_module(module_path)
            cls = getattr(mod, symbol)
            registry[algo_id] = cls
        except Exception:
            continue
    return registry
