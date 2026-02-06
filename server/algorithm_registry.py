from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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
