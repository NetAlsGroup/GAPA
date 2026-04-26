"""
GAPA Workflow Module
====================
User-friendly interface for evolutionary algorithm execution.
This module provides a unified API for both script-based and frontend execution.

**Architecture**: 
    Workflow wraps the existing core engine (Start, CustomController, BasicEvaluator)
    to ensure consistent behavior across all execution contexts.

Core Classes:
    - Algorithm: Base class for evolutionary algorithms  
    - Workflow: Unified orchestrator (wraps existing Start/Controller)
    - Monitor: Tracks evolution progress

Example:
    >>> from gapa.workflow import Workflow, DataLoader, Monitor
    >>> from examples.sixdst_custom import SixDSTAlgorithm
    >>> 
    >>> data = DataLoader.load("A01")
    >>> algo = SixDSTAlgorithm(budget=data.k, pop_size=80)
    >>> monitor = Monitor()
    >>> workflow = Workflow(algo, data, monitor=monitor, mode="m")
    >>> 
    >>> workflow.run(1000)
    >>> print(f"Best: {monitor.get_best_fitness()}")
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

__all__ = [
    "Algorithm",
    "Workflow",
    "Monitor",
    "DataLoader",
]

from gapa.config import get_app_base_url, get_results_dir
from gapa.data_loader import DataLoader

def _results_root() -> Path:
    return get_results_dir(Path(__file__).resolve().parents[1])


def _load_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _benchmark_key(algorithm_id: str, dataset: str, mode: str, remote: bool) -> str:
    return f"{algorithm_id}|{dataset}|{mode}|remote={int(bool(remote))}"


def _compare_and_update_benchmark(
    *,
    algorithm_id: str,
    dataset: str,
    mode: str,
    remote: bool,
    iter_avg_ms: Optional[float],
    iter_seconds: Optional[float],
    regression_threshold: float = 0.25,
) -> Dict[str, Any]:
    key = _benchmark_key(algorithm_id, dataset, mode, remote)
    path = _results_root() / "benchmarks.json"
    db = _load_json_file(path, {})
    if not isinstance(db, dict):
        db = {}
    entry = db.get(key) if isinstance(db.get(key), dict) else {}
    best_avg = entry.get("best_iter_avg_ms")
    best_sec = entry.get("best_iter_seconds")

    current_avg = float(iter_avg_ms) if isinstance(iter_avg_ms, (int, float)) else None
    current_sec = float(iter_seconds) if isinstance(iter_seconds, (int, float)) else None

    regressed = False
    ratio = None
    if current_avg is not None and isinstance(best_avg, (int, float)) and float(best_avg) > 0:
        ratio = (current_avg - float(best_avg)) / float(best_avg)
        regressed = ratio > regression_threshold
    elif current_sec is not None and isinstance(best_sec, (int, float)) and float(best_sec) > 0:
        ratio = (current_sec - float(best_sec)) / float(best_sec)
        regressed = ratio > regression_threshold

    update = dict(entry)
    if current_avg is not None and (not isinstance(best_avg, (int, float)) or current_avg < float(best_avg)):
        update["best_iter_avg_ms"] = current_avg
    if current_sec is not None and (not isinstance(best_sec, (int, float)) or current_sec < float(best_sec)):
        update["best_iter_seconds"] = current_sec
    update["last_iter_avg_ms"] = current_avg
    update["last_iter_seconds"] = current_sec
    update["updated_at"] = datetime.utcnow().isoformat() + "Z"
    db[key] = update
    _save_json_file(path, db)

    return {
        "key": key,
        "benchmark_path": str(path),
        "regressed": bool(regressed),
        "ratio": ratio,
        "threshold": float(regression_threshold),
        "best_iter_avg_ms": update.get("best_iter_avg_ms"),
        "best_iter_seconds": update.get("best_iter_seconds"),
        "last_iter_avg_ms": update.get("last_iter_avg_ms"),
        "last_iter_seconds": update.get("last_iter_seconds"),
    }


def _load_run_report_rows(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    jsonl_path = path or (_results_root() / "run_reports.jsonl")
    if not jsonl_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            # keep summary rows only (benchmark append-only rows don't have algorithm_id)
            if "algorithm_id" in obj and "dataset" in obj:
                rows.append(obj)
    except Exception:
        return []
    return rows


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _resolve_remote_primary_score(result: Dict[str, Any]) -> Optional[float]:
    best = result.get("best_score")
    if isinstance(best, (int, float)):
        return float(best)
    objectives = result.get("objectives") if isinstance(result.get("objectives"), dict) else {}
    best_metrics = result.get("best_metrics") if isinstance(result.get("best_metrics"), dict) else {}
    primary = objectives.get("primary")
    if primary and primary in best_metrics:
        return _to_float(best_metrics.get(primary))
    curves = result.get("curves") if isinstance(result.get("curves"), dict) else {}
    if primary and isinstance(curves.get(primary), list) and curves.get(primary):
        return _to_float(curves.get(primary)[-1])
    return None


def _aggregate_run_trends(
    rows: List[Dict[str, Any]],
    *,
    last_n: int = 20,
) -> Dict[str, Any]:
    if last_n <= 0:
        last_n = 20
    groups: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        algo = str(row.get("algorithm_id") or "")
        ds = str(row.get("dataset") or "")
        mode = str(row.get("resolved_mode") or row.get("mode") or "")
        remote = bool(row.get("remote"))
        key = _benchmark_key(algo, ds, mode, remote)
        if key not in groups:
            groups[key] = {
                "key": key,
                "algorithm_id": algo,
                "dataset": ds,
                "mode": mode,
                "remote": remote,
                "rows": [],
            }
        groups[key]["rows"].append(row)

    out_groups: List[Dict[str, Any]] = []
    for g in groups.values():
        hist = g["rows"][-last_n:]
        avg_list = [_to_float(x.get("iter_avg_ms")) for x in hist]
        sec_list = [_to_float(x.get("iter_seconds")) for x in hist]
        thr_list = [_to_float(x.get("throughput_ips")) for x in hist]
        fit_list = [_to_float(x.get("best_fitness")) for x in hist]
        avg_list = [x for x in avg_list if x is not None]
        sec_list = [x for x in sec_list if x is not None]
        thr_list = [x for x in thr_list if x is not None]
        fit_list = [x for x in fit_list if x is not None]
        regressions = 0
        for x in hist:
            bench = x.get("benchmark")
            if isinstance(bench, dict) and bench.get("regressed"):
                regressions += 1
        def _mean(vals: List[float]) -> Optional[float]:
            return (sum(vals) / len(vals)) if vals else None
        out_groups.append(
            {
                "key": g["key"],
                "algorithm_id": g["algorithm_id"],
                "dataset": g["dataset"],
                "mode": g["mode"],
                "remote": g["remote"],
                "runs": len(hist),
                "regressions": regressions,
                "iter_avg_ms_mean": _mean(avg_list),
                "iter_seconds_mean": _mean(sec_list),
                "throughput_ips_mean": _mean(thr_list),
                "best_fitness_mean": _mean(fit_list),
                "last_run": hist[-1] if hist else None,
            }
        )
    out_groups.sort(key=lambda x: (-int(x.get("runs") or 0), str(x.get("key") or "")))
    return {"groups": out_groups, "total_groups": len(out_groups), "total_rows": len(rows), "last_n": last_n}


def _resolve_default_api_base() -> str:
    env_base = os.getenv("GAPA_API_BASE")
    if env_base:
        return str(env_base).rstrip("/")
    return get_app_base_url().rstrip("/")

import os
import sys
from urllib.parse import urlencode
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import multiprocessing as mp
import json
import uuid

# Suppress known deprecation warnings (before torch imports)
import warnings
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*nvidia-ml-py.*", category=FutureWarning)

import torch
import torch.nn as nn
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency for monitor HTTP
    requests = None

# =============================================================================
# Feature Flags (for PyPI vs Source deployment)
# =============================================================================

# Distributed (MNM) features are loaded lazily. Importing gapa should not touch
# server-side modules because they may initialize runtime state such as SQLite.
def _load_distributed_evaluator():
    try:
        from server.distributed_evaluator import DistributedEvaluator as _DistributedEvaluator
        return _DistributedEvaluator
    except Exception:
        return None


def _has_distributed() -> bool:
    return _load_distributed_evaluator() is not None

# Supported modes in PyPI package vs Source deployment
PYPI_MODES = ["s", "sm", "m"]
SOURCE_MODES = ["s", "sm", "m", "mnm"]


# =============================================================================
# Monitor (Observer Pattern)
# =============================================================================

class _MonitorResourceCompatibility:
    """Legacy resource API compatibility kept for older Monitor callers."""

    def _resource(self):
        from gapa.resource_manager import ResourceManager

        manager = getattr(self, "_resource_manager", None)
        if manager is None:
            manager = ResourceManager(api_base=self.api_base, timeout_s=self.timeout_s)
            self._resource_manager = manager
        return manager

    def _guess_api_base_from_config(self) -> Optional[str]:
        return _resolve_default_api_base()

    def server(self) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        return self._resource().server()

    def server_resource(self, server_id_or_name: str) -> Dict[str, Any]:
        return self._resource().server_resource(server_id_or_name)

    def resources(self, all_servers: bool = True) -> Dict[str, Any]:
        return self._resource().resources(all_servers=all_servers)

    def lock_status(self, scope: str = "all", realtime: bool = True) -> Dict[str, Any]:
        return self._resource().lock_status(scope=scope, realtime=realtime)

    def lock_resource(
        self,
        scope: str = "all",
        duration_s: float = 600.0,
        warmup_iters: int = 2,
        mem_mb: int = 1024,
        strict_idle: bool = False,
        devices: Optional[List[Any]] = None,
        devices_by_server: Optional[Dict[str, List[Any]]] = None,
    ) -> Dict[str, Any]:
        return self._resource().lock_resource(
            scope=scope,
            duration_s=duration_s,
            warmup_iters=warmup_iters,
            mem_mb=mem_mb,
            strict_idle=strict_idle,
            devices=devices,
            devices_by_server=devices_by_server,
        )

    def unlock_resource(self, scope: str = "all") -> Dict[str, Any]:
        return self._resource().unlock_resource(scope=scope)

    def renew_resource(
        self,
        scope: str = "all",
        duration_s: Optional[float] = None,
        lock_id: Optional[str] = None,
        owner: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._resource().renew_resource(
            scope=scope,
            duration_s=duration_s,
            lock_id=lock_id,
            owner=owner,
        )

    def strategy_plan(
        self,
        server_id: Optional[str] = None,
        algorithm: Optional[str] = None,
        dataset: Optional[str] = None,
        mode: Optional[str] = None,
        warmup: int = 0,
        objective: str = "time",
        multi_gpu: bool = True,
        gpu_busy_threshold: Optional[float] = None,
        min_gpu_free_mb: Optional[int] = None,
        tpe_trials: Optional[int] = None,
        tpe_warmup: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self._resource().strategy_plan(
            server_id=server_id,
            algorithm=algorithm,
            dataset=dataset,
            mode=mode,
            warmup=warmup,
            objective=objective,
            multi_gpu=multi_gpu,
            gpu_busy_threshold=gpu_busy_threshold,
            min_gpu_free_mb=min_gpu_free_mb,
            tpe_trials=tpe_trials,
            tpe_warmup=tpe_warmup,
        )

    def distributed_strategy_plan(
        self,
        server_ids: Optional[List[str]] = None,
        servers: Optional[List[str]] = None,
        algorithm: Optional[str] = None,
        dataset: Optional[str] = None,
        mode: Optional[str] = None,
        per_server_gpus: int = 1,
        min_gpu_free_mb: int = 1024,
        gpu_busy_threshold: float = 85.0,
    ) -> Dict[str, Any]:
        return self._resource().distributed_strategy_plan(
            server_ids=server_ids,
            servers=servers,
            algorithm=algorithm,
            dataset=dataset,
            mode=mode,
            per_server_gpus=per_server_gpus,
            min_gpu_free_mb=min_gpu_free_mb,
            gpu_busy_threshold=gpu_busy_threshold,
        )

    def analysis_status(self, server_id: Optional[str] = None) -> Dict[str, Any]:
        return self._resource().analysis_status(server_id=server_id)

    def analysis_queue(self, server_id: Optional[str] = None) -> Dict[str, Any]:
        return self._resource().analysis_queue(server_id=server_id)

    def analysis_start(
        self,
        *,
        algorithm: str,
        dataset: str,
        iterations: int = 20,
        mode: str = "S",
        crossover_rate: float = 0.8,
        mutate_rate: float = 0.2,
        server_id: Optional[str] = None,
        queue_if_busy: bool = False,
        owner: str = "",
        priority: int = 0,
        release_lock_on_finish: bool = True,
        timeout_s: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._resource().analysis_start(
            algorithm=algorithm,
            dataset=dataset,
            iterations=iterations,
            mode=mode,
            crossover_rate=crossover_rate,
            mutate_rate=mutate_rate,
            server_id=server_id,
            queue_if_busy=queue_if_busy,
            owner=owner,
            priority=priority,
            release_lock_on_finish=release_lock_on_finish,
            timeout_s=timeout_s,
            extra=extra,
        )

    def analysis_stop(self, server_id: Optional[str] = None) -> Dict[str, Any]:
        return self._resource().analysis_stop(server_id=server_id)

    def transport_metrics(self) -> Dict[str, Any]:
        return self._resource().transport_metrics()

    def resource_rows(self, resources: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self._resource().resource_rows(resources=resources)

    def resource_dataframe(self, rows: Optional[List[Dict[str, Any]]] = None):
        return self._resource().resource_dataframe(rows=rows)

    def plot_resources(self, metric: str = "cpu_usage_percent", rows: Optional[List[Dict[str, Any]]] = None):
        return self._resource().plot_resources(metric=metric, rows=rows)

    def run_trends(self, last_n: int = 20) -> Dict[str, Any]:
        return self._resource().run_trends(last_n=last_n)

    def lock_mnm(
        self,
        server_inputs: Optional[List[str]] = None,
        duration_s: float = 900.0,
        warmup_iters: int = 1,
        mem_mb: int = 1024,
        owner: str = "",
        print_log: bool = False,
    ) -> Dict[str, Any]:
        return self._resource().lock_mnm(
            server_inputs=server_inputs,
            duration_s=duration_s,
            warmup_iters=warmup_iters,
            mem_mb=mem_mb,
            owner=owner,
            print_log=print_log,
        )

    def unlock_servers(self, server_ids: List[str], api_base: Optional[str] = None, print_log: bool = False) -> Dict[str, Any]:
        return self._resource().unlock_servers(server_ids=server_ids, api_base=api_base, print_log=print_log)

    def renew_mnm(
        self,
        *,
        lock_info: Dict[str, Any],
        duration_s: Optional[float] = None,
        print_log: bool = False,
    ) -> Dict[str, Any]:
        return self._resource().renew_mnm(lock_info=lock_info, duration_s=duration_s, print_log=print_log)


class Monitor(_MonitorResourceCompatibility):
    """
    Evolution progress monitor.
    
    Implements the Observer pattern to track fitness values and solutions.
    Compatible with both Workflow and legacy controller execution.
    
    Example:
        >>> monitor = Monitor()
        >>> workflow = Workflow(algo, data, monitor=monitor)
        >>> workflow.run(1000)
        >>> print(monitor.get_best_fitness())
    """
    
    def __init__(self, opt_direction: str = "min", topk: int = 1, api_base: Optional[str] = None, timeout_s: float = 5.0):
        """
        Initialize the monitor.
        
        Args:
            opt_direction: Optimization direction - "min" or "max". Default: "min"
            topk: Number of top solutions to track. Default: 1
            api_base: Base URL for GAPA backend API. Default: local server.
            timeout_s: HTTP timeout in seconds for monitor requests.
        """
        self.opt_direction = opt_direction
        self.topk = topk
        self.api_base = api_base
        self.timeout_s = float(timeout_s)
        
        self._best_fitness: Optional[float] = None
        self._best_solution: Optional[torch.Tensor] = None
        self._fitness_history: List[float] = []
        self._extra_history: List[Dict[str, Any]] = []
        self._generation: int = 0
        self._remote_result: Optional[Dict[str, Any]] = None
        self._local_timing: Optional[Dict[str, Any]] = None
        self._run_context: Optional[Dict[str, Any]] = None
        self._workflow_state: str = "idle"
        self._workflow_message: str = ""

    def set_run_context(self, context: Dict[str, Any]) -> None:
        self._run_context = dict(context)

    def set_workflow_state(self, state: str, message: str = "") -> None:
        self._workflow_state = str(state or "idle")
        self._workflow_message = str(message or "")

    def reset(self) -> None:
        self._best_fitness = None
        self._best_solution = None
        self._fitness_history = []
        self._extra_history = []
        self._generation = 0
        self._remote_result = None
        self._local_timing = None
        self._run_context = None
        self._workflow_state = "idle"
        self._workflow_message = ""
        self._resource_manager = None

    def record(
        self,
        generation: int,
        fitness_list: torch.Tensor,
        best_gene: torch.Tensor,
        extra: Optional[Dict[str, Any]] = None,
        side: str = "min",
    ) -> None:
        """
        Record callback compatible with existing controller interface.
        
        This method is called by CustomController.calculate() if observer is provided.
        """
        self._generation = generation
        
        if side == "min":
            best_fit = fitness_list.min().item()
            is_better = self._best_fitness is None or best_fit < self._best_fitness
        else:
            best_fit = fitness_list.max().item()
            is_better = self._best_fitness is None or best_fit > self._best_fitness
        
        if is_better:
            self._best_fitness = best_fit
            self._best_solution = best_gene.clone()
        
        self._fitness_history.append(self._best_fitness)
        if extra:
            self._extra_history.append(extra)
    
    def update(
        self,
        population: torch.Tensor,
        fitness: torch.Tensor,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Alternative update method for direct Workflow calls."""
        self._generation += 1
        
        if self.opt_direction == "min":
            best_idx = fitness.argmin()
            best_fit = fitness[best_idx].item()
            is_better = self._best_fitness is None or best_fit < self._best_fitness
        else:
            best_idx = fitness.argmax()
            best_fit = fitness[best_idx].item()
            is_better = self._best_fitness is None or best_fit > self._best_fitness
        
        if is_better:
            self._best_fitness = best_fit
            self._best_solution = population[best_idx].clone()
        
        self._fitness_history.append(self._best_fitness)
    
    def get_best_fitness(self) -> float:
        """Get the best fitness value found so far."""
        return self._best_fitness if self._best_fitness is not None else float('inf')
    
    def get_best_solution(self) -> Optional[torch.Tensor]:
        """Get the best solution found so far."""
        return self._best_solution

    def get_best_gene(self) -> Optional[torch.Tensor]:
        return self.get_best_solution()
    
    def get_fitness_history(self) -> List[float]:
        """Get the full history of best fitness values."""
        return self._fitness_history.copy()
    
    @property
    def best_fitness(self) -> float:
        return self.get_best_fitness()
    
    @property
    def best_solution(self) -> Optional[torch.Tensor]:
        return self.get_best_solution()

    @property
    def best_gene(self) -> Optional[torch.Tensor]:
        return self.get_best_gene()
    
    @property
    def generation(self) -> int:
        return self._generation

    def _iteration_count(self) -> int:
        history_based = max(0, len(self._fitness_history) - 1)
        return max(int(self._generation), history_based)

    @staticmethod
    def _serialize_gene(gene: Optional[torch.Tensor]) -> Optional[List[Any]]:
        if gene is None:
            return None
        try:
            return gene.detach().cpu().tolist()
        except Exception:
            return None

    def status(self) -> Dict[str, Any]:
        best = self._best_fitness if self._best_fitness is not None else None
        return {
            "state": self._workflow_state,
            "iteration": self._iteration_count(),
            "best_fitness": best,
            "message": self._workflow_message,
        }

    def history(self) -> Dict[str, Any]:
        return {
            "fitness": self.get_fitness_history(),
            "extra": list(self._extra_history),
        }

    def result(self) -> Dict[str, Any]:
        report_meta = {}
        if isinstance(self._run_context, dict):
            report_meta = self._run_context.get("reports") if isinstance(self._run_context.get("reports"), dict) else {}
        timing = self._local_timing or {}
        if self._remote_result and isinstance(self._remote_result.get("timing"), dict):
            timing = dict(self._remote_result.get("timing") or {})
        metrics: Dict[str, Any] = {}
        if self._remote_result:
            metrics_block = self._remote_result.get("best_metrics")
            objectives = self._remote_result.get("objectives")
            if isinstance(metrics_block, dict):
                metrics.update(metrics_block)
            if isinstance(objectives, dict):
                metrics["objectives"] = objectives
        return {
            "best_fitness": self._best_fitness,
            "best_gene": self._serialize_gene(self._best_solution),
            "iterations": self._iteration_count(),
            "elapsed_seconds": timing.get("iter_seconds"),
            "metrics": metrics,
            "report_path": report_meta.get("summary_path"),
        }

    def report(self, advanced_verbose: bool = False) -> Dict[str, Any]:
        payload = {
            "status": self.status(),
            "result": self.result(),
            "history": self.history(),
            "run": dict(self._run_context or {}),
        }
        if advanced_verbose:
            payload["advanced"] = self.export_all(pretty=False)
            if self._local_timing:
                payload["local_timing"] = dict(self._local_timing)
        return payload

    def export_all(self, pretty: bool = False) -> Union[Dict[str, Any], str]:
        result = self._remote_result or {}
        objectives = result.get("objectives") or {}
        best_metrics = result.get("best_metrics") or {}
        timing = result.get("timing") or {}
        comm = result.get("comm") or {}
        data = {
            "best_fitness": self.get_best_fitness(),
            "run": self._run_context or {},
            "metrics": {
                "objectives": objectives,
                "best_metrics": best_metrics,
            },
            "timing": {
                "iter_seconds": timing.get("iter_seconds"),
                "iter_avg_ms": timing.get("iter_avg_ms"),
                "throughput_ips": timing.get("throughput_ips"),
            },
            "comm": {
                "avg_ms": comm.get("avg_ms"),
                "per_rank_avg_ms": comm.get("per_rank_avg_ms") or comm.get("per_rank_ms"),
            },
            "raw_result": result,
        }
        if not pretty:
            return data
        return self._format_export_groups(data)

    def save_report(self, path: str, pretty: bool = False) -> str:
        """Save monitor report to JSON or TXT (if pretty=True)."""
        report = self.export_all(pretty=pretty)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if pretty:
            p.write_text(str(report), encoding="utf-8")
        else:
            if isinstance(report, str):
                p.write_text(report, encoding="utf-8")
            else:
                p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(p)

    def _format_export_groups(self, data: Dict[str, Any]) -> str:
        if self._remote_result is None and self._local_timing:
            timing = self._local_timing
            lines = []
            lines.append("=== Local Summary ===")
            lines.append(f"Best Fitness: {data.get('best_fitness')}")
            lines.append(f"Total Iter Time (s): {timing.get('iter_seconds')}")
            lines.append(f"Avg Iter (ms): {timing.get('iter_avg_ms')}")
            lines.append(f"Throughput (iters/s): {timing.get('throughput_ips')}")
            return "\n".join(lines)
        metrics = data.get("metrics") or {}
        objectives = metrics.get("objectives") or {}
        best_metrics = metrics.get("best_metrics") or {}
        timing = data.get("timing") or {}
        comm = data.get("comm") or {}
        lines = []
        lines.append("=== Metrics ===")
        lines.append(f"Best Fitness: {data.get('best_fitness')}")
        lines.append(f"Primary: {objectives.get('primary')}")
        lines.append(f"Secondary: {objectives.get('secondary')}")
        if isinstance(best_metrics, dict) and best_metrics:
            for k, v in best_metrics.items():
                lines.append(f"{k}: {v}")
        lines.append("")
        lines.append("=== Timing ===")
        lines.append(f"Total Iter Time (s): {timing.get('iter_seconds')}")
        lines.append(f"Avg Iter (ms): {timing.get('iter_avg_ms')}")
        lines.append(f"Throughput (ips): {timing.get('throughput_ips')}")
        lines.append("")
        lines.append("=== Comm ===")
        lines.append(f"Avg Comm (ms): {comm.get('avg_ms')}")
        per_rank = comm.get("per_rank_avg_ms")
        if isinstance(per_rank, dict) and per_rank:
            for k, v in per_rank.items():
                lines.append(f"rank {k}: {v} ms")
        return "\n".join(lines)


# =============================================================================
# Algorithm Base Class
# =============================================================================

class Algorithm(nn.Module, ABC):
    """
    Base class for evolutionary algorithms.
    
    Subclasses should implement the required methods to integrate with
    the existing GAPA infrastructure (Controller + Evaluator pattern).
    
    Two integration modes are supported:
    
    1. **Full Custom** (recommended for new algorithms):
       - Implement all abstract methods
       - Workflow creates internal Controller/Evaluator wrappers
    
    2. **Legacy Wrapper** (for existing algorithms):
       - Provide existing Controller/Evaluator instances via get_components()
    
    Example (Full Custom):
        >>> class MyAlgorithm(Algorithm):
        >>>     def create_evaluator(self, data_loader):
        >>>         return MyEvaluator(...)
        >>>     
        >>>     def create_controller(self, data_loader, mode, device):
        >>>         return MyController(...)
    """
    
    def __init__(self):
        super().__init__()
        self.device: torch.device = torch.device("cpu")
        self._data_loader: Optional[DataLoader] = None
    
    @abstractmethod
    def create_evaluator(self, data_loader: DataLoader) -> nn.Module:
        """
        Create the fitness evaluator for this algorithm.
        
        Args:
            data_loader: DataLoader with graph data
            
        Returns:
            A BasicEvaluator subclass instance
        """
        pass
    
    @abstractmethod
    def create_controller(
        self, 
        data_loader: DataLoader, 
        mode: str, 
        device: torch.device
    ) -> "CustomController":
        """
        Create the controller for this algorithm.
        
        Args:
            data_loader: DataLoader with graph data
            mode: Execution mode (s, sm, m, mnm)
            device: Computation device
            
        Returns:
            A CustomController subclass instance
        """
        pass
    
    def create_body(self, data_loader: DataLoader, device: torch.device) -> "Body":
        """
        Create the Body (GA operators) for this algorithm.
        
        Default implementation creates a standard Body.
        Override for custom operators.
        
        Args:
            data_loader: DataLoader with graph data
            device: Computation device
            
        Returns:
            A Body instance
        """
        from gapa.framework.body import Body
        
        # Get controller to access pop_size and budget
        # These are typically set by the Algorithm subclass
        pop_size = getattr(self, 'pop_size', 80)
        budget = getattr(self, 'budget', data_loader.k)
        side = getattr(self, 'side', 'min')
        
        return Body(
            critical_num=data_loader.nodes_num,
            budget=budget,
            pop_size=pop_size,
            fit_side=side,
            device=device,
        )


# =============================================================================
# Workflow (Unified Orchestrator)
# =============================================================================

class Workflow:
    """
    Unified workflow orchestrator for evolutionary algorithms.
    
    **Architecture**: Wraps the existing core engine (Start, CustomController, 
    BasicEvaluator) to ensure consistent behavior across script and frontend.
    
    All execution modes are supported through the same interface:
        - "s": Single process (CPU or GPU)
        - "sm": Single-node multi-GPU (DataParallel)
        - "m": Multi-process GPU (torch.distributed with NCCL)
        - "m_cpu": Maps to "s" mode (CPU single-process)
        - "mnm": Multi-node multi-GPU (DistributedEvaluator)
    
    Example:
        >>> workflow = Workflow(algo, data, mode="m")
        >>> workflow.run(1000)  # Just works!
        >>> print(workflow.monitor.best_fitness)
    """
    
    def __init__(
        self,
        algorithm: Algorithm,
        data_loader: DataLoader,
        monitor: Optional[Monitor] = None,
        mode: str = "s",
        workers: Optional[int] = None,
        auto_select: bool = False,
        servers: Optional[List[str]] = None,
        remote_server: Optional[str] = None,
        remote_devices: Optional[List[int]] = None,
        remote_use_strategy_plan: Optional[bool] = None,
        server_url: str = "",
        fallback_policy: str = "best_effort",
        verbose: bool = True,
    ):
        """
        Initialize the workflow.
        
        Args:
            algorithm: Algorithm instance (subclass of Algorithm)
            data_loader: DataLoader from DataLoader.load()
            monitor: Optional Monitor for tracking progress
            mode: Execution mode - "s", "sm", "m", "m_cpu", "mnm"
            workers: Number of workers for resource discovery (MNM mode)
            auto_select: Auto-select remote servers (for MNM mode)
            servers: List of remote server IDs (for MNM mode)
            remote_server: Single remote server id/name for s/sm/m remote execution
            remote_devices: Explicit remote GPU ids for s/sm/m remote execution
            remote_use_strategy_plan: Whether remote s/sm/m should use StrategyPlan device selection
            server_url: Local GAPA API URL (for MNM mode resource discovery)
            fallback_policy: "best_effort" or "strict" for mode fallback behavior
            verbose: Whether to print progress information
        """
        self.algorithm = algorithm
        self.data_loader = data_loader
        self.monitor = monitor if monitor is not None else Monitor()
        self.verbose = verbose
        self.requested_mode = mode
        self.fallback_policy = (fallback_policy or "best_effort").strip().lower()
        if self.fallback_policy not in ("best_effort", "strict"):
            raise ValueError("fallback_policy must be 'best_effort' or 'strict'")
        
        # Normalize mode
        if mode == "m_cpu":
            mode = "s"  # CPU parallel falls back to single process
            if verbose:
                print("[GAPA] m_cpu mode mapped to 's' mode for CPU execution")
        
        self.mode = mode
        self.workers = workers
        self.auto_select = auto_select
        self.servers = servers or []
        self.remote_server = remote_server
        self.remote_devices = list(remote_devices or [])
        self.remote_use_strategy_plan = remote_use_strategy_plan
        self._server_url_explicit = bool(server_url)
        self.server_url = (server_url or _resolve_default_api_base()).rstrip("/")

        # Resolve canonical algorithm id + capability contract (if registered)
        self.algorithm_id = self.algorithm.__class__.__name__
        self.algorithm_capabilities: Dict[str, Any] = {
            "supported_modes": ["s", "sm", "m", "mnm"],
            "supports_distributed_fitness": True,
            "supports_remote": True,
            "fitness_direction": getattr(self.algorithm, "side", "min"),
        }
        try:
            from server.algorithm_registry import resolve_algorithm_id, get_algorithm_capabilities
            self.algorithm_id = resolve_algorithm_id(self.algorithm_id)
            caps = get_algorithm_capabilities(self.algorithm_id)
            if isinstance(caps, dict):
                self.algorithm_capabilities.update(caps)
        except Exception:
            pass
        
        # Validate mode
        has_distributed = _has_distributed()
        valid_modes = SOURCE_MODES if has_distributed else PYPI_MODES
        if mode not in valid_modes:
            if mode == "mnm" and not has_distributed:
                raise ImportError(
                    "MNM mode requires source deployment with distributed components.\n\n"
                    "PyPI package (pip install gapa) only supports: s, sm, m modes.\n\n"
                    "For MNM mode:\n"
                    "  1. Clone source: git clone https://github.com/NetAlsGroup/GAPA\n"
                    "  2. Deploy with server/ and web/ directories"
                )
            raise ValueError(f"Invalid mode '{mode}'. Available modes: {valid_modes}")

        if self.remote_server and mode not in ("s", "sm", "m"):
            raise ValueError("remote_server only supports s/sm/m modes")

        supported_modes = self.algorithm_capabilities.get("supported_modes")
        if isinstance(supported_modes, list):
            supported_modes_lc = [str(m).strip().lower() for m in supported_modes]
            if mode not in supported_modes_lc:
                raise ValueError(
                    f"algorithm '{self.algorithm_id}' does not support mode '{mode}'. "
                    f"supported={supported_modes_lc}"
                )
        if mode == "mnm" and not bool(self.algorithm_capabilities.get("supports_distributed_fitness", True)):
            raise ValueError(f"algorithm '{self.algorithm_id}' does not support distributed fitness (mnm)")
        if self.remote_server and not bool(self.algorithm_capabilities.get("supports_remote", True)):
            raise ValueError(f"algorithm '{self.algorithm_id}' does not support remote execution")
        
        # =====================================================================
        # Cross-Platform Mode Adaptation
        # =====================================================================
        import platform
        system = platform.system()  # 'Darwin' (Mac), 'Windows', 'Linux'
        skip_platform_adapt = bool(self.remote_server and self.mode in ("s", "sm", "m"))

        def _fallback_to(target_mode: str, reason: str, *, tip: Optional[str] = None) -> None:
            if self.fallback_policy == "strict":
                raise RuntimeError(
                    f"mode '{mode}' is not runnable on current platform. reason={reason}. "
                    f"fallback_policy='strict' blocks fallback."
                )
            if verbose:
                print(f"[GAPA] Falling back '{mode}' -> '{target_mode}': {reason}")
                if tip:
                    print(f"[GAPA] Tip: {tip}")
            self.mode = target_mode
        
        # Select device and world_size
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.world_size = torch.cuda.device_count()
        else:
            self.device = torch.device("cpu")
            self.world_size = 1
        
        # Platform-specific mode handling
        if skip_platform_adapt:
            pass
        elif system == "Darwin":  # MacOS
            if mode == "sm":
                _fallback_to("s", "MacOS + no CUDA for SM mode")
            elif mode == "m":
                # MacOS: No CUDA/NCCL, fall back to S
                # For CPU multiprocessing, use StrategyPlan to evaluate if it's beneficial
                _fallback_to("s", "MacOS + no CUDA for M mode", tip="Use StrategyPlan to evaluate CPU multiprocessing")
        
        elif system == "Windows":
            if mode == "m":
                if torch.cuda.is_available():
                    # Windows with CUDA: use gloo backend
                    os.environ["GAPA_DIST_BACKEND"] = "gloo"
                    if verbose:
                        print("[GAPA] Windows: M mode using 'gloo' backend (NCCL not available).")
                else:
                    # Windows CPU-only: fall back to S
                    _fallback_to("s", "Windows + no CUDA for M mode", tip="Use StrategyPlan to evaluate CPU multiprocessing")
            elif mode == "sm" and not torch.cuda.is_available():
                _fallback_to("s", "Windows SM mode requires CUDA")
        
        elif system == "Linux":
            if mode == "m":
                if torch.cuda.is_available() and self.world_size >= 2:
                    # Linux with multi-GPU: use nccl
                    pass  # Default nccl is fine
                elif torch.cuda.is_available() and self.world_size == 1:
                    # Linux with single GPU: fall back to S (M mode needs >= 2 GPUs)
                    _fallback_to("s", "Linux M mode requires >=2 GPUs", tip="Use StrategyPlan to evaluate CPU multiprocessing")
                else:
                    # Linux CPU-only: fall back to S
                    _fallback_to("s", "Linux M mode requires CUDA", tip="Use StrategyPlan to evaluate CPU multiprocessing")
            elif mode == "sm" and not torch.cuda.is_available():
                _fallback_to("s", "Linux SM mode requires CUDA")
        
        # Final mode confirmation
        if verbose and self.mode != mode and not skip_platform_adapt:
            print(f"[GAPA] Mode adjusted: '{mode}' → '{self.mode}'")
        
        # Confirm actual execution mode
        if verbose and not skip_platform_adapt:
            mode_desc = {
                "s": "Single-process (CPU/GPU)",
                "sm": "DataParallel (multi-GPU)",
                "m": "Distributed (mp.spawn)",
                "mnm": "Multi-node distributed",
            }
            print(f"[GAPA] Execution mode: {mode_desc.get(self.mode, self.mode)} | Device: {self.device}")
        
        # For MNM mode, discover resources if needed
        if self.mode == "mnm" and not self.servers:
            self.servers = self._discover_resources()
            if not self.servers:
                raise RuntimeError(
                    "MNM mode requires at least one online remote server. "
                    "Provide `servers=[...]` or make sure remote agents are online."
                )
        
        # Internal state (created during run)
        self._controller = None
        self._evaluator = None
        self._body = None
        self._state = None  # For step-by-step iteration
        self._workflow_state = "idle"
        self._pause_requested = False
        self._executed_iterations = 0
        self.execution_contract = {
            "algorithm_id": self.algorithm_id,
            "requested_mode": self.requested_mode,
            "resolved_mode": self.mode,
            "fallback_policy": self.fallback_policy,
            "remote_server": self.remote_server,
            "remote_devices": list(self.remote_devices),
            "remote_use_strategy_plan": self.remote_use_strategy_plan,
            "servers": list(self.servers),
            "capabilities": self.algorithm_capabilities,
        }
        self.monitor.set_workflow_state("idle")

    def _set_workflow_state(self, state: str, message: str = "") -> None:
        self._workflow_state = str(state)
        self.monitor.set_workflow_state(state, message)

    def _make_run_context(self, steps: int) -> Dict[str, Any]:
        existing = self.monitor._run_context if isinstance(self.monitor._run_context, dict) else {}
        return {
            "run_id": str(existing.get("run_id") or uuid.uuid4()),
            "started_at": str(existing.get("started_at") or (datetime.utcnow().isoformat() + "Z")),
            "algorithm_id": self.algorithm_id,
            "requested_mode": self.requested_mode,
            "resolved_mode": self.mode,
            "fallback_policy": self.fallback_policy,
            "remote_server": self.remote_server,
            "servers": list(self.servers),
            "steps": int(steps),
            "execution_contract": self.execution_contract,
        }

    def _update_local_timing(self, elapsed_s: float, iterations_delta: int) -> None:
        previous = self.monitor._local_timing or {}
        total_elapsed = float(previous.get("iter_seconds") or 0.0) + max(0.0, float(elapsed_s))
        self._executed_iterations += max(0, int(iterations_delta))
        self.monitor._local_timing = {
            "iter_seconds": total_elapsed,
            "iter_avg_ms": (total_elapsed / max(1, self._executed_iterations)) * 1000.0,
            "throughput_ips": self._executed_iterations / total_elapsed if total_elapsed > 0 else None,
        }

    def _log_report_meta(self, report_meta: Dict[str, Any]) -> None:
        if not self.verbose or not isinstance(report_meta, dict) or not report_meta.get("enabled"):
            return
        print(f"[GAPA] Report saved: {report_meta.get('summary_path')}")
        bench = report_meta.get("benchmark") if isinstance(report_meta.get("benchmark"), dict) else {}
        if bench.get("regressed"):
            print(f"[WARN] Benchmark regression detected: ratio={bench.get('ratio'):.3f} threshold={bench.get('threshold'):.3f}")
    
    def _discover_resources(self) -> List[str]:
        """Discover available remote servers for MNM mode."""
        try:
            import requests

            session = requests.Session()
            session.trust_env = False
            available = []
            resp = session.get(f"{self.server_url}/api/v1/resources/all", timeout=10)
            if not resp.ok:
                resp = session.get(f"{self.server_url}/api/all_resources", timeout=10)
            if resp.ok:
                payload = resp.json()
                if isinstance(payload, dict):
                    for server_id, snap in payload.items():
                        if server_id == "local":
                            continue
                        if isinstance(snap, dict) and snap.get("online"):
                            available.append(server_id)
            
            if available and self.verbose:
                print(f"[GAPA] Discovered {len(available)} available server(s): {available}")
            return available
        except Exception as e:
            if self.verbose:
                print(f"[GAPA] Resource discovery failed: {e}")
            return []
    
    def _setup_components(self):
        """Create controller, evaluator, and body from algorithm."""
        # Create components from Algorithm
        self._evaluator = self.algorithm.create_evaluator(self.data_loader)
        self._controller = self.algorithm.create_controller(
            self.data_loader, 
            self.mode, 
            self.device
        )
        self._body = self.algorithm.create_body(self.data_loader, self.device)
        
        # For MNM mode, wrap evaluator with DistributedEvaluator
        if self.mode == "mnm" and self.servers:
            self._evaluator = self._wrap_for_mnm(self._evaluator)

    def _emit_run_reports(self, run_ctx: Dict[str, Any]) -> Dict[str, Any]:
        auto_report = str(os.getenv("GAPA_AUTO_REPORT", "1")).strip().lower() not in ("0", "false", "off")
        if not auto_report:
            return {"enabled": False}
        report = self.monitor.export_all(pretty=False)
        if not isinstance(report, dict):
            return {"enabled": True, "error": "invalid report"}
        run = report.get("run") if isinstance(report.get("run"), dict) else {}
        run_id = str(run.get("run_id") or run_ctx.get("run_id") or str(uuid.uuid4()))
        results_root = _results_root()
        runs_dir = results_root / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        full_path = runs_dir / f"{run_id}_full.json"
        summary_path = runs_dir / f"{run_id}_summary.json"
        self.monitor.save_report(str(full_path), pretty=False)

        timing = report.get("timing") if isinstance(report.get("timing"), dict) else {}
        comm = report.get("comm") if isinstance(report.get("comm"), dict) else {}
        summary = {
            "run_id": run_id,
            "algorithm_id": self.algorithm_id,
            "dataset": getattr(self.data_loader, "name", None) or getattr(self.data_loader, "dataset", None),
            "requested_mode": self.requested_mode,
            "resolved_mode": self.mode,
            "remote": bool(self.remote_server),
            "remote_server": self.remote_server,
            "servers": list(self.servers),
            "best_fitness": report.get("best_fitness"),
            "iter_seconds": timing.get("iter_seconds"),
            "iter_avg_ms": timing.get("iter_avg_ms"),
            "throughput_ips": timing.get("throughput_ips"),
            "comm_avg_ms": comm.get("avg_ms"),
            "started_at": run_ctx.get("started_at"),
            "ended_at": run_ctx.get("ended_at"),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        jsonl_path = results_root / "run_reports.jsonl"
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

        threshold = float(os.getenv("GAPA_BENCHMARK_REGRESSION", "0.25") or 0.25)
        benchmark = _compare_and_update_benchmark(
            algorithm_id=self.algorithm_id,
            dataset=str(summary.get("dataset") or ""),
            mode=self.mode,
            remote=bool(self.remote_server),
            iter_avg_ms=summary.get("iter_avg_ms"),
            iter_seconds=summary.get("iter_seconds"),
            regression_threshold=threshold,
        )
        summary["benchmark"] = benchmark
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"run_id": run_id, "benchmark": benchmark}, ensure_ascii=False) + "\n")

        return {
            "enabled": True,
            "run_id": run_id,
            "full_path": str(full_path),
            "summary_path": str(summary_path),
            "jsonl_path": str(jsonl_path),
            "benchmark": benchmark,
        }
    
    def _wrap_for_mnm(self, evaluator) -> nn.Module:
        """Wrap evaluator for distributed execution."""
        DistributedEvaluator = _load_distributed_evaluator()
        if DistributedEvaluator is None:
            raise ImportError(
                "MNM mode requires distributed components, but DistributedEvaluator "
                "could not be imported. Run from the source tree or install the "
                "distributed package components."
            )

        wrapped = DistributedEvaluator(
            evaluator,
            algorithm=self.algorithm_id,
            dataset=self.data_loader.name,
            allowed_server_ids=self.servers,
        )
        if self.verbose:
            print(f"[GAPA] MNM mode: {len(self.servers)} remote server(s)")
        return wrapped
    
    def _run_full(self, steps: int) -> None:
        run_ctx = self._make_run_context(steps)
        self.monitor.set_run_context(run_ctx)
        self._set_workflow_state("running")
        try:
            if self.remote_server:
                from gapa.remote_runner import run_remote_task
                from gapa.resource_manager import ResourceManager
                dataset_name = getattr(self.data_loader, "name", None) or getattr(self.data_loader, "dataset", None) or ""
                resource_manager = ResourceManager(api_base=self.server_url if self._server_url_explicit else None)
                result = run_remote_task(
                    resource_manager,
                    self.remote_server,
                    algorithm=self.algorithm_id,
                    dataset=dataset_name,
                    iterations=steps,
                    mode=self.mode,
                    crossover_rate=float(getattr(self.algorithm, "crossover_rate", 0.8)),
                    mutate_rate=float(getattr(self.algorithm, "mutate_rate", 0.2)),
                    pop_size=int(getattr(self.algorithm, "pop_size", 100)) if hasattr(self.algorithm, "pop_size") else None,
                    devices=list(self.remote_devices) if self.remote_devices else None,
                    use_strategy_plan=self.remote_use_strategy_plan,
                    start_timeout_s=max(20.0, float(os.getenv("GAPA_REMOTE_START_TIMEOUT", "120") or 120)),
                )
                if isinstance(result, dict) and result.get("error"):
                    raise RuntimeError(f"remote run failed: {result}")
                remote_result = result.get("result") if isinstance(result.get("result"), dict) else None
                if isinstance(remote_result, dict):
                    best = _resolve_remote_primary_score(remote_result)
                    if isinstance(best, (int, float)):
                        self.monitor._best_fitness = float(best)
                    best_gene = remote_result.get("best_gene")
                    if best_gene is not None:
                        try:
                            self.monitor._best_solution = torch.as_tensor(best_gene)
                        except Exception:
                            self.monitor._best_solution = None
                    points = remote_result.get("points")
                    if isinstance(points, list) and points:
                        last_iter = None
                        for item in points:
                            if isinstance(item, dict) and isinstance(item.get("iter"), int):
                                last_iter = int(item["iter"])
                        if last_iter is not None:
                            self.monitor._generation = last_iter
                        self.monitor._extra_history = [item for item in points if isinstance(item, dict)]
                    elif isinstance(remote_result.get("hyperparams"), dict):
                        hp_iters = remote_result["hyperparams"].get("iterations")
                        if isinstance(hp_iters, int):
                            self.monitor._generation = hp_iters
                    objectives = remote_result.get("objectives") if isinstance(remote_result.get("objectives"), dict) else {}
                    primary = objectives.get("primary")
                    curves = remote_result.get("curves") if isinstance(remote_result.get("curves"), dict) else {}
                    if primary and isinstance(curves.get(primary), list):
                        self.monitor._fitness_history = [float(v) for v in curves.get(primary) if isinstance(v, (int, float))]
                    self.monitor._remote_result = remote_result
                run_ctx["ended_at"] = datetime.utcnow().isoformat() + "Z"
                report_meta = self._emit_run_reports(run_ctx)
                run_ctx["reports"] = report_meta
                self.monitor.set_run_context(run_ctx)
                self._set_workflow_state("completed")
                self._log_report_meta(report_meta)
                return
            if callable(getattr(self.algorithm, "run_full", None)):
                if self.verbose:
                    print(f"[GAPA] Starting {self.algorithm.__class__.__name__} in '{self.mode}' mode")
                    print(f"[GAPA] Generations: {steps}, Device: {self.device}")

                import time

                start_ts = time.perf_counter()
                self.algorithm.run_full(self, int(steps))
                end_ts = time.perf_counter()
                self._update_local_timing(end_ts - start_ts, steps)
                run_ctx["ended_at"] = datetime.utcnow().isoformat() + "Z"
                report_meta = self._emit_run_reports(run_ctx)
                run_ctx["reports"] = report_meta
                self.monitor.set_run_context(run_ctx)
                self._set_workflow_state("completed")
                self._log_report_meta(report_meta)
                if self.verbose:
                    print(f"\n[GAPA] Evolution complete. Best fitness: {self.monitor.best_fitness}")
                return
            from gapa.framework.controller import Start

            self._setup_components()
            if self.verbose:
                print(f"[GAPA] Starting {self.algorithm.__class__.__name__} in '{self.mode}' mode")
                print(f"[GAPA] Generations: {steps}, Device: {self.device}")

            import time
            start_ts = time.perf_counter()
            Start(
                max_generation=steps,
                data_loader=self.data_loader,
                controller=self._controller,
                evaluator=self._evaluator,
                body=self._body,
                world_size=self.world_size,
                verbose=self.verbose,
                observer=self.monitor,
            )
            end_ts = time.perf_counter()
            self._update_local_timing(end_ts - start_ts, steps)
            run_ctx["ended_at"] = datetime.utcnow().isoformat() + "Z"
            report_meta = self._emit_run_reports(run_ctx)
            run_ctx["reports"] = report_meta
            self.monitor.set_run_context(run_ctx)
            self._set_workflow_state("completed")
            self._log_report_meta(report_meta)
            if self.verbose:
                print(f"\n[GAPA] Evolution complete. Best fitness: {self.monitor.best_fitness}")
        except Exception as exc:
            self._set_workflow_state("error", str(exc))
            raise

    def _supports_incremental(self) -> bool:
        return bool(getattr(self.algorithm, "supports_incremental", False))

    def _run_incremental(self, steps: int) -> None:
        if steps <= 0:
            raise ValueError("steps must be a positive integer")
        if self._state is None:
            self.init_step()
        elif self.verbose:
            print("[GAPA] Resuming from current state; call reset() to restart.")

        import time
        run_ctx = self._make_run_context(steps)
        self.monitor.set_run_context(run_ctx)
        self._pause_requested = False
        self._set_workflow_state("running")
        start_ts = time.perf_counter()
        executed = 0
        try:
            for _ in range(int(steps)):
                if self._pause_requested:
                    self._set_workflow_state("paused", "pause requested")
                    break
                self.step()
                executed += 1
        except Exception as exc:
            self._set_workflow_state("error", str(exc))
            raise
        finally:
            end_ts = time.perf_counter()
            self._update_local_timing(end_ts - start_ts, executed)
            run_ctx["ended_at"] = datetime.utcnow().isoformat() + "Z"
            report_meta = self._emit_run_reports(run_ctx)
            run_ctx["reports"] = report_meta
            self.monitor.set_run_context(run_ctx)
            if self._workflow_state != "paused":
                self._set_workflow_state("completed")
            self._log_report_meta(report_meta)

    def run(self, steps: int) -> None:
        """
        Run the full evolution loop.
        
        This is the recommended method for all modes. It uses the existing
        Start() function to ensure consistent behavior with frontend execution.
        
        Args:
            steps: Number of generations to run
        """
        if self.mode == "s" and not self.remote_server and self._supports_incremental():
            self._run_incremental(int(steps))
            return
        self._run_full(int(steps))

    def pause(self) -> None:
        self._pause_requested = True
        if self._state is not None and self._workflow_state != "running":
            self._set_workflow_state("paused", "pause requested")

    def resume(self, steps: int) -> None:
        if self.mode != "s" or self.remote_server:
            raise RuntimeError("resume() is currently supported for local 's' mode workflows only")
        if not self._supports_incremental():
            raise RuntimeError(f"resume() is not available for algorithm '{self.algorithm.__class__.__name__}'")
        if self._state is None:
            raise RuntimeError("no existing workflow state to resume; call run() first")
        if self.verbose:
            print("[GAPA] Continuing from current state.")
        self._run_incremental(int(steps))

    def reset(self) -> None:
        self._controller = None
        self._evaluator = None
        self._body = None
        self._state = None
        self._pause_requested = False
        self._executed_iterations = 0
        self.monitor.reset()
        self._set_workflow_state("idle")
    
    # =========================================================================
    # Step-by-Step Iteration Interface
    # =========================================================================
    
    def init_step(self) -> None:
        """
        Initialize the first generation for step-by-step iteration.
        
        Call this before using step() for manual iteration control.
        
        Example:
            >>> workflow.init_step()
            >>> for i in range(100):
            >>>     result = workflow.step()
            >>>     print(f"Gen {result['generation']}: {result['best_fitness']}")
        """
        # Setup components if not done
        if self._controller is None:
            self._setup_components()
        
        # Call controller setup (applies cutoff, etc.)
        self._evaluator = self._controller.setup(
            data_loader=self.data_loader, 
            evaluator=self._evaluator
        )
        
        # For MNM mode, wrap evaluator
        if self.mode == "mnm" and self.servers and not hasattr(self._evaluator, '_is_distributed'):
            self._evaluator = self._wrap_for_mnm(self._evaluator)
            self._evaluator._is_distributed = True
        
        # Initialize state
        self._state = self._controller.init_state(self._evaluator, self._body)
        
        # Update monitor with initial fitness
        if self._state:
            self.monitor.update(
                self._state["population"],
                self._state["fitness_list"],
            )
        self._set_workflow_state("idle", "initialized")
        
        if self.verbose:
            print(f"[GAPA] Initialized. Ready for step-by-step iteration.")
    
    def step(self) -> Dict[str, Any]:
        """
        Execute a single generation and return results.
        
        Must call init_step() first.
        
        Returns:
            Dict containing:
                - generation: Current generation number
                - best_fitness: Best fitness in current generation
                - best_gene: Best solution tensor
                - metrics: Additional metrics (e.g., PCG)
        
        Example:
            >>> workflow.init_step()
            >>> for i in range(100):
            >>>     result = workflow.step()
            >>>     if result['best_fitness'] < threshold:
            >>>         break
        """
        if self._state is None:
            raise RuntimeError("Call init_step() before step()")
        
        # Execute one generation
        self._set_workflow_state("running")
        self._state = self._controller.single_step(
            self._state,
            self._evaluator,
            self._body,
            observer=self.monitor,
        )
        
        return {
            "generation": self._state["generation"],
            "best_fitness": self._state["best_fitness"],
            "best_gene": self._state["best_gene"],
            "metrics": self._state.get("metrics", {}),
        }
    
    def run_steps(self, num_steps: int, verbose: bool = None) -> None:
        """
        Run a specified number of generations from current state.
        
        Can be called multiple times to continue iteration.
        
        Args:
            num_steps: Number of generations to run
            verbose: Override verbose setting (optional)
        
        Example:
            >>> workflow.init_step()
            >>> workflow.run_steps(500)  # Run 500 generations
            >>> # Check results, adjust parameters...
            >>> workflow.run_steps(500)  # Continue for 500 more
        """
        if verbose is None:
            verbose = self.verbose
        
        if self._state is None:
            self.init_step()
        
        from tqdm import tqdm
        
        start_gen = self._state["generation"]
        iterator = range(num_steps)
        
        if verbose:
            iterator = tqdm(iterator, desc=f"[GAPA] Gen {start_gen} →")
        
        for i in iterator:
            result = self.step()
            if verbose and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix(
                    gen=result['generation'],
                    fitness=f"{result['best_fitness']:.2f}"
                )
        
        if verbose:
            total_gen = self._state["generation"]
            print(f"\n[GAPA] Completed {num_steps} steps (total: {total_gen}). Best: {self.monitor.best_fitness:.4f}")
    
    def get_state(self) -> Optional[Dict]:
        """
        Get current iteration state for serialization/persistence.
        
        Can be used to save state and resume later.
        
        Returns:
            Current state dict or None if not initialized
        """
        return self._state
    
    def set_state(self, state: Dict) -> None:
        """
        Restore iteration state (for resume functionality).
        
        Args:
            state: Previously saved state dict
        """
        self._state = state
    
    def get_result(self) -> Dict[str, Any]:
        """
        Get final results from current iteration.
        
        Returns:
            Dict with best gene, fitness, and statistics
        """
        if self._state is None:
            return {"error": "No iterations completed"}
        return self._controller.get_final_result(self._state)
