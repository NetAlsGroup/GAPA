from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

from .fitness_protocol import dumps, loads


def _require_requests():
    if requests is None:  # pragma: no cover
        raise RuntimeError("requests is required for distributed evaluator")
    return requests


def _require_torch():
    if torch is None or nn is None:  # pragma: no cover
        raise RuntimeError("torch is required for distributed evaluator")
    return torch, nn


@dataclass(frozen=True)
class Worker:
    server_id: str
    base_url: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _servers_file() -> Path:
    return Path(os.getenv("GAPA_SERVERS_FILE", str(_repo_root() / "servers.json")))


def load_workers(exclude_local: bool = True, allowed_ids: Optional[List[str]] = None) -> List[Worker]:
    path = _servers_file()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(raw, dict):
        raw = raw.get("servers", [])
    if not isinstance(raw, list):
        return []
    out: List[Worker] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        sid = entry.get("id") or entry.get("name") or "remote"
        if exclude_local and sid == "local":
            continue
        if allowed_ids is not None and sid not in allowed_ids:
            continue
        base_url = entry.get("base_url") or ""
        if not base_url:
            host = entry.get("ip") or entry.get("host")
            port = entry.get("port")
            protocol = entry.get("protocol") or "http"
            if host:
                base_url = f"{protocol}://{host}{f':{port}' if port else ''}"
        if not base_url:
            continue
        out.append(Worker(server_id=str(sid), base_url=str(base_url).rstrip("/")))
    return out


class AdaptiveWorkerPool:
    def __init__(
        self,
        workers: List[Worker],
        *,
        refresh_interval_s: float = 2.0,
        max_workers: int = 4,
        cpu_busy_threshold: float = 85.0,
        gpu_busy_threshold: float = 85.0,
        min_gpu_free_mb: int = 512,
        request_timeout_s: float = 30.0,
        use_strategy_plan: bool = True,
    ) -> None:
        self._workers = workers
        self._refresh_interval_s = float(refresh_interval_s)
        self._max_workers = int(max_workers)
        self._cpu_busy_threshold = float(cpu_busy_threshold)
        self._gpu_busy_threshold = float(gpu_busy_threshold)
        self._min_gpu_free_mb = int(min_gpu_free_mb)
        self._request_timeout_s = float(request_timeout_s)
        self._use_strategy_plan = bool(use_strategy_plan)

        self._last_refresh = 0.0
        self._scores: Dict[str, float] = {}
        self._ok: Dict[str, bool] = {}
        self._backend: Dict[str, str] = {}

        req = _require_requests()
        self._session = req.Session()
        self._session.trust_env = False

    def _score_from_snapshot(self, snap: Dict[str, Any]) -> float:
        cpu = snap.get("cpu") or {}
        cpu_usage = cpu.get("usage_percent")
        if cpu_usage is None:
            cpu_usage = 50.0
        cpu_ok = float(cpu_usage) <= self._cpu_busy_threshold

        gpus = snap.get("gpus") or []
        gpu_free = 0.0
        gpu_ok = True
        if isinstance(gpus, list) and gpus:
            for g in gpus:
                try:
                    free_mb = float(g.get("free_mb")) if g.get("free_mb") is not None else 0.0
                    gpu_free += max(0.0, free_mb)
                    util = g.get("gpu_util_percent")
                    if util is not None and float(util) > self._gpu_busy_threshold:
                        gpu_ok = False
                except Exception:
                    continue
            if gpu_free < float(self._min_gpu_free_mb):
                gpu_ok = False
        # CPU-only servers are allowed but get lower score.
        base = (gpu_free / 1024.0) if gpu_free > 0 else 0.5
        if not cpu_ok or not gpu_ok:
            base *= 0.1
        return max(base, 0.01)

    def refresh(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_refresh) < self._refresh_interval_s:
            return
        self._last_refresh = now

        scores: Dict[str, float] = {}
        ok: Dict[str, bool] = {}
        backend: Dict[str, str] = {}
        for w in self._workers:
            try:
                resp = self._session.get(w.base_url + "/api/resources", timeout=(2.0, 5.0))
                snap = resp.json() if resp.ok else {"error": f"HTTP {resp.status_code}"}
                if not isinstance(snap, dict) or snap.get("error"):
                    ok[w.server_id] = False
                    scores[w.server_id] = 0.01
                    continue
                base_score = self._score_from_snapshot(snap)
                if self._use_strategy_plan:
                    try:
                        pr = self._session.post(
                            w.base_url + "/api/strategy_plan",
                            json={"warmup": 0, "objective": "time", "multi_gpu": True},
                            timeout=(3.0, 15.0),
                        )
                        plan = pr.json() if pr.ok else {}
                        b = str(plan.get("backend") or plan.get("plan", {}).get("backend") or "").lower()
                        if b:
                            backend[w.server_id] = b
                            if b == "multi-gpu":
                                base_score *= 1.20
                            elif b == "cuda":
                                base_score *= 1.00
                            elif b == "cpu":
                                base_score *= 0.50
                    except Exception:
                        pass
                scores[w.server_id] = base_score
                ok[w.server_id] = True
            except Exception:
                ok[w.server_id] = False
                scores[w.server_id] = 0.01
        self._scores = scores
        self._ok = ok
        self._backend = backend

    def pick(self) -> List[Tuple[Worker, float]]:
        self.refresh()
        ranked = sorted(self._workers, key=lambda w: float(self._scores.get(w.server_id, 0.01)), reverse=True)
        picked: List[Tuple[Worker, float]] = []
        for w in ranked:
            if not self._ok.get(w.server_id, False):
                continue
            picked.append((w, float(self._scores.get(w.server_id, 0.01))))
            if len(picked) >= self._max_workers:
                break
        return picked

    def remote_fitness(self, worker: Worker, *, algorithm: str, dataset: str, population_cpu: Any) -> Any:
        payload = dumps({"algorithm": algorithm, "dataset": dataset, "population": population_cpu})
        resp = self._session.post(
            worker.base_url + "/api/fitness/batch",
            data=payload,
            headers={"Content-Type": "application/octet-stream"},
            timeout=(3.0, self._request_timeout_s),
        )
        if not resp.ok:
            raise RuntimeError(f"remote fitness failed: {worker.server_id} HTTP {resp.status_code}")
        data = loads(resp.content)
        fit = data.get("fitness")
        if fit is None:
            raise RuntimeError(f"remote fitness bad response: {worker.server_id}")
        return fit


class DistributedEvaluator(nn.Module):
    """Evaluator proxy that offloads fitness computation to other servers."""

    def __init__(
        self,
        base_evaluator: nn.Module,
        *,
        algorithm: str,
        dataset: str,
        workers: Optional[List[Worker]] = None,
        allowed_server_ids: Optional[List[str]] = None,
        max_remote_workers: int = 4,
        refresh_interval_s: float = 2.0,
        use_strategy_plan: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self._base = base_evaluator
        self.algorithm = algorithm
        self.dataset = dataset
        if use_strategy_plan is None:
            use_strategy_plan = bool(int(os.getenv("GAPA_MNM_USE_PLAN", "0") or 0))
        self._pool = AdaptiveWorkerPool(
            workers or load_workers(exclude_local=True, allowed_ids=allowed_server_ids),
            refresh_interval_s=refresh_interval_s,
            max_workers=max_remote_workers,
            use_strategy_plan=bool(use_strategy_plan),
        )
        self._comm_total_ms = 0.0
        self._comm_calls = 0

    # ---- attribute forwarding for controller.setup() mutations ----
    def __getattr__(self, name: str) -> Any:  # pragma: no cover (delegation)
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Safely check if _base exists and has the attribute
            base = getattr(self, "_base", None)
            if base is not None:
                return getattr(base, name)
            raise

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover (delegation)
        if name in {"_base", "algorithm", "dataset", "_pool"} or name.startswith("_"):
            return super().__setattr__(name, value)
        
        base = getattr(self, "_base", None)
        if base is not None and hasattr(base, name):
            setattr(base, name, value)
            return
        super().__setattr__(name, value)

    def forward(self, population: Any) -> Any:
        torch, _nn = _require_torch()
        if not isinstance(population, torch.Tensor):
            return self._base(population)

        n = int(population.shape[0])
        if n <= 0:
            return self._base(population)

        picked = self._pool.pick()
        # If no remote workers, fall back to local.
        if not picked:
            return self._base(population)
        if len(picked) > n:
            picked = picked[:n]

        # Split population by weights (keep order).
        weights = [max(1e-6, float(w)) for _wk, w in picked]
        total = sum(weights) or 1.0
        sizes = [max(1, int(n * (w / total))) for w in weights]
        # Distribute remaining items to highest-weight workers.
        remainder = n - sum(sizes)
        if remainder > 0:
            order = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
            for i in order:
                if remainder <= 0:
                    break
                sizes[i] += 1
                remainder -= 1
        # If we overshot due to min=1, trim from lowest-weight workers.
        overshoot = sum(sizes) - n
        if overshoot > 0:
            order = sorted(range(len(weights)), key=lambda i: weights[i])
            for i in order:
                if overshoot <= 0:
                    break
                take = min(overshoot, max(0, sizes[i] - 1))
                sizes[i] -= take
                overshoot -= take
        # Final guard: ensure sum matches n (should be true).
        if sum(sizes) != n:
            sizes = [n] + [0] * (len(sizes) - 1)

        chunks = [c for c in torch.split(population, sizes, dim=0) if int(c.shape[0]) > 0]
        fits: List[torch.Tensor] = []
        for (worker, _w), pop_chunk in zip(picked, chunks):
            try:
                pop_cpu = pop_chunk.detach().to("cpu")
                t0 = time.perf_counter()
                fit_cpu = self._pool.remote_fitness(worker, algorithm=self.algorithm, dataset=self.dataset, population_cpu=pop_cpu)
                self._comm_total_ms += (time.perf_counter() - t0) * 1000.0
                self._comm_calls += 1
                if not isinstance(fit_cpu, torch.Tensor):
                    fit_cpu = torch.tensor(fit_cpu)
                fits.append(fit_cpu.to(population.device))
            except Exception:
                # Fallback to local compute for that chunk.
                fits.append(self._base(pop_chunk))

        return torch.cat(fits, dim=0)

    def comm_stats(self) -> Dict[str, float]:
        avg_ms = (self._comm_total_ms / self._comm_calls) if self._comm_calls else 0.0
        return {
            "type": "http",
            "avg_ms": float(avg_ms),
            "total_ms": float(self._comm_total_ms),
            "calls": int(self._comm_calls),
        }
