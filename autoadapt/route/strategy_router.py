#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Router (clean v2)
--------------------------
- 清晰分层：画像提取 → 候选生成 → 估时模型 → warm‑up 实测 → 口径统一的 Plan
- 支持动态挑选 world_size：在 warm‑up 阶段按 {1,2,4,...,N} 档位枚举多卡候选
- 多卡吞吐采用“并行求和 × 效率折减”，避免被加权平均高估
- 防坑：CPU 0ms 误判保护、warm‑up 0 值保护、轻量校准（measured/predicted）

用法示例（伪代码）
----------------
from evoflow.evoflow.route.perf_profiler import PerformanceProfiler
from evoflow.evoflow.route.strategy_router import StrategyRouter
# wl 需要具备: n_nodes, n_edges, steps, batch_individuals
prof = PerformanceProfiler(quick=True).profile()
router = StrategyRouter(prof)

def run_warmup(plan, iters:int) -> float:
    # 在此根据 plan.backend/devices/world_size 设置环境并做小样本实测
    # 返回毫秒 float
    ...

plan = router.choose_and_warmup(wl, executor=run_warmup,
                                objective='time', multi_gpu=True,
                                power_cap_w=None, warmup_iters=2)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Callable, Any

try:
    import torch
except Exception:
    torch = None

# 统一的方案对象来自 api.schemas（字段：backend, devices, allocation, world_size,
# estimated_time_ms, estimated_energy_j, reason, notes）
from ..api.schemas import Plan


# ---------- module helpers ----------
def _avail_gpus() -> List[int]:
    """返回当前可用 GPU 索引列表（无 CUDA 时为空）"""
    if torch is None:
        return []
    try:
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))
    except Exception:
        return []


def _pow2_schedule(n: int) -> List[int]:
    """生成 [1,2,4,...,2^k]；若 n 不是 2 的幂，结尾补 n"""
    ks: List[int] = []
    p = 1
    while p < n:
        ks.append(p)
        p *= 2
    if n > 0 and (not ks or ks[-1] != n):
        ks.append(n)
    return ks


# ---------- router ----------
class StrategyRouter:
    def __init__(self, profile: Any, *, available_gpus: Optional[List[int]] = None):
        self.p = profile  # 性能画像（CPU/GPU geps、功率、名称等）
        self.available_gpus = (
            sorted({int(i) for i in available_gpus}) if available_gpus else None
        )

        # 建模参数（可按数据微调）
        self.kernel_overhead_ms_small = 0.20  # 小图每步固定开销
        self.kernel_overhead_ms_large = 0.05  # 大图每步固定开销
        self.cpu_ga_overhead_ms = 0.10        # GA 调度开销（ms/千个体）
        self.hysteresis = 0.10                # 切换滞后（10%）

        # 轻量校准：key -> measured/predicted 的指数滑动平均
        self.calib: Dict[str, float] = {}

    # ======== Public APIs ========
    def route(self, wl, *, objective: str = 'time', multi_gpu: bool = True,
              power_cap_w: Optional[float] = None) -> Plan:
        """仅用静态模型挑选方案（不做 warm‑up）"""
        candidates = self._candidates(wl, objective, multi_gpu, power_cap_w)

        # 施加已知校准（measured/pred）
        fixed: List[Tuple[str, Plan]] = []
        for tag, plan in candidates:
            k = self._key(plan)
            scale = self.calib.get(k, 1.0)
            plan.estimated_time_ms = max(1e-6, plan.estimated_time_ms) * scale
            fixed.append((tag, plan))

        return self._pick_by_objective(fixed, objective)

    def choose_and_warmup(self, wl, executor: Callable[[Plan, int], float], *,
                          objective: str = 'time', multi_gpu: bool = True,
                          power_cap_w: Optional[float] = None,
                          warmup_iters: int = 2, max_candidates: int = 6) -> Plan:
        """
        动态 world_size：基于静态估时预选少量候选（CPU/单卡/多卡{1,2,4,...,N}），
        对 shortlist 做 warm‑up 实测，返回最快方案。
        """
        # 画像（geps）
        cpu_geps = self._cpu_geps()
        g_infos = self._gpu_infos()
        gpu_map = {g['index']: g for g in g_infos}
        gpus = _avail_gpus()
        if self.available_gpus is not None:
            allowed = set(self.available_gpus)
            gpus = [i for i in gpus if i in allowed]

        # 候选集
        cand: List[Plan] = []

        # --- CPU 候选 ---
        t_cpu = self._estimate_time_cpu(wl, cpu_geps)
        cand.append(self._plan('cpu', [], {}, 1, t_cpu, self._energy_cpu(t_cpu),
                               reason=f"CPU GEps≈{cpu_geps:.2f}"))

        # --- 单卡候选 ---
        for gi in gpus:
            geps = float(gpu_map.get(gi, {}).get('geps', 1.0))
            t = self._estimate_time_gpu_single(wl, geps)
            cand.append(self._plan('cuda', [gi], {gi: 1.0}, 1, t,
                                   self._energy_gpu([gpu_map.get(gi, {})], t),
                                   reason=f"GPU[{gi}] GEps≈{geps:.2f}"))

        # --- 多卡候选（最强前 k 张，k∈{2,4,...,N}） ---
        if multi_gpu and len(gpus) >= 2:
            ranked = sorted(gpus, key=lambda i: float(gpu_map.get(i, {}).get('geps', 1.0)), reverse=True)
            for k in _pow2_schedule(len(ranked)):
                idxs = ranked[:k]
                geps_eff, alloc = self._compose_multi_geps(gpu_map, idxs)
                t = self._estimate_time_multi_gpu(wl, geps_eff)
                cand.append(self._plan('multi-gpu', idxs, alloc, len(idxs), t,
                                       self._energy_gpu([gpu_map[i] for i in idxs], t),
                                       reason=f"MGPU{idxs} GEps_eff≈{geps_eff:.2f}"))

        # 预选：按预测时间取前 max_candidates
        cand.sort(key=lambda p: p.estimated_time_ms)
        shortlist = cand[:max_candidates] if len(cand) > max_candidates else cand

        # warm‑up 实测
        best, best_ms = None, float('inf')
        for plan in shortlist:
            try:
                m = float(executor(plan, warmup_iters))
            except Exception:
                continue  # 某些方案（尤其多卡）可能起不来，跳过
            m = max(m, 1e-3)  # 0 值保护
            if m < best_ms:
                best, best_ms = plan, m

        # 结果处理
        if best is None:
            best = shortlist[0]  # 回退到预测最快
            best.reason += " | warmup=skipped"
        else:
            # 用实测回填估时值，且更新校准
            best.estimated_time_ms = max(1e-3, min(best.estimated_time_ms, best_ms))
            best.reason += f" | warmup={best_ms:.3f}ms"
            self._update_calib(best, best_ms)

        # CPU=0ms 误判保护：若机器有 GPU 且 CPU 近似 0ms，则退到最快单卡
        if best.backend == 'cpu' and best.estimated_time_ms <= 1e-6 and any(p.backend == 'cuda' for p in shortlist):
            alt = min((p for p in shortlist if p.backend == 'cuda'), key=lambda x: x.estimated_time_ms, default=None)
            if alt is not None:
                alt.reason += " | guard->cuda"
                best = alt

        return best

    def candidate_plans(self, wl, *, objective: str = "time", multi_gpu: bool = True) -> List[Plan]:
        """Generate candidate plans for external optimizers (e.g., TPE)."""
        cpu_geps = self._cpu_geps()
        g_infos = self._gpu_infos()
        gpu_map = {g['index']: g for g in g_infos}
        gpus = _avail_gpus()
        if self.available_gpus is not None:
            allowed = set(self.available_gpus)
            gpus = [i for i in gpus if i in allowed]

        candidates: List[Plan] = []
        t_cpu = self._estimate_time_cpu(wl, cpu_geps)
        candidates.append(self._plan('cpu', [], {}, 1, t_cpu, self._energy_cpu(t_cpu),
                                     reason=f"CPU GEps≈{cpu_geps:.2f}"))

        for gi in gpus:
            geps = float(gpu_map.get(gi, {}).get('geps', 1.0))
            t = self._estimate_time_gpu_single(wl, geps)
            candidates.append(self._plan('cuda', [gi], {gi: 1.0}, 1, t,
                                         self._energy_gpu([gpu_map.get(gi, {})], t),
                                         reason=f"GPU[{gi}] GEps≈{geps:.2f}"))

        if multi_gpu and len(gpus) >= 2:
            ranked = sorted(gpus, key=lambda i: float(gpu_map.get(i, {}).get('geps', 1.0)), reverse=True)
            for k in _pow2_schedule(len(ranked)):
                idxs = ranked[:k]
                geps_eff, alloc = self._compose_multi_geps(gpu_map, idxs)
                t = self._estimate_time_multi_gpu(wl, geps_eff)
                candidates.append(self._plan('multi-gpu', idxs, alloc, len(idxs), t,
                                             self._energy_gpu([gpu_map[i] for i in idxs], t),
                                             reason=f"MGPU{idxs} GEps_eff≈{geps_eff:.2f}"))

        return candidates

    # ======== Candidate generation ========
    def _candidates(self, wl, objective: str, multi_gpu: bool, power_cap_w: Optional[float]):
        out: List[Tuple[str, Plan]] = []
        cpu_geps = self._cpu_geps()
        g_infos = self._gpu_infos()

        # CPU
        t_cpu = self._estimate_time_cpu(wl, cpu_geps)
        out.append(('cpu', self._plan(
            'cpu', [], {}, 1, t_cpu, self._energy_cpu(t_cpu),
            reason=f'CPU GEps≈{cpu_geps:.2f}'
        )))

        # Best single GPU
        if g_infos:
            best_idx, t_gpu = self._estimate_time_best_gpu(wl, g_infos)
            if best_idx is not None:
                g = g_infos[best_idx]
                out.append(('gpu', self._plan(
                    'cuda', [g['index']], {g['index']: 1.0}, 1, t_gpu,
                    self._energy_gpu([g], t_gpu),
                    reason=f"GPU {g['name']} GEps≈{g['geps']:.2f}"
                )))

        # Multi-GPU
        if multi_gpu and g_infos:
            plan_multi = self._estimate_time_multi_gpu_plan(wl, g_infos, objective, power_cap_w)
            out.append(('multi', plan_multi))

        return out

    # ======== Modeling ========
    def _cpu_geps(self) -> float:
        # 优先用画像中的稀疏算子 GE/s；否则退化为 STREAM 推导
        ges = getattr(getattr(self.p, 'cpu', None), 'spmv_ges', None)
        if ges is not None:
            return max(1e-6, float(ges))
        gbps = float(getattr(getattr(self.p, 'cpu', None), 'stream_gbps', 20.0))
        return (gbps * 1e9) / (12.0 * 2.0) / 1e9  # bytes/edge≈12, 2 passes

    def _gpu_infos(self) -> List[Dict]:
        infos: List[Dict] = []
        allowed = set(self.available_gpus) if self.available_gpus is not None else None
        for g in getattr(self.p, 'gpus', []) or []:
            idx = int(getattr(g, 'device_index', 0))
            if getattr(g, 'backend', 'CUDA') == 'MPS':
                idx = 0
            if allowed is not None and idx not in allowed:
                continue
            if getattr(g, 'spmv_ges', None) is not None:
                geps = float(g.spmv_ges)
            elif getattr(g, 'd2d_gbps', None) is not None:
                geps = (float(g.d2d_gbps) * 1e9) / (12.0 * 2.0) / 1e9
            else:
                geps = 5.0  # 合理的兜底
            infos.append({
                'index': idx,
                'name': str(getattr(g, 'name', f'GPU{idx}')),
                'geps': geps,
                'power_w': float(getattr(g, 'avg_power_w', 250.0) or 250.0),
                'score_graph': int(getattr(g, 'score_graph', 0)),
            })
        return infos

    # ---------- Picking helpers ----------
    def _pick_by_objective(self, candidates, objective: str) -> Plan:
        if objective in ('time', 'latency'):
            return min(candidates, key=lambda kv: kv[1].estimated_time_ms)[1]
        elif objective == 'energy':
            return min(candidates, key=lambda kv: (kv[1].estimated_energy_j or float('inf')))[1]
        elif objective in ('edp', 'energy_delay_product'):
            return min(candidates, key=lambda kv: (kv[1].estimated_energy_j or 1.0) * kv[1].estimated_time_ms)[1]
        else:
            return min(candidates, key=lambda kv: kv[1].estimated_time_ms)[1]

    def _estimate_time_cpu(self, wl, geps: float) -> float:
        work_edges = wl.n_edges * wl.steps
        time_s = work_edges / max(1e-9, geps * 1e9)
        overhead_ms = (self.kernel_overhead_ms_small
                       if wl.n_nodes <= getattr(wl, 'small_graph_threshold', 2000)
                       else self.kernel_overhead_ms_large) * wl.steps
        time_s += overhead_ms / 1000.0
        time_s += (getattr(wl, 'batch_individuals', 0) / 1000.0) * (self.cpu_ga_overhead_ms / 1000.0)
        return max(time_s * 1000.0, 0.5)

    def _estimate_time_best_gpu(self, wl, g_infos: List[Dict]) -> Tuple[Optional[int], float]:
        if not g_infos:
            return None, float('inf')
        best_idx, best_t = None, float('inf')
        for i, g in enumerate(g_infos):
            t = self._estimate_time_gpu_single(wl, g['geps'])
            if t < best_t:
                best_t, best_idx = t, i
        return best_idx, best_t

    def _estimate_time_gpu_single(self, wl, geps: float) -> float:
        work_edges = wl.n_edges * wl.steps
        time_s = work_edges / max(1e-9, geps * 1e9)
        overhead = (self.kernel_overhead_ms_small if wl.n_nodes <= getattr(wl, 'small_graph_threshold', 2000)
                    else self.kernel_overhead_ms_large)
        time_s += (overhead / 1000.0) * wl.steps
        return max(time_s * 1000.0, 0.5)

    def _compose_multi_geps(self, gpu_map: Dict[int, Dict], idxs: List[int]):
        """并行吞吐求和 × 效率折减；同时给出 allocation（按各卡 geps 归一化）"""
        geps_list = [float(gpu_map.get(i, {}).get('geps', 1.0)) for i in idxs]
        base = sum(geps_list)
        k = len(idxs)
        eff = max(0.6, 0.9 - 0.05 * (k - 1))
        geps_eff = base * eff
        denom = sum(geps_list) or 1.0
        alloc = {i: (float(gpu_map.get(i, {}).get('geps', 1.0)) / denom) for i in idxs}
        return geps_eff, alloc

    def _estimate_time_multi_gpu(self, wl, geps_eff: float) -> float:
        work_edges = wl.n_edges * wl.steps
        time_s = work_edges / max(1e-9, geps_eff * 1e9)
        overhead = (self.kernel_overhead_ms_small if wl.n_nodes <= getattr(wl, 'small_graph_threshold', 2000)
                    else self.kernel_overhead_ms_large)
        time_s += (overhead / 1000.0) * wl.steps
        return max(time_s * 1000.0, 0.5)

    def _estimate_time_multi_gpu_plan(self, wl, g_infos: List[Dict], objective: str,
                                      power_cap_w: Optional[float]) -> Plan:
        key = (lambda g: g['geps']) if objective == 'time' else (lambda g: g['geps'] / max(1.0, g['power_w']))
        sorted_devs = sorted(g_infos, key=key, reverse=True)
        total_power = 0.0
        chosen: List[Dict] = []
        for g in sorted_devs:
            if power_cap_w is not None and (total_power + g['power_w']) > power_cap_w:
                continue
            chosen.append(g); total_power += g['power_w']
        if not chosen:
            chosen = [sorted_devs[0]]

        idxs = [g['index'] for g in chosen]
        geps_eff, alloc = self._compose_multi_geps({g['index']: g for g in chosen}, idxs)
        t_ms = self._estimate_time_multi_gpu(wl, geps_eff)

        return self._plan('multi-gpu', idxs, alloc, len(idxs), t_ms,
                          self._energy_gpu(chosen, t_ms),
                          reason=f"多GPU按{'吞吐' if objective=='time' else '能效'}优先，设备数={len(chosen)}")

    # ======== Energy (approx) ========
    def _energy_cpu(self, time_ms: float) -> Optional[float]:
        pkg_w = 65.0  # TODO: integrate RAPL
        return pkg_w * (time_ms / 1000.0)

    def _energy_gpu(self, g_list: List[Dict], time_ms: float) -> Optional[float]:
        if not g_list:
            return None
        total_w = sum(g.get('power_w', 250.0) for g in g_list)
        return total_w * (time_ms / 1000.0)

    # ======== Calibration ========
    def _key(self, plan: Plan) -> str:
        if plan.backend == 'cpu':
            return 'cpu'
        if plan.backend == 'cuda' and plan.devices:
            return f'cuda#{plan.devices[0]}'
        if plan.backend == 'multi-gpu':
            return 'multi#' + ",".join(str(d) for d in plan.devices)
        return plan.backend

    def _update_calib(self, plan: Plan, measured_ms: float):
        k = self._key(plan)
        pred = max(1e-6, plan.estimated_time_ms)
        ratio = float(measured_ms) / pred
        ratio = max(0.5, min(2.0, ratio))  # clamp，避免激烈震荡
        old = self.calib.get(k, 1.0)
        self.calib[k] = 0.6 * ratio + 0.4 * old

    # ======== Plan factory ========
    def _plan(self, backend: str, devices: List[int], allocation: Dict[int, float],
              world_size: int, estimated_time_ms: float, estimated_energy_j: Optional[float],
              *, reason: str, notes: str = "") -> Plan:
        # 统一口径构建方案对象（字段完全匹配 Plan 定义）
        return Plan(
            backend=backend,
            devices=list(devices),
            allocation=dict(allocation),
            world_size=int(world_size),
            estimated_time_ms=float(estimated_time_ms),
            estimated_energy_j=estimated_energy_j,
            reason=str(reason),
            notes=str(notes)
        )
