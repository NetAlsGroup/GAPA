#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Profiler (v2): cross‑platform CPU/GPU quick benchmarks with graph‑workload bias
- CPU: FP32 GEMM (compute) + streaming triad (memory) + CSR SpMV (graph‑micro)
- GPU: FP32 GEMM (compute) + device‑to‑device copy BW (memory) + CSR SpMV (graph‑micro)

What changed vs v1?
- Added CSR SpMV micro‑benchmark on CPU & GPU (torch.sparse_csr_tensor)
- Added optional GPU power sampling (NVML) and efficiency hints
- Fixed pretty() printing bug; refined normalizations & GraphScore weights

Outputs:
- Human‑readable summary (compute/memory/graph‑micro + GraphScore)
- JSON for programmatic consumption

Normalization (1000‑scale baselines):
- CPU: 500 GFLOPS, 100 GB/s, 1 GE/s (Giga‑edges/s)
- GPU: 20 TFLOPS, 1500 GB/s, 10 GE/s
- GraphScore = 0.2*Compute + 0.4*Memory + 0.4*GraphMicro  (CPU)
             = 0.3*Compute + 0.3*Memory + 0.4*GraphMicro  (GPU)
These reflect GA/graph workloads that are memory/irregular‑access dominated.
"""

from __future__ import annotations
import os
import time
import json
import platform
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

# Optional deps
try:
    import torch
    _HAS_TORCH = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
except Exception:
    torch = None
    _HAS_TORCH = False

try:
    import cpuinfo  # type: ignore
    _HAS_CPUINFO = True
except Exception:
    cpuinfo = None
    _HAS_CPUINFO = False

try:
    import pynvml  # type: ignore
    _HAS_NVML = True
except Exception:
    pynvml = None
    _HAS_NVML = False

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:
    psutil = None
    _HAS_PSUTIL = False


@dataclass
class DeviceInfo:
    os: str
    python: str
    hostname: str
    cpu_name: str
    cpu_cores_logical: int
    cpu_cores_physical: Optional[int]
    memory_gb: Optional[float]
    simd: Optional[str]
    has_torch: bool
    has_cuda: bool
    has_mps: bool
    gpus: List[Dict[str, Any]]


@dataclass
class CpuBench:
    gemm_gflops: float
    stream_gbps: float
    spmv_ges: Optional[float]  # Giga‑edges/s
    score_compute: int
    score_memory: int
    score_graphmicro: int
    score_graph: int


@dataclass
class GpuBench:
    device_index: int
    name: str
    backend: str  # 'CUDA' or 'MPS'
    fp32_tflops: float
    d2d_gbps: Optional[float]
    spmv_ges: Optional[float]
    avg_power_w: Optional[float]
    score_compute: int
    score_memory: int
    score_graphmicro: int
    score_graph: int
    memory_gb: Optional[float]
    driver: Optional[str]


@dataclass
class ProfileResult:
    device: DeviceInfo
    cpu: CpuBench
    gpus: List[GpuBench]
    overall_hint: str


class PerformanceProfiler:
    def __init__(self, quick: bool = True, seed: int = 1234):
        self.quick = quick
        self.rng = np.random.default_rng(seed)
        # Target runtimes (seconds) per microbench
        if quick:
            self.t_target_cpu = 0.35
            self.t_target_gpu = 0.35
            self.repeats = 2
        else:
            self.t_target_cpu = 0.8
            self.t_target_gpu = 0.8
            self.repeats = 3

    # -------------------- hardware probing --------------------
    def probe_device(self) -> DeviceInfo:
        sys = platform.system()
        pyver = platform.python_version()
        host = platform.node()
        cpu_name = self._cpu_name()
        logical = os.cpu_count() or 1
        phys = None
        mem_gb = None
        if _HAS_PSUTIL:
            try:
                phys = psutil.cpu_count(logical=False)
                mem_bytes = psutil.virtual_memory().total
                mem_gb = round(mem_bytes / (1024**3), 2)
            except Exception:
                pass
        simd = self._cpu_simd_hint()
        has_torch = bool(_HAS_TORCH)
        has_cuda = bool(_HAS_TORCH and torch.cuda.is_available())
        has_mps = bool(_HAS_TORCH and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())

        gpus: List[Dict[str, Any]] = []
        # Prefer NVML
        if _HAS_NVML:
            try:
                pynvml.nvmlInit()
                ng = pynvml.nvmlDeviceGetCount()
                for i in range(ng):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(h).decode('utf-8')
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h).total / (1024**3)
                    drv = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
                    gpus.append({'index': i, 'name': name, 'memory_gb': round(mem, 2), 'driver': drv})
            except Exception:
                pass
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        # Torch fallback
        if not gpus and has_cuda:
            try:
                ng = torch.cuda.device_count()
                for i in range(ng):
                    name = torch.cuda.get_device_name(i)
                    mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    gpus.append({'index': i, 'name': name, 'memory_gb': round(mem, 2), 'driver': None})
            except Exception:
                pass
        # MPS
        if has_mps:
            gpus.append({'index': 0, 'name': 'Apple M‑series (MPS)', 'memory_gb': None, 'driver': 'Metal'})

        return DeviceInfo(
            os=sys,
            python=pyver,
            hostname=host,
            cpu_name=cpu_name,
            cpu_cores_logical=logical,
            cpu_cores_physical=phys,
            memory_gb=mem_gb,
            simd=simd,
            has_torch=has_torch,
            has_cuda=has_cuda,
            has_mps=has_mps,
            gpus=gpus,
        )

    def _cpu_name(self) -> str:
        if _HAS_CPUINFO:
            try:
                info = cpuinfo.get_cpu_info()
                return info.get('brand_raw') or info.get('arch_string_raw') or platform.processor()
            except Exception:
                pass
        # macOS
        try:
            if platform.system() == 'Darwin':
                import subprocess
                out = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
                return out
        except Exception:
            pass
        return platform.processor() or platform.machine()

    def _cpu_simd_hint(self) -> Optional[str]:
        try:
            if platform.system() == 'Darwin':
                return 'NEON/AMX (Apple Silicon)'
            if platform.system() == 'Linux' and os.path.exists('/proc/cpuinfo'):
                txt = open('/proc/cpuinfo','r').read().lower()
                if 'avx512' in txt or 'avx-512' in txt:
                    return 'AVX‑512'
                if 'avx2' in txt:
                    return 'AVX2'
                if 'neon' in txt:
                    return 'NEON'
        except Exception:
            pass
        return None

    # -------------------- CPU benchmarks --------------------
    def bench_cpu(self) -> CpuBench:
        gflops = self._bench_cpu_gemm_fp32()
        gbps = self._bench_cpu_stream_triad()
        ges = self._bench_spmv_cpu()
        sc = self._score_cpu_compute(gflops)
        sm = self._score_cpu_memory(gbps)
        sg_micro = self._score_cpu_graphmicro(ges) if ges is not None else 1000
        sg = int(0.2*sc + 0.4*sm + 0.4*sg_micro)
        return CpuBench(gemm_gflops=gflops, stream_gbps=gbps, spmv_ges=ges, score_compute=sc, score_memory=sm, score_graphmicro=sg_micro, score_graph=sg)

    def _bench_cpu_gemm_fp32(self) -> float:
        N = 1024
        best = 0.0
        for _ in range(self.repeats):
            while True:
                A = self.rng.standard_normal((N, N), dtype=np.float32)
                B = self.rng.standard_normal((N, N), dtype=np.float32)
                C = A @ B  # warmup
                t0 = time.perf_counter(); C = A @ B; t1 = time.perf_counter()
                dt = t1 - t0
                flops = 2.0 * (N**3)
                gflops = flops / dt / 1e9
                if dt < self.t_target_cpu/2 and N < 8192:
                    N *= 2
                    continue
                best = max(best, gflops)
                break
        return float(best)

    def _bench_cpu_stream_triad(self) -> float:
        s = np.float32(1.618)
        n = 20_000_000
        passes = 1
        a = np.zeros(n, dtype=np.float32)
        b = self.rng.standard_normal(n, dtype=np.float32)
        c = self.rng.standard_normal(n, dtype=np.float32)
        np.multiply(c, s, out=a); np.add(b, a, out=a)
        while True:
            t0 = time.perf_counter()
            for _ in range(passes):
                np.multiply(c, s, out=a)
                np.add(b, a, out=a)
            t1 = time.perf_counter()
            dt = t1 - t0
            if dt < self.t_target_cpu/2:
                passes *= 2
                if passes > 64:
                    break
                continue
            break
        bytes_moved_per_pass = (a.nbytes + b.nbytes + c.nbytes)
        gbps = (bytes_moved_per_pass * passes) / dt / 1e9
        return float(gbps)

    def _score_cpu_compute(self, gflops: float) -> int:
        return int(max(1.0, gflops/500.0) * 1000)

    def _score_cpu_memory(self, gbps: float) -> int:
        return int(max(1.0, gbps/100.0) * 1000)

    def _score_cpu_graphmicro(self, ges: float) -> int:
        # 与 GPU 保持一致：10 GE/s ≈ 1000 分
        return int(max(1.0, ges / 10.0) * 1000)

    # -------------------- GPU benchmarks --------------------
    def bench_gpus(self, devinfo: DeviceInfo) -> List[GpuBench]:
        out: List[GpuBench] = []
        if not _HAS_TORCH:
            return out
        if devinfo.has_cuda:
            for g in devinfo.gpus:
                if g.get('driver') == 'Metal':
                    continue
                try:
                    out.append(self._bench_one_cuda(g))
                except Exception:
                    pass
        if devinfo.has_mps:
            try:
                out.append(self._bench_one_mps())
            except Exception:
                pass
        return out

    def _bench_one_cuda(self, gmeta: Dict[str, Any]) -> GpuBench:
        idx = int(gmeta['index'])
        torch.cuda.set_device(idx)
        # --- compute ---
        N = 4096 if self.quick else 6144
        A = torch.randn((N, N), device=f'cuda:{idx}', dtype=torch.float32)
        B = torch.randn((N, N), device=f'cuda:{idx}', dtype=torch.float32)
        C = A @ B; torch.cuda.synchronize(idx)
        best_tflops = 0.0
        for _ in range(self.repeats):
            t0 = time.perf_counter(); C = A @ B; torch.cuda.synchronize(idx); t1 = time.perf_counter()
            dt = t1 - t0
            flops = 2.0 * (N**3)
            tflops = flops / dt / 1e12
            best_tflops = max(best_tflops, tflops)
        # --- memory (D2D) ---
        mem_gb = gmeta.get('memory_gb') or 8.0
        size_gb = min(1.0, max(0.25, 0.2*mem_gb))
        elems = int((size_gb * (1024**3)) // 4)
        x = torch.empty(elems, device=f'cuda:{idx}', dtype=torch.float32)
        y = torch.empty_like(x)
        torch.cuda.synchronize(idx)
        t0 = time.perf_counter(); y.copy_(x); torch.cuda.synchronize(idx); t1 = time.perf_counter()
        dt = t1 - t0
        d2d_gbps = (size_gb * (1024**3)) / dt / 1e9
        # --- graph micro (CSR SpMV) ---
        ges = self._bench_spmv_cuda(idx)
        try:
            del A, B, C, x, y
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        # --- optional power ---
        avg_w = self._nvml_avg_power(idx, 0.2) if _HAS_NVML else None
        sc = self._score_gpu_compute(best_tflops)
        sm = self._score_gpu_memory(d2d_gbps)
        sgm = self._score_gpu_graphmicro(ges) if ges is not None else 1000
        sg = int(0.3*sc + 0.3*sm + 0.4*sgm)
        name = gmeta.get('name') or torch.cuda.get_device_name(idx)
        try:
            driver = gmeta.get('driver') or torch.version.cuda
        except Exception:
            driver = None
        return GpuBench(device_index=idx, name=name, backend='CUDA', fp32_tflops=float(best_tflops), d2d_gbps=float(d2d_gbps), spmv_ges=ges, avg_power_w=avg_w, score_compute=sc, score_memory=sm, score_graphmicro=sgm, score_graph=sg, memory_gb=gmeta.get('memory_gb'), driver=driver)

    def _bench_one_mps(self) -> GpuBench:
        device = torch.device('mps')
        N = 3072 if self.quick else 4096
        A = torch.randn((N, N), device=device, dtype=torch.float32)
        B = torch.randn((N, N), device=device, dtype=torch.float32)
        C = A @ B; torch.mps.synchronize()
        best_tflops = 0.0
        for _ in range(self.repeats):
            t0 = time.perf_counter(); C = A @ B; torch.mps.synchronize(); t1 = time.perf_counter()
            dt = t1 - t0
            flops = 2.0 * (N**3)
            tflops = flops / dt / 1e12
            best_tflops = max(best_tflops, tflops)
        size_gb = 1.0
        elems = int((size_gb * (1024**3)) // 4)
        x = torch.empty(elems, device=device, dtype=torch.float32)
        y = torch.empty_like(x)
        torch.mps.synchronize(); t0 = time.perf_counter(); y.copy_(x); torch.mps.synchronize(); t1 = time.perf_counter()
        dt = t1 - t0
        d2d_gbps = (size_gb * (1024**3)) / dt / 1e9
        ges = self._bench_spmv_mps()
        sc = self._score_gpu_compute(best_tflops)
        sm = self._score_gpu_memory(d2d_gbps)
        sgm = self._score_gpu_graphmicro(ges) if ges is not None else 1000
        sg = int(0.3*sc + 0.3*sm + 0.4*sgm)
        return GpuBench(device_index=0, name='Apple M‑series (MPS)', backend='MPS', fp32_tflops=float(best_tflops), d2d_gbps=float(d2d_gbps), spmv_ges=ges, avg_power_w=None, score_compute=sc, score_memory=sm, score_graphmicro=sgm, score_graph=sg, memory_gb=None, driver='Metal')

    def _score_gpu_compute(self, tflops: float) -> int:
        return int(max(1.0, tflops/20.0) * 1000)

    def _score_gpu_memory(self, gbps: float) -> int:
        return int(max(1.0, gbps/1500.0) * 1000)

    def _score_gpu_graphmicro(self, ges: float) -> int:
        # 10 GE/s ~= 1000
        return int(max(1.0, ges/10.0) * 1000)

    # -------------------- Graph micro‑bench helpers --------------------
    def _make_random_csr(self, n: int, avg_deg: int, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (crow_indices, col_indices, values) for a random CSR with 0/1 values."""
        # degrees ~ Poisson(avg_deg)
        deg = torch.poisson(torch.full((n,), float(avg_deg)))
        deg = torch.clamp(deg, min=1)
        nnz = int(deg.sum().item())
        crow = torch.empty(n+1, dtype=torch.int64)
        torch.cumsum(deg.to(torch.int64), dim=0, out=crow[1:])
        crow[0] = 0
        col = torch.randint(0, n, (nnz,), dtype=torch.int64)
        val = torch.ones(nnz, dtype=torch.float32)
        if device != 'cpu':
            crow = crow.to(device)
            col = col.to(device)
            val = val.to(device)
        return crow, col, val

    def _bench_spmv_cpu(self) -> Optional[float]:
        if not _HAS_TORCH:
            return None
        n = 20000
        avg_deg = 20
        device = 'cpu'
        crow, col, val = self._make_random_csr(n, avg_deg, device)
        A = torch.sparse_csr_tensor(crow, col, val, size=(n, n), device=device, dtype=torch.float32)
        x = torch.randn(n, device=device, dtype=torch.float32)
        # warmup
        y = torch.mv(A.to_dense(), x) if not A.is_sparse_csr else torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
        # adaptive passes
        passes = 1
        t_start = time.perf_counter()
        while True:
            t0 = time.perf_counter()
            for _ in range(passes):
                y = torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
            t1 = time.perf_counter()
            dt = t1 - t0
            if dt < self.t_target_cpu/2:
                passes *= 2
                if passes > 64:
                    break
                continue
            break
        nnz = int(A._nnz()) * passes
        ges = nnz / dt / 1e9
        return float(ges)

    def _bench_spmv_cuda(self, idx: int) -> Optional[float]:
        if not _HAS_TORCH or not torch.cuda.is_available():
            return None
        device = f'cuda:{idx}'
        n = 200000 if not self.quick else 100000
        avg_deg = 20
        crow, col, val = self._make_random_csr(n, avg_deg, device)
        A = torch.sparse_csr_tensor(crow, col, val, size=(n, n), device=device, dtype=torch.float32)
        x = torch.randn(n, device=device, dtype=torch.float32)
        torch.cuda.synchronize(idx)
        # warmup
        y = torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
        torch.cuda.synchronize(idx)
        passes = 1
        while True:
            t0 = time.perf_counter()
            for _ in range(passes):
                y = torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
            torch.cuda.synchronize(idx)
            t1 = time.perf_counter()
            dt = t1 - t0
            if dt < self.t_target_gpu/2:
                passes *= 2
                if passes > 64:
                    break
                continue
            break
        nnz = int(A._nnz()) * passes
        ges = nnz / dt / 1e9
        try:
            del crow, col, val, A, x, y
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return float(ges)

    def _bench_spmv_mps(self) -> Optional[float]:
        if not _HAS_TORCH or not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            return None
        device = 'mps'
        n = 80000 if not self.quick else 40000
        avg_deg = 20
        crow, col, val = self._make_random_csr(n, avg_deg, device)
        A = torch.sparse_csr_tensor(crow, col, val, size=(n, n), device=device, dtype=torch.float32)
        x = torch.randn(n, device=device, dtype=torch.float32)
        # warmup
        y = torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
        torch.mps.synchronize()
        passes = 1
        while True:
            t0 = time.perf_counter()
            for _ in range(passes):
                y = torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
            torch.mps.synchronize()
            t1 = time.perf_counter()
            dt = t1 - t0
            if dt < self.t_target_gpu/2:
                passes *= 2
                if passes > 64:
                    break
                continue
            break
        nnz = int(A._nnz()) * passes
        ges = nnz / dt / 1e9
        return float(ges)

    # -------------------- NVML helpers --------------------
    def _nvml_avg_power(self, index: int, duration_s: float) -> Optional[float]:
        if not _HAS_NVML:
            return None
        try:
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(index)
            t0 = time.perf_counter()
            samples = []
            while (time.perf_counter() - t0) < duration_s:
                p = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0  # mW→W
                samples.append(p)
                time.sleep(0.02)
            return float(np.mean(samples)) if samples else None
        except Exception:
            return None
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    # -------------------- orchestration --------------------
    def profile(self) -> ProfileResult:
        dev = self.probe_device()
        cpu = self.bench_cpu()
        gpus = self.bench_gpus(dev)
        hint = self._recommendation(dev, cpu, gpus)
        try:
            if _HAS_TORCH and torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return ProfileResult(device=dev, cpu=cpu, gpus=gpus, overall_hint=hint)

    def _recommendation(self, dev: DeviceInfo, cpu: CpuBench, gpus: List[GpuBench]) -> str:
        if not gpus:
            return f"建议：本机无可用 GPU，CPU GraphScore {cpu.score_graph}，优先选择 CPU 路线。"
        best = max(gpus, key=lambda x: x.score_graph)
        if cpu.score_graph * 1.1 >= best.score_graph:
            return (f"建议：CPU 与 GPU 图工作负载表现接近（CPU {cpu.score_graph} vs GPU {best.score_graph}），"
                    f"小/中规模可先 CPU；较大规模可用 {best.backend}:{best.name}。")
        else:
            return (f"建议：优先使用 {best.backend} 设备 {best.name}（GPU GraphScore {best.score_graph}），"
                    f"CPU GraphScore {cpu.score_graph}。")

    # -------------------- pretty/JSON --------------------
    @staticmethod
    def pretty(res: ProfileResult) -> str:
        d = res.device
        lines = []
        lines.append("=== 硬件画像 ===\n")
        lines.append(f"OS/Python: {d.os} / {d.python}\n")
        lines.append(f"Host: {d.hostname}\n")
        lines.append(f"CPU: {d.cpu_name} | 逻辑核 {d.cpu_cores_logical} | 物理核 {d.cpu_cores_physical or '?'} | SIMD: {d.simd or '?'}\n")
        lines.append(f"内存: {d.memory_gb or '?'} GB\n")
        if d.gpus:
            lines.append("GPU:")
            for g in d.gpus:
                lines.append(f"  - #{g.get('index')}: {g.get('name')} | 显存 {g.get('memory_gb') or '?'} GB | 驱动 {g.get('driver') or '?'}\n")
        else:
            lines.append("GPU: 无\n")
        lines.append("\n")
        lines.append("=== CPU 基准 ===\n")
        lines.append(f"GEMM: {res.cpu.gemm_gflops:.1f} GFLOPS | STREAM: {res.cpu.stream_gbps:.1f} GB/s | SpMV: {res.cpu.spmv_ges or float('nan'):.2f} GE/s\n")
        lines.append(f"ComputeScore: {res.cpu.score_compute} | MemoryScore: {res.cpu.score_memory} | GraphMicro: {res.cpu.score_graphmicro} | GraphScore: {res.cpu.score_graph}\n")
        lines.append("\n")
        lines.append("=== GPU 基准 ===\n")
        if res.gpus:
            for g in res.gpus:
                pw = f", AvgPower ~{g.avg_power_w:.0f} W" if g.avg_power_w is not None else ""
                lines.append(
                    f"{g.backend}#{g.device_index} {g.name}: FP32 {g.fp32_tflops:.2f} TFLOPS, D2D {g.d2d_gbps or float('nan'):.0f} GB/s, SpMV {g.spmv_ges or float('nan'):.2f} GE/s{pw} | \n"
                    f"Compute {g.score_compute} | Memory {g.score_memory} | GraphMicro {g.score_graphmicro} | GraphScore {g.score_graph} | 显存 {g.memory_gb or '?'} GB\n"
                )
        else:
            lines.append("无可用 GPU\n")
        lines.append("\n")
        lines.append("=== 建议 ===\n")
        lines.append(res.overall_hint)
        return "\n".join(lines)

    @staticmethod
    def to_json(res: ProfileResult) -> str:
        return json.dumps({
            'device': asdict(res.device),
            'cpu': asdict(res.cpu),
            'gpus': [asdict(g) for g in res.gpus],
            'overall_hint': res.overall_hint,
        }, ensure_ascii=False, indent=2)


# -------------------- CLI --------------------
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Performance Profiler (CPU/GPU quick scores, graph‑aware)')
    p.add_argument('--full', action='store_true', help='run longer, more stable benchmarks')
    p.add_argument('--json', action='store_true', help='print JSON only')
    args = p.parse_args()

    prof = PerformanceProfiler(quick=not args.full)
    res = prof.profile()

    if args.json:
        print(PerformanceProfiler.to_json(res))
    else:
        print(PerformanceProfiler.pretty(res))
