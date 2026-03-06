from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict

import torch

from gapa.config import get_remote_servers
from gapa.demo import build_demo_parser, run_demo


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def collect_environment() -> Dict[str, Any]:
    remote_servers = get_remote_servers()
    return {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "torch": {
            "version": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        },
        "optional_modules": {
            "requests": _module_available("requests"),
            "flask": _module_available("flask"),
            "fastapi": _module_available("fastapi"),
            "uvicorn": _module_available("uvicorn"),
            "psutil": _module_available("psutil"),
            "pynvml": _module_available("pynvml"),
        },
        "paths": {
            "remote_servers": remote_servers,
            "remote_server_count": len(remote_servers),
        },
    }


def explain_mode_resolution(requested_mode: str, env: Dict[str, Any]) -> Dict[str, Any]:
    system = str(env["platform"]["system"])
    cuda_available = bool(env["torch"]["cuda_available"])
    cuda_count = int(env["torch"]["cuda_device_count"])

    try:
        from gapa.workflow import HAS_DISTRIBUTED  # type: ignore
    except Exception:
        HAS_DISTRIBUTED = False

    mode = requested_mode
    if requested_mode == "m_cpu":
        mode = "s"
        return {
            "requested_mode": requested_mode,
            "resolved_mode": "s",
            "degraded": True,
            "runnable": True,
            "reason": "m_cpu is mapped to single-process local execution",
            "tip": "Use `s` unless you specifically need CPU fallback semantics.",
        }

    if requested_mode == "mnm":
        if not HAS_DISTRIBUTED:
            return {
                "requested_mode": requested_mode,
                "resolved_mode": "mnm",
                "degraded": False,
                "runnable": False,
                "reason": "MNM requires source deployment with distributed components",
                "tip": "Use `gapa[distributed]` with source checkout and configure remote agents.",
            }
        if int(env["paths"]["remote_server_count"]) <= 0:
            return {
                "requested_mode": requested_mode,
                "resolved_mode": "mnm",
                "degraded": False,
                "runnable": False,
                "reason": "GAPA_REMOTE_SERVERS is empty, so remote MNM targets are not configured",
                "tip": "Add remote agent URLs to `.env` before using MNM.",
            }
        return {
            "requested_mode": requested_mode,
            "resolved_mode": "mnm",
            "degraded": False,
            "runnable": False,
            "reason": "MNM requires live remote agents and is not self-verifiable with the built-in local smoke run",
            "tip": "Use `python examples/run_sixdst.py --mode mnm ...` after remote setup.",
        }

    if mode == "s":
        return {
            "requested_mode": requested_mode,
            "resolved_mode": "s",
            "degraded": False,
            "runnable": True,
            "reason": "",
            "tip": "",
        }

    if mode == "sm":
        if not cuda_available:
            return {
                "requested_mode": requested_mode,
                "resolved_mode": "s",
                "degraded": True,
                "runnable": True,
                "reason": f"{system} SM mode requires CUDA-capable GPU",
                "tip": "Use `s` on this machine or install CUDA-enabled PyTorch on a GPU host.",
            }
        return {
            "requested_mode": requested_mode,
            "resolved_mode": "sm",
            "degraded": False,
            "runnable": True,
            "reason": "",
            "tip": "",
        }

    if mode == "m":
        if system == "Darwin":
            return {
                "requested_mode": requested_mode,
                "resolved_mode": "s",
                "degraded": True,
                "runnable": True,
                "reason": "macOS local distributed mode falls back to single-process execution",
                "tip": "Use `s` locally or move to a supported multi-GPU host for `m`.",
            }
        if system == "Windows":
            if cuda_available:
                return {
                    "requested_mode": requested_mode,
                    "resolved_mode": "m",
                    "degraded": False,
                    "runnable": True,
                    "reason": "Windows M mode uses gloo backend instead of NCCL",
                    "tip": "Keep CUDA available and expect gloo backend on Windows.",
                }
            return {
                "requested_mode": requested_mode,
                "resolved_mode": "s",
                "degraded": True,
                "runnable": True,
                "reason": "Windows M mode without CUDA falls back to single-process execution",
                "tip": "Use `s` locally or move to a CUDA-capable host.",
            }
        if system == "Linux":
            if cuda_available and cuda_count >= 2:
                return {
                    "requested_mode": requested_mode,
                    "resolved_mode": "m",
                    "degraded": False,
                    "runnable": True,
                    "reason": "",
                    "tip": "",
                }
            if cuda_available and cuda_count == 1:
                return {
                    "requested_mode": requested_mode,
                    "resolved_mode": "s",
                    "degraded": True,
                    "runnable": True,
                    "reason": "Linux M mode requires at least 2 CUDA devices",
                    "tip": "Use `s` on a single-GPU host or move to a multi-GPU machine.",
                }
            return {
                "requested_mode": requested_mode,
                "resolved_mode": "s",
                "degraded": True,
                "runnable": True,
                "reason": "Linux M mode requires CUDA",
                "tip": "Use `s` on CPU-only hosts or install CUDA-enabled PyTorch on a GPU host.",
            }

    return {
        "requested_mode": requested_mode,
        "resolved_mode": requested_mode,
        "degraded": False,
        "runnable": False,
        "reason": f"unsupported mode '{requested_mode}'",
        "tip": "Use one of: s, sm, m, m_cpu, mnm.",
    }


def run_doctor(
    *,
    mode: str = "s",
    graph: str = "karate",
    generations: int = 1,
    pop_size: int = 6,
    device: str = "auto",
    output_dir: str | None = None,
    skip_demo: bool = False,
    quiet_demo: bool = True,
    json_output: str | None = None,
) -> Dict[str, Any]:
    env = collect_environment()
    resolution = explain_mode_resolution(mode, env)
    checks = [
        {
            "name": "python",
            "status": "pass" if sys.version_info >= (3, 9) else "fail",
            "detail": f"python={env['python']['version']}",
            "repair": "Install Python 3.9 or newer." if sys.version_info < (3, 9) else "",
        },
        {
            "name": "torch",
            "status": "pass",
            "detail": f"torch={env['torch']['version']}",
            "repair": "",
        },
        {
            "name": "mode_resolution",
            "status": "pass" if resolution["runnable"] else "fail",
            "detail": (
                f"requested={resolution['requested_mode']} resolved={resolution['resolved_mode']}"
                + (f" reason={resolution['reason']}" if resolution["reason"] else "")
            ),
            "repair": resolution.get("tip", ""),
        },
    ]

    demo_result: Dict[str, Any] | None = None
    if skip_demo:
        checks.append(
            {
                "name": "demo_smoke",
                "status": "skip",
                "detail": "smoke run skipped by request",
                "repair": "Run `python -m gapa doctor` without `--skip-demo` for an end-to-end check.",
            }
        )
    elif not resolution["runnable"]:
        checks.append(
            {
                "name": "demo_smoke",
                "status": "fail",
                "detail": resolution["reason"],
                "repair": resolution.get("tip", ""),
            }
        )
    else:
        try:
            demo_result = run_demo(
                graph_name=graph,
                generations=generations,
                pop_size=pop_size,
                mode=mode,
                device=device,
                output_dir=output_dir,
                verbose=not quiet_demo,
            )
            report = demo_result.get("report") if isinstance(demo_result.get("report"), dict) else {}
            checks.append(
                {
                    "name": "demo_smoke",
                    "status": "pass",
                    "detail": (
                        f"resolved={demo_result.get('resolved_mode')} "
                        f"best_fitness={demo_result.get('best_fitness')} "
                        f"summary={report.get('summary_path') or demo_result.get('results_dir')}"
                    ),
                    "repair": "",
                }
            )
        except Exception as exc:
            checks.append(
                {
                    "name": "demo_smoke",
                    "status": "fail",
                    "detail": str(exc),
                    "repair": "Reinstall the core package and re-run `python -m gapa demo`.",
                }
            )

    passed = all(check["status"] in ("pass", "skip") for check in checks) and (
        skip_demo or any(check["name"] == "demo_smoke" and check["status"] == "pass" for check in checks)
    )
    result = {
        "passed": passed,
        "requested_mode": resolution["requested_mode"],
        "resolved_mode": resolution["resolved_mode"],
        "degraded": bool(resolution["degraded"]),
        "reason": resolution["reason"],
        "tip": resolution.get("tip", ""),
        "environment": env,
        "checks": checks,
        "demo": demo_result,
    }
    if json_output:
        result["json_output"] = str(Path(json_output))
        output_path = Path(json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def build_doctor_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate whether GAPA is usable on this machine.")
    demo_defaults = build_demo_parser()
    graph_action = next(action for action in demo_defaults._actions if action.dest == "graph")
    parser.add_argument("--mode", choices=["s", "sm", "m", "m_cpu", "mnm"], default="s")
    parser.add_argument("--graph", choices=graph_action.choices, default="karate")
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--pop-size", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-demo", action="store_true", help="Skip the end-to-end built-in smoke run.")
    parser.add_argument("--quiet-demo", action="store_true", help="Reduce smoke-run logging.")
    parser.add_argument("--json-output", default=None, help="Write the full doctor result to JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_doctor_parser()
    args = parser.parse_args(argv)
    result = run_doctor(
        mode=args.mode,
        graph=args.graph,
        generations=args.generations,
        pop_size=args.pop_size,
        device=args.device,
        output_dir=args.output_dir,
        skip_demo=args.skip_demo,
        quiet_demo=args.quiet_demo,
        json_output=args.json_output,
    )
    print(f"[GAPA] Doctor status: {'PASS' if result['passed'] else 'FAIL'}")
    print(f"[GAPA] Python: {result['environment']['python']['version']}")
    print(
        f"[GAPA] Platform: {result['environment']['platform']['system']} "
        f"{result['environment']['platform']['release']} ({result['environment']['platform']['machine']})"
    )
    print(
        f"[GAPA] Torch: {result['environment']['torch']['version']} "
        f"| cuda={result['environment']['torch']['cuda_available']} "
        f"| cuda_devices={result['environment']['torch']['cuda_device_count']}"
    )
    print(f"[GAPA] Requested mode: {result['requested_mode']}")
    print(f"[GAPA] Resolved mode: {result['resolved_mode']}")
    if result["degraded"] or result["reason"]:
        print(f"[GAPA] Reason: {result['reason']}")
    if result["tip"]:
        print(f"[GAPA] Tip: {result['tip']}")
    for check in result["checks"]:
        print(f"[GAPA] Check[{check['name']}]: {check['status']} | {check['detail']}")
        if check["repair"]:
            print(f"[GAPA] Repair[{check['name']}]: {check['repair']}")
    if result.get("json_output"):
        print(f"[GAPA] JSON report: {result['json_output']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
