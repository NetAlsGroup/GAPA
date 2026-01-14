from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from time import perf_counter


def select_run_mode(mode: str | None, devices: Any) -> Dict[str, Any]:
    """Select requested mode/devices without mutating environment.

    The actual CUDA visibility is applied inside the GA subprocess before torch import/initialization.
    """

    mode = (mode or "AUTO").upper()
    selected: List[int] = []
    if isinstance(devices, list):
        for d in devices:
            try:
                selected.append(int(d))
            except Exception:
                continue
    elif devices is not None:
        try:
            selected = [int(devices)]
        except Exception:
            selected = []

    if mode == "CPU":
        return {"mode": mode, "devices": [], "cuda_visible_devices": None}

    if mode in ("S", "SM"):
        if selected:
            return {"mode": mode, "devices": [selected[0]], "cuda_visible_devices": str(selected[0])}
        return {"mode": mode, "devices": [], "cuda_visible_devices": None}

    if mode in ("M", "MNM"):
        if selected:
            return {"mode": mode, "devices": selected, "cuda_visible_devices": ",".join(str(x) for x in selected)}
        return {"mode": mode, "devices": [], "cuda_visible_devices": None}

    return {"mode": mode, "devices": selected, "cuda_visible_devices": ",".join(str(x) for x in selected) if selected else None}


def ga_worker(
    task_id: str,
    algorithm: str,
    dataset: str,
    iterations: int,
    crossover_rate: float,
    mutate_rate: float,
    selected: Dict[str, Any],
    q: Any,
    resume_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Run GA in a subprocess and emit events to parent via queue."""
    from .db_manager import db_manager
    try:
        cvd = selected.get("cuda_visible_devices")
        if selected.get("mode") == "CPU":
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        elif cvd:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cvd)

        try:
            import psutil as _psutil  # type: ignore
        except Exception:
            _psutil = None
        try:
            import pynvml as _pynvml  # type: ignore
        except Exception:
            _pynvml = None

        dataset_dir = Path(os.getenv("GAPA_DATASET_DIR", str(Path(__file__).resolve().parent.parent / "dataset")))
        repo_root = Path(__file__).resolve().parent.parent

        def _find_dataset_file(name: str) -> Optional[Path]:
            if not name:
                return None
            candidates: List[Path] = []
            candidates.append(dataset_dir / f"{name}.txt")
            candidates.append(dataset_dir / f"{name.lower()}.txt")
            candidates.append(dataset_dir / name / f"{name}.txt")
            candidates.append(dataset_dir / name.lower() / f"{name.lower()}.txt")
            norm = name.replace("_", "-")
            candidates.append(dataset_dir / f"{norm}.txt")
            candidates.append(dataset_dir / f"{norm.lower()}.txt")
            candidates.append(dataset_dir / norm / f"{norm}.txt")
            candidates.append(dataset_dir / norm.lower() / f"{norm.lower()}.txt")
            for p in candidates:
                if p.exists():
                    return p
            target = f"{name}".lower()
            try:
                for p in dataset_dir.glob("**/*.txt"):
                    if p.name.lower() in (f"{target}.txt", f"{norm.lower()}.txt"):
                        return p
            except Exception:
                pass
            return None

        def _find_dataset_gml(name: str) -> Optional[Path]:
            if not name:
                return None
            candidates: List[Path] = []
            candidates.append(dataset_dir / name / f"{name}.gml")
            candidates.append(dataset_dir / name.lower() / f"{name.lower()}.gml")
            norm = name.replace("_", "-")
            candidates.append(dataset_dir / norm / f"{norm}.gml")
            candidates.append(dataset_dir / norm.lower() / f"{norm.lower()}.gml")
            for p in candidates:
                if p.exists():
                    return p
            target = name.lower()
            target2 = norm.lower()
            try:
                for p in dataset_dir.glob("**/*.gml"):
                    if p.name.lower() in (f"{target}.gml", f"{target2}.gml"):
                        return p
            except Exception:
                pass
            return None

        selected_phys = selected.get("devices") or []
        try:
            selected_phys = [int(x) for x in selected_phys]
        except Exception:
            selected_phys = []

        def snapshot() -> Dict[str, Any]:
            cpu_usage = None
            mem_percent = None
            if _psutil:
                try:
                    cpu_usage = _psutil.cpu_percent(interval=0.0)
                    mem_percent = _psutil.virtual_memory().percent
                except Exception:
                    pass
            gpus: List[Dict[str, Any]] = []
            if _pynvml:
                try:
                    _pynvml.nvmlInit()
                    indices = selected_phys if selected_phys else list(range(_pynvml.nvmlDeviceGetCount()))
                    for i in indices:
                        h = _pynvml.nvmlDeviceGetHandleByIndex(i)
                        mem = _pynvml.nvmlDeviceGetMemoryInfo(h)
                        try:
                            util = _pynvml.nvmlDeviceGetUtilizationRates(h)
                            gpu_util = float(util.gpu)
                            mem_util = float(util.memory)
                        except Exception:
                            gpu_util = mem_util = None
                        try:
                            power_w = _pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                        except Exception:
                            power_w = None
                        gpus.append(
                            {
                                "id": i,
                                "used_mb": round(mem.used / (1024**2)),
                                "total_mb": round(mem.total / (1024**2)),
                                "power_w": power_w,
                                "gpu_util_percent": gpu_util,
                                "mem_util_percent": mem_util,
                            }
                        )
                except Exception:
                    gpus = []
                finally:
                    try:
                        _pynvml.nvmlShutdown()
                    except Exception:
                        pass
            return {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "cpu_usage_percent": cpu_usage,
                "memory_percent": mem_percent,
                "gpus": gpus,
            }

        def emit(evt: Dict[str, Any]) -> None:
            evt["task_id"] = task_id
            q.put(evt)

        emit({"type": "log", "line": "[INFO] preprocessing: start"})
        emit(
            {
                "type": "log",
                "line": f"[INFO] 任务 {task_id} 启动，算法={algorithm} dataset={dataset} pc={crossover_rate:.3f} pm={mutate_rate:.3f} iters={iterations}",
            }
        )
        mode = selected.get('mode')
        devices = selected.get('devices') or []
        remote_servers = selected.get('remote_servers') or selected.get('allowed_server_ids') or []
        
        # Build descriptive resource string
        resources = []
        if mode == "CPU" or (mode == "MNM" and not devices):
            resources.append("Local CPU")
        elif devices:
            resources.append(f"Local GPU {','.join(str(d) for d in devices)}")
        
        for srv in remote_servers:
            # Try to get GPU info from server name - for now just add server name
            resources.append(srv)
        
        resource_str = " + ".join(resources) if resources else "未知"
        emit({"type": "log", "line": f"[INFO] 运行模式: {mode} 资源: {resource_str}"})
        emit({"type": "progress", "value": 1})

        try:
            import sys

            sys.path.insert(0, str(repo_root))
            sys.path.insert(0, str(repo_root / "gapa" / "DeepLearning"))
        except Exception:
            pass

        import networkx as nx  # type: ignore
        import torch  # type: ignore
        from gapa.utils.DataLoader import Loader  # type: ignore

        ui_mode = (selected.get("mode") or "AUTO").upper()
        distributed_fitness = ui_mode == "MNM"
        selected_devices = selected.get("devices") or []
        try:
            selected_devices = [int(x) for x in selected_devices]
        except Exception:
            selected_devices = []

        if ui_mode == "CPU":
            algo_mode = "s"
            device = "cpu"
        elif ui_mode == "SM":
            algo_mode = "sm"
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif ui_mode == "S":
            algo_mode = "s"
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif ui_mode == "M":
            algo_mode = "m"
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif ui_mode == "MNM":
            # MNM: multi-machine fitness offload; keep GA loop single-process.
            algo_mode = "mnm"
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            # AUTO: if a device list is provided (typically from evaluation plan), prefer multi-GPU execution.
            if len(selected_devices) >= 2:
                algo_mode = "m"
            else:
                algo_mode = "s"
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        world_size = int(torch.cuda.device_count()) if device.startswith("cuda") else 1
        if algo_mode == "m" and world_size < 2:
            emit({"type": "log", "line": f"[WARN] mode={algo_mode} requires >=2 GPUs; fallback to S"})
            algo_mode = "s"
            world_size = 1
        if distributed_fitness:
            world_size = 1
            emit({"type": "log", "line": f"[INFO] Exec resolved: ui_mode={ui_mode} -> algo_mode={algo_mode} local_device={device} (远程服务器使用其锁定的 GPU)"})
        else:
            emit({"type": "log", "line": f"[INFO] Exec resolved: ui_mode={ui_mode} -> algo_mode={algo_mode} world_size={world_size} device={device}"})

        algo_raw = (algorithm or "").strip()
        algo_key = algo_raw.replace("_", "-")
        algo_map = {
            "cutoff": "CutOff",
            "cut-off": "CutOff",
            "cda-eda": "CDA-EDA",
            "lpa-ga": "LPA-GA",
            "lpaga": "LPA-GA",
            "lpa-eda": "LPA-EDA",
            "ncaga": "NCA-GA",
            "nca-ga": "NCA-GA",
        }
        algo_norm = algo_map.get(algo_key.lower(), algo_raw)

        # Task-specific objective names (for visualization/logs)
        objective = None
        if algo_norm in ("CDA-EDA", "CGN", "QAttack"):
            objective = {"primary": "Q", "secondary": "NMI", "primary_goal": "min"}
        elif algo_norm in ("SixDST", "CutOff", "TDE"):
            objective = {"primary": "PCG", "secondary": "MCN", "primary_goal": "min"}
        elif algo_norm in ("NCA-GA", "SGA", "GANI"):
            objective = {"primary": "Acc", "secondary": "ASR", "primary_goal": "min"}
        elif algo_norm in ("LPA-EDA", "LPA-GA"):
            objective = {"primary": "AUC", "secondary": "Pre", "primary_goal": "min"}
        else:
            objective = {"primary": "fitness", "secondary": None, "primary_goal": "min"}

        def _load_adjlist(name: str, *, sort_nodes: bool) -> Dict[str, Any]:
            ds_file = _find_dataset_file(name)
            if ds_file is None:
                raise FileNotFoundError(f"dataset .txt not found for '{name}' under {dataset_dir}")
            emit({"type": "log", "line": f"[INFO] Load dataset file: {ds_file}"})
            G0 = nx.read_adjlist(str(ds_file), nodetype=int)
            if sort_nodes:
                nodelist0 = sorted(list(G0.nodes()))
                A0 = torch.tensor(nx.to_numpy_array(G0, nodelist=nodelist0), dtype=torch.float32)
                G1 = nx.from_numpy_array(A0.cpu().numpy())
                return {"G": G1, "A": A0.to(device) if device != "cpu" else A0, "nodelist": list(G1.nodes())}
            nodelist0 = list(G0.nodes())
            A0 = torch.tensor(nx.to_numpy_array(G0, nodelist=nodelist0), dtype=torch.float32)
            return {"G": G0, "A": A0.to(device) if device != "cpu" else A0, "nodelist": nodelist0}

        def _load_gml(name: str, *, sort_nodes: bool, rebuild_from_adj: bool) -> Dict[str, Any]:
            gml = _find_dataset_gml(name)
            if gml is None:
                raise FileNotFoundError(f"dataset .gml not found for '{name}' under {dataset_dir}")
            emit({"type": "log", "line": f"[INFO] Load dataset file: {gml}"})
            G0 = nx.read_gml(str(gml), label="id")
            nodelist0 = sorted(list(G0.nodes())) if sort_nodes else list(G0.nodes())
            A0 = torch.tensor(nx.to_numpy_array(G0, nodelist=nodelist0), dtype=torch.float32)
            if rebuild_from_adj:
                G1 = nx.from_numpy_array(A0.cpu().numpy())
                return {"G": G1, "A": A0.to(device) if device != "cpu" else A0, "nodelist": list(G1.nodes())}
            return {"G": G0, "A": A0.to(device) if device != "cpu" else A0, "nodelist": nodelist0}

        result: Dict[str, Any] = {
            "algorithm": algo_norm,
            "dataset": dataset,
            "convergence": [],
            "curves": {},
            "points": [],
            "metrics": [],
            "objectives": objective,
            "hyperparams": {
                "iterations": int(iterations),
                "crossover_rate": float(crossover_rate),
                "mutate_rate": float(mutate_rate),
                "pop_size": int(os.getenv("GAPA_GA_POP_SIZE", "100")),
            },
            "selected": {
                "mode": ui_mode,
                "devices": selected.get("devices"),
                "remote_servers": selected.get("remote_servers") or selected.get("allowed_server_ids"),
            },
            "exec": {"algo_mode": algo_mode, "world_size": world_size, "device": device},
        }
        if objective.get("primary"):
            result["curves"][objective["primary"]] = []
        if objective.get("secondary"):
            result["curves"][objective["secondary"]] = []

        res_lock = threading.Lock()
        iter_t0: Optional[float] = None
        iter_t1: Optional[float] = None

        def on_iter(gen: int, max_gen: int, payload: Any) -> None:
            nonlocal iter_t0, iter_t1
            now = perf_counter()
            with res_lock:
                if gen >= 1 and iter_t0 is None:
                    iter_t0 = now
                    emit({"type": "log", "line": "[INFO] iteration loop started; timing begins (preprocessing excluded)"})
                if iter_t0 is not None:
                    iter_t1 = now

            emit({"type": "progress", "value": int(gen / max(1, max_gen) * 100)})
            metrics = payload if isinstance(payload, dict) else {"fitness": payload}
            p_name = objective.get("primary") or "fitness"
            s_name = objective.get("secondary")
            p_val = metrics.get(p_name)
            s_val = metrics.get(s_name) if s_name else None
            if s_name:
                emit({"type": "log", "line": f"[INFO] iter {gen}/{max_gen} {p_name}={p_val} {s_name}={s_val}"})
            else:
                emit({"type": "log", "line": f"[INFO] iter {gen}/{max_gen} {p_name}={p_val}"})
            with res_lock:
                if p_name in result["curves"] and p_val is not None:
                    result["curves"][p_name].append(float(p_val))
                    result["convergence"].append(float(p_val))  # backward compatible: primary curve
                if s_name and s_name in result["curves"] and s_val is not None:
                    result["curves"][s_name].append(float(s_val))
                point = {"iter": int(gen)}
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and (k == p_name or k == s_name):
                        point[k] = float(v)
                if iter_t0 is not None:
                    point["elapsed_s"] = float(now - iter_t0)
                result["points"].append(point)
                result["metrics"].append({"stage": "iter", "iter": int(gen), "objectives": metrics, **snapshot()})

        obs_path: Optional[Path] = None
        tail_stop = False

        def tail_jsonl(path: Path) -> None:
            nonlocal tail_stop
            last_pos = 0
            while not tail_stop:
                try:
                    if not path.exists():
                        time.sleep(0.2)
                        continue
                    with path.open("r", encoding="utf-8") as f:
                        f.seek(last_pos)
                        while True:
                            line = f.readline()
                            if not line:
                                break
                            last_pos = f.tell()
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                import json

                                obj = json.loads(line)
                                gen = int(obj.get("generation") or 0)
                                mg = int(obj.get("max_generation") or 1)
                                metrics = obj.get("metrics")
                                if isinstance(metrics, dict):
                                    on_iter(gen, mg, metrics)
                                else:
                                    bf = obj.get("best_fitness")
                                    on_iter(gen, mg, float(bf) if bf is not None else None)
                            except Exception:
                                continue
                except Exception:
                    pass
                time.sleep(0.2)

        results_dir = Path(os.getenv("GAPA_RESULTS_DIR", str(repo_root / "results")))
        results_dir.mkdir(parents=True, exist_ok=True)
        fitness_goal = None
        comm_path = results_dir / f"comm_{task_id}.json" if algo_mode == "m" else None

        DISTRIBUTED_FITNESS_SUPPORTED = {
            "SixDST",
            "CutOff",
            "TDE",
            "CDA-EDA",
            "CGN",
            "QAttack",
            "LPA-EDA",
            "LPA-GA",
            "NCA-GA",
            "SGA",
            "GANI",
        }

        def maybe_wrap_distributed(evaluator_obj: Any) -> Any:
            if not distributed_fitness:
                return evaluator_obj
            if algo_norm not in DISTRIBUTED_FITNESS_SUPPORTED:
                emit(
                    {
                        "type": "log",
                        "line": f"[WARN] MNM enabled but algorithm={algo_norm} is not supported for distributed fitness; fallback to local fitness.",
                    }
                )
                return evaluator_obj
            try:
                allowed_ids = selected.get("remote_servers") or selected.get("allowed_server_ids")
                if allowed_ids is not None and not isinstance(allowed_ids, list):
                    allowed_ids = [allowed_ids]
                from server.distributed_evaluator import DistributedEvaluator  # lazy import after CUDA env is set

                use_plan = selected.get("use_strategy_plan")
                if use_plan is None:
                    use_plan = False
                wrapped = DistributedEvaluator(
                    evaluator_obj,
                    algorithm=algo_norm,
                    dataset=dataset,
                    allowed_server_ids=allowed_ids,
                    max_remote_workers=int(os.getenv("GAPA_MNM_MAX_WORKERS", "4")),
                    refresh_interval_s=float(os.getenv("GAPA_MNM_REFRESH_S", "2.0")),
                    use_strategy_plan=bool(use_plan),
                )
                if allowed_ids:
                    emit({"type": "log", "line": f"[INFO] MNM enabled: remote servers={allowed_ids}."})
                else:
                    emit({"type": "log", "line": "[INFO] MNM enabled: distributed fitness offload to remote server agents."})
                return wrapped
            except Exception as exc:
                emit({"type": "log", "line": f"[WARN] MNM setup failed ({exc}); fallback to local fitness."})
                return evaluator_obj

        comm_tracker = None

        def attach_observer(controller_obj: Any) -> None:
            nonlocal obs_path
            if algo_mode in ("m", "mnm"):
                obs_path = Path(str(results_dir / f"obs_{task_id}.jsonl"))
                try:
                    obs_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
                except Exception:
                    pass
                controller_obj.observer = {"type": "jsonl", "path": str(obs_path)}
                threading.Thread(target=tail_jsonl, args=(obs_path,), daemon=True).start()
            else:
                controller_obj.observer = on_iter

        def run_cnd(method: str) -> None:
            nonlocal fitness_goal
            nonlocal comm_tracker
            if method == "SixDST":
                from gapa.algorithm.CND.SixDST import SixDST, SixDSTController, SixDSTEvaluator  # type: ignore

                loaded = _load_adjlist(dataset, sort_nodes=False)
                data_loader = Loader(dataset=dataset, device=device)
                data_loader.G = loaded["G"]
                data_loader.A = loaded["A"]
                data_loader.nodes_num = int(data_loader.A.shape[0])
                data_loader.nodes = torch.tensor(loaded["nodelist"], device=device)
                data_loader.selected_genes_num = int(0.4 * data_loader.nodes_num)
                data_loader.k = int(0.1 * data_loader.nodes_num)
                data_loader.mode = algo_mode
                data_loader.world_size = world_size

                controller = SixDSTController(
                    path=str(results_dir) + "/",
                    pattern="write",
                    cutoff_tag="popGreedy_cutoff_",
                    data_loader=data_loader,
                    loops=1,
                    crossover_rate=float(crossover_rate),
                    mutate_rate=float(mutate_rate),
                    pop_size=int(result["hyperparams"]["pop_size"]),
                    device=device,
                )
                if comm_path:
                    controller.comm_path = str(comm_path)
                evaluator = SixDSTEvaluator(pop_size=int(result["hyperparams"]["pop_size"]), adj=data_loader.A, device=device)
                evaluator = maybe_wrap_distributed(evaluator)
                if hasattr(evaluator, "comm_stats"):
                    comm_tracker = evaluator
                fitness_goal = getattr(controller, "fit_side", None)
                attach_observer(controller)
                emit({"type": "log", "line": f"[INFO] Start SixDST: mode={algo_mode} device={device} world_size={world_size}"})
                result["metrics"].append({"stage": "init", **snapshot()})
                SixDST(mode=algo_mode, max_generation=int(iterations), data_loader=data_loader, controller=controller, evaluator=evaluator, world_size=world_size, verbose=False)
                return

            if method == "CutOff":
                from gapa.algorithm.CND.Cutoff import Cutoff, CutoffController, CutoffEvaluator  # type: ignore

                loaded = _load_adjlist(dataset, sort_nodes=True)
                data_loader = Loader(dataset=dataset, device=device)
                data_loader.G = loaded["G"]
                data_loader.A = loaded["A"]
                data_loader.nodes_num = int(data_loader.A.shape[0])
                data_loader.nodes = torch.tensor(loaded["nodelist"], device=device)
                data_loader.selected_genes_num = int(0.4 * data_loader.nodes_num)
                data_loader.k = int(0.1 * data_loader.nodes_num)
                data_loader.mode = algo_mode
                data_loader.world_size = world_size

                evaluator = CutoffEvaluator(pop_size=int(result["hyperparams"]["pop_size"]), graph=data_loader.G, nodes=data_loader.nodes, device=device)
                evaluator = maybe_wrap_distributed(evaluator)
                if hasattr(evaluator, "comm_stats"):
                    comm_tracker = evaluator
                controller = CutoffController(
                    path=str(results_dir) + "/",
                    pattern="write",
                    cutoff_tag="popGreedy_cutoff_",
                    data_loader=data_loader,
                    loops=1,
                    crossover_rate=float(crossover_rate),
                    mutate_rate=float(mutate_rate),
                    pop_size=int(result["hyperparams"]["pop_size"]),
                    device=device,
                )
                if comm_path:
                    controller.comm_path = str(comm_path)
                fitness_goal = getattr(controller, "fit_side", None)
                attach_observer(controller)
                emit({"type": "log", "line": f"[INFO] Start CutOff: mode={algo_mode} device={device} world_size={world_size}"})
                result["metrics"].append({"stage": "init", **snapshot()})
                Cutoff(mode=algo_mode, max_generation=int(iterations), data_loader=data_loader, controller=controller, evaluator=evaluator, world_size=world_size, verbose=False)
                return

            if method == "TDE":
                from gapa.algorithm.CND.TDE import TDE, TDEController, TDEEvaluator  # type: ignore

                loaded = _load_adjlist(dataset, sort_nodes=True)
                data_loader = Loader(dataset=dataset, device=device)
                data_loader.G = loaded["G"]
                data_loader.A = loaded["A"]
                data_loader.nodes_num = int(data_loader.A.shape[0])
                data_loader.nodes = torch.tensor(loaded["nodelist"], device=device)
                data_loader.selected_genes_num = int(0.4 * data_loader.nodes_num)
                data_loader.k = int(0.1 * data_loader.nodes_num)
                data_loader.mode = algo_mode
                data_loader.world_size = world_size

                evaluator = TDEEvaluator(pop_size=int(result["hyperparams"]["pop_size"]), graph=data_loader.G, budget=data_loader.k, device=device)
                evaluator = maybe_wrap_distributed(evaluator)
                if hasattr(evaluator, "comm_stats"):
                    comm_tracker = evaluator
                controller = TDEController(
                    path=str(results_dir) + "/",
                    pattern="write",
                    data_loader=data_loader,
                    loops=1,
                    crossover_rate=float(crossover_rate),
                    mutate_rate=float(mutate_rate),
                    pop_size=int(result["hyperparams"]["pop_size"]),
                    device=device,
                )
                if comm_path:
                    controller.comm_path = str(comm_path)
                fitness_goal = getattr(controller, "fit_side", None)
                attach_observer(controller)
                emit({"type": "log", "line": f"[INFO] Start TDE: mode={algo_mode} device={device} world_size={world_size}"})
                result["metrics"].append({"stage": "init", **snapshot()})
                TDE(mode=algo_mode, max_generation=int(iterations), data_loader=data_loader, controller=controller, evaluator=evaluator, world_size=world_size, verbose=False)
                return

            raise RuntimeError(f"Unsupported CND algorithm: {method}")

        def run_cda(method: str) -> None:
            nonlocal fitness_goal
            nonlocal comm_tracker
            attack_rate = float(os.getenv("GAPA_CDA_ATTACK_RATE", "0.1"))
            loaded = _load_gml(dataset, sort_nodes=True, rebuild_from_adj=False)
            data_loader = Loader(dataset=dataset, device=device)
            data_loader.G = loaded["G"]
            data_loader.A = loaded["A"]
            data_loader.nodes_num = int(data_loader.A.shape[0])
            data_loader.nodes = torch.tensor(loaded["nodelist"], device=device)
            data_loader.selected_genes_num = int(attack_rate * 4 * data_loader.nodes_num)
            data_loader.k = int(attack_rate * data_loader.nodes_num)
            data_loader.mode = algo_mode
            data_loader.world_size = world_size

            if method == "CDA-EDA":
                from gapa.algorithm.CDA.EDA import EDA as CDA_EDA, EDAController as CDA_EDAController, EDAEvaluator as CDA_EDAEvaluator  # type: ignore

                evaluator = CDA_EDAEvaluator(pop_size=int(result["hyperparams"]["pop_size"]), graph=data_loader.G.copy(), nodes_num=data_loader.nodes_num, adj=data_loader.A, device=device)
                evaluator = maybe_wrap_distributed(evaluator)
                controller = CDA_EDAController(
                    path=str(results_dir) + "/",
                    pattern="write",
                    data_loader=data_loader,
                    loops=1,
                    crossover_rate=float(crossover_rate),
                    mutate_rate=float(mutate_rate),
                    pop_size=int(result["hyperparams"]["pop_size"]),
                    device=device,
                )
                if comm_path:
                    controller.comm_path = str(comm_path)
                if hasattr(evaluator, "comm_stats"):
                    comm_tracker = evaluator
                fitness_goal = getattr(controller, "fit_side", None)
                attach_observer(controller)
                emit({"type": "log", "line": f"[INFO] Start CDA-EDA: mode={algo_mode} device={device} world_size={world_size}"})
                result["metrics"].append({"stage": "init", **snapshot()})
                CDA_EDA(mode=algo_mode, max_generation=int(iterations), data_loader=data_loader, controller=controller, evaluator=evaluator, world_size=world_size, verbose=False)
                return

            if method == "CGN":
                from gapa.algorithm.CDA.CGN import CGN, CGNController, CGNEvaluator  # type: ignore

                evaluator = CGNEvaluator(pop_size=int(result["hyperparams"]["pop_size"]), graph=data_loader.G.copy(), device=device)
                evaluator = maybe_wrap_distributed(evaluator)
                controller = CGNController(
                    path=str(results_dir) + "/",
                    pattern="write",
                    data_loader=data_loader,
                    loops=1,
                    crossover_rate=float(crossover_rate),
                    mutate_rate=float(mutate_rate),
                    pop_size=int(result["hyperparams"]["pop_size"]),
                    device=device,
                )
                if comm_path:
                    controller.comm_path = str(comm_path)
                if hasattr(evaluator, "comm_stats"):
                    comm_tracker = evaluator
                fitness_goal = getattr(controller, "fit_side", None)
                attach_observer(controller)
                emit({"type": "log", "line": f"[INFO] Start CGN: mode={algo_mode} device={device} world_size={world_size}"})
                result["metrics"].append({"stage": "init", **snapshot()})
                CGN(mode=algo_mode, max_generation=int(iterations), data_loader=data_loader, controller=controller, evaluator=evaluator, world_size=world_size, verbose=False)
                return

            if method == "QAttack":
                from gapa.algorithm.CDA.QAttack import QAttack, QAttackController, QAttackEvaluator  # type: ignore

                evaluator = QAttackEvaluator(pop_size=int(result["hyperparams"]["pop_size"]), graph=data_loader.G.copy(), device=device)
                evaluator = maybe_wrap_distributed(evaluator)
                controller = QAttackController(
                    path=str(results_dir) + "/",
                    pattern="write",
                    data_loader=data_loader,
                    loops=1,
                    crossover_rate=float(crossover_rate),
                    mutate_rate=float(mutate_rate),
                    pop_size=int(result["hyperparams"]["pop_size"]),
                    device=device,
                )
                if comm_path:
                    controller.comm_path = str(comm_path)
                if hasattr(evaluator, "comm_stats"):
                    comm_tracker = evaluator
                fitness_goal = getattr(controller, "fit_side", None)
                attach_observer(controller)
                emit({"type": "log", "line": f"[INFO] Start QAttack: mode={algo_mode} device={device} world_size={world_size}"})
                result["metrics"].append({"stage": "init", **snapshot()})
                QAttack(mode=algo_mode, max_generation=int(iterations), data_loader=data_loader, controller=controller, evaluator=evaluator, world_size=world_size, verbose=False)
                return

            raise RuntimeError(f"Unsupported CDA algorithm: {method}")

        def run_lpa(method: str) -> None:
            nonlocal fitness_goal
            nonlocal comm_tracker
            attack_rate = float(os.getenv("GAPA_LPA_ATTACK_RATE", "0.1"))
            try:
                loaded = _load_gml(dataset, sort_nodes=True, rebuild_from_adj=True)
            except Exception:
                loaded = _load_adjlist(dataset, sort_nodes=True)

            data_loader = Loader(dataset=dataset, device=device)
            data_loader.G = loaded["G"]
            data_loader.A = loaded["A"]
            data_loader.nodes_num = int(data_loader.A.shape[0])
            data_loader.nodes = torch.tensor(list(data_loader.G.nodes), device=device)
            data_loader.edges = torch.tensor(list(data_loader.G.edges), device=device)
            data_loader.edges_num = int(len(data_loader.edges))
            data_loader.selected_genes_num = int(attack_rate * 4 * data_loader.nodes_num)
            data_loader.k = float(attack_rate)
            data_loader.mode = algo_mode
            data_loader.world_size = world_size

            if method == "LPA-EDA":
                from gapa.algorithm.LPA.EDA import EDA as LPA_EDA, EDAController as LPA_EDAController, EDAEvaluator as LPA_EDAEvaluator  # type: ignore

                evaluator = LPA_EDAEvaluator(pop_size=int(result["hyperparams"]["pop_size"]), graph=data_loader.G, ratio=0, device=device)
                evaluator = maybe_wrap_distributed(evaluator)
                controller = LPA_EDAController(
                    path=str(results_dir) + "/",
                    pattern="write",
                    data_loader=data_loader,
                    loops=1,
                    mutate_rate=float(mutate_rate),
                    pop_size=int(result["hyperparams"]["pop_size"]),
                    num_eda_pop=int(result["hyperparams"]["pop_size"]),
                    device=device,
                )
                if comm_path:
                    controller.comm_path = str(comm_path)
                if hasattr(evaluator, "comm_stats"):
                    comm_tracker = evaluator
                fitness_goal = getattr(controller, "fit_side", None)
                attach_observer(controller)
                emit({"type": "log", "line": f"[INFO] Start LPA-EDA: mode={algo_mode} device={device} world_size={world_size}"})
                result["metrics"].append({"stage": "init", **snapshot()})
                LPA_EDA(mode=algo_mode, max_generation=int(iterations), data_loader=data_loader, controller=controller, evaluator=evaluator, world_size=world_size, verbose=False)
                return

            if method == "LPA-GA":
                from gapa.algorithm.LPA.LPA_GA import LPA_GA, GAController, GAEvaluator  # type: ignore

                evaluator = GAEvaluator(pop_size=int(result["hyperparams"]["pop_size"]), graph=data_loader.G, ratio=0, device=device)
                evaluator = maybe_wrap_distributed(evaluator)
                controller = GAController(
                    path=str(results_dir) + "/",
                    pattern="write",
                    data_loader=data_loader,
                    loops=1,
                    crossover_rate=float(crossover_rate),
                    mutate_rate=float(mutate_rate),
                    pop_size=int(result["hyperparams"]["pop_size"]),
                    device=device,
                )
                if comm_path:
                    controller.comm_path = str(comm_path)
                if hasattr(evaluator, "comm_stats"):
                    comm_tracker = evaluator
                fitness_goal = getattr(controller, "fit_side", None)
                attach_observer(controller)
                emit({"type": "log", "line": f"[INFO] Start LPA-GA: mode={algo_mode} device={device} world_size={world_size}"})
                result["metrics"].append({"stage": "init", **snapshot()})
                LPA_GA(mode=algo_mode, max_generation=int(iterations), data_loader=data_loader, controller=controller, evaluator=evaluator, world_size=world_size, verbose=False)
                return

            raise RuntimeError(f"Unsupported LPA algorithm: {method}")

        def run_nca(method: str) -> None:
            nonlocal fitness_goal
            from gapa.utils.dataset import load_dataset  # type: ignore
            from gapa.DeepLearning.Classifier import Classifier, load_set  # type: ignore

            ds_key = (dataset or "").strip().lower()
            if ds_key in ("chameleon_filtered", "chameleon"):
                ds_key = "chameleon"
            elif ds_key in ("squirrel_filtered", "squirrel"):
                ds_key = "squirrel"

            data_loader = Loader(ds_key, device)
            ds_obj = load_dataset(ds_key, model="gcn", device=torch.device(device))
            adj, feats, labels = ds_obj.adj, ds_obj.feats, ds_obj.labels
            train_index, val_index, test_index = ds_obj.train_index, ds_obj.val_index, ds_obj.test_index
            num_nodes, num_feats, num_classes = ds_obj.num_nodes, ds_obj.num_feats, ds_obj.num_classes
            num_edge = int(torch.count_nonzero(adj.to_dense()).item())

            data_loader.G = nx.Graph(adj.to_dense().cpu().numpy())
            data_loader.adj = adj
            data_loader.test_index = test_index
            data_loader.feats = feats
            data_loader.labels = labels
            data_loader.num_edge = num_edge
            data_loader.train_index = train_index
            data_loader.val_index = val_index
            data_loader.num_nodes = num_nodes
            data_loader.num_feats = num_feats
            data_loader.num_classes = num_classes
            data_loader.mode = algo_mode
            data_loader.world_size = world_size

            model_dir = Path(os.getenv("GAPA_MODEL_DIR", str(repo_root / "experiment_data" / "Model")))
            model_dir.mkdir(parents=True, exist_ok=True)
            load_set(data_loader.dataset, "gcn", num_nodes=num_nodes, num_edge=num_edge)
            gcn = Classifier(model_name="gcn", input_dim=num_feats, output_dim=num_classes, device=device)
            save_path = model_dir / f"nc_gcn_{data_loader.dataset}.pt"
            retrain = os.getenv("GAPA_NCA_RETRAIN", "0").strip() in ("1", "true", "yes")
            if retrain or not gcn.load_model(str(save_path)):
                emit({"type": "log", "line": f"[INFO] Train classifier for NCA: {ds_key} (may take a while)"})
                gcn.initialize()
                gcn.fit(feats, adj, labels, train_index, val_index, verbose=False)
                gcn.save(str(save_path))

            if method == "NCA-GA":
                from gapa.algorithm.NCA.NCA_GA import NCA_GA, NCA_GAController, NCA_GAEvaluator  # type: ignore

                attack_rate = float(os.getenv("GAPA_NCA_EDGE_ATTACK_RATE", "0.025"))
                data_loader.k = int(attack_rate * num_edge)
                evaluator = NCA_GAEvaluator(classifier=gcn, feats=data_loader.feats, adj=data_loader.adj, test_index=data_loader.test_index, labels=data_loader.labels, pop_size=int(result["hyperparams"]["pop_size"]), device=device)
                evaluator = maybe_wrap_distributed(evaluator)
                controller = NCA_GAController(
                    path=str(results_dir) + "/",
                    pattern="write",
                    data_loader=data_loader,
                    classifier=gcn,
                    loops=1,
                    crossover_rate=float(crossover_rate),
                    mutate_rate=float(mutate_rate),
                    pop_size=int(result["hyperparams"]["pop_size"]),
                    device=device,
                )
                if comm_path:
                    controller.comm_path = str(comm_path)
                if hasattr(evaluator, "comm_stats"):
                    comm_tracker = evaluator
                fitness_goal = getattr(controller, "fit_side", None)
                attach_observer(controller)
                emit({"type": "log", "line": f"[INFO] Start NCA-GA: mode={algo_mode} device={device} world_size={world_size}"})
                result["metrics"].append({"stage": "init", **snapshot()})
                NCA_GA(mode=algo_mode, max_generation=int(iterations), data_loader=data_loader, controller=controller, evaluator=evaluator, world_size=world_size, verbose=False)
                return

            if method in ("SGA", "GANI"):
                from gapa.algorithm.NCA.SGA import SGA, SGAController  # type: ignore

                attack_rate = float(os.getenv("GAPA_SGA_NODE_ATTACK_RATE", "0.05"))
                data_loader.k = int(attack_rate * num_nodes)
                try:
                    from deeprobust.graph.defense import GCN as DR_GCN  # type: ignore
                except Exception as e:
                    raise RuntimeError(f"deeprobust is required for {method}: {e}")

                surrogate = DR_GCN(nfeat=num_feats, nclass=num_classes, nhid=16, dropout=0.5, with_relu=False, with_bias=True, device=device).to(torch.device(device))
                emit({"type": "log", "line": f"[INFO] Train surrogate (deeprobust GCN) for {method}: {ds_key}"})
                surrogate.fit(feats, adj, labels, train_index, val_index, patience=30)

                controller = SGAController(
                    path=str(results_dir) + "/",
                    pattern="write",
                    data_loader=data_loader,
                    classifier=gcn,
                    loops=1,
                    crossover_rate=float(crossover_rate),
                    mutate_rate=float(mutate_rate),
                    pop_size=int(result["hyperparams"]["pop_size"]),
                    device=device,
                )
                if comm_path:
                    controller.comm_path = str(comm_path)
                fitness_goal = getattr(controller, "fit_side", None)
                attach_observer(controller)
                emit({"type": "log", "line": f"[INFO] Start {method}: mode={algo_mode} device={device} world_size={world_size}"})
                result["metrics"].append({"stage": "init", **snapshot()})
                # SGA/GANI internally create evaluators; we pass maybe_wrap_distributed to wrap them
                SGA(mode=algo_mode, max_generation=int(iterations), data_loader=data_loader, controller=controller, surrogate=surrogate, classifier=gcn, homophily_ratio=0.7, world_size=world_size, wrap_evaluator=maybe_wrap_distributed, verbose=False)
                return

            raise RuntimeError(f"Unsupported NCA algorithm: {method}")

        if algo_norm in ("SixDST", "CutOff", "TDE"):
            run_cnd(algo_norm)
        elif algo_norm in ("CDA-EDA", "CGN", "QAttack"):
            run_cda(algo_norm)
        elif algo_norm in ("LPA-EDA", "LPA-GA"):
            run_lpa(algo_norm)
        elif algo_norm in ("NCA-GA", "SGA", "GANI"):
            run_nca(algo_norm)
        else:
            raise RuntimeError(f"Real GA runner not implemented for algorithm={algo_norm}")

        with res_lock:
            if iter_t0 is not None and iter_t1 is not None and iter_t1 >= iter_t0:
                iter_seconds = float(iter_t1 - iter_t0)
            else:
                iter_seconds = None
            result["timing"] = {
                "iter_seconds": iter_seconds,
                "note": "iter_seconds excludes preprocessing; timing starts at first iteration callback and ends at last callback",
            }
            if iter_seconds is not None:
                iters = int(iterations)
                pop_size = int(result["hyperparams"].get("pop_size", 0) or 0)
                iter_avg_ms = (iter_seconds / max(1, iters)) * 1000.0
                throughput = (pop_size * iters / iter_seconds) if iter_seconds > 0 else None
                result["timing"]["iter_avg_ms"] = float(iter_avg_ms)
                result["timing"]["throughput_ips"] = float(throughput) if throughput is not None else None
        if result.get("timing", {}).get("iter_seconds") is not None:
            emit({"type": "log", "line": f"[INFO] iteration loop finished; iter_seconds={result['timing']['iter_seconds']:.3f}s"})
        else:
            emit({"type": "log", "line": "[WARN] iteration timing not available (no per-iteration callbacks captured)"})
        
        # Ensure progress reaches 100% after completion
        emit({"type": "progress", "value": 100})

        if fitness_goal:
            result["fitness_goal"] = fitness_goal
        if comm_path and comm_path.exists():
            try:
                import json

                comm_data = json.loads(comm_path.read_text(encoding="utf-8"))
                if result.get("comm"):
                    result["comm_process"] = comm_data
                else:
                    result["comm"] = comm_data
            except Exception:
                pass
        if comm_tracker is not None:
            try:
                comm_stats = comm_tracker.comm_stats()
                result["comm"] = comm_stats
                # Collect detailed stats if available
                if hasattr(comm_tracker, "detailed_stats"):
                    result["comm_detailed"] = comm_tracker.detailed_stats()
                
                # Emit detailed comm timing log for user visibility
                if comm_stats.get("avg_ms") is not None:
                    emit({"type": "log", "line": f"[INFO] Comm avg: {comm_stats['avg_ms']:.2f}ms total: {comm_stats.get('total_ms', 0):.2f}ms calls: {comm_stats.get('calls', 0)}"})
                
                # Log per-op breakdown for M mode (torch.distributed)
                per_op_ms = comm_stats.get("per_rank_ops", {}).get("0", {}) or {}
                if per_op_ms:
                    op_items = [f"{op} {ms:.1f}ms" for op, ms in per_op_ms.items()]
                    if op_items:
                        emit({"type": "log", "line": f"[INFO] Comm breakdown: {', '.join(op_items)}"})
                
                # Log MNM mode detailed breakdown
                detailed = result.get("comm_detailed") or {}
                if detailed.get("total_comm_ms", 0) > 0:
                    total_ms = detailed.get("total_comm_ms", 0)
                    calls = detailed.get("calls", 0)
                    serialize_ms = detailed.get("total_serialize_ms", 0)
                    network_ms = detailed.get("total_network_ms", 0)
                    compute_ms = detailed.get("total_compute_ms", 0)
                    deserialize_ms = detailed.get("total_deserialize_ms", 0)
                    total_bytes = detailed.get("total_bytes", 0)
                    
                    # Calculate percentages
                    def pct(v): return (v / total_ms * 100) if total_ms > 0 else 0
                    
                    # Format bytes
                    if total_bytes >= 1024 * 1024:
                        bytes_str = f"{total_bytes / (1024 * 1024):.1f} MB"
                    elif total_bytes >= 1024:
                        bytes_str = f"{total_bytes / 1024:.1f} KB"
                    else:
                        bytes_str = f"{total_bytes} B"
                    
                    emit({"type": "log", "line": "─" * 60})
                    emit({"type": "log", "line": f"[INFO] ▼ 通信统计 ({calls}次调用, 总数据量: {bytes_str})"})
                    emit({"type": "log", "line": f"[INFO]   总通信时间: {total_ms:.1f} ms"})
                    emit({"type": "log", "line": f"[INFO]   ├─ 序列化:    {serialize_ms:.1f} ms ({pct(serialize_ms):.1f}%)"})
                    emit({"type": "log", "line": f"[INFO]   ├─ 网络传输: {network_ms:.1f} ms ({pct(network_ms):.1f}%)"})
                    emit({"type": "log", "line": f"[INFO]   ├─ 远程计算: {compute_ms:.1f} ms ({pct(compute_ms):.1f}%)"})
                    emit({"type": "log", "line": f"[INFO]   └─ 反序列化: {deserialize_ms:.1f} ms ({pct(deserialize_ms):.1f}%)"})
                    
                    # Per-worker breakdown
                    per_worker = detailed.get("per_worker", {})
                    if per_worker:
                        emit({"type": "log", "line": f"[INFO]   ▼ 按 Worker 分解"})
                        for wid, ws in per_worker.items():
                            w_calls = ws.get("calls", 0)
                            w_total = ws.get("total_ms", 0)
                            w_avg = ws.get("avg_ms", 0)
                            w_bytes = ws.get("total_bytes", 0)
                            if w_bytes >= 1024 * 1024:
                                w_bytes_str = f"{w_bytes / (1024 * 1024):.1f} MB"
                            elif w_bytes >= 1024:
                                w_bytes_str = f"{w_bytes / 1024:.1f} KB"
                            else:
                                w_bytes_str = f"{w_bytes} B"
                            emit({"type": "log", "line": f"[INFO]     {wid}: {w_calls}次 avg={w_avg:.1f}ms total={w_total:.1f}ms data={w_bytes_str}"})
                    emit({"type": "log", "line": "─" * 60})
            except Exception:
                pass
        else:
            # No comm_tracker available
            if distributed_fitness:
                emit({"type": "log", "line": "[WARN] MNM 模式已启用但无法获取通信统计（evaluator 可能未被包装或不支持）"})

        tail_stop = True
        primary = objective.get("primary") or "fitness"
        secondary = objective.get("secondary")
        primary_vals = (result.get("curves") or {}).get(primary) or []
        secondary_vals = (result.get("curves") or {}).get(secondary) or []
        if primary_vals:
            if objective.get("primary_goal") == "max":
                idx = max(range(len(primary_vals)), key=lambda i: primary_vals[i])
            else:
                idx = min(range(len(primary_vals)), key=lambda i: primary_vals[i])
            best = {primary: primary_vals[idx]}
            if secondary and idx < len(secondary_vals):
                best[secondary] = secondary_vals[idx]
            result["best_metrics"] = best
            if secondary and secondary in best:
                result["best_score"] = f"{primary}={best[primary]:.6g}, {secondary}={best[secondary]:.6g}"
            else:
                result["best_score"] = f"{primary}={best[primary]:.6g}"
        else:
            result["best_metrics"] = None
            result["best_score"] = None

        emit({"type": "log", "line": "[INFO] 分析完成。"})
        
        # Save GA state for potential resume
        try:
            db_manager.save_ga_state(task_id, algorithm, dataset, {
                "last_result": result,
                "timestamp": time.time()
            })
        except Exception as e:
            emit({"type": "log", "line": f"[WARN] 状态保存失败: {e}"})

        emit({"type": "result", "result": result})
        emit({"type": "state", "state": "completed"})
    except Exception as exc:
        try:
            q.put({"type": "log", "line": f"[ERROR] {exc}", "task_id": task_id})
            q.put({"type": "state", "state": "error", "error": str(exc), "task_id": task_id})
        except Exception:
            pass
