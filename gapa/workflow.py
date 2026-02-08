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

Utility Functions:
    - load_dataset: Load graph dataset by name

Example:
    >>> from gapa.workflow import Workflow, load_dataset, Monitor
    >>> from examples.sixdst_custom import SixDSTAlgorithm
    >>> 
    >>> data = load_dataset("ForestFire_n500")
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
    "load_dataset",
]


def _results_root() -> Path:
    return Path(os.getenv("GAPA_RESULTS_DIR", str(Path(__file__).resolve().parents[1] / "results")))


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
    cfg = os.getenv("GAPA_SERVERS_FILE")
    path = Path(cfg) if cfg else Path(__file__).resolve().parents[1] / "servers.json"
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            servers = raw.get("servers", []) if isinstance(raw, dict) else []
            for entry in servers:
                if not isinstance(entry, dict):
                    continue
                port = entry.get("port")
                protocol = entry.get("protocol") or "http"
                if port:
                    return f"{protocol}://127.0.0.1:{port}"
        except Exception:
            pass
    return "http://127.0.0.1:5000"

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
import networkx as nx
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency for monitor HTTP
    requests = None

# =============================================================================
# Feature Flags (for PyPI vs Source deployment)
# =============================================================================

# Check if distributed (MNM) features are available
# These are only available in source deployment, not in PyPI package
try:
    from server.distributed_evaluator import DistributedEvaluator
    HAS_DISTRIBUTED = True
except ImportError:
    HAS_DISTRIBUTED = False
    DistributedEvaluator = None

# Supported modes in PyPI package vs Source deployment
PYPI_MODES = ["s", "sm", "m"]
SOURCE_MODES = ["s", "sm", "m", "mnm"]


# =============================================================================
# Data Loading
# =============================================================================

class DataLoader:
    """
    Container for graph dataset.
    
    Compatible with the existing GAPA data loader interface.
    """
    
    def __init__(
        self,
        name: str,
        G: nx.Graph,
        A: torch.Tensor,
        nodes: torch.Tensor,
        k: int,
        selected_genes_num: int,
        device: torch.device,
    ):
        self.name = name
        self.dataset = name  # Alias for compatibility with existing code
        self.G = G
        self.A = A
        self.nodes = nodes
        self.nodes_num = len(nodes)
        self.k = k
        self.selected_genes_num = selected_genes_num
        self.device = device


def load_dataset(
    name: str,
    *,
    detection_rate: float = 0.1,
    selected_genes_rate: float = 0.4,
    device: str = "auto",
    sort_nodes: bool = True,
    paths: Optional[List[str]] = None,
    format: Optional[str] = None,
) -> DataLoader:
    """
    Load a graph dataset by name or path.
    
    Supports multiple common graph formats with auto-detection.
    
    Supported Formats:
        - **edgelist**: Two-column file (source target), whitespace/comma separated
        - **csv**: CSV with 'source,target' or first two columns as edges
        - **adjlist**: Adjacency list format
        - **gml**: Graph Modeling Language
        - **graphml**: GraphML XML format
        - **gexf**: GEXF format (Gephi)
        - **npz**: NumPy adjacency matrix
    
    Args:
        name: Dataset name (e.g., "ForestFire_n500") or full file path
        detection_rate: Fraction of nodes to select as budget (k). Default: 0.1
        selected_genes_rate: Fraction of nodes for gene pool. Default: 0.4
        device: Computation device - "auto", "cuda", or "cpu". Default: "auto"
        sort_nodes: Whether to sort node labels for consistent indexing. Default: True
        paths: Custom search paths (optional). Will be searched first.
        format: Force specific format. Auto-detected if None.
    
    Returns:
        DataLoader with graph data
    
    Example:
        >>> # By name (searches in standard locations)
        >>> data = load_dataset("ForestFire_n500")
        
        >>> # Direct path with edge list
        >>> data = load_dataset("/path/to/edges.txt", format="edgelist")
        
        >>> # CSV file
        >>> data = load_dataset("interactions.csv", format="csv")
    """
    import numpy as np
    
    # Determine device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Check if name is a direct path
    direct_path = Path(name)
    if direct_path.exists() and direct_path.is_file():
        graph_path = direct_path
        dataset_name = direct_path.stem
    else:
        # Build search paths
        project_root = Path(__file__).parent.parent
        
        # Supported extensions
        extensions = [".txt", ".csv", ".edgelist", ".gml", ".graphml", ".gexf", ".npz"]
        
        search_paths = []
        if paths:
            for p in paths:
                for ext in extensions:
                    search_paths.append(Path(p) / f"{name}{ext}")
        
        for ext in extensions:
            search_paths.extend([
                project_root / "datasets" / f"{name}{ext}",
                project_root / "dataset" / f"{name}{ext}",
                project_root / "datasets" / name / f"{name}{ext}",
                project_root / "dataset" / name / f"{name}{ext}",
                Path.cwd() / "datasets" / f"{name}{ext}",
                Path.cwd() / f"{name}{ext}",
            ])
        
        graph_path = None
        for p in search_paths:
            if p.exists():
                graph_path = p
                break
        
        if graph_path is None:
            tried_paths = "\n  - ".join(str(p) for p in search_paths[:8])
            raise FileNotFoundError(
                f"Dataset '{name}' not found.\n\n"
                f"Searched in:\n  - {tried_paths}\n\n"
                f"Supported formats: edgelist, csv, adjlist, gml, graphml, gexf, npz\n\n"
                f"Suggestions:\n"
                f"  1. Provide full path: load_dataset('/path/to/data.csv')\n"
                f"  2. Specify format: load_dataset('name', format='edgelist')\n"
                f"  3. Place file in ./datasets/ directory"
            )
        dataset_name = name
    
    print(f"[GAPA] Loading dataset: {graph_path}")
    
    # Auto-detect format from extension if not specified
    suffix = graph_path.suffix.lower()
    if format is None:
        format_map = {
            ".txt": "auto",
            ".csv": "csv",
            ".edgelist": "edgelist",
            ".gml": "gml",
            ".graphml": "graphml",
            ".gexf": "gexf",
            ".npz": "npz",
        }
        format = format_map.get(suffix, "auto")
    
    # Load graph based on format
    G = _load_graph(graph_path, format)
    
    # Process graph
    if sort_nodes:
        nodelist = sorted(list(G.nodes()))
        A = torch.tensor(nx.to_numpy_array(G, nodelist=nodelist), device=device, dtype=torch.float32)
        G = nx.from_numpy_array(A.cpu().numpy())
    else:
        A = torch.tensor(nx.to_numpy_array(G), device=device, dtype=torch.float32)
    
    nodes = torch.tensor(list(G.nodes()), device=device)
    nodes_num = len(nodes)
    k = max(1, int(detection_rate * nodes_num))
    selected_genes_num = max(1, int(selected_genes_rate * nodes_num))
    
    print(f"[GAPA] Graph loaded: {nodes_num} nodes, {len(G.edges())} edges")
    print(f"[GAPA] Budget (k): {k}, Gene pool: {selected_genes_num}")
    
    return DataLoader(
        name=dataset_name,
        G=G,
        A=A,
        nodes=nodes,
        k=k,
        selected_genes_num=selected_genes_num,
        device=device,
    )


def _load_graph(path: Path, format: str) -> nx.Graph:
    """
    Load graph from file with format auto-detection.
    
    Supports: edgelist, csv, adjlist, gml, graphml, gexf, npz
    """
    import numpy as np
    
    path_str = str(path)
    
    if format == "gml":
        return nx.read_gml(path_str, label="id")
    
    elif format == "graphml":
        return nx.read_graphml(path_str)
    
    elif format == "gexf":
        return nx.read_gexf(path_str)
    
    elif format == "npz":
        data = np.load(path_str)
        # Try common keys for adjacency matrix
        for key in ["A", "adj", "adjacency", "matrix", data.files[0]]:
            if key in data.files:
                adj = data[key]
                return nx.from_numpy_array(adj)
        raise ValueError(f"NPZ file does not contain adjacency matrix. Keys: {data.files}")
    
    elif format == "csv":
        return _load_edgelist_csv(path)
    
    elif format == "edgelist":
        return _load_edgelist(path)
    
    elif format == "auto":
        # Try to auto-detect based on content
        return _load_auto(path)
    
    else:
        # Default: try adjlist
        try:
            return nx.read_adjlist(path_str, nodetype=int)
        except:
            return _load_edgelist(path)


def _load_edgelist(path: Path) -> nx.Graph:
    """Load edge list (two columns: source target)."""
    edges = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('%'):
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) >= 2:
                try:
                    src, tgt = int(parts[0]), int(parts[1])
                    edges.append((src, tgt))
                except ValueError:
                    # Try as string nodes
                    edges.append((parts[0], parts[1]))
    
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def _load_edgelist_csv(path: Path) -> nx.Graph:
    """Load CSV edge list."""
    import csv
    
    edges = []
    with open(path, 'r', newline='') as f:
        # Skip BOM if present
        sample = f.read(1024)
        f.seek(0)
        
        # Detect delimiter
        dialect = csv.Sniffer().sniff(sample, delimiters=',\t;')
        has_header = csv.Sniffer().has_header(sample)
        
        reader = csv.reader(f, dialect)
        if has_header:
            next(reader)  # Skip header
        
        for row in reader:
            if len(row) >= 2:
                try:
                    src, tgt = int(row[0]), int(row[1])
                except ValueError:
                    src, tgt = row[0].strip(), row[1].strip()
                edges.append((src, tgt))
    
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def _load_auto(path: Path) -> nx.Graph:
    """Auto-detect format and load."""
    with open(path, 'r') as f:
        first_lines = [f.readline() for _ in range(5)]
    
    content = ''.join(first_lines)
    
    # Check for GML markers
    if 'graph [' in content.lower() or 'node [' in content.lower():
        return nx.read_gml(str(path), label="id")
    
    # Check for GraphML
    if '<?xml' in content and 'graphml' in content.lower():
        return nx.read_graphml(str(path))
    
    # Check if it's CSV-like
    if ',' in content and not content.strip().startswith('#'):
        return _load_edgelist_csv(path)
    
    # Try edge list first (most common)
    try:
        G = _load_edgelist(path)
        if G.number_of_edges() > 0:
            return G
    except:
        pass
    
    # Fall back to adjlist
    try:
        return nx.read_adjlist(str(path), nodetype=int)
    except:
        pass
    
    # Last resort: edge list without type conversion
    return _load_edgelist(path)


# =============================================================================
# Monitor (Observer Pattern)
# =============================================================================

class Monitor:
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

    def set_run_context(self, context: Dict[str, Any]) -> None:
        self._run_context = dict(context)

    def _resolve_api_base(self) -> str:
        if self.api_base:
            return self.api_base.rstrip("/")
        return _resolve_default_api_base()

    def _guess_api_base_from_servers_file(self) -> Optional[str]:
        cfg = os.getenv("GAPA_SERVERS_FILE")
        if cfg:
            path = Path(cfg)
        else:
            path = Path(__file__).resolve().parents[1] / "servers.json"
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        servers = raw.get("servers", []) if isinstance(raw, dict) else []
        for entry in servers:
            if not isinstance(entry, dict):
                continue
            port = entry.get("port")
            protocol = entry.get("protocol") or "http"
            if port:
                return f"{protocol}://127.0.0.1:{port}"
        return None

    def _http_get_json(self, url: str) -> Dict[str, Any]:
        if requests is None:
            return {"error": "requests not available"}
        try:
            resp = requests.get(url, timeout=self.timeout_s)
        except Exception as exc:
            return {"error": str(exc)}
        if not getattr(resp, "ok", False):
            try:
                body = resp.json()
            except Exception:
                body = {"raw": getattr(resp, "text", "")}
            return {"error": f"HTTP {resp.status_code}", "body": body}
        try:
            return resp.json()
        except Exception as exc:
            return {"error": f"invalid json: {exc}"}

    def _http_get_json_with_params(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not params:
            return self._http_get_json(url)
        query = urlencode({k: v for k, v in params.items() if v is not None})
        target = f"{url}?{query}" if query else url
        return self._http_get_json(target)

    def _http_post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if requests is None:
            return {"error": "requests not available"}
        try:
            session = requests.Session()
            session.trust_env = False
            resp = session.post(url, json=payload, timeout=(3.0, max(5.0, self.timeout_s)))
        except Exception as exc:
            return {"error": str(exc)}
        body: Any
        try:
            body = resp.json()
        except Exception:
            body = {"raw": (getattr(resp, "text", "") or "")[:500]}
        if not getattr(resp, "ok", False):
            return {"error": f"HTTP {resp.status_code}", "body": body}
        return body if isinstance(body, dict) else {"data": body}

    def _post_server_direct(self, server_id: str, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Directly call remote server endpoint by server id/name, bypassing local app proxy."""
        if requests is None:
            return {"error": "requests not available"}
        servers = self.server()
        if not isinstance(servers, list):
            return {"error": "server list unavailable", "detail": servers}
        target = None
        for s in servers:
            sid = str(s.get("id") or "")
            name = str(s.get("name") or "")
            if server_id in (sid, name) or server_id.lower() in (sid.lower(), name.lower()):
                target = s
                break
        if target is None:
            return {"error": "server not found", "server_id": server_id}
        base_url = str(target.get("base_url") or "").rstrip("/")
        if not base_url:
            return {"error": "missing base_url", "server_id": server_id, "server": target}
        try:
            session = requests.Session()
            session.trust_env = False
            resp = session.post(base_url + endpoint, json=payload, timeout=(3.0, max(5.0, self.timeout_s)))
        except Exception as exc:
            return {"error": str(exc), "server_id": server_id, "url": base_url + endpoint}
        try:
            body = resp.json()
        except Exception:
            body = {"raw": (getattr(resp, "text", "") or "")[:500]}
        if not getattr(resp, "ok", False):
            return {"error": f"HTTP {resp.status_code}", "body": body, "server_id": server_id, "url": base_url + endpoint}
        return body if isinstance(body, dict) else {"data": body}

    def _normalize_server_id(self, raw: str, servers: List[Dict[str, Any]]) -> Optional[str]:
        token = str(raw or "").strip()
        if not token:
            return None
        token_lower = token.lower()
        aliases = [token, f"Server {token}", f"server {token}"]
        alias_lower = [a.lower() for a in aliases]
        for item in servers:
            sid = str(item.get("id") or "")
            name = str(item.get("name") or "")
            sid_lower = sid.lower()
            name_lower = name.lower()
            if token in (sid, name) or token_lower in (sid_lower, name_lower):
                return sid
            if sid in aliases or sid_lower in alias_lower:
                return sid
            if name in aliases or name_lower in alias_lower:
                return sid
        return None

    def _fetch_snapshots(self) -> Dict[str, Any]:
        base = self._resolve_api_base()
        data = self._http_get_json(f"{base}/api/v1/resources/all")
        if isinstance(data, dict) and "error" not in data:
            return data
        legacy = self._http_get_json(f"{base}/api/all_resources")
        if isinstance(legacy, dict) and "error" not in legacy:
            return legacy
        return data if isinstance(data, dict) else {"error": "invalid snapshots"}

    def server(self) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Return configured servers with activation status."""
        base = self._resolve_api_base()
        servers = self._http_get_json(f"{base}/api/servers")
        if not isinstance(servers, list):
            return {"error": "invalid servers response", "body": servers}
        snapshots = self._fetch_snapshots()
        if isinstance(snapshots, dict) and "error" in snapshots:
            snapshots = {}
        results: List[Dict[str, Any]] = []
        for item in servers:
            if not isinstance(item, dict):
                continue
            sid = item.get("id") or item.get("name") or item.get("host")
            name = item.get("name") or item.get("id") or sid
            base_url = item.get("base_url")
            if not base_url:
                host = item.get("ip") or item.get("host")
                port = item.get("port")
                protocol = item.get("protocol") or "http"
                if host:
                    base_url = f"{protocol}://{host}{f':{port}' if port else ''}"
            snap = snapshots.get(sid) if isinstance(snapshots, dict) else None
            online = bool(snap and snap.get("online"))
            status = "Activate" if online else "Deactivate"
            results.append(
                {
                    "id": sid,
                    "name": name,
                    "base_url": base_url,
                    "status": status,
                    "online": online,
                }
            )
        return results

    def server_resource(self, server_id_or_name: str) -> Dict[str, Any]:
        """Return current resource snapshot for a server by id or name."""
        servers = self.server()
        if isinstance(servers, dict) and "error" in servers:
            return servers
        target_id = None
        lookup = str(server_id_or_name or "")
        for item in servers:
            sid = item.get("id")
            name = item.get("name")
            if lookup == sid or lookup == name:
                target_id = sid
                break
        if target_id is None:
            for item in servers:
                sid = item.get("id")
                name = item.get("name")
                if name and name.lower() == lookup.lower():
                    target_id = sid
                    break
        if not target_id:
            return {"error": "server not found", "input": server_id_or_name}
        snapshots = self._fetch_snapshots()
        if isinstance(snapshots, dict) and "error" in snapshots:
            return snapshots
        if isinstance(snapshots, dict) and target_id in snapshots:
            return snapshots[target_id]
        return {"error": "server not found in snapshots", "server_id": target_id}

    def resources(self, all_servers: bool = True) -> Dict[str, Any]:
        """Fetch resource snapshots."""
        base = self._resolve_api_base()
        if all_servers:
            data = self._http_get_json(f"{base}/api/v1/resources/all")
            if isinstance(data, dict) and "error" not in data:
                return data
            return self._http_get_json(f"{base}/api/all_resources")
        return self._http_get_json(f"{base}/api/resources")

    def lock_status(self, scope: str = "all", realtime: bool = True) -> Dict[str, Any]:
        """Query resource lock status."""
        base = self._resolve_api_base()
        return self._http_get_json_with_params(
            f"{base}/api/resource_lock/status",
            {
                "scope": scope,
                "realtime": str(bool(realtime)).lower(),
            },
        )

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
        """Acquire resource lock via app API."""
        base = self._resolve_api_base()
        payload: Dict[str, Any] = {
            "scope": scope,
            "duration_s": float(duration_s),
            "warmup_iters": int(warmup_iters),
            "mem_mb": int(mem_mb),
            "strict_idle": bool(strict_idle),
        }
        if devices is not None:
            payload["devices"] = devices
        if devices_by_server is not None:
            payload["devices_by_server"] = devices_by_server
        return self._http_post_json(f"{base}/api/resource_lock", payload)

    def unlock_resource(self, scope: str = "all") -> Dict[str, Any]:
        """Release resource lock via app API."""
        base = self._resolve_api_base()
        return self._http_post_json(f"{base}/api/resource_lock/release", {"scope": scope})

    def renew_resource(
        self,
        scope: str = "all",
        duration_s: Optional[float] = None,
        lock_id: Optional[str] = None,
        owner: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Renew active resource lock via app API."""
        base = self._resolve_api_base()
        payload: Dict[str, Any] = {"scope": scope}
        if duration_s is not None:
            payload["duration_s"] = float(duration_s)
        if lock_id:
            payload["lock_id"] = str(lock_id)
        if owner:
            payload["owner"] = str(owner)
        return self._http_post_json(f"{base}/api/resource_lock/renew", payload)

    def strategy_plan(
        self,
        server_id: str = "local",
        algorithm: Optional[str] = None,
        warmup: int = 0,
        objective: str = "time",
        multi_gpu: bool = True,
        gpu_busy_threshold: Optional[float] = None,
        min_gpu_free_mb: Optional[int] = None,
        tpe_trials: Optional[int] = None,
        tpe_warmup: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Request adaptive strategy plan."""
        base = self._resolve_api_base()
        payload: Dict[str, Any] = {
            "server_id": server_id,
            "warmup": int(warmup),
            "objective": objective,
            "multi_gpu": bool(multi_gpu),
        }
        if algorithm:
            payload["algorithm"] = algorithm
        if gpu_busy_threshold is not None:
            payload["gpu_busy_threshold"] = float(gpu_busy_threshold)
        if min_gpu_free_mb is not None:
            payload["min_gpu_free_mb"] = int(min_gpu_free_mb)
        if tpe_trials is not None:
            payload["tpe_trials"] = int(tpe_trials)
        if tpe_warmup is not None:
            payload["tpe_warmup"] = int(tpe_warmup)
        return self._http_post_json(f"{base}/api/strategy_plan", payload)

    def distributed_strategy_plan(
        self,
        servers: Optional[List[str]] = None,
        per_server_gpus: int = 1,
        min_gpu_free_mb: int = 1024,
        gpu_busy_threshold: float = 85.0,
    ) -> Dict[str, Any]:
        """Request distributed strategy plan."""
        base = self._resolve_api_base()
        payload: Dict[str, Any] = {
            "server_ids": servers or [],
            "per_server_gpus": int(per_server_gpus),
            "min_gpu_free_mb": int(min_gpu_free_mb),
            "gpu_busy_threshold": float(gpu_busy_threshold),
        }
        return self._http_post_json(f"{base}/api/distributed_strategy_plan", payload)

    def analysis_status(self, server_id: Optional[str] = None) -> Dict[str, Any]:
        """Query analysis runtime status for local/remote server."""
        base = self._resolve_api_base()
        return self._http_get_json_with_params(
            f"{base}/api/analysis/status",
            {"server_id": server_id} if server_id else None,
        )

    def analysis_queue(self, server_id: Optional[str] = None) -> Dict[str, Any]:
        """Query queued analysis tasks for local/remote server."""
        base = self._resolve_api_base()
        return self._http_get_json_with_params(
            f"{base}/api/analysis/queue",
            {"server_id": server_id} if server_id else None,
        )

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
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Start (or queue) an analysis task via app API."""
        base = self._resolve_api_base()
        payload: Dict[str, Any] = {
            "algorithm": algorithm,
            "dataset": dataset,
            "iterations": int(iterations),
            "mode": mode,
            "crossover_rate": float(crossover_rate),
            "mutate_rate": float(mutate_rate),
            "queue_if_busy": bool(queue_if_busy),
            "owner": str(owner or ""),
            "priority": int(priority),
            "release_lock_on_finish": bool(release_lock_on_finish),
        }
        if server_id:
            payload["server_id"] = server_id
        if isinstance(extra, dict):
            payload.update(extra)
        return self._http_post_json(f"{base}/api/analysis/start", payload)

    def analysis_stop(self, server_id: Optional[str] = None) -> Dict[str, Any]:
        """Stop analysis task for local/remote server."""
        base = self._resolve_api_base()
        payload: Dict[str, Any] = {}
        if server_id:
            payload["server_id"] = server_id
        return self._http_post_json(f"{base}/api/analysis/stop", payload)

    def resource_rows(self, resources: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Flatten resource snapshots for table display."""
        snap = resources or self.resources(all_servers=True)
        if not isinstance(snap, dict):
            return []
        rows: List[Dict[str, Any]] = []
        for sid, val in snap.items():
            if not isinstance(val, dict):
                continue
            cpu = val.get("cpu") or {}
            memory = val.get("memory") or {}
            gpus = val.get("gpus") or []
            gpu_count = len(gpus) if isinstance(gpus, list) else 0
            gpu_util = None
            if isinstance(gpus, list) and gpus:
                util_values = [g.get("gpu_util_percent") for g in gpus if isinstance(g, dict)]
                util_values = [float(x) for x in util_values if isinstance(x, (int, float))]
                if util_values:
                    gpu_util = sum(util_values) / len(util_values)
            rows.append(
                {
                    "server_id": sid,
                    "name": val.get("name"),
                    "online": val.get("online"),
                    "time": val.get("time"),
                    "cpu_usage_percent": cpu.get("usage_percent"),
                    "memory_percent": memory.get("percent"),
                    "gpu_count": gpu_count,
                    "gpu_util_avg_percent": gpu_util,
                }
            )
        return rows

    def resource_dataframe(self, rows: Optional[List[Dict[str, Any]]] = None):
        """Return pandas DataFrame when pandas is available."""
        try:
            import pandas as pd  # type: ignore
        except Exception:
            return None
        return pd.DataFrame(rows or self.resource_rows())

    def plot_resources(self, metric: str = "cpu_usage_percent", rows: Optional[List[Dict[str, Any]]] = None):
        """Plot one resource metric across servers."""
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return None, None
        data = rows or self.resource_rows()
        labels = [str(r.get("server_id")) for r in data]
        vals: List[float] = []
        for r in data:
            val = r.get(metric)
            vals.append(float(val) if isinstance(val, (int, float)) else 0.0)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(labels, vals)
        ax.set_ylabel(metric)
        ax.set_title("GAPA Resource Snapshot")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return fig, ax

    def run_trends(self, last_n: int = 20) -> Dict[str, Any]:
        """Aggregate run reports into trend summary grouped by algo/dataset/mode/remote."""
        rows = _load_run_report_rows()
        return _aggregate_run_trends(rows, last_n=last_n)

    def lock_mnm(
        self,
        server_inputs: Optional[List[str]] = None,
        duration_s: float = 900.0,
        warmup_iters: int = 1,
        mem_mb: int = 1024,
        owner: str = "",
        print_log: bool = False,
    ) -> Dict[str, Any]:
        """
        Acquire resource locks for MNM mode.

        Args:
            server_inputs: Optional server ids/names. If empty, all online remote servers are targeted.
            duration_s: Lock duration in seconds.
            warmup_iters: Warmup iterations hint for lock benchmark.
            mem_mb: Memory hint in MB.
            print_log: Whether to print lock logs.
        """
        server_list = self.server()
        if not isinstance(server_list, list):
            return {"error": "server list error", "detail": server_list}
        online = [s for s in server_list if s.get("online") and str(s.get("id")) != "local"]
        if server_inputs:
            targets: List[str] = []
            for raw in server_inputs:
                sid = self._normalize_server_id(str(raw), online)
                if not sid:
                    return {"error": "server offline or unknown", "server": raw}
                targets.append(sid)
        else:
            targets = [str(s.get("id")) for s in online]

        if not targets:
            return {"error": "no online remote servers for mnm lock"}

        api_base = self._resolve_api_base().rstrip("/")
        if print_log:
            print(f"[GAPA] MNM lock enabled. api_base={api_base}, targets={targets}")

        locked: List[str] = []
        details: Dict[str, Any] = {}
        for sid in targets:
            payload = {
                "scope": sid,
                "duration_s": float(duration_s),
                "warmup_iters": int(warmup_iters),
                "mem_mb": int(mem_mb),
                "owner": str(owner or ""),
            }
            body = self._http_post_json(f"{api_base}/api/resource_lock", payload)
            details[sid] = body
            result = body.get("results", {}).get(sid, {}) if isinstance(body, dict) else {}
            active = bool(isinstance(result, dict) and result.get("active"))
            if print_log:
                state = "ok" if active else "failed"
                print(f"[GAPA] MNM lock -> {sid}: {state}")
            if active:
                locked.append(sid)

        if not locked:
            return {
                "error": "mnm lock failed on all targets",
                "api_base": api_base,
                "targets": targets,
                "details": details,
            }
        lock_ids: Dict[str, str] = {}
        for sid in locked:
            one = details.get(sid, {})
            if isinstance(one, dict):
                lock_obj = one.get("results", {}).get(sid, {})
                if isinstance(lock_obj, dict) and lock_obj.get("lock_id"):
                    lock_ids[sid] = str(lock_obj.get("lock_id"))
        return {
            "api_base": api_base,
            "targets": targets,
            "locked": locked,
            "lock_ids": lock_ids,
            "owner": str(owner or ""),
            "details": details,
        }

    def unlock_servers(
        self,
        server_ids: List[str],
        api_base: Optional[str] = None,
        print_log: bool = False,
    ) -> Dict[str, Any]:
        """Release resource locks for the given servers."""
        base = (api_base or self._resolve_api_base()).rstrip("/")
        results: Dict[str, Any] = {}
        for sid in server_ids:
            body = self._http_post_json(f"{base}/api/resource_lock/release", {"scope": sid})
            results[sid] = body
            if print_log:
                ok = bool(isinstance(body, dict) and body.get("results", {}).get(sid))
                print(f"[GAPA] MNM unlock -> {sid}: {'ok' if ok else 'failed'}")
        return {"api_base": base, "results": results}

    def renew_mnm(
        self,
        *,
        lock_info: Dict[str, Any],
        duration_s: Optional[float] = None,
        print_log: bool = False,
    ) -> Dict[str, Any]:
        """Renew MNM lock(s) based on lock_info returned by lock_mnm()."""
        api_base = str(lock_info.get("api_base") or self._resolve_api_base()).rstrip("/")
        owner = lock_info.get("owner")
        lock_ids = lock_info.get("lock_ids") or {}
        servers = lock_info.get("locked") or lock_info.get("targets") or []
        results: Dict[str, Any] = {}
        for sid in servers:
            payload: Dict[str, Any] = {"scope": sid}
            if duration_s is not None:
                payload["duration_s"] = float(duration_s)
            if isinstance(lock_ids, dict) and lock_ids.get(sid):
                payload["lock_id"] = str(lock_ids.get(sid))
            if owner:
                payload["owner"] = owner
            body = self._http_post_json(f"{api_base}/api/resource_lock/renew", payload)
            # Compatibility fallback:
            # If local proxy app does not expose /api/resource_lock/renew yet, call remote agent directly.
            if (
                isinstance(body, dict)
                and str(body.get("error") or "").startswith("HTTP 404")
            ):
                body = self._post_server_direct(str(sid), "/api/resource_lock/renew", payload)
            results[sid] = body
            if print_log:
                node = body.get("results", {}).get(sid, {}) if isinstance(body, dict) else {}
                if isinstance(node, dict) and node:
                    ok = bool(node.get("active"))
                elif isinstance(body, dict) and body.get("active") is not None:
                    ok = bool(body.get("active"))
                else:
                    ok = False
                print(f"[GAPA] MNM renew -> {sid}: {'ok' if ok else 'failed'}")
        return {"api_base": api_base, "results": results}
    
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
    def generation(self) -> int:
        return self._generation

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
        remote_use_strategy_plan: Optional[bool] = None,
        server_url: str = "",
        fallback_policy: str = "best_effort",
        verbose: bool = True,
    ):
        """
        Initialize the workflow.
        
        Args:
            algorithm: Algorithm instance (subclass of Algorithm)
            data_loader: DataLoader from load_dataset()
            monitor: Optional Monitor for tracking progress
            mode: Execution mode - "s", "sm", "m", "m_cpu", "mnm"
            workers: Number of workers for resource discovery (MNM mode)
            auto_select: Auto-select remote servers (for MNM mode)
            servers: List of remote server IDs (for MNM mode)
            remote_server: Single remote server id/name for s/sm/m remote execution
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
        self.remote_use_strategy_plan = remote_use_strategy_plan
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
        valid_modes = SOURCE_MODES if HAS_DISTRIBUTED else PYPI_MODES
        if mode not in valid_modes:
            if mode == "mnm" and not HAS_DISTRIBUTED:
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
        self.execution_contract = {
            "algorithm_id": self.algorithm_id,
            "requested_mode": self.requested_mode,
            "resolved_mode": self.mode,
            "fallback_policy": self.fallback_policy,
            "remote_server": self.remote_server,
            "remote_use_strategy_plan": self.remote_use_strategy_plan,
            "servers": list(self.servers),
            "capabilities": self.algorithm_capabilities,
        }
    
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
        if not HAS_DISTRIBUTED or DistributedEvaluator is None:
            if self.verbose:
                print("[GAPA] Warning: DistributedEvaluator not available. Using local evaluation.")
            return evaluator

        wrapped = DistributedEvaluator(
            evaluator,
            algorithm=self.algorithm_id,
            dataset=self.data_loader.name,
            allowed_server_ids=self.servers,
        )
        if self.verbose:
            print(f"[GAPA] MNM mode: {len(self.servers)} remote server(s)")
        return wrapped
    
    def run(self, generations: int) -> None:
        """
        Run the full evolution loop.
        
        This is the recommended method for all modes. It uses the existing
        Start() function to ensure consistent behavior with frontend execution.
        
        Args:
            generations: Number of generations to run
        """
        run_id = str(uuid.uuid4())
        run_ctx = {
            "run_id": run_id,
            "started_at": datetime.utcnow().isoformat() + "Z",
            "algorithm_id": self.algorithm_id,
            "requested_mode": self.requested_mode,
            "resolved_mode": self.mode,
            "fallback_policy": self.fallback_policy,
            "remote_server": self.remote_server,
            "servers": list(self.servers),
            "generations": int(generations),
            "execution_contract": self.execution_contract,
        }
        self.monitor.set_run_context(run_ctx)

        if self.remote_server:
            from gapa.remote_runner import run_remote_task
            dataset_name = getattr(self.data_loader, "name", None) or getattr(self.data_loader, "dataset", None) or ""
            result = run_remote_task(
                self.monitor,
                self.remote_server,
                algorithm=self.algorithm_id,
                dataset=dataset_name,
                iterations=generations,
                mode=self.mode,
                crossover_rate=0.8,
                mutate_rate=0.2,
                use_strategy_plan=self.remote_use_strategy_plan,
            )
            if isinstance(result, dict) and result.get("error"):
                raise RuntimeError(f"remote run failed: {result}")
            run_ctx["ended_at"] = datetime.utcnow().isoformat() + "Z"
            report_meta = self._emit_run_reports(run_ctx)
            run_ctx["reports"] = report_meta
            self.monitor.set_run_context(run_ctx)
            if self.verbose and isinstance(report_meta, dict) and report_meta.get("enabled"):
                print(f"[GAPA] Report saved: {report_meta.get('summary_path')}")
                bench = report_meta.get("benchmark") if isinstance(report_meta.get("benchmark"), dict) else {}
                if bench.get("regressed"):
                    print(f"[WARN] Benchmark regression detected: ratio={bench.get('ratio'):.3f} threshold={bench.get('threshold'):.3f}")
            return
        # Import the core execution function
        from gapa.framework.controller import Start
        
        # Setup components
        self._setup_components()

        if self.verbose:
            print(f"[GAPA] Starting {self.algorithm.__class__.__name__} in '{self.mode}' mode")
            print(f"[GAPA] Generations: {generations}, Device: {self.device}")

        import time
        start_ts = time.perf_counter()

        # Call the unified core engine
        Start(
            max_generation=generations,
            data_loader=self.data_loader,
            controller=self._controller,
            evaluator=self._evaluator,
            body=self._body,
            world_size=self.world_size,
            verbose=self.verbose,
            observer=self.monitor,  # Pass monitor as observer
        )
        end_ts = time.perf_counter()
        try:
            total = max(0.0, end_ts - start_ts)
            if generations > 0:
                self.monitor._local_timing = {
                    "iter_seconds": total,
                    "iter_avg_ms": (total / generations) * 1000.0,
                    "throughput_ips": generations / total if total > 0 else None,
                }
        except Exception:
            pass
        run_ctx["ended_at"] = datetime.utcnow().isoformat() + "Z"
        report_meta = self._emit_run_reports(run_ctx)
        run_ctx["reports"] = report_meta
        self.monitor.set_run_context(run_ctx)
        if self.verbose and isinstance(report_meta, dict) and report_meta.get("enabled"):
            print(f"[GAPA] Report saved: {report_meta.get('summary_path')}")
            bench = report_meta.get("benchmark") if isinstance(report_meta.get("benchmark"), dict) else {}
            if bench.get("regressed"):
                print(f"[WARN] Benchmark regression detected: ratio={bench.get('ratio'):.3f} threshold={bench.get('threshold'):.3f}")
        
        if self.verbose:
            print(f"\n[GAPA] Evolution complete. Best fitness: {self.monitor.best_fitness}")
    
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
