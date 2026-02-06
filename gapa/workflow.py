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

__all__ = [
    "Algorithm",
    "Workflow", 
    "Monitor",
    "load_dataset",
]

import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import multiprocessing as mp
import json

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

    def _resolve_api_base(self) -> str:
        if self.api_base:
            return self.api_base.rstrip("/")
        env_base = os.getenv("GAPA_API_BASE")
        if env_base:
            return str(env_base).rstrip("/")
        guessed = self._guess_api_base_from_servers_file()
        if guessed:
            return guessed
        return "http://127.0.0.1:5000"

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
        server_url: str = "http://localhost:5000",
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
            server_url: Local GAPA API URL (for MNM mode resource discovery)
            verbose: Whether to print progress information
        """
        self.algorithm = algorithm
        self.data_loader = data_loader
        self.monitor = monitor if monitor is not None else Monitor()
        self.verbose = verbose
        
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
        self.server_url = server_url
        
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
        
        # =====================================================================
        # Cross-Platform Mode Adaptation
        # =====================================================================
        import platform
        system = platform.system()  # 'Darwin' (Mac), 'Windows', 'Linux'
        skip_platform_adapt = bool(self.remote_server and self.mode in ("s", "sm", "m"))
        
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
                if verbose:
                    print("[GAPA] MacOS: SM mode not supported (no CUDA). Falling back to 'S' mode.")
                self.mode = "s"
            elif mode == "m":
                # MacOS: No CUDA/NCCL, fall back to S
                # For CPU multiprocessing, use StrategyPlan to evaluate if it's beneficial
                if verbose:
                    print("[GAPA] MacOS: M mode not supported (no CUDA).")
                    print("[GAPA] Tip: Use StrategyPlan to evaluate if CPU multiprocessing is beneficial for your workload.")
                    print("[GAPA] Falling back to 'S' mode.")
                self.mode = "s"
        
        elif system == "Windows":
            if mode == "m":
                if torch.cuda.is_available():
                    # Windows with CUDA: use gloo backend
                    os.environ["GAPA_DIST_BACKEND"] = "gloo"
                    if verbose:
                        print("[GAPA] Windows: M mode using 'gloo' backend (NCCL not available).")
                else:
                    # Windows CPU-only: fall back to S
                    if verbose:
                        print("[GAPA] Windows: M mode not supported (no CUDA).")
                        print("[GAPA] Tip: Use StrategyPlan to evaluate if CPU multiprocessing is beneficial.")
                        print("[GAPA] Falling back to 'S' mode.")
                    self.mode = "s"
            elif mode == "sm" and not torch.cuda.is_available():
                if verbose:
                    print("[GAPA] Windows: SM mode requires CUDA. Falling back to 'S' mode.")
                self.mode = "s"
        
        elif system == "Linux":
            if mode == "m":
                if torch.cuda.is_available() and self.world_size >= 2:
                    # Linux with multi-GPU: use nccl
                    pass  # Default nccl is fine
                elif torch.cuda.is_available() and self.world_size == 1:
                    # Linux with single GPU: fall back to S (M mode needs >= 2 GPUs)
                    if verbose:
                        print("[GAPA] Linux: M mode requires >= 2 GPUs.")
                        print("[GAPA] Tip: Use StrategyPlan to evaluate if CPU multiprocessing is beneficial.")
                        print("[GAPA] Falling back to 'S' mode.")
                    self.mode = "s"
                else:
                    # Linux CPU-only: fall back to S
                    if verbose:
                        print("[GAPA] Linux: M mode requires CUDA.")
                        print("[GAPA] Tip: Use StrategyPlan to evaluate if CPU multiprocessing is beneficial.")
                        print("[GAPA] Falling back to 'S' mode.")
                    self.mode = "s"
            elif mode == "sm" and not torch.cuda.is_available():
                if verbose:
                    print("[GAPA] Linux: SM mode requires CUDA. Falling back to 'S' mode.")
                self.mode = "s"
        
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
        if self.mode == "mnm" and self.auto_select and not self.servers:
            self.servers = self._discover_resources()
        
        # Internal state (created during run)
        self._controller = None
        self._evaluator = None
        self._body = None
        self._state = None  # For step-by-step iteration
    
    def _discover_resources(self) -> List[str]:
        """Discover available remote servers for MNM mode."""
        try:
            import requests
            resp = requests.get(
                f"{self.server_url}/api/resource_lock/status",
                params={"scope": "all", "realtime": "true"},
                timeout=10
            )
            resp.raise_for_status()
            
            available = []
            for server in resp.json().get("status", []):
                status = server.get("status", "")
                if "locked" not in status.lower():
                    server_id = server.get("server_id") or server.get("id")
                    if server_id:
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
    
    def _wrap_for_mnm(self, evaluator) -> nn.Module:
        """Wrap evaluator for distributed execution."""
        if not HAS_DISTRIBUTED or DistributedEvaluator is None:
            if self.verbose:
                print("[GAPA] Warning: DistributedEvaluator not available. Using local evaluation.")
            return evaluator
        
        wrapped = DistributedEvaluator(
            evaluator,
            algorithm=self.algorithm.__class__.__name__,
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
        if self.remote_server:
            from gapa.remote_runner import run_remote_task, resolve_algorithm_id
            dataset_name = getattr(self.data_loader, "name", None) or getattr(self.data_loader, "dataset", None) or ""
            algo_id = resolve_algorithm_id(self.algorithm)
            result = run_remote_task(
                self.monitor,
                self.remote_server,
                algorithm=algo_id,
                dataset=dataset_name,
                iterations=generations,
                mode=self.mode,
                crossover_rate=0.8,
                mutate_rate=0.2,
            )
            if isinstance(result, dict) and result.get("error"):
                raise RuntimeError(f"remote run failed: {result}")
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
