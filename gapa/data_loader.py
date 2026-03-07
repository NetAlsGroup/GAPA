from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import torch


class DataLoader:
    """Unified public dataset container and registry-backed loader."""

    def __init__(
        self,
        name: str,
        G: nx.Graph,
        A: torch.Tensor,
        nodes: torch.Tensor,
        k: int,
        selected_genes_num: int,
        device: torch.device,
        *,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.dataset = name
        self.G = G
        self.A = A
        self.nodes = nodes
        self.nodes_num = len(nodes)
        self.k = k
        self.selected_genes_num = selected_genes_num
        self.device = device
        self.meta = dict(meta or {})

    @classmethod
    def load(
        cls,
        name: str,
        task: str | None = None,
        *,
        detection_rate: float = 0.1,
        selected_genes_rate: float = 0.4,
        device: str = "auto",
        sort_nodes: bool = True,
    ) -> "DataLoader":
        registry = cls._load_registry()
        datasets = registry.get("datasets", {})
        if name not in datasets:
            raise FileNotFoundError(
                f"Dataset '{name}' is not registered in {cls._registry_path()}."
            )
        meta = dict(datasets[name])
        tasks = cls._entry_tasks(meta)
        if task and task not in tasks:
            raise ValueError(
                f"Dataset '{name}' does not belong to task '{task}'. Registered tasks: {tasks}"
            )

        target_device = cls._resolve_device(device)
        data_path = cls._datasets_root() / str(meta.get("path", "")).strip()
        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset file for '{name}' is missing: {data_path}"
            )

        graph = cls._load_graph(data_path, str(meta.get("format") or "edgelist"))
        if sort_nodes:
            nodelist = sorted(list(graph.nodes()))
            adjacency = torch.tensor(
                nx.to_numpy_array(graph, nodelist=nodelist),
                device=target_device,
                dtype=torch.float32,
            )
            graph = nx.from_numpy_array(adjacency.cpu().numpy())
            node_values = list(graph.nodes())
        else:
            nodelist = list(graph.nodes())
            adjacency = torch.tensor(
                nx.to_numpy_array(graph, nodelist=nodelist),
                device=target_device,
                dtype=torch.float32,
            )
            node_values = nodelist

        nodes = torch.tensor(node_values, device=target_device)
        nodes_num = len(nodes)
        k = max(1, int(detection_rate * nodes_num))
        selected_genes_num = max(1, int(selected_genes_rate * nodes_num))

        return cls(
            name=name,
            G=graph,
            A=adjacency,
            nodes=nodes,
            k=k,
            selected_genes_num=selected_genes_num,
            device=target_device,
            meta={"path": str(data_path), **meta},
        )

    @classmethod
    def list(cls, task: str | None = None) -> List[Dict[str, Any]]:
        registry = cls._load_registry()
        rows: List[Dict[str, Any]] = []
        for name, raw in registry.get("datasets", {}).items():
            meta = dict(raw)
            tasks = cls._entry_tasks(meta)
            if task and task not in tasks:
                continue
            rows.append({"name": name, **meta, "tasks": tasks})
        rows.sort(key=lambda item: (str(item.get("task") or ""), int(item.get("nodes") or 0), str(item["name"])))
        return rows

    @classmethod
    def describe(cls, name: str) -> Dict[str, Any]:
        registry = cls._load_registry()
        datasets = registry.get("datasets", {})
        if name not in datasets:
            raise FileNotFoundError(
                f"Dataset '{name}' is not registered in {cls._registry_path()}."
            )
        meta = dict(datasets[name])
        return {"name": name, **meta, "tasks": cls._entry_tasks(meta)}

    @classmethod
    def _load_registry(cls) -> Dict[str, Any]:
        path = cls._registry_path()
        if not path.exists():
            raise FileNotFoundError(f"Dataset registry does not exist: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Dataset registry must be a JSON object: {path}")
        return raw

    @classmethod
    def _registry_path(cls) -> Path:
        return cls._datasets_root() / "registry.json"

    @classmethod
    def _datasets_root(cls) -> Path:
        repo_root = Path(__file__).resolve().parents[1] / "datasets"
        if repo_root.exists():
            return repo_root
        package_root = Path(__file__).resolve().parent / "datasets"
        return package_root

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _entry_tasks(meta: Dict[str, Any]) -> List[str]:
        tasks = meta.get("tasks")
        if isinstance(tasks, list):
            return [str(item) for item in tasks if str(item).strip()]
        task = meta.get("task")
        return [str(task)] if task else []

    @classmethod
    def _load_graph(cls, path: Path, fmt: str) -> nx.Graph:
        normalized = fmt.strip().lower()
        if normalized == "gml":
            return nx.read_gml(str(path), label="id")
        if normalized == "graphml":
            return nx.read_graphml(str(path))
        if normalized == "gexf":
            return nx.read_gexf(str(path))
        if normalized == "npz":
            return cls._load_npz_graph(path)
        if normalized == "csv":
            return cls._load_csv_edgelist(path)
        if normalized == "adjlist":
            return nx.read_adjlist(str(path), nodetype=int)
        return cls._load_edgelist(path)

    @staticmethod
    def _load_edgelist(path: Path) -> nx.Graph:
        edges: List[Any] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("%"):
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) < 2:
                    continue
                try:
                    src, dst = int(parts[0]), int(parts[1])
                except ValueError:
                    src, dst = parts[0], parts[1]
                edges.append((src, dst))
        graph = nx.Graph()
        graph.add_edges_from(edges)
        return graph

    @staticmethod
    def _load_csv_edgelist(path: Path) -> nx.Graph:
        import csv

        edges: List[Any] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            sample = handle.read(1024)
            handle.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
            has_header = csv.Sniffer().has_header(sample)
            reader = csv.reader(handle, dialect)
            if has_header:
                next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    src, dst = int(row[0]), int(row[1])
                except ValueError:
                    src, dst = row[0].strip(), row[1].strip()
                edges.append((src, dst))
        graph = nx.Graph()
        graph.add_edges_from(edges)
        return graph

    @staticmethod
    def _load_npz_graph(path: Path) -> nx.Graph:
        data = np.load(path, allow_pickle=True)
        files = set(data.files)

        if "edges" in files:
            graph = nx.Graph()
            graph.add_edges_from(data["edges"])
            return graph

        csr_keys = {"adj_data", "adj_indices", "adj_indptr", "adj_shape"}
        if csr_keys.issubset(files):
            try:
                from scipy import sparse  # type: ignore

                shape = tuple(int(item) for item in data["adj_shape"])
                matrix = sparse.csr_matrix(
                    (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
                    shape=shape,
                )
                return nx.from_scipy_sparse_array(matrix)
            except Exception as exc:
                raise ValueError(f"Failed to load CSR adjacency from {path}: {exc}") from exc

        for key in ["A", "adj", "adjacency", "matrix"]:
            if key not in files:
                continue
            matrix = data[key]
            if getattr(matrix, "ndim", 0) == 2 and matrix.shape[0] == matrix.shape[1]:
                return nx.from_numpy_array(matrix)

        raise ValueError(
            f"Unsupported NPZ graph layout in {path}. Available keys: {sorted(files)}"
        )
