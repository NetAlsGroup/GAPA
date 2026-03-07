from __future__ import annotations
import os, json
from typing import Any, Dict
import networkx as nx
import numpy as np
import torch
from ..api.schemas import JobSpec
from .DataLoader import Loader as UserLoader
def _load_graph_from_config(cfg: Dict[str, Any]) -> nx.Graph:
    dtype = cfg.get('type','edgelist'); path = cfg.get('path'); directed = cfg.get('directed', False); delimiter = cfg.get('delimiter', None)
    if dtype=='edgelist':
        if not path or not os.path.exists(path): raise FileNotFoundError(f'edgelist 路径不存在: {path}')
        G = nx.read_edgelist(path, delimiter=delimiter, nodetype=int, create_using=nx.DiGraph() if directed else nx.Graph())
        return G.to_undirected() if not directed else G
    if dtype=='nx_gpickle':
        if not path or not os.path.exists(path): raise FileNotFoundError(f'nx_gpickle 路径不存在: {path}')
        return nx.read_gpickle(path)
    if dtype=='adj_npy':
        if not path or not os.path.exists(path): raise FileNotFoundError(f'adj_npy 路径不存在: {path}')
        try:
            A = np.load(path); A=(A>0).astype(np.float32); G = nx.from_numpy_array(A); return G
        except:
            G = nx.read_adjlist(path, nodetype=int); A = nx.to_numpy_array(G, nodelist=sorted(list(G.nodes()))); G = nx.from_numpy_array(A); return G
    raise ValueError('未知数据类型')
def build_loader(job: JobSpec) -> Any:
    dscfg = job.artifacts.get('dataset', {})
    G = _load_graph_from_config(dscfg)
    loader = UserLoader(dataset=dscfg.get('name','unknown'), device='cpu')
    loader.G = G; loader.n_nodes = G.number_of_nodes(); loader.n_edges = G.number_of_edges()
    loader.nodes_num = loader.n_nodes
    loader.nodes = torch.tensor(list(loader.G.nodes))
    loader.selected_genes_num = int(0.4 * loader.nodes_num)
    loader.k = int(0.1 * loader.nodes_num)
    if dscfg.get('type') == 'adj_npy':
        A = nx.to_numpy_array(G).astype(np.float32); loader.adj = torch.from_numpy(A); loader.A = loader.adj
    return loader
