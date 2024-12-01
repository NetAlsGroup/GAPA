import matplotlib.pyplot as plt
import networkx as nx
import torch
import dgl
import subprocess
from tqdm import tqdm


def draw_graph(graph: nx.Graph):
    pos = nx.spring_layout(graph, k=0.4, iterations=100)
    nx.draw_networkx_nodes(G=graph, pos=pos, node_color="white", node_size=200)
    nx.draw_networkx_edges(G=graph, pos=pos, alpha=0.5)
    nx.draw_networkx_labels(G=graph, pos=pos, font_size=5)
    # Set margins for the axes so that nodes aren't clipped
    plt.axis("off")
    plt.show()


def init_device():
    if torch.cuda.is_available():
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                                         universal_newlines=True)
        gpu_memory = [int(x) for x in result.strip().split('\n')]

        best_gpu_index = gpu_memory.index(min(gpu_memory))
        device = f'cuda:{best_gpu_index}'
        torch.cuda.set_device(best_gpu_index)
    else:
        device = 'cpu'

    return device


def generate_rwr_subgraph(
        dgl_graph,  # a graph
        subgraph_size
):
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    traces = dgl.sampling.random_walk(dgl_graph, all_idx, length=subgraph_size * 3)
    subv = []

    for i, trace in enumerate(traces[0]):
        subv.append(torch.unique(trace, sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = dgl.sampling.random_walk(dgl_graph, nodes=[i], length=subgraph_size)
            subv[i] = torch.unique(cur_trace[0], sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= 2) and (retry_time > 10):
                subv[i] = (subv[i] * reduced_size)
        subv[i] = subv[i][:reduced_size]
        subv[i].append(i)

    return subv


def generate_subgraph(graph: nx.Graph, node, sample_size=4):
    neighbors_1 = list(graph.neighbors(node))

    if len(neighbors_1) >= sample_size:
        sampled_nodes = neighbors_1[:sample_size]
    else:
        neighbors_2 = set()
        for neighbor in neighbors_1:
            neighbors_2.update(graph.neighbors(neighbor))

        neighbors_2.discard(node)
        neighbors_2 = list(neighbors_2 - set(neighbors_1))

        sampled_nodes = neighbors_1

        additional_needed = sample_size - len(neighbors_1)

        if len(neighbors_2) >= additional_needed:
            sampled_nodes.extend(neighbors_2[:additional_needed])
        else:
            sampled_nodes.extend(neighbors_2)

    sampled_nodes.append(node)
    return sampled_nodes


def normalize_adj(adj: torch.Tensor):
    """Symmetrically normalize adjacency matrix."""
    row_sum = torch.sum(adj.to_dense(), dim=1)
    d_inv_sqrt = torch.float_power(row_sum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.matmul(torch.matmul(adj, d_mat_inv_sqrt).t(), d_mat_inv_sqrt)


def generate_features(graph: nx.Graph, subgraph_size, device):
    # _A = torch.tensor(nx.to_numpy_array(graph, nodelist=sorted(list(graph.nodes()))), device=device)
    # _A_normalize = normalize_adj(_A)
    # D_list = torch.sum(_A, dim=1)
    feats_dict = {}
    _pbar = tqdm(graph.nodes(), total=len(graph.nodes()))
    _pbar.set_description(f"Generate features...")
    for node in _pbar:
        node_list = generate_subgraph(graph, node, sample_size=subgraph_size)
        node_adj = torch.tensor(nx.to_numpy_array(graph, node_list), device=device, dtype=torch.float)
        D_list = torch.tensor([graph.degree[_node] for _node in node_list], device=device, dtype=torch.float)
        node_adj[0] = node_adj[0] * D_list
        node_adj_t = node_adj.t()
        node_adj_t[0] = node_adj_t[0] * D_list
        feats_dict[node] = node_adj_t.t() + torch.diag(D_list)
        if len(feats_dict[node]) < subgraph_size:
            temp = torch.zeros(size=(subgraph_size, subgraph_size), dtype=torch.float, device=device)
            temp[:len(feats_dict[node]), :len(feats_dict[node])] = feats_dict[node]
            feats_dict[node] = temp
        feats_dict[node] = feats_dict[node].unsqueeze(dim=0)
        _pbar.update(1)

    # dgl_graph = dgl.from_networkx(graph)
    # sub_v = generate_rwr_subgraph(dgl_graph, subgraph_size=subgraph_size)
    return feats_dict
