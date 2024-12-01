import os
import random
import argparse
import torch
# import cugraph
import networkx as nx
import scipy.sparse as sp
from datetime import datetime
from igraph import Graph as IG
from igraph.clustering import compare_communities
import numpy as np
import torch.distributed as dist
from typing import Tuple, Union, Optional
from scipy.sparse import coo_matrix, lil_matrix
import torch.sparse as tsp
import torch.multiprocessing as mp


def Parsers():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--method", required=True, type=str)
    parser.add_argument("--pop_size", required=True, type=int)
    parser.add_argument("--mode", required=True, type=str)
    parser.add_argument("--world_size", type=int, default=2)
    args = parser.parse_args()
    return args


def init_dist(rank, world_size, async_op=False):
    dist.init_process_group(
        backend='nccl',
        init_method=f"tcp://127.0.0.1:12355",
        rank=rank,
        world_size=world_size
    )
    dist.barrier(async_op=async_op)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(rank)
    return device


def current_time():
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_datetime


def save(path, pattern, dataset, gene, best_fitness_list, time_list, method, pop_size, crossover_rate, mutate_rate, **kwargs):
    metrics = []
    for key, value in kwargs.items():
        metrics.append(value)
    if not os.path.exists(path):
        os.makedirs(path)
    elif pattern == 'overwrite':
        delete_files_in_folder(path)
    elif pattern == 'write':
        pass
    save_path = path + dataset + '_crossover_rate_' + str(crossover_rate) + '_mutate_rate_' + str(mutate_rate) + f'_{method}.txt'
    with open(save_path, 'a+') as f:
        f.write(current_time())
        f.write(f"\nCurrent mode: None. Current pop_size: {pop_size}\n")
    with open(save_path, 'a+') as f:
        f.write(str([i for i in metrics[0]]) + '\n')
    with open(save_path, 'a+') as f:
        f.write(str([i for i in metrics[1]]) + '\n')
    with open(save_path, 'a+') as f:
        f.write(str([i for i in gene]) + '\n')
    with open(save_path, 'a+') as f:
        f.write(str(time_list) + '\n')


class MissingModule:
    """
    Copy from cuGraph

    Raises RuntimeError when any attribute is accessed on instances of this
    class.

    Instances of this class are returned by import_optional() when a module
    cannot be found, which allows for code to import optional dependencies, and
    have only the code paths that use the module affected.
    """

    def __init__(self, mod_name):
        self.name = mod_name

    def __getattr__(self, attr):
        raise RuntimeError(f"This feature requires the {self.name} " "package/module")


def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            os.remove(filepath)


def ga_dist_function(rank, world_size, population, algorithm, result_queue):
    try:
        dist.init_process_group(
            backend='nccl',
            init_method=f"tcp://127.0.0.1:12355",
            rank=rank,
            world_size=world_size
        )
        dist.barrier()
        torch.cuda.set_device(rank)
        rows_per_proc = len(population) // world_size
        start_index = rank * rows_per_proc
        end_index = start_index + rows_per_proc if rank != world_size - 1 else len(population)
        local_tensor = population[start_index:end_index]
        fitness = algorithm.fitness_calculate(local_tensor)
        result_queue.put(fitness.cpu().clone())
    finally:
        del local_tensor, fitness
        torch.cuda.empty_cache()
        dist.destroy_process_group()


def ga_start_dist(world_size, population, algorithm):
    mp.set_start_method('spawn', force=True)
    with mp.Manager() as manager:
        result_queue = manager.Queue()
        mp.spawn(ga_dist_function, args=(world_size, population, algorithm, result_queue), nprocs=world_size, join=True)
        all_scores = []
        for _ in range(world_size):
            score = result_queue.get()
            all_scores.append(score)
    all_scores = torch.cat(all_scores)
    return all_scores


def scipy_sparse_to_tensor(sp_matrix: Union[coo_matrix, lil_matrix], dtype: Optional[torch.dtype] = None,
                           device: Optional[torch.device] = None) -> tsp.Tensor:
    if isinstance(sp_matrix, lil_matrix):
        sp_matrix = sp_matrix.tocoo()
    _indices = torch.tensor([sp_matrix.row.tolist(), sp_matrix.col.tolist()])
    _values = sp_matrix.data
    return torch.sparse_coo_tensor(
        indices=_indices,
        values=_values,
        dtype=dtype,
        size=sp_matrix.shape,
        device=device
    )


def eye(m: int, dtype: Union[torch.dtype, None], device=Union[torch.device, None]) -> tsp.Tensor:
    """Returns a sparse matrix with ones on the diagonal and zeros elsewhere.

    Args:
        m (int): The first dimension of sparse matrix.
        dtype (`torch.dtype`, optional): The desired data type of returned
            value vector. (default is set by `torch.set_default_tensor_type()`)
        device (`torch.device`, optional): The desired device of returned
            tensors. (default is set by `torch.set_default_tensor_type()`)

    :rtype: layout==torch.sparse_coo
    """

    row = torch.arange(m, dtype=torch.long, device=device)
    index = torch.stack([row, row], dim=0)
    value = torch.ones(m, device=device)
    Identity = torch.sparse_coo_tensor(
        indices=index,
        values=value,
        dtype=dtype,
    )
    del row, index, value
    return Identity


def gcn_filter(adj: tsp.Tensor, power=-0.5) -> tsp.Tensor:
    """
    计算重归一化拉普拉斯矩阵

    :param adj:  邻接矩阵
    :param power: 度矩阵乘的幂数
    :return: 重归一化拉普拉斯矩阵
    """
    Identity = eye(adj.shape[0], dtype=torch.float32, device=adj.device)
    _filter = adj + Identity
    # 合并filter
    _filter = _filter.coalesce()
    """计算D^{-1/2}"""
    d = tsp.sum(_filter, dim=1)
    d = torch.pow(d, power)
    d = torch.sparse_coo_tensor(
        indices=torch.cat((d.indices(), d.indices()), dim=0),
        values=d.values(),
        size=(d.shape[0], d.shape[0]),
        device=adj.device
    )
    """计算D^{-1/2} @ A @ D^{-1/2}"""
    _filter = torch.sparse.mm(d, torch.sparse.mm(_filter, d))
    del Identity, d
    return _filter


def gcn_filter_I(adj: tsp.Tensor, power=-0.5) -> tsp.Tensor:
    """
    计算重归一化拉普拉斯矩阵

    :param adj:  邻接矩阵
    :param power: 度矩阵乘的幂数
    :return: 重归一化拉普拉斯矩阵
    """
    Identity = eye(adj.shape[0], dtype=torch.float32, device=adj.device)
    _filter = adj + Identity
    # 合并filter
    _filter = _filter.coalesce()
    """计算D^{-1/2}"""
    d = tsp.sum(_filter, dim=1)
    d = torch.pow(d, power).to_dense()
    d[torch.isinf(d)] = 0
    d = d.to_sparse_coo()
    d = torch.sparse_coo_tensor(
        indices=torch.cat((d.indices(), d.indices()), dim=0),
        values=d.values(),
        size=(d.shape[0], d.shape[0]),
        device=adj.device
    )
    """计算(D + I)^{-1/2} @ A @ (D + I)^{-1/2}"""
    _filter = torch.sparse.mm(d, torch.sparse.mm(_filter, d))
    del Identity, d
    return _filter


def set_seed(seed: int):
    """
    设置随机数种子

    :param seed: 随机种子
    :return: None
    """
    # seed = 80230
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    print(f"Random seed set with {seed}. File run in utils.functions.set_seed().")


def tensorToSparse(tensor: torch.Tensor):
    tensor = tensor.cpu()
    coo = sp.coo_matrix((tensor.values(), (tensor.indices()[0], tensor.indices()[1])), shape=tensor.shape)
    return coo.tocsr()


def csr_matrix_to_tensor(csr_matrix, device):
    csr_matrix = csr_matrix.tocoo()

    row = torch.tensor(csr_matrix.row, device=device)
    col = torch.tensor(csr_matrix.col, device=device)
    data = torch.tensor(csr_matrix.data, device=device)
    tensor = torch.sparse_coo_tensor(torch.stack([row, col]), data, csr_matrix.shape, device=device)
    return tensor


def adjReshapeAddDim(adj, target_size, device):
    assert len(adj) < target_size, f"adj dim {len(adj)} is bigger than target_size {target_size}."
    temp = torch.zeros(size=(target_size, target_size), dtype=torch.float, device=device)
    temp[:len(adj), :len(adj)] = adj.clone().to_dense()
    return temp.to_sparse_coo()


def CNDTest(graph: nx.Graph, critical_nodes, pattern="cc"):
    if pattern == "cc":
        copy_g = graph.copy()
        for node in critical_nodes:
            try:
                copy_g.remove_node(node.item())
            except:
                pass
        total = 0
        sub_graph_list = list(nx.connected_components(copy_g))
        for sub_graph_i in range(len(sub_graph_list)):
            total += len(sub_graph_list[sub_graph_i]) * (len(sub_graph_list[sub_graph_i]) - 1) / 2
        return total
    elif pattern == "ccn":
        try:
            device = critical_nodes.device
        except:
            device = 'cpu'
        copy_A = torch.tensor(nx.to_numpy_array(graph.copy(), nodelist=list(graph.nodes())), device=device)
        I = torch.eye(len(copy_A), device=device)
        copy_A[critical_nodes, :] = 0
        copy_A[:, critical_nodes] = 0
        matrix2 = torch.matmul((copy_A + I), (copy_A + I))
        matrix4 = torch.matmul(matrix2, matrix2)
        matrix8 = torch.matmul(matrix4, matrix4)
        matrix16 = torch.matmul(matrix8, matrix8)
        matrix17 = torch.matmul(matrix16, copy_A)
        final_matrix = matrix17 + matrix16
        return torch.count_nonzero(final_matrix, dim=1).max().item()


def Q_Test(graph, method="louvain"):
    # if torch.cuda.is_available():
    #     if method == "louvain":
    #         Q = cugraph.louvain(graph)[1]
    #         return Q
    # else:
    G_i = IG(directed=False)
    G_i = G_i.from_networkx(graph)
    if method == "louvain":
        Q = IG.community_multilevel(G_i).modularity
        return Q


def NMI_Test(ori_graph, graph, method="louvain"):
    vitim_G = IG(directed=False)
    vitim_G = vitim_G.from_networkx(graph)
    ori_G = IG(directed=False)
    ori_G = ori_G.from_networkx(ori_graph)

    if method == "louvain":
        vitim_community = IG.community_multilevel(vitim_G)
        ori_community = IG.community_multilevel(ori_G)
        return compare_communities(ori_community, vitim_community, 'nmi')


def AS_Rate(feats, modify_feats, adj, modify_adj, test_index, model):
    try:
        adj = adj.to_sparse_coo()
        modify_adj = modify_adj.to_sparse_coo()
    except:
        pass
    adj_norm = gcn_filter(adj)
    modify_adj_norm = gcn_filter(modify_adj)
    model.eval()
    output = torch.argmax(model(feats, adj_norm), dim=1)
    modify_output = torch.argmax(model(modify_feats, modify_adj_norm), dim=1)
    attack_failure_num = torch.eq(output[test_index], modify_output[test_index]).sum()
    attack_success_rate = 1 - attack_failure_num.item() * 1.0 / test_index.shape[0]
    return attack_success_rate


def Acc(modify_feats, modify_adj, labels, test_index, model):
    try:
        modify_adj = modify_adj.to_sparse_coo()
    except:
        pass
    modify_adj_norm = gcn_filter(modify_adj)
    model.eval()
    output = model(modify_feats, modify_adj_norm)
    predicts = torch.argmax(output, dim=1)
    correct = torch.eq(predicts[test_index], labels[test_index]).sum()
    return correct.item() * 1.0 / test_index.shape[0]
