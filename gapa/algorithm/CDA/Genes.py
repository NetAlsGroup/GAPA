import torch
import itertools
from igraph import Graph as IG
from copy import deepcopy
import networkx as nx


class Nodes:
    def __init__(self, graph: nx.Graph, device, mode=None, k=None):
        copy_graph = graph.copy()
        nodes_list = list(copy_graph.nodes)
        greedy_node = set()
        if mode == "cutoff":
            for i in range(k):
                degree = nx.degree_centrality(copy_graph)
                greedy_node.add(max(degree, key=degree.get))
                copy_graph.remove_node(max(degree, key=degree.get))
            copy_graph = graph.copy()
        neighbors = []
        non_neighbors = []
        _remove_node = []
        for node in nodes_list:
            if mode == "cutoff":
                neighbor = set(sorted(copy_graph.neighbors(node))) & greedy_node
                non_neighbor = (set(nodes_list) - set(neighbor)) & greedy_node
            else:
                neighbor = sorted(copy_graph.neighbors(node))
                non_neighbor = set(nodes_list) - set(neighbor)
            neighbor = list(neighbor)
            non_neighbor = list(non_neighbor)
            if len(neighbor) == 0 or len(non_neighbor) == 0:
                _remove_node.append(node)
            else:
                neighbors.append(torch.tensor(neighbor, device=device))
                non_neighbors.append(torch.tensor(non_neighbor, device=device))
        for node in _remove_node:
            nodes_list.remove(node)
        self.nodes_index = torch.arange(len(nodes_list), device=device)
        self.nodes = torch.tensor(nodes_list, device=device)
        self.neighbors = neighbors
        self.non_neighbors = non_neighbors
        self.device = device

    def to(self, device):
        self.device = device
        self.nodes = self.nodes.to(device)
        self.nodes_index = self.nodes_index.to(device)
        self.neighbors = [neighbor.to(device) for neighbor in self.neighbors]
        self.non_neighbors = [non_neighbor.to(device) for non_neighbor in self.non_neighbors]


class Gene:
    def __init__(self, node, neighbor, non_neighbor, index, device, copy=False):
        self.node = node.clone()
        self.neighbor = neighbor.clone()
        self.non_neighbor = non_neighbor.clone()
        self.index = index
        self.device = device
        if not copy:
            _add_node, _remove_node = self.reconnect_edge()
            self.current_gene = torch.tensor([node, _add_node, _remove_node, index], device=device)

    def add_edge(self, _remove_node=None):
        node = self._random_node(self.non_neighbor, _remove_node)
        return node

    def remove_edge(self, _add_node=None):
        node = self._random_node(self.neighbor, _add_node)
        return node

    def reconnect_edge(self, _add_node=None, _remove_node=None):
        _add_node = self.add_edge(_add_node)
        _remove_node = self.remove_edge([_add_node, _remove_node])
        return _add_node, _remove_node

    def _random_node(self, tensor, node):
        node_index = torch.randperm(len(tensor), device=self.device)
        new_node = tensor[node_index[0]]
        if node is not None and new_node in node:
            if len(tensor) == 1:
                return new_node
            else:
                return tensor[node_index[1]]
        else:
            return new_node

    def __eq__(self, other):
        if torch.equal(self.current_gene, other.current_gene):
            return True
        else:
            return False

    def __hash__(self):
        current_gene_hash = tuple(self.current_gene.cpu().numpy())
        return hash(current_gene_hash)

    def copy(self):
        copy_gene = Gene(self.node, self.neighbor, self.non_neighbor, self.index, self.device, copy=True)
        copy_gene.neighbor = deepcopy(self.neighbor)
        copy_gene.non_neighbor = deepcopy(self.non_neighbor)
        copy_gene.current_gene = deepcopy(self.current_gene)
        return copy_gene


def Generate_Genes(nodes: Nodes, nums, device):
    gene_list = []
    curren_gene_list = torch.zeros(size=(nums, 4), device=device)
    node_index = torch.randint(len(nodes.nodes), size=(nums,), device=device)
    for i, index in enumerate(node_index):
        while True:
            gene = Gene(nodes.nodes[index], nodes.neighbors[index], nodes.non_neighbors[index], index, device=device)
            if gene not in gene_list:
                gene_list.append(gene)
                curren_gene_list[i] = gene.current_gene
                break

    return curren_gene_list


def Gain_Edge_Set(graph: nx.Graph, budget, device):
    edges = []
    edges_in_final = []
    copy_graph = graph.copy()
    G_i = IG(directed=False)
    G_i: IG = G_i.from_networkx(copy_graph)
    community = G_i.community_multilevel()
    for k in range(len(community)):
        node_in = set(community[k])
        node_out = set(copy_graph.nodes) - node_in

        for i in itertools.product(node_in, node_out):
            i = sorted(list(i))
            i = tuple(i)

            edges.append(i)

        edges_out = []
        for i in copy_graph.edges(list(node_out)):
            i = sorted(list(i))
            i = tuple(i)
            edges_out.append(i)

        edges_in = list(copy_graph.edges - edges_out)
        if len(edges_in) >= budget:
            edges_in_final = edges_in.copy()
    edges_out = list(set(edges) - set(copy_graph.edges))

    return torch.tensor(edges_in_final, dtype=torch.int64, device=device), torch.tensor(edges_out, dtype=torch.int64, device=device)


def generate_candidate_edge(edge_list, num, pattern, device):
    edges = edge_list[(torch.randperm(len(edge_list))[:num])]
    if pattern == 1:
        edges = torch.hstack((edges, torch.ones(size=(num, 1), device=device)))
    elif pattern == 0:
        edges = torch.hstack((edges, torch.zeros(size=(num, 1), device=device)))
    else:
        ValueError(f"No such pattern: {pattern}")
    return edges


def Generate_Pop(budget, edges_in, edges_out, device):
    add_num = torch.randint(budget, size=(1,), device=device).squeeze()
    del_num = budget - add_num
    del_edge = generate_candidate_edge(edges_in, del_num, 1, device)
    add_edge = generate_candidate_edge(edges_out, add_num, 0, device)
    pop = torch.cat((del_edge, add_edge))
    pop = pop[torch.randperm(len(pop))]
    # pop = torch.hstack((pop, torch.arange(budget, device=device)))
    return pop



