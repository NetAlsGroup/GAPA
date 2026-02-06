import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import networkx as nx
from time import time
from absolute_path import dataset_path
from gapa.utils.init_device import init_device
import numpy as np
from copy import deepcopy
from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm
from evotorch.logging import StdOutLogger
from evotorch.operators import GaussianMutation, SimulatedBinaryCrossOver
from evotorch.operators import CrossOver, CopyingOperator
from typing import Iterable, Optional, Union
from collections import Counter
from evotorch.decorators import vectorized, on_aux_device


device, _ = init_device()
# Circuit A01 humanDiseasome yeast1
data_set = 'yeast1'

if data_set == "dolphins" or data_set == "football" or data_set == "email-Eu-core":
    G = nx.read_gml(os.path.join(dataset_path, data_set, data_set + '.gml'), label="id")
else:
    G = nx.read_adjlist(os.path.join(dataset_path, data_set + '.txt'), nodetype=int)
A = torch.tensor(nx.to_numpy_array(G, nodelist=sorted(list(G.nodes()))), device=device)
G = nx.from_numpy_array(A.cpu().numpy())
nodes = torch.tensor(list(G.nodes()), device=device)
nodes_num = len(A)
edge_number = len(G.edges())
edge_list = torch.tensor(list(G.edges()), device=device)
k = int(0.1 * nodes_num)
selected_genes_rate = 0.4
selected_genes_num = int(nodes_num * selected_genes_rate)
pop_size = 80
Eye = torch.eye(nodes_num, device=device)
ONE = torch.ones((pop_size, k), dtype=torch.float32)
crossover_rate = 0.6
mutate_rate = 0.2


def CNDTest(graph, critical_nodes):
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


def pop_greedy_cutoff(graph, genes, pop_num):
    greedy_indi = []
    for _i in range(pop_num):
        temp_fitness = []
        temp_population = []
        for j in range(pop_size):
            temp_genes = torch.randperm(nodes_num, device=device)[:k]
            temp_population.append(temp_genes)
            temp_fitness.append(CNDTest(graph, genes[temp_genes]))
        top_index = torch.tensor(temp_fitness, device=device).argsort()[:1]
        greedy_indi.append(temp_population[top_index.item()])

    genes = []
    for pop in greedy_indi:
        genes = genes + pop.tolist()
    data = dict(Counter(genes))
    data = sorted(data.items(), key=lambda x: x[1])[::-1][:int(selected_genes_num / 2)]
    genes = torch.tensor([data[i][0] for i in range(len(data))], device=device)
    genes = nodes[genes].tolist()
    degree = nx.degree_centrality(graph)
    degree = sorted(degree.items(), key=lambda x: x[1])[::-1]
    degree = [degree[i][0] for i in range(len(degree))]
    count_i = 0
    while len(genes) < selected_genes_num:
        if degree[count_i] not in genes:
            genes.append(degree[count_i])
        count_i = count_i + 1
    return torch.tensor(genes, device=device)


@vectorized
@on_aux_device
def matrix_matmul(population) -> torch.Tensor:
    population = population.int()
    fitness_list = []
    for pop in population:
        copy_A = A.clone()
        copy_A[nodes[pop], :] = 0
        copy_A[:, nodes[pop]] = 0
        matrix2 = torch.matmul((copy_A + Eye), (copy_A + Eye))
        matrix4 = torch.matmul(matrix2, matrix2)
        matrix6 = torch.matmul(matrix4, matrix2)
        nonzero = torch.count_nonzero(matrix6, dim=1)
        fitness_list.append(nonzero.max())
    fitness = torch.tensor(fitness_list, device=population.device)
    return fitness


class crossover(CrossOver):
    def __init__(
            self,
            problem: Problem,
            *,
            tournament_size: int,
            obj_index: Optional[int] = None,
            num_children: Optional[int] = None,
            cross_over_rate: Optional[float] = None,
    ):
        super().__init__(
            problem,
            tournament_size=tournament_size,
            obj_index=obj_index,
            num_children=num_children,
            cross_over_rate=cross_over_rate,
        )

    def _do_cross_over(self, new_population1, new_population2):
        inner_device = new_population1.device
        one = torch.ones(new_population1.shape, dtype=torch.float32, device=inner_device)
        crossover_matrix = torch.tensor(np.random.choice([0, 1], size=new_population1.shape, p=[1 - crossover_rate, crossover_rate]), device=inner_device)
        crossover_population = new_population1 * (one - crossover_matrix) + new_population2 * crossover_matrix
        result = self._make_children_batch(crossover_population)
        return result


class mutation(CopyingOperator):
    def __init__(self, problem: Problem):
        super().__init__(problem)

    def _do(self, batch):
        result = deepcopy(batch)
        data = result.access_values()
        crossover_population = result.values
        inner_device = result.device
        one = torch.ones(crossover_population.shape, dtype=torch.float32, device=inner_device)
        mutation_matrix = torch.tensor(np.random.choice([0, 1], size=crossover_population.shape, p=[1 - mutate_rate, mutate_rate]), device=inner_device)
        mutation_population = crossover_population * (one - mutation_matrix) + torch.randint(0, nodes_num, size=crossover_population.shape, device=inner_device) * mutation_matrix

        data[:] = self._respect_bounds(mutation_population)
        return result


start = time()
nodes = pop_greedy_cutoff(G, nodes, pop_num=10)
nodes_num = len(nodes)

problem = Problem("min", matrix_matmul, solution_length=k, initial_bounds=(0, nodes_num-1), num_actors=4, num_gpus_per_actor=0.5)

"""
Considering that the proportion of the Mutation operator in the overall computation time 
is much smaller than that of the fitness calculation, 
and if we use the mutation function, the performance of the algorithm will become poor, 
we use the GaussianMutation provided by Evotorch here.
"""
searcher = GeneticAlgorithm(
    problem,
    popsize=80,
    operators=[
        crossover(problem, tournament_size=4, cross_over_rate=crossover_rate),
        GaussianMutation(problem, stdev=0.03),
    ])
logger = StdOutLogger(searcher, interval=100)
searcher.run(5000)
end = time()
print("Final status:\n", searcher.status)
print(f"Total Time: {end-start}")







