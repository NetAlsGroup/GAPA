import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import networkx as nx
from time import time
from absolute_path import dataset_path
from gapa.utils.init_device import init_device
import numpy as np
from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm
from evotorch.logging import StdOutLogger
from evotorch.operators import GaussianMutation, SimulatedBinaryCrossOver
from evotorch.operators import CrossOver, CopyingOperator
from typing import Iterable, Optional, Union
from collections import Counter
from copy import deepcopy
from evotorch.decorators import vectorized, on_aux_device


device, _ = init_device()
# karate dolphins football email-Eu-core
data_set = 'email-Eu-core'

if data_set == "dolphins" or data_set == "football" or data_set == "email-Eu-core":
    G = nx.read_gml(os.path.join(dataset_path, data_set, data_set + '.gml'), label="id")
else:
    G = nx.read_adjlist(os.path.join(dataset_path, data_set + '.txt'), nodetype=int)
A = torch.tensor(nx.to_numpy_array(G, nodelist=sorted(list(G.nodes()))), device=device)
G = nx.from_numpy_array(A.cpu().numpy())
nodes_num = len(A)
edge_number = len(G.edges())
edge_list = torch.tensor(list(G.edges()), device=device)
k = int(0.1 * nodes_num)
pop_size = 100
Eye = torch.eye(nodes_num, device=device)
ONE = torch.ones((pop_size, k), dtype=torch.float32, device=device)
crossover_rate = 0.8
mutate_rate = 0.5

ONE_distance = torch.ones((nodes_num, nodes_num), dtype=torch.float32, device=device)
ONE_distance.fill_diagonal_(0)
D_inverse = torch.diag(torch.float_power(torch.sum(A + Eye, dim=1), -1))
normalized_matrix = torch.matmul(D_inverse.float(), A.float())
approximate_deepwalk_matrix = torch.mul(1 / 2, torch.add(normalized_matrix, torch.matmul(normalized_matrix, normalized_matrix)))
u, s, v = torch.svd_lowrank(approximate_deepwalk_matrix, niter=10)

# Calculate distance matrix
embedding = torch.matmul(u, torch.diag(torch.sqrt(s)))
E_dots = torch.sum(torch.mul(embedding, embedding), dim=1).reshape(nodes_num, 1)

distance_org = torch.mul(
    torch.float_power(
        torch.abs(torch.add(
            torch.matmul(E_dots, torch.ones(size=(1, nodes_num), device=device, dtype=torch.float32)),
            torch.matmul(torch.ones(size=(nodes_num, 1), device=device, dtype=torch.float32),
                         E_dots.T))
                  - 2 * torch.matmul(embedding, embedding.T)),
        1 / 2).float(),
    ONE_distance.float())


@vectorized
@on_aux_device
def CalFit(population) -> torch.Tensor:
    population = population.int()
    inner_device = population.device
    # graph = G.copy()
    IMatrix = Eye.clone().to(inner_device)
    inner_edge_list = edge_list.clone().to(inner_device)
    inner_ONE_distance = ONE_distance.clone().to(inner_device)
    inner_distance_org = distance_org.clone().to(inner_device)
    fitness_list = torch.tensor([], device=inner_device)
    for i, pop in enumerate(population):
        copy_A = A.clone().float().to(device)
        del_idx = inner_edge_list[pop]
        copy_A[del_idx[:, 0], del_idx[:, 1]] = 0
        copy_A[del_idx[:, 1], del_idx[:, 0]] = 0
        normalized_matrix = torch.matmul(
            torch.diag(torch.float_power(torch.sum(copy_A + IMatrix, dim=1), -1)).float(),
            copy_A.float())  # D^(-1)@A
        approximate_deepwalk_matrix = torch.mul(1 / 2, torch.add(normalized_matrix, torch.matmul(normalized_matrix, normalized_matrix)))
        u, s, v = torch.svd_lowrank(approximate_deepwalk_matrix)
        embedding = torch.matmul(u, torch.diag(torch.sqrt(s)))
        # Calculate distance
        E_dots = torch.sum(torch.mul(embedding, embedding), dim=1).reshape(nodes_num, 1)
        distance = torch.mul(
            torch.float_power(
                torch.abs(
                    torch.add(
                        torch.matmul(E_dots, torch.ones(size=(1, nodes_num), device=inner_device)),
                        torch.matmul(torch.ones(size=(nodes_num, 1), device=inner_device),
                                     E_dots.T))
                    - 2 * torch.matmul(embedding, embedding.T)),
                1 / 2),
            inner_ONE_distance)
        # Calculate fitness
        fitness = 1 - abs(
            torch.corrcoef(
                input=torch.vstack((distance.flatten(), inner_distance_org.flatten()))
            )[0, 1]
        ).unsqueeze(0)
        fitness_list = torch.cat((fitness_list, fitness))
    return fitness_list


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
problem = Problem("max", CalFit, solution_length=k, initial_bounds=(0, edge_number-1), num_actors=4, num_gpus_per_actor=0.5)

"""
Considering that the proportion of the Mutation operator in the overall computation time 
is much smaller than that of the fitness calculation, 
and if we use the mutation function, the performance of the algorithm will become poor, 
we use the GaussianMutation provided by Evotorch here.
"""
searcher = GeneticAlgorithm(
    problem,
    popsize=100,
    operators=[
        crossover(problem, tournament_size=4, cross_over_rate=crossover_rate),
        GaussianMutation(problem, stdev=0.03),
    ])
logger = StdOutLogger(searcher, interval=100)
searcher.run(1500)
end = time()
print("Final status:\n", searcher.status)
print(f"Total Time: {end-start}")







