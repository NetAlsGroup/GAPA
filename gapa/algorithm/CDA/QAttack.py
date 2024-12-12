import torch
import torch.multiprocessing as mp
import torch.distributed as dist
# import cugraph
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from time import time
from igraph import Graph as ig
from gapa.framework.body import BasicBody
from gapa.framework.controller import BasicController
from gapa.framework.evaluator import BasicEvaluator
from gapa.utils.functions import Q_Test, NMI_Test
from gapa.utils.functions import current_time
from gapa.utils.functions import init_dist
from gapa.algorithm.CDA.Genes import Gene, Nodes
from gapa.algorithm.CDA.Genes import Generate_Genes


class QAttackEvaluator(BasicEvaluator):
    def __init__(self, pop_size, graph, device, cal_mode="cpu"):
        super().__init__(
            pop_size=pop_size,
            device=device
        )
        self.Nodes = None
        self.cal_mode = cal_mode
        self.graph = graph.copy()

    def forward(self, population):
        fitness_list = []
        copy_G = self.graph.copy()
        for i, pop in enumerate(population):
            genes = torch.stack([gene for gene in pop])
            add_edges = torch.stack((genes[:, 0], genes[:, 1]), dim=1)
            del_edges = torch.stack((genes[:, 0], genes[:, 2]), dim=1)
            copy_G.remove_edges_from(del_edges.tolist())
            copy_G.add_edges_from(add_edges.tolist())
            if self.cal_mode == "gpu":
                # fitness_list.append(cugraph.louvain(copy_G)[1])
                pass
            else:
                G_i = ig(directed=False)
                G_i = G_i.from_networkx(copy_G)
                fitness_list.append(ig.community_multilevel(G_i).modularity)
            copy_G.add_edges_from(del_edges.tolist())
            copy_G.remove_edges_from(add_edges.tolist())
        fitness_list = torch.exp(-1 * torch.tensor(fitness_list, device=self.device))
        return fitness_list


class QAttackBody(BasicBody):
    def __init__(self, nodes, budget, pop_size, fit_side, device):
        super().__init__()
        self.budget = budget
        self.pop_size = pop_size
        self.fit_side = fit_side
        self.device = device
        self.Nodes = nodes
        self.Nodes.to(device)

    def init_population(self):
        init_population = torch.zeros((self.pop_size, self.budget, 4), device=self.device)
        for i in range(self.pop_size):
            init_population[i] = Generate_Genes(self.Nodes, self.budget, device=self.device)
        ONE = torch.ones((self.pop_size, self.budget), dtype=torch.int32, device=self.device)
        return ONE, init_population

    def selection(self, population, fitness_list):
        copy_pop = population.clone()
        normalize_fit = fitness_list / fitness_list.sum()
        normalize_fit[normalize_fit < 0] = 0
        samples = torch.multinomial(normalize_fit, len(normalize_fit), replacement=True)
        return copy_pop[samples]

    def crossover(self, population, new_population1, new_population2, crossover_rate, one):
        crossover_matrix = torch.tensor(np.random.choice([0, 1], size=(self.pop_size, self.budget), p=[1 - crossover_rate, crossover_rate]), device=self.device)
        crossover_population_index = new_population1 * (one - crossover_matrix) + new_population2 * crossover_matrix
        return population.view(-1, population.shape[-1])[crossover_population_index]

    def mutation(self, population, crossover_population, mutate_rate, one):
        mutation_matrix = torch.tensor(np.random.choice([0, 1], size=(self.pop_size, self.budget), p=[1 - mutate_rate, mutate_rate]), device=self.device)
        mutation_population_index = crossover_population * (one - mutation_matrix) + torch.randint(2, 5, size=(self.pop_size, self.budget), device=self.device) * mutation_matrix
        return self._point_mutation(population, mutation_population_index)

    def elitism(self, population, mutation_population, fitness_list, new_fitness_list, best_fitness_list):
        stack_population = torch.vstack((population, mutation_population))
        stack_fitness_list = torch.hstack((fitness_list, new_fitness_list))
        top_index = None
        if self.fit_side == 'max':
            top_index = torch.argsort(stack_fitness_list)[len(stack_fitness_list) - self.pop_size:]
        elif self.fit_side == 'min':
            top_index = torch.argsort(stack_fitness_list)[:self.pop_size]
        population = stack_population[top_index]
        fitness_list = stack_fitness_list[top_index]
        if self.fit_side == 'max':
            best_fitness_list = torch.hstack((best_fitness_list, max(fitness_list)))
        elif self.fit_side == 'min':
            best_fitness_list = torch.hstack((best_fitness_list, min(fitness_list)))
        else:
            raise ValueError(f"No such fit side: {self.fit_side}")
        return population, fitness_list, best_fitness_list

    def _point_mutation(self, crossover_population, mutation_population_index):
        for i in range(len(mutation_population_index)):
            for j in range(len(mutation_population_index[0])):
                idx = mutation_population_index[i][j]
                if idx.item() == 2:
                    rand_index = torch.randperm(len(self.Nodes.nodes), device=self.device)[0]
                    crossover_population[i][j] = Gene(self.Nodes.nodes[rand_index], self.Nodes.neighbors[rand_index], self.Nodes.non_neighbors[rand_index], rand_index, device=self.device).current_gene
                elif idx.item() == 3:
                    index = crossover_population[i][j][3].int().item()
                    add_node = crossover_population[i][j][1]
                    _add_node = Gene(self.Nodes.nodes[index], self.Nodes.neighbors[index], self.Nodes.non_neighbors[index], index, device=self.device).add_edge(add_node)
                    if _add_node == add_node:
                        rand_index = torch.randperm(len(self.Nodes.nodes), device=self.device)[0]
                        crossover_population[i][j] = Gene(self.Nodes.nodes[rand_index], self.Nodes.neighbors[rand_index], self.Nodes.non_neighbors[rand_index], rand_index, device=self.device).current_gene
                    else:
                        crossover_population[i][j][1] = _add_node
                elif idx.item() == 4:
                    index = crossover_population[i][j][3].int().item()
                    remove_node = crossover_population[i][j][2]
                    _remove_node = Gene(self.Nodes.nodes[index], self.Nodes.neighbors[index], self.Nodes.non_neighbors[index], index, device=self.device).remove_edge(remove_node)
                    if _remove_node == remove_node:
                        rand_index = torch.randperm(len(self.Nodes.nodes), device=self.device)[0]
                        crossover_population[i][j] = Gene(self.Nodes.nodes[rand_index], self.Nodes.neighbors[rand_index], self.Nodes.non_neighbors[rand_index], rand_index, device=self.device).current_gene
                    else:
                        crossover_population[i][j][2] = _remove_node

        return crossover_population


class QAttackController(BasicController):
    def __init__(self, path, pattern, data_loader, loops, crossover_rate, mutate_rate, pop_size, device, fit_side="max"):
        super().__init__(
            path,
            pattern,
        )
        self.loops = loops
        self.crossover_rate = crossover_rate
        self.mutate_rate = mutate_rate
        self.pop_size = pop_size
        self.device = device
        self.fit_side = fit_side
        self.dataset = data_loader.dataset
        self.budget = data_loader.k
        self.graph = data_loader.G.copy()
        self.Nodes = None

    def setup(self, data_loader, evaluator: QAttackEvaluator):
        self.Nodes = Nodes(data_loader.G, device=self.device)
        # self.Nodes = Nodes(data_loader.G, device=self.device, mode="cutoff", k=data_loader.selected_genes_num)
        evaluator.Nodes = self.Nodes
        print(f"Original Q: {Q_Test(data_loader.G)}")
        return evaluator

    def calculate(self, max_generation, evaluator):
        best_Q = []
        best_NMI = []
        best_genes = []
        time_list = []
        body = QAttackBody(self.Nodes, self.budget, self.pop_size, self.fit_side, evaluator.device)
        for loop in range(self.loops):
            start = time()
            ONE, population = body.init_population()
            if self.mode == "sm":
                evaluator = torch.nn.DataParallel(evaluator)
            fitness_list = evaluator(population)
            best_fitness_list = torch.tensor(data=[], device=self.device)
            best_fitness_list = torch.hstack((best_fitness_list, torch.max(fitness_list)))
            with tqdm(total=max_generation) as pbar:
                pbar.set_description(f'Training....{self.dataset} in Loop: {loop}...')
                for generation in range(max_generation):
                    new_population_index_1 = torch.arange(self.pop_size * self.budget, device=self.device).reshape(self.pop_size, self.budget)
                    new_population_index_2 = body.selection(new_population_index_1, fitness_list)
                    crossover_population = body.crossover(population, new_population_index_1, new_population_index_2, self.crossover_rate, ONE)
                    mutation_population_index = torch.ones(size=(len(crossover_population), len(crossover_population[0])), device=self.device)
                    mutation_population = body.mutation(crossover_population, mutation_population_index, self.mutate_rate, ONE)
                    # mutation_population = self._remove_repeat(mutation_population)
                    new_fitness_list = evaluator(mutation_population)
                    population, fitness_list, best_fitness_list = body.elitism(population, mutation_population, fitness_list, new_fitness_list, best_fitness_list)
                    if generation % 30 == 0 or (generation + 1) == max_generation:
                        index = torch.argsort(fitness_list)[-1]
                        best_Q.append(-torch.log(fitness_list[index]).item())
                        pop = population[index]
                        genes = torch.stack([gene for gene in pop])
                        add_edges = torch.stack((genes[:, 0], genes[:, 1]), dim=1)
                        del_edges = torch.stack((genes[:, 0], genes[:, 2]), dim=1)
                        copy_G = self.graph.copy()
                        copy_G.remove_edges_from(del_edges.tolist())
                        copy_G.add_edges_from(add_edges.tolist())
                        best_NMI.append(NMI_Test(self.graph.copy(), copy_G))
                        best_genes.append(genes)
                        end = time()
                        time_list.append(end - start)
                    pbar.set_postfix(fitness=max(fitness_list).item(), Q=min(best_Q), NMI=min(best_NMI))
                    pbar.update(1)
            top_index = best_Q.index(min(best_Q))
            print(f"Q after attack: {best_Q[top_index]}, NMI after attack: {best_NMI[top_index]}")
            self.save(self.dataset, best_genes[top_index], [best_Q[top_index], best_NMI[top_index], time_list[-1]], time_list, "QAttack", bestQ=best_Q, bestNMI=best_NMI)
            print(f"Loop {loop} finished. Data saved in {self.path}...")

    def mp_calculate(self, rank, max_generation, evaluator, world_size):
        device = init_dist(rank, world_size)
        best_Q = []
        best_NMI = []
        best_genes = []
        time_list = []
        component_size = self.pop_size // world_size
        body = QAttackBody(self.Nodes, self.budget, component_size, self.fit_side, device)
        for loop in range(self.loops):
            start = time()
            ONE, component_population = body.init_population()
            if self.mode == "mnm":
                evaluator = torch.nn.DataParallel(evaluator)
            component_fitness_list = evaluator(component_population).to(device)
            best_fitness_list = torch.tensor(data=[], device=device)
            best_fitness_list = torch.hstack((best_fitness_list, torch.max(component_fitness_list)))
            if rank == 0:
                population = [torch.zeros(component_population.shape, dtype=component_population.dtype, device=device) for _ in range(world_size)]
                fitness_list = [torch.empty((component_size,), dtype=component_fitness_list.dtype, device=device) for _ in range(world_size)]
            else:
                population = None
                fitness_list = None
            dist.gather(component_population, population, dst=0)
            dist.gather(component_fitness_list, fitness_list, dst=0)
            if rank == 0:
                population = torch.cat(population)
                fitness_list = torch.cat(fitness_list)
            with tqdm(total=max_generation, position=rank) as pbar:
                pbar.set_description(f'Rank {rank} in {self.dataset} in Loop: {loop}')
                for generation in range(max_generation):
                    if rank == 0:
                        new_population_index_1 = torch.arange(self.pop_size * self.budget, device=device).reshape(self.pop_size, self.budget)
                        new_population_index_2 = body.selection(new_population_index_1, fitness_list)
                        body.pop_size = self.pop_size
                        crossover_ONE = torch.ones((self.pop_size, self.budget), dtype=ONE.dtype, device=device)
                        crossover_population = body.crossover(population, new_population_index_1, new_population_index_2, self.crossover_rate, crossover_ONE)
                        body.pop_size = component_size
                    if rank == 0:
                        crossover_population = torch.stack(torch.split(crossover_population, component_size))
                    else:
                        crossover_population = torch.stack([torch.zeros(component_population.shape, dtype=component_population.dtype, device=device) for _ in range(world_size)])
                    dist.broadcast(crossover_population, src=0)
                    mutation_population_index = torch.ones(size=(component_size, self.budget), device=device)
                    mutation_population = body.mutation(crossover_population[rank], mutation_population_index, self.mutate_rate, ONE)
                    # mutation_population = self._remove_repeat(mutation_population)
                    new_component_fitness_list = evaluator(mutation_population).to(device)

                    if rank == 0:
                        elitism_population = [torch.zeros(mutation_population.shape, dtype=component_population.dtype, device=device) for _ in range(world_size)]
                        elitism_fitness_list = [torch.empty((component_size,), dtype=new_component_fitness_list.dtype, device=device) for _ in range(world_size)]
                    else:
                        elitism_population = None
                        elitism_fitness_list = None
                    dist.gather(mutation_population, elitism_population, dst=0)
                    dist.gather(new_component_fitness_list, elitism_fitness_list, dst=0)
                    if rank == 0:
                        elitism_population = torch.cat(elitism_population)
                        elitism_fitness_list = torch.cat(elitism_fitness_list)
                        body.pop_size = self.pop_size
                        population, fitness_list, best_fitness_list = body.elitism(population, elitism_population, fitness_list, elitism_fitness_list, best_fitness_list)
                        top_index = torch.argsort(fitness_list)[self.pop_size-component_size:]
                        component_population = population[top_index]
                        component_fitness_list = fitness_list[top_index]
                        body.pop_size = component_size
                    else:
                        component_population = torch.zeros(component_population.shape, dtype=component_population.dtype, device=device)
                        component_fitness_list = torch.empty((component_size,), dtype=component_fitness_list.dtype, device=device)
                    dist.broadcast(component_population, src=0)
                    dist.broadcast(component_fitness_list, src=0)
                    if generation % 30 == 0 or (generation + 1) == max_generation:
                        index = torch.argsort(component_fitness_list)[-1]
                        best_Q.append(-torch.log(component_fitness_list[index]).item())
                        pop = component_population[index]
                        genes = torch.stack([gene for gene in pop])
                        add_edges = torch.stack((genes[:, 0], genes[:, 1]), dim=1)
                        del_edges = torch.stack((genes[:, 0], genes[:, 2]), dim=1)
                        copy_G = self.graph.copy()
                        copy_G.remove_edges_from(del_edges.tolist())
                        copy_G.add_edges_from(add_edges.tolist())
                        best_NMI.append(NMI_Test(self.graph.copy(), copy_G))
                        best_genes.append(genes)
                        end = time()
                        time_list.append(end - start)
                    pbar.set_postfix(fitness=max(component_fitness_list).item(), Q=min(best_Q), NMI=min(best_NMI))
                    pbar.update(1)
            best_genes = torch.stack(best_genes)
            best_Q = torch.tensor(best_Q, device=device)
            best_NMI = torch.tensor(best_NMI, device=device)
            if rank == 0:
                whole_genes = [torch.zeros(best_genes.shape, dtype=best_genes.dtype, device=device) for _ in range(world_size)]
                whole_NMI = [torch.empty(best_NMI.shape, device=device) for _ in range(world_size)]
                whole_Q = [torch.empty(best_Q.shape, device=device) for _ in range(world_size)]
            else:
                whole_genes = None
                whole_NMI = None
                whole_Q = None
            dist.barrier()
            dist.gather(best_genes, whole_genes, dst=0)
            dist.gather(best_Q, whole_Q, dst=0)
            dist.gather(best_NMI, whole_NMI, dst=0)
            if rank == 0:
                whole_genes = torch.cat(whole_genes)
                whole_Q = torch.cat(whole_Q)
                whole_NMI = torch.cat(whole_NMI)
                top_index = torch.argsort(whole_NMI)[0]
                print(f"Q after attack: {whole_Q[top_index]}, NMI after attack: {whole_NMI[top_index]}")
                self.save(self.dataset, whole_genes[top_index], [whole_Q[top_index].item(), whole_NMI[top_index].item(), time_list[-1]], time_list, "QAttack", bestQ=best_Q.tolist(), bestNMI=best_NMI.tolist())
        torch.cuda.empty_cache()
        dist.destroy_process_group()
        torch.cuda.synchronize()

    def save(self, dataset, gene, best_metric, time_list, method, **kwargs):
        save_path = self.path + dataset + '_crossover_rate_' + str(self.crossover_rate) + '_mutate_rate_' + str(self.mutate_rate) + f'_{method}.txt'
        with open(save_path, 'a+') as f:
            f.write(current_time())
            f.write(f"\nCurrent mode: {self.mode}. Current pop_size: {self.pop_size}\n")
        with open(save_path, 'a+') as f:
            f.write(str([i for i in kwargs['bestQ']]) + '\n')
        with open(save_path, 'a+') as f:
            f.write(str([i for i in kwargs['bestNMI']]) + '\n')
        with open(save_path, 'a+') as f:
            f.write(str([i.tolist() for i in gene]) + '\n')
        with open(save_path, 'a+') as f:
            f.write(str(time_list) + '\n')
        with open(save_path, 'a+') as f:
            f.write(str(best_metric) + '\n')


    def _remove_repeat(self, population: torch.Tensor):
        for i, pop in enumerate(population):
            clean = set([tuple(k) for k in pop.tolist()])
            out = torch.tensor(list(clean), device=population.device)
            lens = len(clean)
            while lens != len(pop):
                edge = self._generate_node(clean, population.device)
                out = torch.cat((out, edge.unsqueeze(0)))
                lens += 1
            population[i] = out
        return population

    def _generate_node(self, edges, device):
        rand_index = torch.randperm(len(self.Nodes.nodes), device=self.device)[0]
        edge = Gene(self.Nodes.nodes[rand_index], self.Nodes.neighbors[rand_index], self.Nodes.non_neighbors[rand_index], rand_index, device=self.device).current_gene
        if tuple(edge.tolist()) in edges:
            edge = self._generate_node(edges, device)
            return edge
        else:
            return edge


def QAttack(mode, max_generation, data_loader, controller: QAttackController, evaluator, world_size):
    controller.mode = mode
    evaluator = controller.setup(data_loader=data_loader, evaluator=evaluator)
    if mode == "s" or mode == "sm":
        controller.calculate(max_generation=max_generation, evaluator=evaluator)
    elif mode == "m" or mode == "mnm":
        mp.spawn(controller.mp_calculate, args=(max_generation, deepcopy(evaluator), world_size), nprocs=world_size, join=True)
    else:
        raise ValueError(f"No such mode. Please choose ss, sm, ms or mm.")


