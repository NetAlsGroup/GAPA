import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import networkx as nx
import random
from copy import deepcopy
from tqdm import tqdm
from time import time
from gapa.framework.body import Body
from gapa.framework.controller import BasicController
from gapa.framework.evaluator import BasicEvaluator
from gapa.utils.functions import CNDTest
from gapa.utils.functions import current_time
from gapa.utils.functions import init_dist
from igraph import Graph as ig


def phenotype2genotype(embeds, phenotype):
    return embeds[phenotype]


def genotype2phenotype(genotype, nodes, nodes_num, budget, device):
    # 计算每一节点与所有节点间两两距离
    # genotype = [1, budget], nodes = [1, nodes_num]
    fit_distance = torch.abs(genotype.t().repeat(1, nodes_num) - nodes.repeat(budget, 1)).cpu().numpy()
    # 计算每一节点最接近节点，无重复
    phenotype = []
    for i in range(len(fit_distance)):
        distance_sort = fit_distance[i].argsort()
        j = random.randint(0, list(fit_distance[i]).count(fit_distance[i][distance_sort[0]]) - 1)
        while True:
            if distance_sort[j] not in phenotype:
                phenotype.append(distance_sort[j])
                break
            j = j + 1
    return torch.tensor(phenotype, dtype=torch.int, device=device)


class TDEEvaluator(BasicEvaluator):
    def __init__(self, pop_size, graph, budget, device):
        super().__init__(
            pop_size=pop_size,
            device=device
        )
        self.graph = graph.copy()
        self.budget = budget
        self.nodes = None
        self.embeds = None
        self.embeds_num = None
        self.R0 = None

    def forward(self, population):

        def calculate_r(graph):
            r = 0
            len_copy_g = graph.vcount()

            for z in range(len_copy_g - 1):
                degree = graph.degree()

                node_to_remove = degree.index(max(degree))
                graph.delete_vertices(node_to_remove)

                largest_cc_size = max(graph.clusters().sizes())
                r += largest_cc_size / len_copy_g
            return r / (len_copy_g - 1)

        device = population.device
        copy_embeds = self.embeds.clone().to(device)
        fitness_list = torch.zeros(size=(len(population),), device=device)
        pop_embeds = population.unsqueeze(dim=1)
        for i, pop_embed in enumerate(pop_embeds):
            copy_g = self.graph.copy()
            pop_nodes = genotype2phenotype(pop_embed, copy_embeds, self.embeds_num, self.budget, device)
            for node in pop_nodes:
                copy_g.remove_node(node.item())
            graph_i = ig.from_networkx(copy_g)
            fitness_list[i] = self.R0 - torch.tensor(calculate_r(graph=graph_i), device=device)
        return fitness_list

    @staticmethod
    def calculate_r(graph):
        r = 0
        len_copy_g = graph.vcount()

        for z in range(len_copy_g - 1):
            degree = graph.degree()

            node_to_remove = degree.index(max(degree))
            graph.delete_vertices(node_to_remove)

            largest_cc_size = max(graph.clusters().sizes())
            r += largest_cc_size / len_copy_g
        return r / (len_copy_g - 1)

        # r = 0
        # len_copy_g = len(graph)
        # for z in range(0, len_copy_g - 1):
        #     degree = nx.degree_centrality(graph)
        #     graph.remove_node(max(degree, key=degree.get))
        #     r = r + len(max(nx.connected_components(graph), key=len)) / len_copy_g
        # return r / (len_copy_g - 1)


class TDEBody(Body):
    def __init__(self, critical_num, budget, pop_size, fit_side, device):
        super().__init__(
            critical_num=critical_num,
            budget=budget,
            pop_size=pop_size,
            fit_side=fit_side,
            device=device
        )

    @staticmethod
    def mutation_rewrite(population, selected_population_1, selected_population_2, mutate_rate):
        return population + mutate_rate * (selected_population_1 - selected_population_2)


class TDEController(BasicController):
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
        self.selected_genes_num = data_loader.selected_genes_num
        self.graph = data_loader.G
        self.nodes = None
        self.nodes_num = None
        self.mode = None
        self.embeds = None

    def setup(self, data_loader, evaluator: TDEEvaluator):
        self.nodes = data_loader.nodes
        self.nodes_num = data_loader.nodes_num
        degree_list = [nx.degree(self.graph)[i] for i in range(len(self.graph))]
        c = max(degree_list)
        d = min(degree_list)
        if c == d:
            degree_list = [1 for _ in range(len(self.graph))]
        else:
            degree_list = list((np.array(degree_list) - d) / (c - d))
        embeds = torch.tensor(degree_list, device=self.device)
        evaluator.nodes = data_loader.nodes
        evaluator.embeds = embeds
        evaluator.embeds_num = len(embeds)

        graph_i = ig.from_networkx(self.graph.copy())
        evaluator.R0 = evaluator.calculate_r(graph_i)
        self.embeds = embeds
        print(evaluator.R0)
        return evaluator

    def calculate(self, max_generation, evaluator):
        best_PCG = []
        best_MCN = []
        best_genes = []
        time_list = []
        body = TDEBody(self.nodes_num, self.budget, self.pop_size, self.fit_side, evaluator.device)
        for loop in range(self.loops):
            start = time()
            ONE, population = body.init_population()
            fitness_list = torch.empty(size=(self.pop_size,), device=self.device)
            for i, pop in enumerate(population):
                copy_graph = self.graph.copy()
                for node in pop:
                    copy_graph.remove_node(node.item())
                graph_i = ig.from_networkx(copy_graph)
                fitness_list[i] = evaluator.R0 - torch.tensor(evaluator.calculate_r(graph_i), device=self.device)
            if self.mode == "sm":
                evaluator = torch.nn.DataParallel(evaluator)
            population_embed = phenotype2genotype(self.embeds, population)
            best_fitness_list = torch.tensor(data=[], device=self.device)
            best_fitness_list = torch.hstack((best_fitness_list, torch.max(fitness_list)))
            with tqdm(total=max_generation) as pbar:
                pbar.set_description(f'Training....{self.dataset} in Loop: {loop}...')
                for generation in range(max_generation):
                    rotary_table_1 = self._calc_rotary_table(fitness_list, self.device)
                    rotary_table_2 = self._calc_rotary_table(fitness_list, self.device, pattern="trans")
                    new_population_1 = population_embed[[self._roulette(rotary_table_1) for _ in range(self.pop_size)]]
                    new_population_2 = population_embed[[self._roulette(rotary_table_2) for _ in range(self.pop_size)]]
                    mutation_population_embed = body.mutation_rewrite(population_embed, new_population_1, new_population_2, self.mutate_rate)
                    new_population_3 = population_embed[[self._roulette(rotary_table_2) for _ in range(self.pop_size)]]
                    crossover_population_embed = body.crossover(mutation_population_embed, new_population_3, self.crossover_rate, ONE)
                    # crossover_population_embed = self._remove_repeat(crossover_population_embed, self.embeds)
                    crossover_population_embed[crossover_population_embed > 1] = 1
                    crossover_population_embed[crossover_population_embed < 0] = 0
                    new_fitness_list = evaluator(crossover_population_embed)
                    population_embed, fitness_list, best_fitness_list = body.elitism(population_embed, crossover_population_embed, fitness_list, new_fitness_list, best_fitness_list)
                    if generation % 50 == 0 or (generation+1) == max_generation:
                        genes_embed = population_embed[torch.argsort(fitness_list.clone(), descending=True)[0]].unsqueeze(dim=0)
                        critical_nodes = genotype2phenotype(genes_embed, self.embeds, len(self.embeds), self.budget, self.device)
                        best_PCG.append(CNDTest(self.graph, critical_nodes))
                        best_MCN.append(CNDTest(self.graph, critical_nodes, pattern="ccn"))
                        best_genes.append(critical_nodes)
                        end = time()
                        time_list.append(end - start)
                    pbar.set_postfix(fitness=max(fitness_list).item(), PCG=min(best_PCG), MCN=min(best_MCN))
                    pbar.update(1)
            top_index = best_PCG.index(min(best_PCG))
            print(f"Best PC(G): {best_PCG[top_index]}. Best connected num: {best_MCN[top_index]}.")
            self.save(self.dataset, best_genes[top_index], [best_PCG[top_index], best_MCN[top_index], time_list[-1]], time_list, "TDE", bestPCG=best_PCG, bestMCN=best_MCN)
            print(f"Loop {loop} finished. Data saved in {self.path}...")

    def mp_calculate(self, rank, max_generation, evaluator, world_size):
        device = init_dist(rank, world_size)
        best_PCG = []
        best_MCN = []
        best_genes = []
        time_list = []
        embeds = self.embeds.to(device)
        component_size = self.pop_size // world_size
        if self.pop_size == 2:
            component_size = 1
        body = TDEBody(self.nodes_num, self.budget, component_size, self.fit_side, device)
        for loop in range(self.loops):
            start = time()
            ONE, component_population = body.init_population()
            if self.pop_size == 2:
                component_population = component_population.unsqueeze(0)
            component_fitness_list = torch.empty(size=(component_size,), device=device)
            for i, pop in enumerate(component_population):
                copy_graph = self.graph.copy()
                for node in pop:
                    copy_graph.remove_node(node.item())
                graph_i = ig(directed=False)
                graph_i = graph_i.from_networkx(copy_graph)
                component_fitness_list[i] = evaluator.R0 - torch.tensor(evaluator.calculate_r(graph_i), device=device)
                # fitness_list[i] = evaluator.R0 - torch.tensor(evaluator.calculate_r(copy_graph), device=device)
            if self.mode == "mnm":
                evaluator = torch.nn.DataParallel(evaluator)
            component_population_embed = phenotype2genotype(embeds, component_population)
            best_fitness_list = torch.tensor(data=[], device=device)
            best_fitness_list = torch.hstack((best_fitness_list, torch.max(component_fitness_list)))
            if rank == 0:
                population_embed = [torch.zeros((component_size, self.budget), dtype=component_population_embed.dtype, device=device) for _ in range(world_size)]
                fitness_list = [torch.empty((component_size,), dtype=component_fitness_list.dtype, device=device) for _ in range(world_size)]
            else:
                population_embed = None
                fitness_list = None

            dist.gather(component_population_embed, population_embed, dst=0)
            dist.gather(component_fitness_list, fitness_list, dst=0)
            if rank == 0:
                population_embed = torch.cat(population_embed)
                fitness_list = torch.cat(fitness_list)
            with tqdm(total=max_generation, position=rank) as pbar:
                pbar.set_description(f'Rank {rank} in {self.dataset} in Loop: {loop}')
                for generation in range(max_generation):
                    if rank == 0:
                        body.pop_size = self.pop_size
                        rotary_table_1 = self._calc_rotary_table(fitness_list, device)
                        rotary_table_2 = self._calc_rotary_table(fitness_list, device, pattern="trans")
                        new_population_1 = population_embed[[self._roulette(rotary_table_1) for _ in range(self.pop_size)]]
                        new_population_2 = population_embed[[self._roulette(rotary_table_2) for _ in range(self.pop_size)]]
                        mutation_population_embed = body.mutation_rewrite(population_embed, new_population_1, new_population_2, self.mutate_rate)
                        new_population_3 = population_embed[[self._roulette(rotary_table_2) for _ in range(self.pop_size)]]
                        crossover_ONE = torch.ones((self.pop_size, self.budget), dtype=torch.int64, device=device)
                        crossover_population_embed = body.crossover(mutation_population_embed, new_population_3, self.crossover_rate, crossover_ONE)
                        crossover_population_embed[crossover_population_embed > 1] = 1
                        crossover_population_embed[crossover_population_embed < 0] = 0
                        elitism_population_embed = crossover_population_embed.clone()
                    if rank == 0:
                        crossover_population_embed = torch.stack(torch.split(crossover_population_embed, component_size))
                    else:
                        crossover_population_embed = torch.stack([torch.zeros((component_size, self.budget), dtype=component_population_embed.dtype, device=device) for _ in range(world_size)])
                    dist.broadcast(crossover_population_embed, src=0)
                    new_fitness_list = evaluator(crossover_population_embed[rank]).to(device)

                    if rank == 0:
                        elitism_fitness_list = [torch.empty((component_size,), dtype=new_fitness_list.dtype, device=device) for _ in range(world_size)]
                    else:
                        elitism_population_embed = None
                        elitism_fitness_list = None
                    dist.gather(new_fitness_list, elitism_fitness_list, dst=0)
                    if rank == 0:
                        elitism_fitness_list = torch.cat(elitism_fitness_list)
                        population_embed, fitness_list, best_fitness_list = body.elitism(population_embed, elitism_population_embed, fitness_list, elitism_fitness_list, best_fitness_list)
                        top_index = torch.argsort(fitness_list)[self.pop_size - component_size:]
                        component_population_embed = population_embed[top_index]
                        component_fitness_list = fitness_list[top_index]
                    else:
                        component_population_embed = torch.zeros((component_size, self.budget), dtype=component_population_embed.dtype, device=device)
                        component_fitness_list = torch.empty((component_size,), dtype=component_fitness_list.dtype, device=device)
                    dist.broadcast(component_population_embed, src=0)
                    dist.broadcast(component_fitness_list, src=0)
                    if generation % 50 == 0 or (generation + 1) == max_generation:
                        genes_embed = component_population_embed[torch.argsort(component_fitness_list.clone(), descending=True)[0]].unsqueeze(dim=0)
                        critical_nodes = genotype2phenotype(genes_embed, embeds, len(embeds), self.budget, device)
                        best_PCG.append(CNDTest(self.graph, critical_nodes))
                        best_MCN.append(CNDTest(self.graph, critical_nodes, pattern="ccn"))
                        best_genes.append(critical_nodes)
                        end = time()
                        time_list.append(end - start)
                    pbar.set_postfix(fitness=max(component_fitness_list).item(), PCG=min(best_PCG), MCN=min(best_MCN))
                    pbar.update(1)

            best_genes = torch.stack(best_genes)
            best_PCG = torch.tensor(best_PCG, device=device)
            best_MCN = torch.tensor(best_MCN, device=device)
            if rank == 0:
                whole_genes = [torch.zeros(best_genes.shape, dtype=best_genes.dtype, device=device) for _ in range(world_size)]
                whole_PCG = [torch.empty(best_PCG.shape, device=device) for _ in range(world_size)]
                whole_MCN = [torch.zeros(best_MCN.shape, dtype=best_MCN.dtype, device=device) for _ in range(world_size)]
            else:
                whole_genes = None
                whole_PCG = None
                whole_MCN = None
            dist.barrier()
            dist.gather(best_genes, whole_genes, dst=0)
            dist.gather(best_PCG, whole_PCG, dst=0)
            dist.gather(best_MCN, whole_MCN, dst=0)
            if rank == 0:
                whole_genes = torch.cat(whole_genes)
                whole_PCG = torch.cat(whole_PCG)
                whole_MCN = torch.cat(whole_MCN)
                top_index = torch.argsort(whole_PCG)[0]
                print(f"Best PC(G): {whole_PCG[top_index]}. Best connected num: {whole_MCN[top_index]}.")
                self.save(self.dataset, whole_genes[top_index], [whole_PCG[top_index].item(), whole_MCN[top_index].item(), time_list[-1]], time_list, "TDE", bestPCG=best_PCG, bestMCN=best_MCN)
                print(f"Loop {loop} finished. Data saved in {self.path}...")

    def save(self, dataset, gene, best_metric, time_list, method, **kwargs):
        save_path = self.path + dataset + '_crossover_rate_' + str(self.crossover_rate) + '_mutate_rate_' + str(self.mutate_rate) + f'_{method}.txt'
        with open(save_path, 'a+') as f:
            f.write(current_time())
            f.write(f"\nCurrent mode: {self.mode}. Current pop_size: {self.pop_size}\n")
        with open(save_path, 'a+') as f:
            f.write(str([i for i in kwargs['bestPCG']]) + '\n')
        with open(save_path, 'a+') as f:
            f.write(str([i for i in kwargs['bestMCN']]) + '\n')
        with open(save_path, 'a+') as f:
            f.write(str([i.tolist() for i in gene]) + '\n')
        with open(save_path, 'a+') as f:
            f.write(str(time_list) + '\n')
        with open(save_path, 'a+') as f:
            f.write(str(best_metric) + '\n')

    def _calc_rotary_table(self, fit, device, pattern="normal"):
        fit = fit.clone()
        if (max(fit) - min(fit)) != 0:
            if pattern == "trans":
                fit = 1 - fit
            fit = (fit - min(fit)) / (max(fit) - min(fit))
            fit_interval = fit / fit.sum()
        else:
            fit_interval = torch.tensor([1 / self.pop_size for _ in range(self.pop_size)], device=device)
        rotary_table = []
        add = 0
        for i in range(len(fit_interval)):
            add = add + fit_interval[i]
            rotary_table.append(add)
        return rotary_table

    @staticmethod
    def _roulette(rotary_table):
        value = random.random()
        for i, roulette_i in enumerate(rotary_table):
            if roulette_i >= value:
                return i

    def _remove_repeat(self, population: torch.Tensor, embeds):
        for i, pop in enumerate(population):
            clean = pop.unique()
            while len(clean) != len(pop):
                edge = self._generate_node(clean, embeds)
                clean = torch.cat((clean, edge.unsqueeze(0)))
            population[i] = clean
        return population

    def _generate_node(self, edges, embeds):
        edge = embeds[torch.randperm(self.nodes_num, device=edges.device)[0]]
        if edge in edges:
            edge = self._generate_node(edges, embeds)
            return edge
        else:
            return edge


def TDE(mode, max_generation, data_loader, controller: TDEController, evaluator, world_size):
    controller.mode = mode
    evaluator = controller.setup(data_loader=data_loader, evaluator=evaluator)
    if mode == "s" or mode == "sm":
        controller.calculate(max_generation=max_generation, evaluator=evaluator)
    elif mode == "m" or mode == "mnm":
        mp.spawn(controller.mp_calculate, args=(max_generation, deepcopy(evaluator), world_size), nprocs=world_size, join=True)
    else:
        raise ValueError(f"No such mode. Please choose ss, sm, ms or mm.")


