import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import networkx as nx
from igraph import Graph as ig
from copy import deepcopy
from tqdm import tqdm
from time import time
from gapa.framework.body import Body
from gapa.framework.controller import BasicController
from gapa.framework.evaluator import BasicEvaluator
from gapa.utils.functions import CNDTest
from gapa.utils.functions import current_time
from gapa.utils.functions import init_dist
from collections import Counter


class CutoffEvaluator(BasicEvaluator):
    def __init__(self, pop_size, graph, nodes, device):
        super().__init__(
            pop_size=pop_size,
            device=device
        )
        self.graph = ig.from_networkx(graph.copy())
        self.nodes = nodes.clone()
        self.genes_index = None

    def forward(self, population: torch.Tensor):
        copy_nodes = self.nodes.clone().to(population.device)
        fitness_list = torch.zeros(size=(population.shape[0],), device=population.device)
        pop_nodes = copy_nodes[population]
        for i, nodes in enumerate(pop_nodes):
            copy_g: ig = self.graph.copy()
            # for node in nodes:
            copy_g.delete_vertices(nodes.tolist())
            # sub_graph_list = list(nx.connected_components(copy_g))
            # copy_g = ig.from_networkx(copy_g)
            sub_graph_list = list(copy_g.clusters())
            _fitness = 0
            for sub_graph_i in range(len(sub_graph_list)):
                _fitness += len(sub_graph_list[sub_graph_i]) * (len(sub_graph_list[sub_graph_i]) - 1) / 2
            fitness_list[i] = torch.tensor(_fitness, device=population.device)

        return fitness_list


class CutoffController(BasicController):
    def __init__(self, path, pattern, data_loader, loops, crossover_rate, mutate_rate, pop_size, device,
                 cutoff_tag="popGA_cutoff_", fit_side="min"):
        super().__init__(
            path,
            pattern,
        )
        self.cutoff_tag = cutoff_tag
        self.loops = loops
        self.crossover_rate = crossover_rate
        self.mutate_rate = mutate_rate
        self.pop_size = pop_size
        self.device = device
        self.fit_side = fit_side
        self.dataset = data_loader.dataset
        self.budget = data_loader.k
        self.selected_genes_num = data_loader.selected_genes_num
        self.graph = data_loader.G.copy()
        self.nodes = None
        self.nodes_num = None
        self.mode = None

    def setup(self, data_loader, evaluator):
        self.nodes = data_loader.nodes
        self.nodes_num = data_loader.nodes_num
        genes = data_loader.nodes.clone()
        print(f"Cutoff with {self.cutoff_tag}. Original genes: ", len(genes))
        genes = self.genes_cutoff(data_loader.G, genes, evaluator)
        genes_index = torch.tensor([torch.nonzero(data_loader.nodes == node) for node in genes], device=self.device).squeeze()
        print(f"Cutoff finished. Genes after cutoff: ", len(genes))
        evaluator.nodes = genes
        evaluator.genes_index = genes_index
        self.nodes = genes
        self.nodes_num = len(genes)
        return evaluator

    def calculate(self, max_generation, evaluator):
        best_PCG = []
        best_MCN = []
        best_genes = []
        time_list = []
        genes_index = evaluator.genes_index
        body = Body(self.nodes_num, self.budget, self.pop_size, self.fit_side, evaluator.device)
        for loop in range(self.loops):
            start = time()
            ONE, population = body.init_population()
            if self.mode == "sm":
                evaluator = torch.nn.DataParallel(evaluator)
            fitness_list = evaluator(population)
            best_fitness_list = torch.tensor(data=[], device=self.device)
            best_fitness_list = torch.hstack((best_fitness_list, torch.min(fitness_list)))
            with tqdm(total=max_generation) as pbar:
                pbar.set_description(f'Training....{self.dataset} in Loop: {loop}...')
                for generation in range(max_generation):
                    try:
                        rotary_table = self._calc_rotary_table(fitness_list, self.device)
                        new_population1 = population[[self._roulette(rotary_table, self.device) for _ in range(self.pop_size)]]
                        new_population2 = population[[self._roulette(rotary_table, self.device) for _ in range(self.pop_size)]]
                    except:
                        new_population1 = population.clone()
                        new_population2 = body.selection(population, fitness_list)
                    crossover_population = body.crossover(new_population1, new_population2, self.crossover_rate, ONE)
                    mutation_population = body.mutation(crossover_population, self.mutate_rate, ONE)
                    mutation_population = self._remove_repeat(mutation_population)
                    new_fitness_list = evaluator(mutation_population)
                    population, fitness_list, best_fitness_list = body.elitism(population, mutation_population, fitness_list, new_fitness_list, best_fitness_list)
                    if generation % 50 == 0 or (generation+1) == max_generation:
                        # population_copy = self._remove_repeat(population.clone())
                        nodes_index = genes_index[population[torch.argsort(fitness_list.clone())[0]]]
                        critical_nodes = self.nodes[population[torch.argsort(fitness_list.clone())[0]]]
                        best_MCN.append(CNDTest(self.graph, nodes_index, pattern='ccn'))
                        best_PCG.append(best_fitness_list[-1].item())
                        best_genes.append(critical_nodes)
                        end = time()
                        time_list.append(end - start)
                    pbar.set_postfix(MCN=min(best_MCN), PCG=min(best_PCG), fitness=min(fitness_list).item())
                    pbar.update(1)
            top_index = best_PCG.index(min(best_PCG))
            print(f"Best PC(G): {best_PCG[top_index]}. Best connected num: {best_MCN[top_index]}.")
            self.save(self.dataset, best_genes[top_index], [best_PCG[top_index], best_MCN[top_index], time_list[-1]], time_list, "CutOff", bestPCG=best_PCG, bestMCN=best_MCN)
            print(f"Loop {loop} finished. Data saved in {self.path}...")

    def mp_calculate(self, rank, max_generation, evaluator, world_size):
        device = init_dist(rank, world_size)
        best_PCG = []
        best_MCN = []
        best_genes = []
        time_list = []
        genes_index = evaluator.genes_index.to(device)
        nodes = self.nodes.clone().to(device)
        component_size = self.pop_size // world_size
        body = Body(self.nodes_num, self.budget, component_size, self.fit_side, device)
        for loop in range(self.loops):
            start = time()
            ONE, component_population = body.init_population()
            if self.mode == "mnm":
                evaluator = torch.nn.DataParallel(evaluator)
            component_fitness_list = evaluator(component_population).to(device)
            best_fitness_list = torch.tensor(data=[], device=device)
            best_fitness_list = torch.hstack((best_fitness_list, min(component_fitness_list)))
            if rank == 0:
                population = [torch.zeros((component_size, self.budget), dtype=torch.int64, device=device) for _ in range(world_size)]
                fitness_list = [torch.empty((component_size,), dtype=torch.float32, device=device) for _ in range(world_size)]
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
                        try:
                            rotary_table = self._calc_rotary_table(fitness_list, self.device)
                            new_population1 = population[[self._roulette(rotary_table, self.device) for _ in range(self.pop_size)]]
                            new_population2 = population[[self._roulette(rotary_table, self.device) for _ in range(self.pop_size)]]
                        except:
                            new_population1 = population.clone()
                            new_population2 = body.selection(population, fitness_list)
                        body.pop_size = self.pop_size
                        crossover_ONE = torch.ones((self.pop_size, self.budget), dtype=torch.int64, device=device)
                        crossover_population = body.crossover(new_population1, new_population2, self.crossover_rate, crossover_ONE)
                        body.pop_size = component_size
                    if rank == 0:
                        crossover_population = torch.stack(torch.split(crossover_population, component_size))
                    else:
                        crossover_population = torch.stack([torch.zeros((component_size, self.budget), dtype=torch.int64, device=device) for _ in range(world_size)])
                    dist.broadcast(crossover_population, src=0)
                    mutation_population = body.mutation(crossover_population[rank], self.mutate_rate, ONE)
                    mutation_population = self._remove_repeat(mutation_population)
                    new_component_fitness_list = evaluator(mutation_population).to(device)

                    if rank == 0:
                        elitism_population = [torch.zeros((component_size, self.budget), dtype=component_population.dtype, device=device) for _ in range(world_size)]
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
                        top_index = torch.argsort(fitness_list)[:component_size]
                        component_population = population[top_index]
                        component_fitness_list = fitness_list[top_index]
                        body.pop_size = component_size
                    else:
                        component_population = torch.zeros((component_size, self.budget), dtype=torch.int64, device=device)
                        component_fitness_list = torch.empty((component_size,), dtype=torch.float32, device=device)
                    dist.broadcast(component_population, src=0)
                    dist.broadcast(component_fitness_list, src=0)
                    if generation % 50 == 0 or (generation+1) == max_generation:
                        # component_population = self._remove_repeat(component_population.clone())
                        nodes_index = genes_index[component_population[torch.argsort(component_fitness_list.clone())[0]]]
                        critical_nodes = nodes[component_population[torch.argsort(component_fitness_list.clone())[0]]]
                        best_MCN.append(CNDTest(self.graph, nodes_index, pattern='ccn'))
                        best_PCG.append(best_fitness_list[-1].item())
                        best_genes.append(critical_nodes)
                        end = time()
                        time_list.append(end - start)
                    pbar.set_postfix(MCN=min(best_MCN), PCG=min(best_PCG), fitness=min(component_fitness_list).item())
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
                print(f"Best PC(G): {best_PCG[top_index]}. Best connected num: {whole_MCN[top_index]}.")
                self.save(self.dataset, whole_genes[top_index], [whole_PCG[top_index].item(), whole_MCN[top_index].item(), time_list[-1]], time_list, "CutOff", bestPCG=best_PCG, bestMCN=best_MCN)
                print(f"Loop {loop} finished. Data saved in {self.path}...")
        torch.cuda.empty_cache()
        dist.destroy_process_group()
        torch.cuda.synchronize()

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

    def _calc_rotary_table(self, fit, device):
        if (max(fit) - min(fit)) != 0:
            fit = 1 - (fit - min(fit)) / (max(fit) - min(fit))
            fit_interval = fit / fit.sum()
        else:
            fit_interval = torch.tensor([1 / self.pop_size for _i in range(self.pop_size)], device=device)
        rotary_table = []
        add = 0
        for i in range(len(fit_interval)):
            add = add + fit_interval[i]
            rotary_table.append(add)
        return rotary_table

    def _roulette(self, rotary_table, device):
        value = torch.rand(1, device=device).squeeze()
        for i, roulette_i in enumerate(rotary_table):
            if roulette_i >= value:
                return i

    def genes_cutoff(self, graph, genes, evaluator):
        if self.cutoff_tag == "no_cutoff_":
            pass
        if self.cutoff_tag == "random_cutoff_":
            genes = self._random_cutoff(genes)
        if self.cutoff_tag == "greedy_cutoff_":
            genes = self._greedy_cutoff(graph, genes)
        if self.cutoff_tag == "popGreedy_cutoff_":
            genes = self._pop_greedy_cutoff(graph, genes, 10)
        if self.cutoff_tag == "popGA_cutoff_":
            genes = self._local_optimal_cutoff(graph, 10, evaluator)
        return genes

    def _random_cutoff(self, genes):
        genes = genes[torch.randperm(len(genes), device=self.device)[:self.selected_genes_num]]
        return genes

    def _greedy_cutoff(self, graph, genes):
        copy_G = graph.copy()
        genes = genes.tolist()
        while len(genes) > self.selected_genes_num:
            degree = nx.degree_centrality(copy_G)
            genes.remove(min(degree, key=degree.get))
            copy_G.remove_node(min(degree, key=degree.get))
        return torch.tensor(genes, device=self.device)

    def _pop_greedy_cutoff(self, graph, genes, pop_num):
        greedy_indi = []
        body = Body(self.nodes_num, self.budget, self.pop_size, self.fit_side, self.device)
        for _i in range(pop_num):
            temp_fitness = []
            _, temp_population = body.init_population()
            temp_population = self._remove_repeat(temp_population)
            for j in temp_population:
                temp_fitness.append(CNDTest(graph, genes[j]))
            top_index = torch.tensor(temp_fitness, device=self.device).argsort()[:1]
            greedy_indi.append(temp_population[top_index])

        genes = []
        for pop in greedy_indi:
            genes = genes + pop[0].tolist()
        data = dict(Counter(genes))
        data = sorted(data.items(), key=lambda x: x[1])[::-1][:int(self.selected_genes_num / 2)]
        genes = [data[i][0] for i in range(len(data))]
        # genes = list(set(genes + list(history[0])))
        genes = self.nodes[genes].tolist()
        # 在基因池中添加度值最高的节点
        degree = nx.degree_centrality(graph)
        degree = sorted(degree.items(), key=lambda x: x[1])[::-1]
        degree = [degree[i][0] for i in range(len(degree))]
        count_i = 0
        while len(genes) < self.selected_genes_num:
            if degree[count_i] not in genes:
                genes.append(degree[count_i])
            count_i = count_i + 1
        return torch.tensor(genes, device=self.device)

    def _local_optimal_cutoff(self, graph, pop_num, evaluator):
        greedy_indi = []
        body = Body(self.nodes_num, self.budget, self.pop_size, self.fit_side, self.device)
        with tqdm(total=pop_num * 100) as cbar:
            for _i in range(pop_num):
                cbar.set_description(f"Genetic cutoff in {_i}")
                ONE, temp_population = body.init_population()
                temp_population = self._remove_repeat(temp_population)
                temp_fitness = evaluator(temp_population)
                best_fitness_list = torch.tensor(data=[], device=self.device)
                best_fitness_list = torch.hstack((best_fitness_list, torch.min(temp_fitness)))
                for _generation in range(100):
                    rotary_table = self._calc_rotary_table(temp_fitness, self.device)
                    new_population1 = temp_population[self._roulette(rotary_table, self.device)].clone()
                    new_population2 = temp_population[self._roulette(rotary_table, self.device)].clone()

                    crossover_population = body.crossover(new_population1, new_population2, self.crossover_rate, ONE)
                    mutation_population = body.mutation(crossover_population, self.mutate_rate, ONE)
                    mutation_population = self._remove_repeat(mutation_population)
                    new_fitness_list = evaluator(mutation_population)
                    temp_population, temp_fitness, best_fitness_list = body.elitism(temp_population, mutation_population, temp_fitness, new_fitness_list, best_fitness_list)
                    cbar.update(1)
                top_index = temp_fitness.argsort()[0:1]
                greedy_indi.append(temp_population[top_index])

        genes = []
        for pop in greedy_indi:
            genes = genes + pop[0].tolist()
        data = dict(Counter(genes))
        data = sorted(data.items(), key=lambda x: x[1])[::-1][:int(self.selected_genes_num / 2)]
        genes = [data[i][0] for i in range(len(data))]
        # genes = list(set(genes + list(history[0])))
        genes = self.nodes[genes].tolist()
        # 在基因池中添加度值最高的节点
        degree = nx.degree_centrality(graph)
        degree = sorted(degree.items(), key=lambda x: x[1])[::-1]
        degree = [degree[i][0] for i in range(len(degree))]
        count_i = 0
        while len(genes) < self.selected_genes_num:
            if degree[count_i] not in genes:
                genes.append(degree[count_i])
            count_i = count_i + 1
        return torch.tensor(genes, device=self.device)

    def _remove_repeat(self, population: torch.Tensor):
        for i, pop in enumerate(population):
            clean = pop.unique()
            while len(clean) != len(pop):
                node = self._generate_node(clean)
                clean = torch.cat((clean, node.unsqueeze(0)))
            population[i] = clean
        return population

    def _generate_node(self, nodes):
        node = torch.randperm(self.nodes_num, device=nodes.device)[0]
        if node in nodes:
            node = self._generate_node(nodes)
            return node
        else:
            return node


def Cutoff(mode, max_generation, data_loader, controller: CutoffController, evaluator, world_size):
    controller.mode = mode
    evaluator = controller.setup(data_loader=data_loader, evaluator=evaluator)
    if mode == "s" or mode == "sm":
        controller.calculate(max_generation=max_generation, evaluator=evaluator)
    elif mode == "m" or mode == "mnm":
        mp.spawn(controller.mp_calculate, args=(max_generation, deepcopy(evaluator), world_size), nprocs=world_size, join=True)
    else:
        raise ValueError(f"No such mode. Please choose ss, sm, ms or mm.")


