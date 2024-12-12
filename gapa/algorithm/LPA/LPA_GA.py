import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import networkx as nx
from copy import deepcopy
from itertools import combinations
from tqdm import tqdm
from time import time
from gapa.framework.body import Body
from gapa.framework.controller import BasicController
from gapa.framework.evaluator import BasicEvaluator
from gapa.utils.functions import current_time
from gapa.utils.functions import init_dist


def igraph_to_nx_mapping(edges):
    mapping = {}
    next_index = 0
    for edge in edges:
        for node in edge:
            if node not in mapping:
                mapping[node] = next_index
                next_index += 1

    return mapping, list(mapping.keys())


class GAEvaluator(BasicEvaluator):
    def __init__(self, pop_size, graph, ratio, device):
        super().__init__(
            pop_size=pop_size,
            device=device
        )
        self.graph = graph.copy()
        self.budget = None
        self.ratio = ratio
        self.nodes_num = None
        self.train_G = None
        self.len_non_exist_edges = None
        self.neg_fitness = None
        self.train_edges_set = None
        self.test_edges = None
        self.test_edges_set = None
        self.init_proximity_dic = None
        self.decedent_order = None
        self.train_G_neighbor = None

    def forward(self, population):
        device = population.device
        fitness_list = torch.zeros(size=(len(population),), device=device)
        for i, pop in enumerate(population):
            fitness = self.increment_link_prediction(pop, self.neg_fitness)
            fitness_list[i] = fitness
        return torch.exp(-fitness_list)

    def increment_link_prediction(self, pop, neg_fitness):
        tensor_del_edges = pop[:self.budget]
        tensor_add_edges = pop[self.budget:2*self.budget]
        del_edges = {tuple(each) for each in tensor_del_edges.tolist()}
        add_edges = {tuple(each) for each in tensor_add_edges.tolist()}

        recalc = set()
        init_edges = []
        init_edges_set = []
        testList = set()
        for edge in del_edges:
            if edge not in init_edges:
                tL, recalc = self._perturb_recalc_delete(edge, recalc)
                init_edges.append(edge)
                init_edges_set.append(tL)
                testList |= tL
            else:
                index = init_edges.index(edge)
                testList |= init_edges_set[index]

        for edge in add_edges:
            if edge not in init_edges:
                tL, recalc = self._perturb_recalc_add(edge, recalc)
                init_edges.append(edge)
                init_edges_set.append(tL)
                testList |= tL
            else:
                index = init_edges.index(edge)
                testList |= init_edges_set[index]

        perturb_G = self.train_G.copy()
        perturb_G.remove_edges_from(del_edges)
        perturb_G.add_edges_from(add_edges)
        test = testList | del_edges | self.test_edges_set

        perturb_G_nodes = list(perturb_G.nodes())
        if len(perturb_G_nodes) < self.nodes_num:
            for each in test:
                if each[0] not in perturb_G_nodes:
                    perturb_G.add_node(each[0])
                if each[1] not in perturb_G_nodes:
                    perturb_G.add_node(each[1])

        pred_dic = self._RA(perturb_G, test)
        fitness = self._test_fuc(del_edges, add_edges, pred_dic, neg_fitness, testList)
        return fitness

    def _test_fuc(self, del_edges, add_edges, pred_dic, neg_fitness, testList):
        max_test = 0
        max_count = 0
        pos_fitness = 0
        for each in self.test_edges:
            pos_fitness += pred_dic[each]
            if pred_dic[each] > max_test:
                max_test = pred_dic[each]
        pos_fitness /= len(self.test_edges)
        for each in testList:  # 需要重新计算相似度的不存在的连边/the non-existent links which need to recalculate the similarity index
            neg_fitness -= self.init_proximity_dic[each]
            neg_fitness += pred_dic[each]
            if pred_dic[each] >= max_test:
                max_count += 1
        for each in del_edges:
            neg_fitness += pred_dic[each]
            if pred_dic[each] >= max_test:
                max_count += 1
        for each in add_edges:
            neg_fitness -= self.init_proximity_dic[each]
        neg_fitness /= self.len_non_exist_edges
        temp = testList | add_edges
        for each in self.decedent_order:
            if each[1] > max_test:
                if each[0] not in temp:
                    max_count += 1
            else:
                break
        fitness = pos_fitness - neg_fitness
        fitness -= self.ratio * max_count
        return fitness

    def _RA(self, graph: nx.Graph, edges):
        RA_dic = dict()
        neighbors_num = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes())).sum(1)
        for each in edges:
            # common = nx.common_neighbors(graph, each[0], each[1])
            common = set(graph.neighbors(each[0])) & set(graph.neighbors(each[1]))
            a = sum(1.0 / neighbors_num[w] for w in common)
            RA_dic[each] = a
        return RA_dic

    def _perturb_recalc_add(self, edge, recalc):
        testList = set()
        # neighborhood_x = sorted(self.train_G.neighbors(self.train_nx_to_igraph_mapping_dic[edge[0]]))
        # neighborhood_y = sorted(self.train_G.neighbors(self.train_nx_to_igraph_mapping_dic[edge[1]]))
        neighborhood_x = self.train_G_neighbor[edge[0]]
        neighborhood_y = self.train_G_neighbor[edge[1]]
        for each in list(combinations(neighborhood_x, r=2))[:10]:
            # mid = (self.train_igraph_to_nx_mapping_list[each[0]], self.train_igraph_to_nx_mapping_list[each[1]])
            if each not in self.train_edges_set:
                if each not in recalc:
                    testList.add(each)
                    recalc.add(each)
        for each in list(combinations(neighborhood_y, r=2))[:10]:
            # mid = (self.train_igraph_to_nx_mapping_list[each[0]], self.train_igraph_to_nx_mapping_list[each[1]])
            if each not in self.train_edges_set:
                if each not in recalc:
                    testList.add(each)
                    recalc.add(each)

        for each in neighborhood_x:
            min_node = min(each, edge[1])
            max_node = max(each, edge[1])
            # mid = (self.train_igraph_to_nx_mapping_list[min_node], self.train_igraph_to_nx_mapping_list[max_node])
            if (min_node, max_node) not in self.train_edges_set:
                if (min_node, max_node) not in recalc:
                    testList.add((min_node, max_node))
                    recalc.add((min_node, max_node))

        for each in neighborhood_y:
            min_node = min(each, edge[1])
            max_node = max(each, edge[1])
            # mid = (self.train_igraph_to_nx_mapping_list[min_node], self.train_igraph_to_nx_mapping_list[max_node])
            if (min_node, max_node) not in self.train_edges_set:
                if (min_node, max_node) not in recalc:
                    testList.add((min_node, max_node))
                    recalc.add((min_node, max_node))
        return testList, recalc

    def _perturb_recalc_delete(self, edge, recalc):
        testList = set()
        # neighborhood_x = sorted(self.train_G.neighbors(self.train_nx_to_igraph_mapping_dic[edge[0]]))
        # neighborhood_y = sorted(self.train_G.neighbors(self.train_nx_to_igraph_mapping_dic[edge[1]]))
        neighborhood_x = self.train_G_neighbor[edge[0]]
        neighborhood_y = self.train_G_neighbor[edge[1]]
        for each in list(combinations(neighborhood_x, r=2))[:50]:
            # mid = (self.train_igraph_to_nx_mapping_list[each[0]], self.train_igraph_to_nx_mapping_list[each[1]])
            if each not in self.train_edges_set:
                if each not in recalc:
                    testList.add(each)
                    recalc.add(each)
        for each in list(combinations(neighborhood_y, r=2))[:50]:
            # mid = (self.train_igraph_to_nx_mapping_list[each[0]], self.train_igraph_to_nx_mapping_list[each[1]])
            if each not in self.train_edges_set:
                if each not in recalc:
                    testList.add(each)
                    recalc.add(each)

        return testList, recalc


class GABody(Body):
    def __init__(self, critical_num, budget, pop_size, train_edges_index, non_exist_edges_index, all_edges: torch.Tensor, nodes_num, fit_side, device):
        super().__init__(
            critical_num=critical_num,
            budget=budget,
            pop_size=pop_size,
            fit_side=fit_side,
            device=device
        )
        self.nodes_num = nodes_num
        self.all_edges = all_edges
        self.train_edges = all_edges[train_edges_index]
        self.non_exist_edges = all_edges[non_exist_edges_index]
        self.len_train_edges = len(train_edges_index)
        self.train_edges_index: torch.Tensor = train_edges_index
        self.len_non_exist_edges = len(non_exist_edges_index)
        self.non_exist_edges_index: torch.Tensor = non_exist_edges_index

    def init_population_rewrite(self):
        population = torch.tensor([], dtype=torch.int, device=self.device)
        for i in range(self.pop_size):
            del_edges = self.train_edges_index[torch.randperm(len(self.train_edges_index), device=self.device)[:self.budget]]
            add_edges = self.non_exist_edges_index[torch.randperm(len(self.non_exist_edges_index), device=self.device)[:self.budget]]
            population = torch.cat((population, self._one_pop(del_edges, add_edges).unsqueeze(dim=0)))
        one = torch.ones(size=(self.pop_size, self.budget), device=self.device)
        return one, population

    def _one_pop(self, del_edges, add_edges):
        # mask = ~torch.any(torch.all(self.train_edges_index[:, None] == del_edges, dim=-1), dim=-1)
        mask = ~torch.isin(self.train_edges_index, del_edges)
        kk = self.train_edges_index[mask]
        edges = torch.cat((self.train_edges_index[mask], add_edges))
        return torch.cat((del_edges, add_edges, edges))

    def _one_pop_old(self, del_edges, add_edges):
        mask = ~torch.any(torch.all(self.train_edges[:, None] == del_edges, dim=-1), dim=-1)
        edges = torch.unique(torch.vstack((self.train_edges[mask], add_edges)), dim=0)
        return torch.vstack((del_edges, add_edges, edges))

    @staticmethod
    def selection_rewrite(population, fitness_list, sample_num, replacement=True):
        normalize_fit = fitness_list / fitness_list.sum()
        normalize_fit[normalize_fit < 0] = 0
        samples = torch.multinomial(normalize_fit, sample_num, replacement=replacement)
        return population[samples]

    def eda(self, population, fitness_list, num_eda_pop):
        selected_eda_pop: torch.Tensor = self.selection_rewrite(population, fitness_list, 3*num_eda_pop)
        del_edges = selected_eda_pop[:, :self.budget]
        del_edges_index, del_edges_count = del_edges.unique(return_counts=True)
        add_edges = selected_eda_pop[:, self.budget:2*self.budget]
        add_edges_index, add_edges_count = add_edges.unique(return_counts=True)

        normalize_del = del_edges_count / del_edges_count.sum()
        normalize_add = del_edges_count / del_edges_count.sum()

        new_population = torch.tensor([], dtype=torch.int, device=self.device)
        for i in range(num_eda_pop):
            samples_del = torch.multinomial(normalize_del, self.budget, replacement=False)
            eda_del_edges = del_edges_index[samples_del]
            samples_add = torch.multinomial(normalize_add, self.budget, replacement=False)
            eda_add_edges = add_edges_index[samples_add]
            new_population = torch.cat((new_population, self._one_pop(eda_del_edges, eda_add_edges).unsqueeze(dim=0)))
        return new_population

    def del_mutation(self, population, mutate_rate, one):
        mutation_matrix = torch.tensor(np.random.choice([0, 1], size=(self.pop_size, self.budget), p=[1 - mutate_rate, mutate_rate]), device=self.device)
        mutation_population = population * (one - mutation_matrix) + torch.randint(0, self.len_train_edges, size=(self.pop_size, self.budget), device=self.device) * mutation_matrix
        return mutation_population.int()

    def add_mutation(self, population, mutate_rate, one):
        mutation_matrix = torch.tensor(np.random.choice([0, 1], size=(self.pop_size, self.budget), p=[1 - mutate_rate, mutate_rate]), device=self.device)
        mutation_population = population * (one - mutation_matrix) + torch.randint(self.len_train_edges, self.len_non_exist_edges, size=(self.pop_size, self.budget), device=self.device) * mutation_matrix
        return mutation_population.int()


class GAController(BasicController):
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
        self.budget = None
        self.selected_genes_num = data_loader.selected_genes_num
        self.graph = data_loader.G
        self.min_node = None
        self.nodes = None
        self.nodes_num = None
        self.mode = None
        self.train_G = None
        self.test_G = None
        self.ori_G_edges = None
        self.train_edges = torch.tensor([])
        self.train_edges_set = None
        self.test_edges = torch.tensor([])
        self.test_edges_set = None
        self.complete_edges = torch.tensor([])
        self.non_exist_edges = None
        self.non_exist_edges_set = None
        self.init_recalc = {}
        self.neg_fitness = None
        self.train_edges_index = None
        self.train_index_len = None
        self.non_exist_edges_index = None
        self.non_exist_index_len = None
        self.all_edges = None

    def setup(self, data_loader, evaluator: GAEvaluator):
        ori_G_edges = np.array(data_loader.G.edges())
        np.random.shuffle(ori_G_edges)
        train_edges = ori_G_edges[:int(len(ori_G_edges) * 0.8)]
        test_edges = ori_G_edges[int(len(ori_G_edges) * 0.8):]
        self.train_G = nx.from_edgelist(train_edges)
        self.test_G = nx.from_edgelist(test_edges)

        self.min_node = min(list(data_loader.G.nodes()))
        self.nodes_num = len(data_loader.G.nodes())
        self.train_G.add_nodes_from(range(self.min_node, self.nodes_num+self.min_node))
        self.ori_G_edges = torch.tensor([(min(each[0], each[1]), max(each[0], each[1])) for each in data_loader.G.edges()], dtype=torch.int, device=self.device)
        self.train_edges = torch.tensor([(min(each[0], each[1]), max(each[0], each[1])) for each in self.train_G.edges()], dtype=torch.int, device=self.device)
        self.train_index_len = len(self.train_edges)
        self.train_edges_index = torch.arange(len(self.train_edges), dtype=torch.int, device=self.device)
        self.train_edges_set = torch.unique(self.train_edges, dim=0)
        self.test_edges = torch.tensor([(min(each[0], each[1]), max(each[0], each[1])) for each in self.test_G.edges()], dtype=torch.int, device=self.device)
        self.test_edges_set = torch.unique(self.test_edges, dim=0)
        self.complete_edges = torch.tensor(list(combinations(sorted(data_loader.G.nodes()), 2)), dtype=torch.int, device=self.device)
        mask = ~torch.any(torch.all(self.complete_edges[:, None] == self.ori_G_edges, dim=-1), dim=-1)
        self.non_exist_edges = self.complete_edges[mask]
        self.non_exist_index_len = len(self.non_exist_edges)
        self.non_exist_edges_index = torch.arange(len(self.train_edges), len(self.non_exist_edges), dtype=torch.int, device=self.device)
        self.non_exist_edges_set = torch.unique(self.non_exist_edges, dim=0)
        self.all_edges = torch.cat((self.train_edges, self.non_exist_edges))
        perturb = self.train_edges_set.clone()
        ori_results = self.calc_pre_and_auc(perturb, device=self.device)
        print(f"Pre: {ori_results[0]}. AUC: {ori_results[1]}")
        init_proximity_generator = self._calc_similarity(perturb, self.complete_edges)
        init_proximity_adj = -torch.abs(torch.ones(size=(self.nodes_num, self.nodes_num), device=self.device))
        num_index = 0
        init_proximity_dic = {}
        for s, t, v in init_proximity_generator:
            init_proximity_adj[s, t] = v
            init_proximity_dic[(s, t)] = v
            num_index += 1
        _, indices = torch.topk(init_proximity_adj.flatten(), num_index)
        decedent_order = torch.vstack((indices // self.nodes_num, indices % self.nodes_num)).T
        self.neg_fitness = init_proximity_adj[self.non_exist_edges[:, 0], self.non_exist_edges[:, 1]].sum().item()

        self.budget = int(data_loader.k * len(train_edges))
        evaluator.budget = self.budget
        evaluator.train_G = self.train_G.copy()
        evaluator.train_G_neighbor = [sorted(set(self.train_G.neighbors(n))) for n in sorted(self.train_G.nodes())]

        evaluator.init_proximity_dic = init_proximity_dic
        evaluator.nodes_num = self.nodes_num
        evaluator.len_non_exist_edges = len(self.non_exist_edges)
        evaluator.neg_fitness = self.neg_fitness
        evaluator.train_edges_set = {tuple(each) for each in self.train_edges_set.tolist()}
        evaluator.test_edges = {tuple(each) for each in self.test_edges.tolist()}
        evaluator.test_edges_set = {tuple(each) for each in self.test_edges_set.tolist()}
        evaluator.decedent_order = {tuple(each) for each in decedent_order.tolist()}

        return evaluator

    def calculate(self, max_generation, evaluator):
        best_Pre = []
        best_AUC = []
        best_genes = []
        time_list = []
        body = GABody(self.nodes_num, self.budget, self.pop_size, self.train_edges_index, self.non_exist_edges_index, self.all_edges, self.nodes_num, self.fit_side, self.device)
        for loop in range(self.loops):
            start = time()
            ONE, population = body.init_population_rewrite()
            crossover_one = torch.hstack((ONE, ONE))
            if self.mode == "sm":
                evaluator = torch.nn.DataParallel(evaluator)
            fitness_list = evaluator(self.all_edges[population])
            best_fitness_list = torch.tensor(data=[], device=self.device)
            best_fitness_list = torch.hstack((best_fitness_list, torch.max(fitness_list)))
            with tqdm(total=max_generation) as pbar:
                pbar.set_description(f'Training....{self.dataset} in Loop: {loop}...')
                for generation in range(max_generation):
                    new_population_1 = population.clone()
                    new_population_2 = body.selection(population, fitness_list)

                    new_del_add_population_1 = new_population_1[:, :2*self.budget]
                    new_del_add_population_2 = new_population_2[:, :2*self.budget]
                    body.budget = 2 * self.budget
                    crossover_population = body.crossover(new_del_add_population_1, new_del_add_population_2, self.crossover_rate, crossover_one)
                    body.budget = self.budget
                    del_population = crossover_population[:, :self.budget]
                    del_mutation_population = body.del_mutation(del_population, self.mutate_rate, ONE)
                    add_population = crossover_population[:, self.budget:]
                    add_mutation_population = body.add_mutation(add_population, self.mutate_rate, ONE)

                    del_pop = self._remove_repeat(del_mutation_population, self.train_index_len)
                    add_pop = self._remove_repeat(add_mutation_population, self.non_exist_index_len, pattern="add", add=self.train_index_len)

                    mutation_population = torch.tensor([], dtype=torch.int, device=self.device)
                    for i in range(self.pop_size):
                        mutation_population = torch.cat((mutation_population, body._one_pop(del_pop[i], add_pop[i]).unsqueeze(dim=0)))

                    new_fitness_list = evaluator(self.all_edges[mutation_population])
                    population, fitness_list, best_fitness_list = body.elitism(population, mutation_population, fitness_list, new_fitness_list, best_fitness_list)
                    if generation % 10 == 0 or (generation+1) == max_generation:
                        genes = population[torch.argsort(fitness_list.clone(), descending=True)[0]]
                        perturb_edges = genes[2*self.budget:]
                        pre, auc = self.calc_pre_and_auc(self.all_edges[perturb_edges], device=self.device)
                        best_Pre.append(pre)
                        best_AUC.append(auc)
                        best_genes.append(genes)
                        end = time()
                        time_list.append(end - start)
                    pbar.set_postfix(fitness=max(fitness_list).item(), Pre=min(best_Pre), AUC=min(best_AUC))
                    pbar.update(1)

            top_index = best_AUC.index(min(best_AUC))
            print(f"Best Pre: {best_Pre[top_index]}. Best AUC: {best_AUC[top_index]}.")
            self.save(self.dataset, self.all_edges[best_genes[top_index]], [best_Pre[top_index], best_AUC[top_index], time_list[-1]], time_list, "EDA", bestPre=best_Pre, bestAUC=best_AUC)
            print(f"Loop {loop} finished. Data saved in {self.path}...")

    def mp_calculate(self, rank, max_generation, evaluator, world_size):
        device = init_dist(rank, world_size)
        best_Pre = []
        best_AUC = []
        best_genes = []
        time_list = []
        train_edges_index = self.train_edges_index.to(device)
        non_exist_edges_index = self.non_exist_edges_index.to(device)
        all_edges = self.all_edges.to(device)
        component_size = self.pop_size // world_size
        body = GABody(self.nodes_num, self.budget, component_size, train_edges_index, non_exist_edges_index, all_edges, self.nodes_num, self.fit_side, device)
        for loop in range(self.loops):
            start = time()
            ONE, component_population = body.init_population_rewrite()
            if self.mode == "mnm":
                evaluator = torch.nn.DataParallel(evaluator)
            component_fitness_list = evaluator(all_edges[component_population]).to(device)
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
                        new_population_1 = population.clone()
                        new_population_2 = body.selection(population, fitness_list)
                        body.pop_size = self.pop_size
                        body.budget = 2 * self.budget
                        new_del_add_population_1 = new_population_1[:, :2 * self.budget]
                        new_del_add_population_2 = new_population_2[:, :2 * self.budget]
                        crossover_one = torch.ones((self.pop_size, 2*self.budget), dtype=component_population.dtype, device=device)
                        crossover_population = body.crossover(new_del_add_population_1, new_del_add_population_2, self.crossover_rate, crossover_one).int()
                        body.pop_size = component_size
                        body.budget = self.budget
                    if rank == 0:
                        crossover_population = torch.stack(torch.split(crossover_population, component_size))
                    else:
                        crossover_population = torch.stack([torch.zeros((component_size, 2 * self.budget), dtype=component_population.dtype, device=device) for _ in range(world_size)])

                    dist.broadcast(crossover_population, src=0)
                    del_population = crossover_population[rank][:, :self.budget]
                    del_mutation_population = body.del_mutation(del_population, self.mutate_rate, ONE)
                    add_population = crossover_population[rank][:, self.budget:]
                    add_mutation_population = body.add_mutation(add_population, self.mutate_rate, ONE)

                    del_pop = self._remove_repeat(del_mutation_population, self.train_index_len)
                    add_pop = self._remove_repeat(add_mutation_population, self.non_exist_index_len, pattern="add", add=self.train_index_len)
                    component_mutation_population = torch.tensor([], dtype=torch.int, device=device)
                    for i in range(component_size):
                        component_mutation_population = torch.cat((component_mutation_population, body._one_pop(del_pop[i], add_pop[i]).unsqueeze(dim=0)))

                    new_component_fitness_list = evaluator(all_edges[component_mutation_population]).to(device)

                    if rank == 0:
                        elitism_population = [torch.zeros(component_population.shape, dtype=component_population.dtype, device=device) for _ in range(world_size)]
                        elitism_fitness_list = [torch.empty((component_size,), dtype=new_component_fitness_list.dtype, device=device) for _ in range(world_size)]
                    else:
                        elitism_population = None
                        elitism_fitness_list = None
                    dist.gather(component_mutation_population, elitism_population, dst=0)
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

                    if generation % 10 == 0 or (generation + 1) == max_generation:
                        genes = component_population[torch.argsort(component_fitness_list.clone(), descending=True)[0]]
                        perturb_edges = genes[2 * self.budget:]
                        pre, auc = self.calc_pre_and_auc(all_edges[perturb_edges], device=device)
                        best_Pre.append(pre)
                        best_AUC.append(auc)
                        best_genes.append(genes)
                        end = time()
                        time_list.append(end - start)
                    pbar.set_postfix(fitness=max(component_fitness_list).item(), Pre=min(best_Pre), AUC=min(best_AUC))
                    pbar.update(1)
            best_genes = torch.stack(best_genes)
            best_Pre = torch.tensor(best_Pre, device=device)
            best_AUC = torch.tensor(best_AUC, device=device)
            if rank == 0:
                whole_genes = [torch.zeros(best_genes.shape, dtype=best_genes.dtype, device=device) for _ in range(world_size)]
                whole_Pre = [torch.empty(best_Pre.shape, device=device) for _ in range(world_size)]
                whole_AUC = [torch.empty(best_AUC.shape, device=device) for _ in range(world_size)]
            else:
                whole_genes = None
                whole_Pre = None
                whole_AUC = None
            dist.barrier()
            dist.gather(best_genes, whole_genes, dst=0)
            dist.gather(best_Pre, whole_Pre, dst=0)
            dist.gather(best_AUC, whole_AUC, dst=0)
            if rank == 0:
                whole_genes = torch.cat(whole_genes)
                whole_Pre = torch.cat(whole_Pre)
                whole_AUC = torch.cat(whole_AUC)
                top_index = torch.argsort(whole_Pre)[0]
                print(f"Best Pre: {whole_Pre[top_index]}. Best AUC: {whole_AUC[top_index]}.")
                self.save(self.dataset, all_edges[whole_genes[top_index]], [whole_Pre[top_index].item(), whole_AUC[top_index].item(), time_list[-1]], time_list, "EDA", bestPre=best_Pre, bestAUC=best_AUC)
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
            f.write(str([i for i in kwargs['bestPre']]) + '\n')
        with open(save_path, 'a+') as f:
            f.write(str([i for i in kwargs['bestAUC']]) + '\n')
        with open(save_path, 'a+') as f:
            f.write(str([i.tolist() for i in gene]) + '\n')
        with open(save_path, 'a+') as f:
            f.write(str(time_list) + '\n')
        with open(save_path, 'a+') as f:
            f.write(str(best_metric) + '\n')

    def calc_pre_and_auc(self, edges, device):
        test_edges = self.test_edges.to(device)
        complete_edges = self.complete_edges.to(device)
        pred = self._calc_similarity(edges, complete_edges)
        value_adj = -torch.abs(torch.ones(size=(self.nodes_num, self.nodes_num), device=device))
        for s, t, v in pred:
            value_adj[s, t] = v
        _, indices = torch.topk(value_adj.flatten(), len(test_edges))
        raTest = torch.vstack((indices // self.nodes_num, indices % self.nodes_num)).T
        mask = torch.any(torch.all(raTest[:, None] == test_edges, dim=-1), dim=-1)
        items = raTest[mask]
        precision = float(len(items)) / len(test_edges)
        mask = ~torch.any(torch.all(complete_edges[:, None] == torch.cat((edges, test_edges)), dim=-1), dim=-1)
        edgesNotExit = complete_edges[mask]
        auc = 0.0
        for te in test_edges:
            auc += (value_adj[te[0], te[1]] > value_adj[edgesNotExit[:, 0], edgesNotExit[:, 1]]).sum().item()
            auc += (value_adj[te[0], te[1]] == value_adj[edgesNotExit[:, 0], edgesNotExit[:, 1]]).sum().item() / 2

        auc /= len(test_edges) * len(edgesNotExit)
        return precision, auc

    def _calc_similarity(self, edges, complete_edges: torch.Tensor):
        train_G = nx.Graph(edges.tolist())
        train_G.add_nodes_from(range(self.min_node, self.nodes_num+self.min_node))
        mask = ~torch.any(torch.all(complete_edges[:, None] == edges, dim=-1), dim=-1)
        testList = complete_edges[mask]
        pred = nx.resource_allocation_index(train_G, testList.tolist())
        return pred

    @staticmethod
    def resource_allocation_index(graph, edge_list):
        ra_dict = {}
        for edge in edge_list:
            node_a, node_b = edge
            # 获取共同邻居
            neighbors_a = set(graph.neighbors(node_a))
            neighbors_b = set(graph.neighbors(node_b))
            common_neighbors = neighbors_a.intersection(neighbors_b)

            # 计算资源分配指数
            ra_value = sum(1.0 / len(graph.neighbors(neighbor)) for neighbor in common_neighbors if len(graph.neighbors(neighbor)) > 0)

            ra_dict[tuple(edge)] = ra_value

        return ra_dict

    def _remove_repeat(self, population: torch.Tensor, ranges, pattern="del", add=None):
        for i, pop in enumerate(population):
            clean = pop.unique()
            while len(clean) != len(pop):
                if pattern == "add":
                    edge = self._generate_node(clean, ranges) + add
                else:
                    edge = self._generate_node(clean, ranges)
                clean = torch.cat((clean, edge.unsqueeze(0)))
            population[i] = clean
        return population

    def _generate_node(self, edges, ranges):
        edge = torch.randperm(ranges, device=edges.device)[0]
        if edge in edges:
            edge = self._generate_node(edges, ranges)
            return edge
        else:
            return edge


def LPA_GA(mode, max_generation, data_loader, controller: GAController, evaluator, world_size):
    controller.mode = mode
    evaluator = controller.setup(data_loader=data_loader, evaluator=evaluator)
    if mode == "s" or mode == "sm":
        controller.calculate(max_generation=max_generation, evaluator=evaluator)
    elif mode == "m" or mode == "mnm":
        mp.spawn(controller.mp_calculate, args=(max_generation, deepcopy(evaluator), world_size), nprocs=world_size, join=True)
    else:
        raise ValueError(f"No such mode. Please choose ss, sm, ms or mm.")

















