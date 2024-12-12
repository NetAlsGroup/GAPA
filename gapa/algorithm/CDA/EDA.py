import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import networkx as nx
from copy import deepcopy
from tqdm import tqdm
from time import time
from gapa.framework.body import Body
from gapa.framework.controller import BasicController
from gapa.framework.evaluator import BasicEvaluator
from gapa.utils.functions import Q_Test, NMI_Test
from gapa.utils.functions import current_time
from gapa.utils.functions import init_dist


class EDAEvaluator(BasicEvaluator):
    def __init__(self, pop_size, graph, adj, nodes_num, device):
        super().__init__(
            pop_size=pop_size,
            device=device
        )
        self.G = graph.copy()
        self.AMatrix = adj.clone()
        self.IMatrix = torch.eye(len(adj), device=device).to_sparse_coo()
        self.edge_list = None
        self.nodes_num = nodes_num
        self.ONE_distance = torch.ones((nodes_num, nodes_num), dtype=torch.int32, device=device)
        self.ONE_distance.fill_diagonal_(0)
        self.distance_org = self._calculate_distance_org(self.AMatrix, self.IMatrix)

    def forward(self, population):
        device = population.device
        graph = self.G.copy()
        IMatrix = self.IMatrix.clone().to(device)
        edge_list = self.edge_list.clone().to(device)
        ONE_distance = self.ONE_distance.clone().to(device)
        distance_org = self.distance_org.clone().to(device)
        fitness_list = torch.tensor([], device=device)
        for i, pop in enumerate(population):
            del_edge_list = edge_list[pop].tolist()
            graph.remove_edges_from(del_edge_list)
            copy_A = torch.tensor(nx.adjacency_matrix(graph).toarray(), device=device)
            normalized_matrix = torch.matmul(
                torch.diag(torch.float_power(torch.sum(copy_A + IMatrix, dim=1), -1)).float(),
                copy_A.float())  # D^(-1)@A
            approximate_deepwalk_matrix = torch.mul(1 / 2, torch.add(normalized_matrix, torch.matmul(normalized_matrix, normalized_matrix)))
            u, s, v = torch.svd_lowrank(approximate_deepwalk_matrix)
            embedding = torch.matmul(u, torch.diag(torch.sqrt(s)))
            # Calculate distance
            E_dots = torch.sum(torch.mul(embedding, embedding), dim=1).reshape(self.nodes_num, 1)
            distance = torch.mul(
                torch.float_power(
                    torch.abs(
                        torch.add(
                            torch.matmul(E_dots, torch.ones(size=(1, self.nodes_num), device=device)),
                            torch.matmul(torch.ones(size=(self.nodes_num, 1), device=device),
                                         E_dots.T))
                        - 2 * torch.matmul(embedding, embedding.T)),
                    1 / 2),
                ONE_distance)
            # Calculate fitness
            fitness = 1 - abs(
                torch.corrcoef(
                    input=torch.vstack((distance.flatten(), distance_org.flatten()))
                )[0, 1]
            ).unsqueeze(0)
            fitness_list = torch.cat((fitness_list, fitness))
            graph.add_edges_from(del_edge_list)
        return fitness_list

    def _calculate_distance_org(self, AMatirx, IMatrix):
        D_inverse = torch.diag(torch.float_power(torch.sum(AMatirx + IMatrix, dim=1), -1))
        normalized_matrix = torch.matmul(D_inverse.float(), AMatirx.float())
        approximate_deepwalk_matrix = torch.mul(1 / 2, torch.add(normalized_matrix, torch.matmul(normalized_matrix, normalized_matrix)))
        u, s, v = torch.svd_lowrank(approximate_deepwalk_matrix, niter=10)

        # Calculate distance matrix
        embedding = torch.matmul(u, torch.diag(torch.sqrt(s)))
        E_dots = torch.sum(torch.mul(embedding, embedding), dim=1).reshape(self.nodes_num, 1)

        distance_org = torch.mul(
            torch.float_power(
                torch.abs(torch.add(
                    torch.matmul(E_dots, torch.ones(size=(1, self.nodes_num), device=self.device, dtype=torch.float32)),
                    torch.matmul(torch.ones(size=(self.nodes_num, 1), device=self.device, dtype=torch.float32),
                                 E_dots.T))
                          - 2 * torch.matmul(embedding, embedding.T)),
                1 / 2).float(),
            self.ONE_distance.float())

        return distance_org


class EDAController(BasicController):
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
        self.edge_list = None
        self.edge_num = None
        self.graph = data_loader.G
        self.mode = None

    def setup(self, data_loader, evaluator: EDAEvaluator):
        copy_graph = data_loader.G.copy()
        # greedy_node = []
        # for i in range(data_loader.selected_genes_num):
        #     degree = nx.degree_centrality(copy_graph)
        #     greedy_node.append(max(degree, key=degree.get))
        #     copy_graph.remove_node(max(degree, key=degree.get))
        # edge_list = []
        # for node in greedy_node:
        #     edge_list.extend(list(data_loader.G.edges(node)))
        self.edge_num = len(copy_graph.edges())
        self.edge_list = torch.tensor(list(copy_graph.edges()), device=evaluator.device)
        evaluator.edge_list = self.edge_list.clone()
        print(f"Original Q: {Q_Test(data_loader.G)}")
        return evaluator

    def calculate(self, max_generation, evaluator):
        best_Q = []
        best_NMI = []
        best_genes = []
        time_list = []
        body = Body(self.edge_num, self.budget, self.pop_size, self.fit_side, evaluator.device)
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
                    new_population1 = population.clone()
                    new_population2 = body.selection(population, fitness_list)
                    crossover_population = body.crossover(new_population1, new_population2, self.crossover_rate, ONE)
                    mutation_population = body.mutation(crossover_population, self.mutate_rate, ONE)
                    mutation_population = self._remove_repeat(mutation_population)
                    new_fitness_list = evaluator(mutation_population)
                    population, fitness_list, best_fitness_list = body.elitism(population, mutation_population, fitness_list, new_fitness_list, best_fitness_list)
                    if generation % 50 == 0 or (generation+1) == max_generation:
                        critical_edges = self.edge_list[population[torch.argsort(fitness_list, descending=True)[0]]]
                        copy_G = self.graph.copy()
                        copy_G.remove_edges_from(critical_edges.tolist())
                        best_Q.append(Q_Test(copy_G))
                        best_NMI.append(NMI_Test(self.graph.copy(), copy_G))
                        best_genes.append(critical_edges)
                        end = time()
                        time_list.append(end - start)
                    pbar.set_postfix(fitness=max(fitness_list).item(), Q=min(best_Q), NMI=min(best_NMI))
                    pbar.update(1)
            end = time()
            global_time = end - start
            top_index = best_Q.index(min(best_Q))
            print(f"Q after attack: {best_Q[top_index]}, NMI after attack: {best_NMI[top_index]}")
            self.save(self.dataset, best_genes[top_index], [best_Q[top_index], best_NMI[top_index], time_list[-1]], time_list, "EDA", bestQ=best_Q, bestNMI=best_NMI)

    def mp_calculate(self, rank, max_generation, evaluator, world_size):
        device = init_dist(rank, world_size)
        best_Q = []
        best_NMI = []
        best_genes = []
        time_list = []
        edge_list = self.edge_list.to(device)
        component_size = self.pop_size // world_size
        body = Body(self.edge_num, self.budget, component_size, self.fit_side, device)
        for loop in range(self.loops):
            start = time()
            ONE, component_population = body.init_population()
            if self.mode == "mnm":
                evaluator = torch.nn.DataParallel(evaluator)
            component_fitness_list = evaluator(component_population).to(device)
            best_fitness_list = torch.tensor(data=[], device=device)
            best_fitness_list = torch.hstack((best_fitness_list, torch.max(component_fitness_list)))
            if rank == 0:
                population = [torch.zeros((component_size, self.budget), dtype=torch.int64, device=device) for _ in range(world_size)]
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
                        new_population1 = population.clone()
                        new_population2 = body.selection(new_population1, fitness_list)
                        body.pop_size = self.pop_size
                        crossover_ONE = torch.ones((self.pop_size, self.budget), dtype=torch.int64, device=device)
                        crossover_population = body.crossover(new_population1, new_population2, self.crossover_rate, crossover_ONE)
                        body.pop_size = component_size
                    if rank == 0:
                        crossover_population = torch.stack(torch.split(crossover_population, component_size))
                    else:
                        crossover_population = torch.stack( [torch.zeros((component_size, self.budget), dtype=torch.int64, device=device) for _ in range(world_size)])
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
                        top_index = torch.argsort(fitness_list)[self.pop_size - component_size:]
                        component_population = population[top_index]
                        component_fitness_list = fitness_list[top_index]
                        body.pop_size = component_size
                    else:
                        component_population = torch.zeros((component_size, self.budget), dtype=torch.int64, device=device)
                        component_fitness_list = torch.empty((component_size,), dtype=torch.float64, device=device)
                    dist.broadcast(component_population, src=0)
                    dist.broadcast(component_fitness_list, src=0)
                    if generation % 30 == 0 or (generation+1) == max_generation:
                        component_population = self._remove_repeat(component_population.clone())
                        component_edges = edge_list[component_population[torch.argsort(component_fitness_list, descending=True)[0]]]
                        copy_G = self.graph.copy()
                        copy_G.remove_edges_from(component_edges.tolist())
                        best_Q.append(Q_Test(copy_G))
                        best_NMI.append(NMI_Test(self.graph.copy(), copy_G))
                        best_genes.append(component_edges)
                        end = time()
                        time_list.append(end - start)
                    pbar.set_postfix(fitness=max(component_fitness_list).item(), Q=min(best_Q), NMI=min(best_NMI))
                    pbar.update(1)
            end = time()
            global_time = end - start
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
                top_index = torch.argsort(whole_Q)[0]
                print(f"Q after attack: {whole_Q[top_index]}, NMI after attack: {whole_NMI[top_index]}")
                self.save(self.dataset, whole_genes[top_index], [whole_Q[top_index].item(), whole_NMI[top_index].item(), time_list[-1]], time_list, "EDA", bestQ=best_Q.tolist(), bestNMI=best_NMI.tolist())
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
            clean = pop.unique()
            while len(clean) != len(pop):
                edge = self._generate_node(clean)
                clean = torch.cat((clean, edge.unsqueeze(0)))
            population[i] = clean
        return population

    def _generate_node(self, edges):
        edge = torch.randperm(self.edge_num, device=edges.device)[0]
        if edge in edges:
            edge = self._generate_node(edges)
            return edge
        else:
            return edge


def EDA(mode, max_generation, data_loader, controller: EDAController, evaluator, world_size):
    controller.mode = mode
    evaluator = controller.setup(data_loader=data_loader, evaluator=evaluator)
    if mode == "s" or mode == "sm":
        controller.calculate(max_generation=max_generation, evaluator=evaluator)
    elif mode == "m" or mode == "mnm":
        mp.spawn(controller.mp_calculate, args=(max_generation, deepcopy(evaluator), world_size), nprocs=world_size, join=True)
    else:
        raise ValueError(f"No such mode. Please choose ss, sm, ms or mm.")







