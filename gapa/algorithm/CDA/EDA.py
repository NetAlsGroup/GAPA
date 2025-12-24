import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import networkx as nx
from copy import deepcopy
from typing import Any
from tqdm import tqdm
from time import time
from gapa.framework.body import Body
from gapa.framework.controller import BasicController
from gapa.framework.evaluator import BasicEvaluator
from gapa.utils.functions import Q_Test, NMI_Test, Num2Chunks
from gapa.utils.functions import current_time
from gapa.utils.functions import init_dist
from gapa.utils.comm_stats import CommTimer, finalize_comm_stats, timed_call


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
        self.one = torch.ones(size=(1, self.nodes_num), device=device)
        self.one_t = torch.ones(size=(self.nodes_num, 1), device=device)
        self.distance_org = self._calculate_distance_org(self.AMatrix, self.IMatrix)

    def forward(self, population):
        device = population.device
        A = self.AMatrix.to(device).float()
        edge_list = self.edge_list.clone().to(device)
        ONE_distance = self.ONE_distance.clone().to(device)
        distance_org = self.distance_org.clone().to(device)
        fitness_list = []
        for i, pop in enumerate(population):
            copy_A = A.clone()
            del_idx = edge_list[pop]
            copy_A[del_idx[:, 0], del_idx[:, 1]] = 0
            copy_A[del_idx[:, 1], del_idx[:, 0]] = 0
            normalized_matrix = torch.diag(torch.float_power(copy_A.sum(dim=1) + 1, -1)).float() @ copy_A
            approximate_deepwalk_matrix = 1/2 * (normalized_matrix + normalized_matrix @ normalized_matrix)
            u, s, v = torch.svd_lowrank(approximate_deepwalk_matrix)
            embedding = u @ torch.diag(s.sqrt())
            # Calculate distance
            E_dots = torch.sum(torch.mul(embedding, embedding), dim=1).reshape(self.nodes_num, 1)
            distance = torch.mul(
                torch.float_power(
                    torch.abs(
                        torch.add(
                            torch.matmul(E_dots, self.one.to(device)),
                            torch.matmul(self.one_t.to(device), E_dots.T))
                        - 2 * torch.matmul(embedding, embedding.T)),
                    1 / 2),
                ONE_distance)

            # Calculate fitness
            fitness = 1 - abs(
                torch.corrcoef(
                    input=torch.vstack((distance.flatten(), distance_org.flatten()))
                )[0, 1]
            )
            fitness_list.append(fitness)

        return torch.tensor(fitness_list, device=device)

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
                    torch.matmul(E_dots, self.one),
                    torch.matmul(self.one_t, E_dots.T)
                ) - 2 * torch.matmul(embedding, embedding.T)),
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
        self.observer = None

    def _emit_observer(self, generation: int, max_generation: int, payload: Any) -> None:
        obs = self.observer
        if obs is None:
            return
        if callable(obs):
            obs(generation, max_generation, payload)
            return
        if isinstance(obs, dict) and obs.get("type") == "jsonl" and obs.get("path"):
            try:
                import json

                with open(obs["path"], "a", encoding="utf-8") as f:
                    if isinstance(payload, dict):
                        f.write(json.dumps({"generation": generation, "max_generation": max_generation, "metrics": payload}) + "\n")
                    else:
                        f.write(json.dumps({"generation": generation, "max_generation": max_generation, "best_fitness": payload}) + "\n")
            except Exception:
                pass

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
            with tqdm(total=max_generation) as pbar:
                pbar.set_description(f'Training....{self.dataset} in Loop: {loop}...')
                for generation in range(max_generation):
                    new_population1 = population.clone()
                    new_population2 = body.selection(population, fitness_list)
                    crossover_population = body.crossover(new_population1, new_population2, self.crossover_rate, ONE)
                    mutation_population = body.mutation(crossover_population, self.mutate_rate, ONE)
                    mutation_population = self._remove_repeat(mutation_population)
                    new_fitness_list = evaluator(mutation_population)
                    population, fitness_list = body.elitism(population, mutation_population, fitness_list, new_fitness_list)
                    if generation % 50 == 0 or (generation+1) == max_generation:
                        critical_edges = self.edge_list[population[torch.argsort(fitness_list, descending=True)[0]]]
                        copy_G = self.graph.copy()
                        copy_G.remove_edges_from(critical_edges.tolist())
                        best_Q.append(Q_Test(copy_G))
                        best_NMI.append(NMI_Test(self.graph.copy(), copy_G))
                        best_genes.append(critical_edges)
                        end = time()
                        time_list.append(end - start)
                        try:
                            self._emit_observer(
                                generation + 1,
                                max_generation,
                                {"Q": float(best_Q[-1]), "NMI": float(best_NMI[-1])},
                            )
                        except Exception:
                            pass
                    pbar.set_postfix(fitness=max(fitness_list).item(), Q=min(best_Q), NMI=min(best_NMI))
                    pbar.update(1)
            end = time()
            global_time = end - start
            top_index = best_Q.index(min(best_Q))
            print(f"Q after attack: {best_Q[top_index]}, NMI after attack: {best_NMI[top_index]}")
            self.save(self.dataset, best_genes[top_index], [best_Q[top_index], best_NMI[top_index], time_list[-1]], time_list, "EDA", bestQ=best_Q, bestNMI=best_NMI)

    def mp_calculate(self, rank, max_generation, evaluator, world_size, component_size_list):
        device = init_dist(rank, world_size)
        comm_timer = CommTimer()
        best_Q = []
        best_NMI = []
        best_genes = []
        time_list = []
        edge_list = self.edge_list.to(device)
        body = Body(self.edge_num, self.budget, component_size_list[rank], self.fit_side, device)
        for loop in range(self.loops):
            start = time()
            ONE, component_population = body.init_population()
            if self.mode == "mnm":
                evaluator = torch.nn.DataParallel(evaluator)
            component_fitness_list = evaluator(component_population).to(device)
            population = [torch.zeros((component_size, self.budget), dtype=component_population.dtype, device=device) for component_size in component_size_list]
            fitness_list = [torch.empty((component_size,), dtype=component_fitness_list.dtype, device=device) for component_size in component_size_list]
            timed_call(comm_timer, "all_gather_init_pop", dist.all_gather, population, component_population)
            timed_call(comm_timer, "all_gather_init_fit", dist.all_gather, fitness_list, component_fitness_list)

            population = torch.cat(population)
            fitness_list = torch.cat(fitness_list)

            with tqdm(total=max_generation, position=rank) as pbar:
                pbar.set_description(f'Rank {rank} in {self.dataset} in Loop: {loop}')
                for generation in range(max_generation):
                    if rank == 0:
                        new_population1 = population.clone()
                        new_population2 = body.selection(new_population1, fitness_list)
                        body.pop_size = self.pop_size
                        crossover_ONE = torch.ones((self.pop_size, self.budget), dtype=component_population.dtype, device=device)
                        crossover_population = body.crossover(new_population1, new_population2, self.crossover_rate, crossover_ONE)
                        body.pop_size = component_size_list[rank]
                    if rank == 0:
                        crossover_population = list(torch.split(crossover_population, component_size_list))
                    else:
                        crossover_population = [None for _ in range(world_size)]

                    component_crossover_population = [torch.tensor([0])]
                    timed_call(comm_timer, "scatter_crossover", dist.scatter_object_list, component_crossover_population, crossover_population, src=0)
                    component_crossover_population = component_crossover_population[0].to(device)
                    mutation_population = body.mutation(component_crossover_population, self.mutate_rate, ONE)
                    # mutation_population = self._remove_repeat(mutation_population)
                    new_component_fitness_list = evaluator(mutation_population).to(device)

                    elitism_population = [torch.zeros((component_size, self.budget), dtype=component_population.dtype, device=device) for component_size in component_size_list]
                    elitism_fitness_list = [torch.empty((component_size,), dtype=new_component_fitness_list.dtype, device=device) for component_size in component_size_list]
                    timed_call(comm_timer, "all_gather_elitism_pop", dist.all_gather, elitism_population, mutation_population)
                    timed_call(comm_timer, "all_gather_elitism_fit", dist.all_gather, elitism_fitness_list, new_component_fitness_list)

                    if rank == 0:
                        elitism_population = torch.cat(elitism_population)
                        elitism_fitness_list = torch.cat(elitism_fitness_list)
                        body.pop_size = self.pop_size
                        population, fitness_list = body.elitism(population, elitism_population, fitness_list, elitism_fitness_list)
                        body.pop_size = component_size_list[rank]
                    else:
                        population = torch.zeros(population.shape, dtype=population.dtype, device=device)
                        fitness_list = torch.empty(fitness_list.shape, dtype=fitness_list.dtype, device=device)

                    timed_call(comm_timer, "broadcast_pop", dist.broadcast, population, src=0)
                    timed_call(comm_timer, "broadcast_fit", dist.broadcast, fitness_list, src=0)

                    top_index = torch.argsort(fitness_list)[self.pop_size - component_size_list[rank]:]
                    component_population = population[top_index]
                    component_fitness_list = fitness_list[top_index]

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
                        if rank == 0:
                            try:
                                self._emit_observer(
                                    generation + 1,
                                    max_generation,
                                    {"Q": float(best_Q[-1]), "NMI": float(best_NMI[-1])},
                                )
                            except Exception:
                                pass
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
            timed_call(comm_timer, "barrier", dist.barrier)
            timed_call(comm_timer, "gather_genes", dist.gather, best_genes, whole_genes, dst=0)
            timed_call(comm_timer, "gather_q", dist.gather, best_Q, whole_Q, dst=0)
            timed_call(comm_timer, "gather_nmi", dist.gather, best_NMI, whole_NMI, dst=0)
            if rank == 0:
                whole_genes = torch.cat(whole_genes)
                whole_Q = torch.cat(whole_Q)
                whole_NMI = torch.cat(whole_NMI)
                top_index = torch.argsort(whole_Q)[0]
                print(f"Q after attack: {whole_Q[top_index]}, NMI after attack: {whole_NMI[top_index]}")
                self.save(self.dataset, whole_genes[top_index], [whole_Q[top_index].item(), whole_NMI[top_index].item(), time_list[-1]], time_list, "EDA", bestQ=best_Q.tolist(), bestNMI=best_NMI.tolist())
                print(f"Loop {loop} finished. Data saved in {self.path}...")
        torch.cuda.empty_cache()
        finalize_comm_stats(comm_timer, rank, world_size, getattr(self, "comm_path", None))
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


def EDA(mode, max_generation, data_loader, controller: EDAController, evaluator, world_size, verbose=True):
    controller.mode = mode
    evaluator = controller.setup(data_loader=data_loader, evaluator=evaluator)
    if mode in ("s", "sm", "mnm"):
        controller.calculate(max_generation=max_generation, evaluator=evaluator)
    elif mode == "m":
        component_size_list = Num2Chunks(controller.pop_size, world_size)
        if verbose:
            print(f"Component Size List: {component_size_list}")
        mp.spawn(controller.mp_calculate, args=(max_generation, deepcopy(evaluator), world_size, component_size_list), nprocs=world_size, join=True)
    else:
        raise ValueError(f"No such mode. Please choose s, sm, m or mnm.")
