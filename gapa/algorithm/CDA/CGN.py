import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from copy import deepcopy
import os
import time as time_mod
import numpy as np
from igraph import Graph as ig
from typing import Any
from tqdm import tqdm
from time import time
from igraph.clustering import compare_communities
from gapa.framework.body import BasicBody
from gapa.framework.controller import BasicController
from gapa.framework.evaluator import BasicEvaluator
from gapa.utils.functions import Q_Test, Num2Chunks
from gapa.utils.functions import current_time
from gapa.utils.functions import init_dist
from gapa.utils.comm_stats import CommTimer, finalize_comm_stats, timed_call
from gapa.algorithm.CDA.Genes import Gain_Edge_Set, Generate_Pop
from gapa.algorithm.CDA.Genes import generate_candidate_edge


class CGNEvaluator(BasicEvaluator):
    def __init__(self, pop_size, graph, device):
        super().__init__(
            pop_size=pop_size,
            device=device
        )
        self.G = graph.copy()
        self.ori_community = None

    def forward(self, population):
        device = population.device
        modified_graph = self.G.copy()
        fitness_list = torch.tensor([], device=device)
        for i, pop in enumerate(population):
            del_edge = pop[pop[:, 2] == 1][:, :2].tolist()
            add_edge = pop[pop[:, 2] == 0][:, :2].tolist()
            modified_graph.remove_edges_from(del_edge)
            modified_graph.add_edges_from(add_edge)

            vitim_G = ig.from_networkx(modified_graph)
            vitim_community = ig.community_multilevel(vitim_G)

            fitness = torch.tensor(compare_communities(self.ori_community, vitim_community, 'nmi'), device=device).unsqueeze(0)
            modified_graph.remove_edges_from(add_edge)
            modified_graph.add_edges_from(del_edge)
            fitness_list = torch.cat((fitness_list, fitness))
        return fitness_list


class CGNBody(BasicBody):
    def __init__(self, pop_size, budget, fit_side, edge_in, edge_out, device):
        super().__init__()
        self.fit_side = fit_side
        self.pop_size = pop_size
        self.budget = budget
        self.edge_in = edge_in.clone()
        self.edge_out = edge_out.clone()
        self.device = device

    def init_population(self):
        init_population = Generate_Pop(self.budget, self.edge_in, self.edge_out, self.device).unsqueeze(dim=0)
        for i in range(self.pop_size - 1):
            init_population = torch.vstack((init_population, Generate_Pop(self.budget, self.edge_in, self.edge_out, self.device).unsqueeze(dim=0)))
        ONE = torch.ones((self.pop_size, self.budget), dtype=torch.int, device=self.device)
        return ONE, init_population

    def selection(self, population, fitness_list):
        copy_pop = population.clone()
        normalize_fit = fitness_list / fitness_list.sum()
        samples = torch.multinomial(normalize_fit, len(normalize_fit), replacement=True)
        return copy_pop[samples]

    def mutation(self, population, population_index, mutate_rate, one):
        mutation_matrix = torch.tensor(np.random.choice([0, 1], size=(self.pop_size, self.budget), p=[1 - mutate_rate, mutate_rate]), device=self.device)
        mutation_population_index = population_index * (one - mutation_matrix) + torch.randint(2, 4, size=(self.pop_size, self.budget), device=self.device) * mutation_matrix
        return self._point_mutation(population, mutation_population_index)

    def crossover(self, population, new_population1, new_population2, crossover_rate, one):
        crossover_matrix = torch.tensor(np.random.choice([0, 1], size=(self.pop_size, self.budget), p=[1 - crossover_rate, crossover_rate]), dtype=torch.int, device=self.device)
        crossover_population_index = new_population1 * (one - crossover_matrix) + new_population2 * crossover_matrix
        return population.view(-1, population.shape[-1])[crossover_population_index]

    def elitism(self, population, mutation_population, fitness_list, new_fitness_list):
        stack_population = torch.vstack((population, mutation_population))
        stack_fitness_list = torch.hstack((fitness_list, new_fitness_list))
        top_index = None
        if self.fit_side == 'max':
            top_index = torch.argsort(stack_fitness_list, descending=True)[:self.pop_size]
        elif self.fit_side == 'min':
            top_index = torch.argsort(stack_fitness_list)[:self.pop_size]
        population = stack_population[top_index]
        fitness_list = stack_fitness_list[top_index]
        return population, fitness_list

    def _point_mutation(self, crossover_population, mutation_population_index):
        for i in range(len(mutation_population_index)):
            for j in range(len(mutation_population_index[0])):
                idx = mutation_population_index[i][j]
                if idx.item() == 2:
                    crossover_population[i][j] = generate_candidate_edge(edge_list=self.edge_out, num=1, pattern=0, device=self.device)
                elif idx.item() == 3:
                    crossover_population[i][j] = generate_candidate_edge(edge_list=self.edge_in, num=1, pattern=1, device=self.device)

        return crossover_population


class CGNController(BasicController):
    def __init__(self, path, pattern, data_loader, loops, crossover_rate, mutate_rate, pop_size, device, fit_side="min"):
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
        self.edge_in = None
        self.edge_out = None
        self.ori_community = None
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

    def setup(self, data_loader, evaluator: CGNEvaluator):
        # copy_graph = data_loader.G.copy()
        self.edge_in, self.edge_out = Gain_Edge_Set(data_loader.G, self.budget, self.device)
        ori_G = ig.from_networkx(self.graph)
        self.ori_community = ig.community_multilevel(ori_G)
        evaluator.ori_community = self.ori_community
        print(f"Original Q: {Q_Test(data_loader.G)}")
        return evaluator

    def calculate(self, max_generation, evaluator):
        best_Q = []
        best_NMI = []
        best_genes = []
        time_list = []
        body = CGNBody(self.pop_size, self.budget, self.fit_side, self.edge_in, self.edge_out, self.device)
        for loop in range(self.loops):
            start = time()
            ONE, population = body.init_population()
            if self.mode == "sm":
                evaluator = torch.nn.DataParallel(evaluator)
            fitness_list = evaluator(population).to(self.device)
            with tqdm(total=max_generation) as pbar:
                pbar.set_description(f'Training....{self.dataset} in Loop: {loop}...')
                for generation in range(max_generation):
                    t_gen_start = time_mod.perf_counter()
                    new_population_index_1 = torch.arange(self.pop_size * self.budget, device=self.device).reshape(self.pop_size, self.budget)
                    new_population_index_2 = body.selection(new_population_index_1, fitness_list)
                    crossover_population = body.crossover(population, new_population_index_1, new_population_index_2, self.crossover_rate, ONE)
                    mutation_population_index = torch.ones(size=(self.pop_size, self.budget), device=self.device)
                    mutation_population = body.mutation(crossover_population, mutation_population_index, self.mutate_rate, ONE)
                    # mutation_population = self._remove_repeat(mutation_population)
                    new_fitness_list = evaluator(mutation_population).to(self.device)
                    population, fitness_list = body.elitism(population, mutation_population, fitness_list, new_fitness_list)
                    if generation % 30 == 0 or (generation+1) == max_generation:
                        pop = population[torch.argsort(fitness_list)[0].item()]
                        copy_G = self.graph.copy()
                        del_edges = pop[pop[:, 2] == 1][:, :2].tolist()
                        add_edges = pop[pop[:, 2] == 0][:, :2].tolist()
                        copy_G.remove_edges_from(del_edges)
                        copy_G.add_edges_from(add_edges)
                        best_NMI.append(min(fitness_list).item())
                        best_Q.append(Q_Test(copy_G))
                        best_genes.append(pop)
                        end = time()
                        time_list.append(end-start)
                        try:
                            self._emit_observer(
                                generation + 1,
                                max_generation,
                                {"Q": float(best_Q[-1]), "NMI": float(best_NMI[-1])},
                            )
                        except Exception:
                            pass
                if self.mode == "mnm" and (generation % 30 == 0 or (generation + 1) == max_generation):
                    t_total = time_mod.perf_counter() - t_gen_start
                    comm = evaluator.comm_stats() if hasattr(evaluator, "comm_stats") else {}
                    avg_ms = comm.get("avg_ms", 0.0)
                    total_ms = comm.get("total_ms", 0.0)
                    print(
                        f"[MNM-LOG] gen={generation} total={t_total:.3f}s comm_avg={avg_ms:.3f}ms comm_total={total_ms/1000.0:.3f}s",
                        flush=True,
                    )
                pbar.set_postfix(NMI=min(fitness_list).item(), Q=min(best_Q))
                pbar.update(1)
            top_index = best_NMI.index(min(best_NMI))
            print(f"Q after attack: {best_Q[top_index]}, NMI after attack: {best_NMI[top_index]}")
            self.save(self.dataset, best_genes[top_index], [best_Q[top_index], best_NMI[top_index], time_list[-1]], time_list, "CGN", bestQ=best_Q, bestNMI=best_NMI)

    def mp_calculate(self, rank, max_generation, evaluator, world_size, component_size_list):
        device = init_dist(rank, world_size)
        comm_timer = CommTimer()
        best_Q = []
        best_NMI = []
        best_genes = []
        time_list = []
        edge_in, edge_out = self.edge_in.to(device), self.edge_out.to(device)
        body = CGNBody(component_size_list[rank], self.budget, self.fit_side, edge_in, edge_out, device)
        for loop in range(self.loops):
            start = time()
            ONE, component_population = body.init_population()
            if self.mode == "mnm":
                evaluator = torch.nn.DataParallel(evaluator)
            component_fitness_list = evaluator(component_population).to(device)
            population = [torch.zeros(size=(component_size,) + component_population.shape[1:], dtype=component_population.dtype, device=device) for component_size in component_size_list]
            fitness_list = [torch.empty((component_size,), dtype=component_fitness_list.dtype, device=device) for component_size in component_size_list]
            timed_call(comm_timer, "all_gather_init_pop", dist.all_gather, population, component_population)
            timed_call(comm_timer, "all_gather_init_fit", dist.all_gather, fitness_list, component_fitness_list)

            population = torch.cat(population)
            fitness_list = torch.cat(fitness_list)
            with tqdm(total=max_generation, position=rank) as pbar:
                pbar.set_description(f'Rank {rank} in {self.dataset} in Loop: {loop}')
                for generation in range(max_generation):
                    if rank == 0:
                        new_population_index_1 = torch.arange(self.pop_size * self.budget, device=self.device).reshape(self.pop_size, self.budget)
                        new_population_index_2 = body.selection(new_population_index_1, fitness_list)
                        body.pop_size = self.pop_size
                        crossover_ONE = torch.ones((self.pop_size, self.budget), dtype=new_population_index_1.dtype, device=device)
                        crossover_population = body.crossover(population, new_population_index_1, new_population_index_2, self.crossover_rate, crossover_ONE)
                        body.pop_size = component_size_list[rank]
                    max_comp = max(component_size_list)
                    if rank == 0:
                        chunks = list(torch.split(crossover_population, component_size_list))
                        scatter_list = []
                        for chunk, size in zip(chunks, component_size_list):
                            if size == max_comp:
                                scatter_list.append(chunk.contiguous())
                                continue
                            pad_shape = (max_comp,) + chunk.shape[1:]
                            padded = torch.zeros(pad_shape, dtype=chunk.dtype, device=chunk.device)
                            padded[:size] = chunk
                            scatter_list.append(padded)
                    else:
                        chunks = None
                        scatter_list = None
                    recv_shape = (max_comp,) + component_population.shape[1:]
                    component_crossover_population = torch.empty(recv_shape, dtype=component_population.dtype, device=device)
                    try:
                        timed_call(comm_timer, "scatter_crossover", dist.scatter, component_crossover_population, scatter_list, 0)
                        component_crossover_population = component_crossover_population[: component_size_list[rank]]
                    except Exception:
                        component_crossover_population = [torch.tensor([0])]
                        timed_call(
                            comm_timer,
                            "scatter_crossover",
                            dist.scatter_object_list,
                            component_crossover_population,
                            chunks if rank == 0 else [None for _ in range(world_size)],
                            0,
                        )
                        component_crossover_population = component_crossover_population[0].to(device)
                    mutation_population_index = torch.ones(size=(component_size_list[rank], self.budget), device=device)
                    mutation_population = body.mutation(component_crossover_population, mutation_population_index, self.mutate_rate, ONE)
                    # mutation_population = self._remove_repeat(mutation_population)
                    new_component_fitness_list = evaluator(mutation_population).to(device)

                    elitism_population = [torch.zeros(size=(component_size,) + mutation_population.shape[1:], dtype=mutation_population.dtype, device=device) for component_size in component_size_list]
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

                    top_index = torch.argsort(fitness_list)[:component_size_list[rank]]
                    component_population = population[top_index]
                    component_fitness_list = fitness_list[top_index]

                    if generation % 30 == 0 or (generation+1) == max_generation:
                        pop = component_population[torch.argsort(component_fitness_list)[0].item()]
                        copy_G = self.graph.copy()
                        del_edges = pop[pop[:, 2] == 1][:, :2].tolist()
                        add_edges = pop[pop[:, 2] == 0][:, :2].tolist()
                        copy_G.remove_edges_from(del_edges)
                        copy_G.add_edges_from(add_edges)
                        best_NMI.append(min(component_fitness_list).item())
                        best_Q.append(Q_Test(copy_G))
                        best_genes.append(pop)
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
                    pbar.set_postfix(NMI=min(component_fitness_list).item(), Q=min(best_Q))
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
                top_index = torch.argsort(whole_NMI)[0]
                print(f"Q after attack: {whole_Q[top_index]}, NMI after attack: {whole_NMI[top_index]}")
                self.save(self.dataset, whole_genes[top_index], [whole_Q[top_index].item(), whole_NMI[top_index].item(), time_list[-1]], time_list, "CGN", bestQ=best_Q.tolist(), bestNMI=best_NMI.tolist())
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


def CGN(mode, max_generation, data_loader, controller: CGNController, evaluator, world_size, verbose=True):
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
