from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import time
from typing import Dict, List
import os
import time as time_mod

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn
from tqdm import tqdm

from gapa.utils.functions import Num2Chunks, current_time, delete_files_in_folder, init_dist


class BasicController(nn.Module):
    """Base class controlling the GA lifecycle (selection, crossover, mutation, evaluation)."""

    def __init__(self, path: str | None, pattern: str = "overwrite", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path
        self.crossover_rate = None
        self.mutate_rate = None
        if path and pattern is not None:
            path_obj = Path(path)
            path_obj.mkdir(parents=True, exist_ok=True)
            if pattern == "overwrite":
                delete_files_in_folder(path)

    def setup(self, **kwargs):
        pass

    def calculate(self, **kwargs):
        pass

    def mp_calculate(self, **kwargs):
        pass

    def save(self, dataset, gene, best_metric, time_list, method, **kwargs):
        if not self.path:
            return
        filename = f"{dataset}_crossover_rate_{self.crossover_rate}_mutate_rate_{self.mutate_rate}_{method}.txt"
        save_path = Path(self.path) / filename
        with open(save_path, "a+", encoding="utf-8") as f:
            f.write(f"{current_time()}\n")
            f.write(f"{[i.item() for i in best_metric]}\n")
            try:
                f.write(f"{[i.item() for i in gene]}\n")
            except Exception:
                f.write(f"{[i.tolist() for i in gene]}\n")
            f.write(f"{time_list}\n")


class CustomController(BasicController):
    """Reference controller implementing a simple GA training loop."""

    def __init__(
        self,
        budget: int,
        pop_size: int,
        mode: str,
        side: str,
        num_to_eval: int,
        device: torch.device,
        save: bool = False,
        path: str | None = None,
        pattern: str | None = None,
        **kwargs,
    ):
        super().__init__(path, pattern)
        self.budget = budget
        self.pop_size = pop_size
        self.num_to_eval = num_to_eval
        self.save_flag = save
        self.side = side
        if side not in ("min", "max"):
            raise ValueError("No such side. Please choose 'max' or 'min'.")
        self.mode = mode
        self.device = device
        self.dataset = None

    def setup(self, data_loader, evaluator, **kwargs):
        self.dataset = data_loader.dataset
        return evaluator

    def init(self, body):
        return body.init_population()

    def SelectionAndCrossover(self, body, population, fitness_list, ONE):
        new_population1 = population.clone()
        new_population2 = body.selection(population, fitness_list)
        return body.crossover(new_population1, new_population2, self.crossover_rate, ONE)

    def Mutation(self, body, crossover_population, ONE):
        return body.mutation(crossover_population, self.mutate_rate, ONE)

    def Eval(self, generation, population, fitness_list, critical_genes):
        return {"generation": generation}

    def _best_index(self, fitness_list: Tensor) -> int:
        return torch.argmax(fitness_list).item() if self.side == "max" else torch.argmin(fitness_list).item()

    def _best_metric(self, fitness_list: Tensor) -> Tensor:
        return torch.max(fitness_list) if self.side == "max" else torch.min(fitness_list)

    def calculate(self, max_generation, body, evaluator, observer=None, **kwargs):
        best_genes: List[Tensor] = []
        time_list: List[float] = []
        start = time()
        ONE, population = self.init(body)
        if self.mode == "sm":
            evaluator = torch.nn.DataParallel(evaluator)
        fitness_list = evaluator(population)
        log_every = int(os.getenv("GAPA_M_LOG_EVERY", "1") or 20)
        best_fitness_list: List[Tensor] = []
        with tqdm(total=max_generation) as pbar:
            pbar.set_description(f'Training....{self.dataset}')
            for generation in range(max_generation):
                t_gen_start = time_mod.perf_counter()
                crossover_population = self.SelectionAndCrossover(body, population, fitness_list, ONE)
                mutation_population = self.Mutation(body, crossover_population, ONE)
                new_fitness_list = evaluator(mutation_population)
                population, fitness_list = body.elitism(population, mutation_population, fitness_list, new_fitness_list)

                best_metric = self._best_metric(fitness_list)
                best_fitness_list.append(best_metric)
                best_gene_idx = self._best_index(fitness_list)
                best_genes.append(population[best_gene_idx])
                elapsed = time() - start
                time_list.append(elapsed)

                results: Dict[str, float] = {}
                if generation % self.num_to_eval == 0 or generation + 1 == max_generation:
                    results = self.Eval(generation, population, fitness_list, population[best_gene_idx])
                results["fitness"] = best_metric.item()

                if observer:
                    observer.record(
                        generation=generation,
                        fitness_list=fitness_list.detach(),
                        best_gene=population[best_gene_idx].detach(),
                        extra=results,
                        side=self.side,
                    )
                if self.mode == "mnm" and log_every > 0 and (generation % log_every == 0 or generation + 1 == max_generation):
                    t_total = time_mod.perf_counter() - t_gen_start
                    comm = evaluator.comm_stats() if hasattr(evaluator, "comm_stats") else {}
                    avg_ms = comm.get("avg_ms", 0.0)
                    total_ms = comm.get("total_ms", 0.0)
                    print(
                        f"[MNM-LOG] gen={generation} total={t_total:.3f}s comm_avg={avg_ms:.3f}ms comm_total={total_ms/1000.0:.3f}s",
                        flush=True,
                    )
                pbar.set_postfix(results)
                pbar.update(1)
        top_index = torch.argmax(torch.stack(best_fitness_list)) if self.side == "max" else torch.argmin(torch.stack(best_fitness_list))
        if self.save_flag:
            self.save(self.dataset, best_genes[top_index], [time_list[-1]], time_list, "Custom")
            print(f"Data saved in {self.path}...")
        else:
            pass

    def mp_calculate(self, rank, max_generation, evaluator, body, world_size, component_size_list, observer=None):
        device = init_dist(rank, world_size)
        best_genes: List[Tensor] = []
        time_list: List[float] = []
        start = time()
        log_every = int(os.getenv("GAPA_M_LOG_EVERY", "1") or 20)
        body.device = device
        body.pop_size = component_size_list[rank]
        ONE, component_population = self.init(body)
        if self.mode == "mnm":
            evaluator = torch.nn.DataParallel(evaluator)
        component_fitness_list = evaluator(component_population).to(device)
        population = [torch.zeros((component_size,) + component_population.shape[1:], dtype=component_population.dtype, device=device) for component_size in component_size_list]
        fitness_list = [torch.empty((component_size,), dtype=component_fitness_list.dtype, device=device) for component_size in component_size_list]
        dist.all_gather(population, component_population)
        dist.all_gather(fitness_list, component_fitness_list)
        population = torch.cat(population)
        fitness_list = torch.cat(fitness_list)
        best_fitness_list: List[Tensor] = []

        with tqdm(total=max_generation, position=rank) as pbar:
            pbar.set_description(f'Rank {rank} in {self.dataset}')
            for generation in range(max_generation):
                t_gen_start = time_mod.perf_counter()
                t_sel = t_scatter = t_eval = t_gather = t_elit = t_bcast = 0.0
                if rank == 0:
                    body.pop_size = self.pop_size
                    crossover_ONE = torch.ones((self.pop_size, self.budget), dtype=component_population.dtype, device=device)
                    t_sel_start = time_mod.perf_counter()
                    crossover_population = self.SelectionAndCrossover(body, population, fitness_list, crossover_ONE)
                    t_sel = time_mod.perf_counter() - t_sel_start
                    body.pop_size = component_size_list[rank]
                max_comp = max(component_size_list)
                if rank == 0:
                    scatter_list = []
                    chunks = list(torch.split(crossover_population, component_size_list))
                    for chunk, size in zip(chunks, component_size_list):
                        if size == max_comp:
                            scatter_list.append(chunk.contiguous())
                            continue
                        pad_shape = (max_comp,) + chunk.shape[1:]
                        padded = torch.zeros(pad_shape, dtype=chunk.dtype, device=chunk.device)
                        padded[:size] = chunk
                        scatter_list.append(padded)
                else:
                    scatter_list = None
                recv_shape = (max_comp,) + component_population.shape[1:]
                component_crossover_population = torch.empty(recv_shape, dtype=component_population.dtype, device=device)
                t_scatter_start = time_mod.perf_counter()
                try:
                    dist.scatter(component_crossover_population, scatter_list, src=0)
                    t_scatter = time_mod.perf_counter() - t_scatter_start
                    component_crossover_population = component_crossover_population[: component_size_list[rank]]
                except Exception:
                    # Fallback for older dist backends or mismatched shapes
                    t_scatter = time_mod.perf_counter() - t_scatter_start
                    component_crossover_population = [torch.tensor([0])]
                    dist.scatter_object_list(component_crossover_population, chunks if rank == 0 else [None for _ in range(world_size)], src=0)
                    component_crossover_population = component_crossover_population[0].to(device)
                t_eval_start = time_mod.perf_counter()
                mutation_population = self.Mutation(body, component_crossover_population, ONE)
                new_component_fitness_list = evaluator(mutation_population).to(device)
                t_eval = time_mod.perf_counter() - t_eval_start
                elitism_population = [torch.zeros((component_size,) + mutation_population.shape[1:], dtype=mutation_population.dtype, device=device) for component_size in component_size_list]
                elitism_fitness_list = [torch.empty((component_size,), dtype=new_component_fitness_list.dtype, device=device) for component_size in component_size_list]
                t_gather_start = time_mod.perf_counter()
                dist.all_gather(elitism_population, mutation_population)
                dist.all_gather(elitism_fitness_list, new_component_fitness_list)
                t_gather = time_mod.perf_counter() - t_gather_start
                if rank == 0:
                    t_elit_start = time_mod.perf_counter()
                    elitism_population = torch.cat(elitism_population)
                    elitism_fitness_list = torch.cat(elitism_fitness_list)
                    body.pop_size = self.pop_size
                    population, fitness_list = body.elitism(population, elitism_population, fitness_list, elitism_fitness_list)
                    best_fitness_list.append(self._best_metric(fitness_list))
                    t_elit = time_mod.perf_counter() - t_elit_start
                    body.pop_size = component_size_list[rank]
                else:
                    population = torch.zeros(population.shape, dtype=population.dtype, device=device)
                    fitness_list = torch.empty(fitness_list.shape, dtype=fitness_list.dtype, device=device)

                t_bcast_start = time_mod.perf_counter()
                dist.broadcast(population, src=0)
                dist.broadcast(fitness_list, src=0)
                t_bcast = time_mod.perf_counter() - t_bcast_start

                top_index = (
                    torch.argsort(fitness_list, descending=True)[self.pop_size - component_size_list[rank] :]
                    if self.side == "max"
                    else torch.argsort(fitness_list)[: component_size_list[rank]]
                )
                component_population = population[top_index]
                component_fitness_list = fitness_list[top_index]
                best_gene = population[self._best_index(fitness_list)]
                if rank == 0:
                    best_genes.append(best_gene)
                elapsed = time() - start
                time_list.append(elapsed)
                results: Dict[str, float] = {}
                if generation % self.num_to_eval == 0 or generation + 1 == max_generation:
                    results = self.Eval(generation, component_population, component_fitness_list, best_gene)
                results["fitness"] = self._best_metric(component_fitness_list).item()

                if rank == 0 and observer:
                    observer.record(
                        generation=generation,
                        fitness_list=fitness_list.detach(),
                        best_gene=best_gene.detach(),
                        extra=results,
                        side=self.side,
                    )
                if rank == 0 and log_every > 0 and (generation % log_every == 0 or generation + 1 == max_generation):
                    t_total = time_mod.perf_counter() - t_gen_start
                    print(
                        f"[M-LOG] gen={generation} total={t_total:.3f}s sel={t_sel:.3f}s "
                        f"scatter={t_scatter:.3f}s eval={t_eval:.3f}s gather={t_gather:.3f}s "
                        f"elitism={t_elit:.3f}s bcast={t_bcast:.3f}s",
                        flush=True,
                    )
                pbar.set_postfix(results)
                pbar.update(1)
        if rank == 0:
            top_index = torch.argmax(torch.stack(best_fitness_list)) if self.side == "max" else torch.argmin(torch.stack(best_fitness_list))
            if self.save_flag:
                self.save(self.dataset, best_genes[top_index], [time_list[-1]], time_list, "Custom")
                print(f"Data saved in {self.path}...")
        torch.cuda.empty_cache()
        dist.destroy_process_group()
        torch.cuda.synchronize()


def Start(max_generation, data_loader, controller, evaluator, body, world_size, verbose=True, observer=None):
    evaluator = controller.setup(data_loader=data_loader, evaluator=evaluator)
    if controller.mode in ("s", "sm", "mnm"):
        controller.calculate(max_generation=max_generation, evaluator=evaluator, body=body, observer=observer)
    elif controller.mode == "m":
        if observer is not None and verbose:
            print("Observer recording is currently supported in single-process mode. Distributed runs will skip observer writes.")
        if world_size < 1:
            raise ValueError(f"Error in world_size -> {world_size} <- Since your device may not support for m mode, please re-choose s mode.")
        component_size_list = Num2Chunks(controller.pop_size, world_size)
        if verbose:
            print(f"Component Size List: {component_size_list}")
        mp.spawn(
            controller.mp_calculate,
            args=(max_generation, deepcopy(evaluator), deepcopy(body), world_size, component_size_list, None),
            nprocs=world_size,
            join=True,
        )
    else:
        raise ValueError("No such mode. Please choose s, sm, m or mnm.")
