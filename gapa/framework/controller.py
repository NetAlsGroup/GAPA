import os
import torch
from time import time
from torch import nn
from tqdm import tqdm
from copy import deepcopy
import torch.distributed as dist
# from gapa.framework.body import Body
import torch.multiprocessing as mp
from gapa.utils.functions import init_dist
from gapa.utils.functions import current_time, Num2Chunks
from gapa.utils.functions import delete_files_in_folder


class BasicController(nn.Module):
    def __init__(self, path, pattern='overwrite', *args, **kwargs):
        super().__init__(*args, **kwargs)
        if pattern is None:
            pass
        else:
            if not os.path.exists(path):
                os.makedirs(path)
            elif pattern == 'overwrite':
                delete_files_in_folder(path)
            elif pattern == 'write':
                pass
        self.path = path
        self.crossover_rate = None
        self.mutate_rate = None

    def setup(self, **kwargs):
        pass

    def calculate(self, **kwargs):
        pass

    def mp_calculate(self, **kwargs):
        pass

    def save(self, dataset, gene, best_metric, time_list, method, **kwargs):
        save_path = self.path + dataset + '_crossover_rate_' + str(self.crossover_rate) + '_mutate_rate_' + str(self.mutate_rate) + f'_{method}.txt'
        with open(save_path, 'a+') as f:
            f.write(current_time())
            f.write("\n")
        with open(save_path, 'a+') as f:
            f.write(str([i.item() for i in best_metric]) + '\n')
        with open(save_path, 'a+') as f:
            try:
                f.write(str([i.item() for i in gene]) + '\n')
            except:
                f.write(str([i.tolist() for i in gene]) + '\n')
        with open(save_path, 'a+') as f:
            f.write(str(time_list) + '\n')


class CustomController(BasicController):
    def __init__(self, budget, pop_size, mode, side, num_to_eval, device, save=False, path=None, pattern=None, **kwargs):
        super().__init__(
            path,
            pattern,
        )
        self.budget = budget
        self.pop_size = pop_size
        self.num_to_eval = num_to_eval
        self.save_flag = save
        self.side = side
        func_map = {
            "min": min,
            "max": max
        }
        if self.side == "max" or self.side == "min":
            self.func = func_map.get(self.side)
        else:
            raise ValueError(f"No such side. Please choose 'max' or 'min'.")
        self.mode = mode
        self.device = device
        self.dataset = None

    def setup(self, data_loader, evaluator, **kwargs):
        self.dataset = data_loader.dataset
        return evaluator

    def init(self, body):
        ONE, population = body.init_population()
        return ONE, population

    def SelectionAndCrossover(self, body, population, fitness_list, ONE):
        new_population1 = population.clone()
        new_population2 = body.selection(population, fitness_list)
        crossover_population = body.crossover(new_population1, new_population2, self.crossover_rate, ONE)
        return crossover_population

    def Mutation(self, body, crossover_population, ONE):
        mutation_population = body.mutation(crossover_population, self.mutate_rate, ONE)
        return mutation_population

    def Eval(self, generation, population, fitness_list, critical_genes):
        return {
            "generation": generation
        }

    def calculate(self, max_generation, body, evaluator, **kwargs):
        best_genes = []
        time_list = []
        start = time()
        ONE, population = self.init(body)
        if self.mode == "sm":
            evaluator = torch.nn.DataParallel(evaluator)
        fitness_list = evaluator(population)
        best_fitness_list = torch.tensor(data=[], device=self.device)
        best_fitness_list = torch.hstack((best_fitness_list, torch.min(fitness_list)))
        with tqdm(total=max_generation) as pbar:
            pbar.set_description(f'Training....{self.dataset}')
            for generation in range(max_generation):
                crossover_population = self.SelectionAndCrossover(body, population, fitness_list, ONE)
                mutation_population = self.Mutation(body, crossover_population, ONE)
                new_fitness_list = evaluator(mutation_population)
                population, fitness_list = body.elitism(population, mutation_population, fitness_list, new_fitness_list)
                best_fitness_list = torch.hstack((best_fitness_list, self.func(fitness_list)))
                critical_genes = population[torch.argsort(fitness_list.clone())[0]]
                best_genes.append(critical_genes)
                end = time()
                time_list.append(end - start)
                if generation % self.num_to_eval == 0 or generation + 1 == max_generation:
                    results = self.Eval(generation, population, fitness_list, critical_genes)
                results["fitness"] = self.func(fitness_list).item()
                pbar.set_postfix(results)
                pbar.update(1)
        top_index = torch.argsort(best_fitness_list)[-1 if self.side == "max" else 0]
        if self.save_flag:
            self.save(self.dataset, best_genes[top_index], [time_list[-1]], time_list, "Custom")
            print(f"Data saved in {self.path}...")
        else:
            pass

    def mp_calculate(self, rank, max_generation, evaluator, body, world_size, component_size_list):
        device = init_dist(rank, world_size)
        best_genes = []
        time_list = []
        start = time()

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

        best_fitness_list = torch.tensor(data=[], device=device)
        best_fitness_list = torch.hstack((best_fitness_list, self.func(component_fitness_list)))

        with tqdm(total=max_generation, position=rank) as pbar:
            pbar.set_description(f'Rank {rank} in {self.dataset}')
            for generation in range(max_generation):
                if rank == 0:
                    body.pop_size = self.pop_size
                    crossover_ONE = torch.ones((self.pop_size, self.budget), dtype=component_population.dtype, device=device)
                    crossover_population = self.SelectionAndCrossover(body, population, fitness_list, crossover_ONE)
                    body.pop_size = component_size_list[rank]
                if rank == 0:
                    crossover_population = list(torch.split(crossover_population, component_size_list))
                else:
                    crossover_population = [None for _ in range(world_size)]
                component_crossover_population = [torch.tensor([0])]
                dist.scatter_object_list(component_crossover_population, crossover_population, src=0)
                component_crossover_population = component_crossover_population[0].to(device)

                mutation_population = self.Mutation(body, component_crossover_population, ONE)
                new_component_fitness_list = evaluator(mutation_population).to(device)

                elitism_population = [torch.zeros((component_size,) + mutation_population.shape[1:], dtype=mutation_population.dtype, device=device) for component_size in component_size_list]
                elitism_fitness_list = [torch.empty((component_size,), dtype=new_component_fitness_list.dtype, device=device) for component_size in component_size_list]
                dist.all_gather(elitism_population, mutation_population)
                dist.all_gather(elitism_fitness_list, new_component_fitness_list)

                if rank == 0:
                    elitism_population = torch.cat(elitism_population)
                    elitism_fitness_list = torch.cat(elitism_fitness_list)
                    body.pop_size = self.pop_size
                    population, fitness_list = body.elitism(population, elitism_population, fitness_list, elitism_fitness_list)
                    best_fitness_list = torch.hstack((best_fitness_list, self.func(fitness_list)))
                    body.pop_size = component_size_list[rank]
                else:
                    population = torch.zeros(population.shape, dtype=population.dtype, device=device)
                    fitness_list = torch.empty(fitness_list.shape, dtype=fitness_list.dtype, device=device)

                dist.broadcast(population, src=0)
                dist.broadcast(fitness_list, src=0)

                if self.side == "max":
                    top_index = torch.argsort(fitness_list)[self.pop_size - component_size_list[rank]:]
                else:
                    top_index = torch.argsort(fitness_list)[:component_size_list[rank]]
                component_population = population[top_index]
                component_fitness_list = fitness_list[top_index]

                critical_genes = component_population[torch.argsort(component_fitness_list.clone())[0]]
                best_genes.append(critical_genes)
                end = time()
                time_list.append(end - start)
                if generation % self.num_to_eval == 0 or generation + 1 == max_generation:
                    results = self.Eval(generation, component_population, component_fitness_list, critical_genes)
                results["fitness"] = self.func(component_fitness_list).item()
                pbar.set_postfix(results)
                pbar.update(1)
        best_genes = torch.stack(best_genes)
        if rank == 0:
            whole_genes = [torch.zeros(best_genes.shape, dtype=best_genes.dtype, device=device) for _ in range(world_size)]
        else:
            whole_genes = None
        dist.barrier()
        dist.gather(best_genes, whole_genes, dst=0)
        if rank == 0:
            whole_genes = torch.cat(whole_genes)
            top_index = torch.argsort(best_fitness_list)[-1 if self.side == "max" else 0]
            if self.save_flag:
                self.save(self.dataset, best_genes[top_index], [time_list[-1]], time_list, "Custom")
                print(f"Data saved in {self.path}...")
            else:
                pass
        torch.cuda.empty_cache()
        dist.destroy_process_group()
        torch.cuda.synchronize()


def Start(max_generation, data_loader, controller, evaluator, body, world_size, verbose=True):
    evaluator = controller.setup(data_loader=data_loader, evaluator=evaluator)
    if controller.mode == "s" or controller.mode == "sm":
        controller.calculate(max_generation=max_generation, evaluator=evaluator, body=body)
    elif controller.mode == "m" or controller.mode == "mnm":
        if world_size < 1:
            raise ValueError(f"Error in world_size -> {world_size} <- Since your device may not support for m or mnm mode, please re-choose s mode.")
        component_size_list = Num2Chunks(controller.pop_size, world_size)
        if verbose:
            print(f"Component Size List: {component_size_list}")
        mp.spawn(controller.mp_calculate, args=(max_generation, deepcopy(evaluator), deepcopy(body), world_size, component_size_list), nprocs=world_size, join=True)
    else:
        raise ValueError(f"No such mode. Please choose ss, sm, ms or mm.")

