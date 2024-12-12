import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from copy import deepcopy
from tqdm import tqdm
from time import time
from gapa.framework.body import Body
from gapa.framework.controller import BasicController
from gapa.framework.evaluator import BasicEvaluator
from gapa.utils.functions import AS_Rate, current_time, init_dist, Acc


class NCA_GAEvaluator(BasicEvaluator):
    def __init__(self, classifier, feats, adj, test_index, labels, pop_size, device):
        super().__init__(
            pop_size=pop_size,
            device=device
        )
        self.classifier = classifier
        self.edge_list = None
        self.feats = feats.clone()
        self.adj = adj.to_dense().clone()
        self.test_index = test_index
        self.labels = labels

    def forward(self, population):
        device = population.device
        model = deepcopy(self.classifier.model).to(device)
        feats = self.feats.clone().to(device)
        copy_adj = self.adj.clone().to(device)
        test_index = self.test_index.clone().to(device)
        labels = self.labels.clone().to(device)
        edge_list = self.edge_list.clone().to(device)
        fitness_list = torch.tensor([], device=device)
        for i in range(len(population)):
            del_edge_list = edge_list[population[i]]
            copy_adj[del_edge_list[:, 0], del_edge_list[:, 1]] = 0
            output = F.log_softmax(model(feats, copy_adj), dim=1)
            loss_test = F.nll_loss(output[test_index], labels[test_index])
            fitness_list = torch.cat((fitness_list, torch.tensor([loss_test], device=device)), dim=0)
            copy_adj[del_edge_list[:, 0], del_edge_list[:, 1]] = 1
        return fitness_list


class NCA_GAController(BasicController):
    def __init__(self, path, pattern, data_loader, classifier, loops, crossover_rate, mutate_rate, pop_size, device, fit_side="max"):
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
        self.classifier = classifier
        self.edge_list = None
        self.edge_num = None
        self.mode = None
        self.adj = None
        self.feats = None
        self.test_index = None
        self.labels = None

    def setup(self, data_loader, evaluator: NCA_GAEvaluator):
        copy_graph = data_loader.G.copy()
        self.edge_num = len(copy_graph.edges())
        self.edge_list = torch.tensor(list(copy_graph.edges()), device=evaluator.device)
        self.feats = data_loader.feats
        self.adj = evaluator.adj.clone()
        self.test_index = data_loader.test_index
        self.labels = data_loader.labels
        evaluator.edge_list = self.edge_list.clone()
        return evaluator

    def calculate(self, max_generation, evaluator):
        best_as_rate = []
        best_acc = []
        best_genes = []
        time_list = []
        body = Body(self.edge_num, self.budget, self.pop_size, self.fit_side, evaluator.device)
        for loop in range(self.loops):
            start = time()
            ONE, population = body.init_population()
            if self.mode == "sm":
                evaluator = torch.nn.DataParallel(evaluator)
            fitness_list = evaluator(torch.clone(population))
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
                    if generation % 30 == 0 or (generation+1) == max_generation:
                        critical_edges = self.edge_list[population[torch.argsort(fitness_list, descending=True)[0]]]
                        copy_adj = self.adj.clone()
                        copy_adj[critical_edges[:, 0], critical_edges[:, 1]] = 0
                        best_as_rate.append(AS_Rate(feats=self.feats, modify_feats=self.feats, adj=self.adj, modify_adj=copy_adj, model=self.classifier.model, test_index=self.test_index))
                        best_acc.append(Acc(modify_feats=self.feats, modify_adj=copy_adj, model=self.classifier.model, test_index=self.test_index, labels=self.labels))
                        best_genes.append(critical_edges)
                        end = time()
                        time_list.append(end - start)
                    pbar.set_postfix(Loss=max(fitness_list).item(), Acc=min(best_acc), AS_Rate=max(best_as_rate))
                    pbar.update(1)
            # Test
            top_index = best_acc.index(min(best_acc))
            print(f"Acc: {best_acc[top_index]}, AS_Rate: {best_as_rate[top_index]}")
            self.save(self.dataset, best_genes[top_index], [best_acc[top_index], best_as_rate[top_index], time_list[-1]], time_list, "NCA_GA", bestAcc=best_acc, bestASR=best_as_rate)
            print(f"Loop {loop} finished. Data saved in {self.path}...")

    def mp_calculate(self, rank, max_generation, evaluator, world_size):
        device = init_dist(rank, world_size)
        best_as_rate = []
        best_acc = []
        best_genes = []
        time_list = []
        # send to current device
        edge_list = self.edge_list.to(device)
        feats = self.feats.to(device)
        adj = self.adj.to(device)
        test_index = self.test_index.to(device)
        labels = self.labels.to(device)
        model = deepcopy(self.classifier.model).to(device)
        evaluator.edge_list = edge_list.clone()

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
                population = [torch.zeros((component_size, self.budget), dtype=component_population.dtype, device=device) for _ in range(world_size)]
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
                        crossover_ONE = torch.ones((self.pop_size, self.budget), dtype=new_population1.dtype, device=device)
                        crossover_population = body.crossover(new_population1, new_population2, self.crossover_rate, crossover_ONE)
                        body.pop_size = component_size
                    if rank == 0:
                        crossover_population = torch.stack(torch.split(crossover_population, component_size))
                    else:
                        crossover_population = torch.stack([torch.zeros((component_size, self.budget), dtype=component_population.dtype, device=device) for _ in range(world_size)])
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
                        component_population = torch.zeros((component_size, self.budget), dtype=component_population.dtype, device=device)
                        component_fitness_list = torch.empty((component_size,), dtype=component_fitness_list.dtype, device=device)
                    dist.broadcast(component_population, src=0)
                    dist.broadcast(component_fitness_list, src=0)
                    if generation % 30 == 0 or (generation + 1) == max_generation:
                        component_population = self._remove_repeat(component_population.clone())
                        component_edges = edge_list[component_population[torch.argsort(component_fitness_list, descending=True)[0]]]
                        copy_adj = adj.clone()
                        copy_adj[component_edges[:, 0], component_edges[:, 1]] = 0
                        best_as_rate.append(AS_Rate(feats=feats, modify_feats=feats, adj=adj, modify_adj=copy_adj, model=model, test_index=test_index))
                        best_acc.append(Acc(modify_feats=feats, modify_adj=copy_adj, model=model, test_index=test_index, labels=labels))
                        best_genes.append(component_edges)
                        end = time()
                        time_list.append(end - start)
                    pbar.set_postfix(Loss=max(component_fitness_list).item(), Acc=min(best_acc), AS_Rate=max(best_as_rate))
                    pbar.update(1)
            best_genes = torch.stack(best_genes)
            best_acc = torch.tensor(best_acc, device=device)
            best_as_rate = torch.tensor(best_as_rate, device=device)
            if rank == 0:
                whole_genes = [torch.zeros(best_genes.shape, dtype=best_genes.dtype, device=device) for _ in range(world_size)]
                whole_acc = [torch.empty(best_acc.shape, device=device) for _ in range(world_size)]
                whole_as_rate = [torch.empty(best_as_rate.shape, device=device) for _ in range(world_size)]
            else:
                whole_genes = None
                whole_acc = None
                whole_as_rate = None
            dist.barrier()
            dist.gather(best_genes, whole_genes, dst=0)
            dist.gather(best_acc, whole_acc, dst=0)
            dist.gather(best_as_rate, whole_as_rate, dst=0)
            if rank == 0:
                whole_genes = torch.cat(whole_genes)
                whole_acc = torch.cat(whole_acc)
                whole_as_rate = torch.cat(whole_as_rate)
                top_index = torch.argsort(whole_acc)[0]
                print(f"Acc: {whole_acc[top_index]}, AS_Rate: {whole_as_rate[top_index]}")
                self.save(self.dataset, whole_genes[top_index], [whole_acc[top_index].item(), whole_as_rate[top_index].item(), time_list[-1]], time_list, "NCA_GA", bestAcc=best_acc.tolist(), bestASR=best_as_rate.tolist())
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
            f.write(str([i for i in kwargs['bestAcc']]) + '\n')
        with open(save_path, 'a+') as f:
            f.write(str([i for i in kwargs['bestASR']]) + '\n')
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


def NCA_GA(mode, max_generation, data_loader, controller: NCA_GAController, evaluator, world_size):
    controller.mode = mode
    evaluator = controller.setup(data_loader=data_loader, evaluator=evaluator)
    if mode == "s" or mode == "sm":
        controller.calculate(max_generation=max_generation, evaluator=evaluator)
    elif mode == "m" or mode == "mnm":
        mp.spawn(controller.mp_calculate, args=(max_generation, deepcopy(evaluator), world_size), nprocs=world_size, join=True)
    else:
        raise ValueError(f"No such mode. Please choose ss, sm, ms or mm.")

