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
from gapa.utils.functions import CNDTest
from gapa.utils.functions import current_time
from gapa.utils.functions import init_dist
from collections import Counter


class SixDSTEvaluator(BasicEvaluator):
    def __init__(self, pop_size, adj: torch.Tensor, device):
        super().__init__(
            pop_size=pop_size,
            device=device
        )
        self.AMatrix = adj.clone()
        self.IMatrix = torch.eye(len(adj)).to_sparse_coo()
        self.genes_index = None

    def forward(self, population):
        population_component_list = []
        genes_index = self.genes_index.clone().to(population.device)
        AMatrix = self.AMatrix.clone().to(population.device)
        IMatrix = self.IMatrix.clone().to(population.device)
        for i in range(len(population)):
            copy_A = AMatrix.clone()
            copy_A[genes_index[population[i]], :] = 0
            copy_A[:, genes_index[population[i]]] = 0
            # copy_A = copy_A.to_sparse_coo()
            matrix2 = torch.matmul((copy_A + IMatrix), (copy_A + IMatrix))
            matrix4 = torch.matmul(matrix2, matrix2)
            matrix6 = torch.matmul(matrix4, matrix2)
            # matrix8 = torch.matmul(matrix4, matrix4)
            # matrix16 = torch.sparse.mm(matrix8, matrix8)
            # matrix17 = torch.matmul(matrix16, copy_A)
            # final_matrix = matrix17 + matrix16
            population_component_list.append(torch.count_nonzero(matrix6, dim=1))
        fitness_list = torch.max(torch.stack(population_component_list), dim=1).values.float()
        return fitness_list


class SixDSTController(BasicController):
    def __init__(self, path, pattern, data_loader, loops, crossover_rate, mutate_rate, pop_size, device, cutoff_tag="popGA_cutoff_", fit_side="min"):
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
        self.graph = data_loader.G
        self.nodes = None
        self.nodes_num = None
        self.mode = None

    def setup(self, data_loader, evaluator: SixDSTEvaluator):
        self.nodes = data_loader.nodes
        self.nodes_num = data_loader.nodes_num
        genes = data_loader.nodes.clone()
        print(f"Cutoff with {self.cutoff_tag}. Original genes: ", len(genes))
        genes = self.genes_cutoff(data_loader.G, genes)
        genes_index = torch.tensor([torch.nonzero(data_loader.nodes == node) for node in genes], device=self.device).squeeze()
        print(f"Cutoff finished. Genes after cutoff: ", len(genes))
        evaluator.genes_index = genes_index
        self.nodes = genes
        self.nodes_num = len(genes)
        return evaluator

    def calculate(self, max_generation, evaluator):
        best_PCG = []
        best_MCN = []
        best_genes = []
        time_list = []
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
                    new_population1 = population.clone()
                    new_population2 = body.selection(population, fitness_list)
                    crossover_population = body.crossover(new_population1, new_population2, self.crossover_rate, ONE)
                    mutation_population = body.mutation(crossover_population, self.mutate_rate, ONE)
                    # mutation_population = self._remove_repeat(mutation_population)
                    new_fitness_list = evaluator(mutation_population)
                    population, fitness_list, best_fitness_list = body.elitism(population, mutation_population, fitness_list, new_fitness_list, best_fitness_list)
                    if generation % 50 == 0 or (generation+1) == max_generation:
                        # population_copy = self._remove_repeat(population.clone())
                        critical_nodes = self.nodes[population[torch.argsort(fitness_list.clone())[0]]]
                        best_PCG.append(CNDTest(self.graph, critical_nodes))
                        best_MCN.append(min(fitness_list).item())
                        best_genes.append(critical_nodes)
                        end = time()
                        time_list.append(end - start)
                    pbar.set_postfix(fitness=min(fitness_list).item(), PCG=min(best_PCG))
                    pbar.update(1)
            top_index = best_PCG.index(min(best_PCG))
            print(f"Best PC(G): {best_PCG[top_index]}. Best connected num: {best_MCN[top_index]}.")
            self.save(self.dataset, best_genes[top_index], [best_PCG[top_index], best_MCN[top_index], time_list[-1]], time_list, "SixDST", bestPCG=best_PCG, bestMCN=best_MCN)
            print(f"Loop {loop} finished. Data saved in {self.path}...")

    def mp_calculate(self, rank, max_generation, evaluator, world_size):
        device = init_dist(rank, world_size)
        best_PCG = []
        best_MCN = []
        best_genes = []
        time_list = []
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
                        crossover_ONE = torch.ones((self.pop_size, self.budget), dtype=component_population.dtype, device=device)
                        crossover_population = body.crossover(new_population1, new_population2, self.crossover_rate, crossover_ONE)
                        body.pop_size = component_size
                    if rank == 0:
                        crossover_population = torch.stack(torch.split(crossover_population, component_size))
                    else:
                        crossover_population = torch.stack([torch.zeros((component_size, self.budget), dtype=component_population.dtype, device=device) for _ in range(world_size)])
                    dist.broadcast(crossover_population, src=0)
                    mutation_population = body.mutation(crossover_population[rank], self.mutate_rate, ONE)
                    # mutation_population = self._remove_repeat(mutation_population)
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
                        component_population = torch.zeros((component_size, self.budget), dtype=component_population.dtype, device=device)
                        component_fitness_list = torch.empty((component_size,), dtype=component_fitness_list.dtype, device=device)
                    dist.broadcast(component_population, src=0)
                    dist.broadcast(component_fitness_list, src=0)
                    if generation % 50 == 0 or (generation+1) == max_generation:
                        # component_population = self._remove_repeat(component_population.clone())
                        critical_nodes = nodes[component_population[torch.argsort(component_fitness_list.clone())[0]]]
                        best_PCG.append(CNDTest(self.graph, critical_nodes))
                        best_MCN.append(min(component_fitness_list).item())
                        best_genes.append(critical_nodes)
                        end = time()
                        time_list.append(end - start)
                    pbar.set_postfix(MCN=min(component_fitness_list).item(), PCG=min(best_PCG))
                    pbar.update(1)
            best_genes = torch.stack(best_genes)
            best_PCG = torch.tensor(best_PCG, device=device)
            best_MCN = torch.tensor(best_MCN, device=device)
            if rank == 0:
                whole_genes = [torch.zeros(best_genes.shape, dtype=best_genes.dtype, device=device) for _ in range(world_size)]
                whole_PCG = [torch.empty(best_PCG.shape, device=device) for _ in range(world_size)]
                whole_MCN = [torch.empty(best_MCN.shape, device=device) for _ in range(world_size)]
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
                self.save(self.dataset, whole_genes[top_index], [whole_PCG[top_index].item(), whole_MCN[top_index].item(), time_list[-1]], time_list, "SixDST", world_size=world_size, bestPCG=best_PCG, bestMCN=best_MCN)
                print(f"Loop {loop} finished. Data saved in {self.path}...")
        torch.cuda.empty_cache()
        dist.destroy_process_group()
        torch.cuda.synchronize()

    def save(self, dataset, gene, best_metric, time_list, method, world_size=1, **kwargs):
        save_path = self.path + dataset + '_crossover_rate_' + str(self.crossover_rate) + '_mutate_rate_' + str(self.mutate_rate) + f'_{method}.txt'
        with open(save_path, 'a+') as f:
            f.write(current_time())
            f.write(f"\nCurrent mode: {self.mode}. Current pop_size: {self.pop_size}. World size: {world_size}\n")
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

    def genes_cutoff(self, graph, genes):
        if self.cutoff_tag == "no_cutoff_":
            pass
        if self.cutoff_tag == "popGreedy_cutoff_":
            genes = self._pop_greedy_cutoff(graph, genes, 10)
        if self.cutoff_tag == "matrixGreedy_cutoff_":
            genes = self._matrix_greedy_cutoff(graph, genes)
        return genes

    def _matrix_greedy_cutoff(self, graph, genes):
        copy_A = torch.tensor(nx.to_numpy_array(graph, nodelist=list(graph.nodes())), device=self.device)
        IMatrix = torch.eye(len(copy_A)).to_sparse_coo()
        matrix2 = torch.matmul((copy_A + IMatrix), (copy_A + IMatrix))
        matrix4 = torch.matmul(matrix2, matrix2)
        matrix8 = torch.matmul(matrix4, matrix4)
        matrix9 = torch.matmul(matrix8, copy_A)
        final_matrix = matrix8 + matrix9
        component = torch.count_nonzero(final_matrix, dim=1)
        top_index = component.argsort()[len(component)-self.selected_genes_num // 2:]
        genes = self.nodes[top_index].tolist()
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


def SixDST(mode, max_generation, data_loader, controller: SixDSTController, evaluator, world_size):
    controller.mode = mode
    evaluator = controller.setup(data_loader=data_loader, evaluator=evaluator)
    if mode == "s" or mode == "sm":
        controller.calculate(max_generation=max_generation, evaluator=evaluator)
    elif mode == "m" or mode == "mnm":
        mp.spawn(controller.mp_calculate, args=(max_generation, deepcopy(evaluator), world_size), nprocs=world_size, join=True)
    else:
        raise ValueError(f"No such mode. Please choose ss, sm, ms or mm.")

