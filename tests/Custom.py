import torch
from gafama.framework.evaluator import BasicEvaluator
from gafama.framework.controller import CustomController
from gafama.utils.functions import CNDTest


class ExampleEvaluator(BasicEvaluator):
    def __init__(self, pop_size, adj: torch.Tensor, device):
        super().__init__(
            pop_size=pop_size,
            device=device
        )
        self.AMatrix = adj.clone()
        self.IMatrix = torch.eye(len(adj)).to_sparse_coo()

    def forward(self, population):
        population_component_list = []
        AMatrix = self.AMatrix.to(population.device)
        IMatrix = self.IMatrix.to(population.device)
        for i in range(len(population)):
            copy_A = AMatrix.clone()
            copy_A[population[i], :] = 0
            copy_A[:, population[i]] = 0
            matrix2 = torch.matmul((copy_A + IMatrix), (copy_A + IMatrix))
            matrix4 = torch.matmul(matrix2, matrix2)
            matrix6 = torch.matmul(matrix4, matrix2)
            population_component_list.append(torch.count_nonzero(matrix6, dim=1))
        fitness_list = torch.max(torch.stack(population_component_list), dim=1).values.float()
        return fitness_list


class ExampleController(CustomController):
    def __init__(self, budget, pop_size, pc, pm, side, mode, num_to_eval, device):
        super().__init__(
            side=side,
            mode=mode,
            budget=budget,
            pop_size=pop_size,
            num_to_eval=num_to_eval,
            device=device
        )
        self.crossover_rate = pc
        self.mutate_rate = pm
        self.graph = None

    def setup(self, data_loader, evaluator, **kwargs):
        # CutOff gene pool here
        self.graph = data_loader.G.copy()
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
        pcg = CNDTest(self.graph, critical_genes)
        return {
            "PCG": pcg
        }
