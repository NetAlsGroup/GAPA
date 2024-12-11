import torch
import numpy as np


class BasicBody:
    def __init__(self, **kwargs):
        pass

    def init_population(self, **kwargs):
        pass

    def selection(self, **kwargs):
        pass

    def mutation(self, **kwargs):
        pass

    def crossover(self, **kwargs):
        pass

    def elitism(self, **kwargs):
        pass


class Body(BasicBody):
    def __init__(self, critical_num, budget, pop_size, fit_side, device):
        super().__init__()
        self.critical_num = critical_num
        self.budget = budget
        self.pop_size = pop_size
        self.fit_side = fit_side
        self.device = device

    def init_population(self, **kwargs):
        init_population = torch.randperm(self.critical_num, device=self.device)[:self.budget]
        for i in range(self.pop_size - 1):
            init_population = torch.vstack((init_population, torch.randperm(self.critical_num, device=self.device)[:self.budget]))
        ONE = torch.ones((self.pop_size, self.budget), dtype=torch.int, device=self.device)
        return ONE, init_population

    def selection(self, population, fitness_list):
        copy_pop = population.clone()
        normalize_fit = fitness_list / fitness_list.sum()
        normalize_fit[normalize_fit < 0] = 0
        samples = torch.multinomial(normalize_fit, len(normalize_fit), replacement=True)
        return copy_pop[samples]

    def mutation(self, crossover_population, mutate_rate, one):
        mutation_matrix = torch.tensor(np.random.choice([0, 1], size=(self.pop_size, self.budget), p=[1 - mutate_rate, mutate_rate]), device=self.device)
        mutation_population = crossover_population * (one - mutation_matrix) + torch.randint(0, self.critical_num, size=(self.pop_size, self.budget), device=self.device) * mutation_matrix
        return mutation_population

    def crossover(self, new_population1, new_population2, crossover_rate, one):
        crossover_matrix = torch.tensor(np.random.choice([0, 1], size=(self.pop_size, self.budget), p=[1 - crossover_rate, crossover_rate]), device=self.device)
        crossover_population = new_population1 * (one - crossover_matrix) + new_population2 * crossover_matrix
        return crossover_population

    def elitism(self, population, mutation_population, fitness_list, new_fitness_list, best_fitness_list):
        stack_population = torch.vstack((population, mutation_population))
        stack_fitness_list = torch.hstack((fitness_list, new_fitness_list))
        top_index = None
        if self.fit_side == 'max':
            top_index = torch.argsort(stack_fitness_list)[len(stack_fitness_list) - self.pop_size:]
        elif self.fit_side == 'min':
            top_index = torch.argsort(stack_fitness_list)[:self.pop_size]
        population = stack_population[top_index]
        fitness_list = stack_fitness_list[top_index]
        if self.fit_side == 'max':
            best_fitness_list = torch.hstack((best_fitness_list, max(fitness_list)))
        elif self.fit_side == 'min':
            best_fitness_list = torch.hstack((best_fitness_list, min(fitness_list)))
        else:
            raise ValueError(f"No such fit side: {self.fit_side}")
        return population, fitness_list, best_fitness_list
