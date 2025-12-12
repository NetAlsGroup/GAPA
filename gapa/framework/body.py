from __future__ import annotations

import torch
from torch import Tensor


class BasicBody:
    """Minimal interface for genetic algorithm operators."""

    def __init__(self, **_: object) -> None:
        pass

    def init_population(self, **_: object):
        raise NotImplementedError

    def selection(self, **_: object):
        raise NotImplementedError

    def mutation(self, **_: object):
        raise NotImplementedError

    def crossover(self, **_: object):
        raise NotImplementedError

    def elitism(self, **_: object):
        raise NotImplementedError


class Body(BasicBody):
    """Genetic algorithm operators implemented with Torch for GPU acceleration."""

    def __init__(self, critical_num: int, budget: int, pop_size: int, fit_side: str, device: torch.device):
        super().__init__()
        self.critical_num = critical_num
        self.budget = budget
        self.pop_size = pop_size
        self.fit_side = fit_side
        self.device = device

    def init_population(self, **_: object):
        """Randomly initialize a population with unique candidates per row."""
        population = torch.stack(
            [torch.randperm(self.critical_num, device=self.device)[: self.budget] for _ in range(self.pop_size)]
        )
        ones = torch.ones((self.pop_size, self.budget), dtype=torch.int64, device=self.device)
        return ones, population

    def selection(self, population: Tensor, fitness_list: Tensor) -> Tensor:
        """Roulette wheel selection with safe fallback when fitness collapses."""
        weights = fitness_list.clone()
        if self.fit_side == "min":
            weights = -weights
        weights = weights - weights.min()
        weights = torch.clamp(weights, min=0)
        if torch.sum(weights) <= 0:
            weights = torch.ones_like(weights, device=population.device)
        normalize_fit = weights / weights.sum()
        samples = torch.multinomial(normalize_fit, len(normalize_fit), replacement=True)
        return population[samples]

    def mutation(self, crossover_population: Tensor, mutate_rate: float, _: Tensor | None = None) -> Tensor:
        """Mutate genes with a Bernoulli mask generated on-device."""
        mutation_mask = torch.rand((self.pop_size, self.budget), device=self.device) < mutate_rate
        mutated_genes = torch.randint(0, self.critical_num, size=(self.pop_size, self.budget), device=self.device)
        return torch.where(mutation_mask, mutated_genes, crossover_population)

    def crossover(self, new_population1: Tensor, new_population2: Tensor, crossover_rate: float, _: Tensor | None = None) -> Tensor:
        """Uniform crossover between two parent populations."""
        crossover_mask = torch.rand((self.pop_size, self.budget), device=self.device) < crossover_rate
        return torch.where(crossover_mask, new_population2, new_population1)

    def elitism(self, population: Tensor, mutation_population: Tensor, fitness_list: Tensor, new_fitness_list: Tensor):
        """Keep the best individuals from parents + offspring."""
        stack_population = torch.vstack((population, mutation_population))
        stack_fitness_list = torch.hstack((fitness_list, new_fitness_list))
        descending = self.fit_side == "max"
        top_index = torch.argsort(stack_fitness_list, descending=descending)[: self.pop_size]
        return stack_population[top_index], stack_fitness_list[top_index]
