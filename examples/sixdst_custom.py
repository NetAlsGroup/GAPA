"""
SixDST Algorithm - Custom Implementation
========================================
Demonstrates how to implement an algorithm using the unified GAPA workflow.

This algorithm uses the same core engine as the frontend (Start, CustomController),
ensuring consistent behavior across all execution contexts.

Usage:
    from gapa.workflow import Workflow, load_dataset, Monitor
    from examples.sixdst_custom import SixDSTAlgorithm
    
    data = load_dataset("ForestFire_n500")
    algo = SixDSTAlgorithm(pop_size=80)
    workflow = Workflow(algo, data, mode="m")
    workflow.run(1000)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import networkx as nx
from collections import Counter
from typing import Optional, Dict, Any

from gapa.workflow import Algorithm, DataLoader
from gapa.framework.body import Body
from gapa.framework.controller import CustomController
from gapa.framework.evaluator import BasicEvaluator
from gapa.utils.functions import CNDTest


# =============================================================================
# SixDST Evaluator (Fitness Function)
# =============================================================================

class SixDSTEvaluator(BasicEvaluator):
    """
    SixDST Evaluator: 6-hop matrix power fitness calculation.
    
    This is the same evaluator used by the frontend/backend.
    """
    
    def __init__(self, pop_size: int, adj: torch.Tensor, device: torch.device):
        super().__init__(pop_size=pop_size, device=device)
        self.AMatrix = adj.clone().to(device)
        self.IMatrix = torch.eye(len(adj), device=device).to_sparse_coo()
        self.nodes = None
    
    def forward(self, population: torch.Tensor) -> torch.Tensor:
        device = population.device
        AMatrix = self.AMatrix.to(device)
        IMatrix = self.IMatrix.to(device)
        nodes = self.nodes.to(device) if self.nodes is not None else None
        
        fitness_list = []
        for pop in population:
            copy_A = AMatrix.clone()
            if nodes is not None:
                node_indices = nodes[pop.long()]
            else:
                node_indices = pop.long()
            
            copy_A[node_indices, :] = 0
            copy_A[:, node_indices] = 0
            
            matrix2 = torch.matmul((copy_A + IMatrix), (copy_A + IMatrix))
            matrix4 = torch.matmul(matrix2, matrix2)
            matrix6 = torch.matmul(matrix4, matrix2)
            
            fitness_list.append(torch.count_nonzero(matrix6, dim=1).max())
        
        return torch.stack(fitness_list).float()


# =============================================================================
# SixDST Controller (GA Workflow)
# =============================================================================

class SixDSTController(CustomController):
    """
    SixDST Controller: Custom GA workflow with popGreedy cutoff.
    
    Inherits from CustomController to reuse the existing calculate/mp_calculate
    logic that handles all execution modes (s, sm, m, mnm).
    """
    
    def __init__(
        self,
        budget: int,
        pop_size: int,
        mode: str,
        device: torch.device,
        crossover_rate: float = 0.6,
        mutate_rate: float = 0.2,
        cutoff_enabled: bool = True,
        cutoff_rounds: int = 10,
        **kwargs
    ):
        super().__init__(
            side="min",
            mode=mode,
            budget=budget,
            pop_size=pop_size,
            num_to_eval=50,
            device=device,
            **kwargs
        )
        self.crossover_rate = crossover_rate
        self.mutate_rate = mutate_rate
        self.cutoff_enabled = cutoff_enabled
        self.cutoff_rounds = cutoff_rounds
        
        self.graph = None
        self.nodes = None
        self.nodes_num = None
        self.selected_genes_num = None
    
    def setup(self, data_loader, evaluator, **kwargs):
        """Setup with popGreedy cutoff."""
        self.dataset = data_loader.name
        self.graph = data_loader.G.copy()
        self.nodes = data_loader.nodes.clone()
        self.nodes_num = data_loader.nodes_num
        self.selected_genes_num = getattr(data_loader, 'selected_genes_num', self.nodes_num)
        
        if self.cutoff_enabled:
            print(f"[SixDST] Applying popGreedy cutoff (original: {len(self.nodes)} genes)")
            self.nodes = self._pop_greedy_cutoff(
                self.graph, 
                self.nodes, 
                rounds=self.cutoff_rounds
            )
            self.nodes_num = len(self.nodes)
            print(f"[SixDST] Genes after cutoff: {self.nodes_num}")
        
        evaluator.nodes = self.nodes
        return evaluator
    
    def Eval(self, generation, population, fitness_list, critical_genes):
        """Compute domain-specific metrics."""
        if self.nodes is not None:
            # Ensure device compatibility for multi-GPU (M mode)
            nodes_device = self.nodes.device
            critical_genes_on_device = critical_genes.to(nodes_device)
            critical_nodes = self.nodes[critical_genes_on_device.long()]
        else:
            critical_nodes = critical_genes
        
        # CNDTest expects CPU tensors
        if critical_nodes.is_cuda:
            critical_nodes = critical_nodes.cpu()
        
        pcg = CNDTest(self.graph, critical_nodes)
        
        return {
            "generation": generation,
            "PCG": float(pcg),
        }
    
    def _pop_greedy_cutoff(self, graph, genes, rounds: int = 10):
        """PopGreedy gene pool cutoff."""
        greedy_indi = []
        
        for _ in range(rounds):
            temp_fitness = []
            temp_population = []
            
            for _ in range(self.pop_size):
                temp_genes = torch.randperm(len(genes), device=self.device)[:self.budget]
                temp_population.append(temp_genes)
                temp_fitness.append(CNDTest(graph, genes[temp_genes]))
            
            best_idx = torch.tensor(temp_fitness, device=self.device).argmin()
            greedy_indi.append(temp_population[best_idx].tolist())
        
        all_genes = []
        for indi in greedy_indi:
            all_genes.extend(indi)
        
        gene_counts = Counter(all_genes)
        top_genes = [g for g, _ in gene_counts.most_common(self.selected_genes_num // 2)]
        
        result_genes = genes[top_genes].tolist()
        degree = nx.degree_centrality(graph)
        degree_sorted = sorted(degree.items(), key=lambda x: x[1], reverse=True)
        
        for node, _ in degree_sorted:
            if len(result_genes) >= self.selected_genes_num:
                break
            if node not in result_genes:
                result_genes.append(node)
        
        return torch.tensor(result_genes, device=self.device)


# =============================================================================
# SixDST Algorithm (Unified Interface)
# =============================================================================

class SixDSTAlgorithm(Algorithm):
    """
    SixDST Algorithm for Critical Node Detection.
    
    This is the unified interface that works with both script and frontend.
    
    Args:
        pop_size: Population size (default: 80)
        crossover_rate: Crossover probability (default: 0.6)
        mutate_rate: Mutation probability (default: 0.2)
        cutoff_enabled: Whether to apply popGreedy cutoff (default: True)
        cutoff_rounds: Number of cutoff rounds (default: 10)
    
    Example:
        >>> data = load_dataset("ForestFire_n500")
        >>> algo = SixDSTAlgorithm(pop_size=80)
        >>> workflow = Workflow(algo, data, mode="mnm", auto_select=True)
        >>> workflow.run(1000)
    """
    
    def __init__(
        self,
        pop_size: int = 80,
        crossover_rate: float = 0.6,
        mutate_rate: float = 0.2,
        cutoff_enabled: bool = True,
        cutoff_rounds: int = 10,
    ):
        super().__init__()
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutate_rate = mutate_rate
        self.cutoff_enabled = cutoff_enabled
        self.cutoff_rounds = cutoff_rounds
        self.side = "min"
    
    def create_evaluator(self, data_loader: DataLoader) -> nn.Module:
        """Create the SixDST evaluator."""
        return SixDSTEvaluator(
            pop_size=self.pop_size,
            adj=data_loader.A,
            device=data_loader.device,
        )
    
    def create_controller(
        self, 
        data_loader: DataLoader, 
        mode: str, 
        device: torch.device
    ) -> CustomController:
        """Create the SixDST controller."""
        return SixDSTController(
            budget=data_loader.k,
            pop_size=self.pop_size,
            mode=mode,
            device=device,
            crossover_rate=self.crossover_rate,
            mutate_rate=self.mutate_rate,
            cutoff_enabled=self.cutoff_enabled,
            cutoff_rounds=self.cutoff_rounds,
        )
    
    def create_body(self, data_loader: DataLoader, device: torch.device) -> Body:
        """
        Create the Body (GA operators).
        
        Note: Uses selected_genes_num (gene pool size after cutoff) as critical_num
        to ensure population indices stay within bounds.
        """
        # Use selected_genes_num as critical_num since cutoff will reduce the gene pool
        # If cutoff is disabled, this still works because selected_genes_num <= nodes_num
        critical_num = data_loader.selected_genes_num if self.cutoff_enabled else data_loader.nodes_num
        
        return Body(
            critical_num=critical_num,
            budget=data_loader.k,
            pop_size=self.pop_size,
            fit_side=self.side,
            device=device,
        )
