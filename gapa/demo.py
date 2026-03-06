from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator

import networkx as nx
import torch
import torch.nn as nn

from gapa.framework.body import Body
from gapa.framework.controller import CustomController
from gapa.framework.evaluator import BasicEvaluator
from gapa.utils.functions import CNDTest
from gapa.data_loader import DataLoader
from gapa.workflow import Algorithm, Monitor, Workflow


DEMO_GRAPHS = {
    "karate": lambda: nx.karate_club_graph(),
    "barbell": lambda: nx.barbell_graph(8, 2),
    "watts_strogatz": lambda: nx.watts_strogatz_graph(30, 4, 0.2, seed=42),
}


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_demo_data(
    graph_name: str = "karate",
    *,
    detection_rate: float = 0.12,
    selected_genes_rate: float = 0.5,
    device: str = "auto",
) -> DataLoader:
    if graph_name not in DEMO_GRAPHS:
        raise ValueError(f"unknown demo graph '{graph_name}'. choices={sorted(DEMO_GRAPHS)}")
    resolved_device = _resolve_device(device)
    graph = DEMO_GRAPHS[graph_name]()
    nodelist = sorted(graph.nodes())
    adjacency = torch.tensor(
        nx.to_numpy_array(graph, nodelist=nodelist),
        dtype=torch.float32,
        device=resolved_device,
    )
    normalized_graph = nx.from_numpy_array(adjacency.cpu().numpy())
    nodes = torch.tensor(list(normalized_graph.nodes()), device=resolved_device)
    nodes_num = len(nodes)
    budget = max(1, int(detection_rate * nodes_num))
    selected_genes_num = max(budget + 1, int(selected_genes_rate * nodes_num))
    return DataLoader(
        name=f"demo_{graph_name}",
        G=normalized_graph,
        A=adjacency,
        nodes=nodes,
        k=budget,
        selected_genes_num=selected_genes_num,
        device=resolved_device,
    )


class DemoSixDSTEvaluator(BasicEvaluator):
    """Small built-in evaluator used by the official quickstart demo."""

    def __init__(self, pop_size: int, adj: torch.Tensor, device: torch.device):
        super().__init__(pop_size=pop_size, device=device)
        self.AMatrix = adj.clone().to(device)
        self.IMatrix = torch.eye(len(adj), device=device).to_sparse_coo()
        self.nodes: torch.Tensor | None = None

    def forward(self, population: torch.Tensor) -> torch.Tensor:
        device = population.device
        adjacency = self.AMatrix.to(device)
        identity = self.IMatrix.to(device)
        node_map = self.nodes.to(device) if self.nodes is not None else None
        fitness_values = []
        for individual in population:
            working_adj = adjacency.clone()
            indices = node_map[individual.long()] if node_map is not None else individual.long()
            working_adj[indices, :] = 0
            working_adj[:, indices] = 0
            matrix2 = torch.matmul(working_adj + identity, working_adj + identity)
            matrix4 = torch.matmul(matrix2, matrix2)
            matrix6 = torch.matmul(matrix4, matrix2)
            fitness_values.append(torch.count_nonzero(matrix6, dim=1).max())
        return torch.stack(fitness_values).float()


class DemoSixDSTController(CustomController):
    """Minimal controller for the official quickstart demo."""

    def __init__(
        self,
        *,
        budget: int,
        pop_size: int,
        mode: str,
        device: torch.device,
        crossover_rate: float = 0.6,
        mutate_rate: float = 0.2,
    ):
        super().__init__(
            budget=budget,
            pop_size=pop_size,
            mode=mode,
            side="min",
            num_to_eval=1,
            device=device,
            save=False,
        )
        self.crossover_rate = crossover_rate
        self.mutate_rate = mutate_rate
        self.graph = None
        self.nodes = None

    def setup(self, data_loader, evaluator, **kwargs):
        evaluator = super().setup(data_loader=data_loader, evaluator=evaluator, **kwargs)
        self.graph = data_loader.G.copy()
        self.nodes = data_loader.nodes.clone()
        evaluator.nodes = self.nodes
        return evaluator

    def Eval(self, generation, population, fitness_list, critical_genes):
        if self.nodes is None:
            return {"generation": generation}
        critical_nodes = self.nodes[critical_genes.long()]
        if critical_nodes.is_cuda:
            critical_nodes = critical_nodes.cpu()
        return {
            "generation": generation,
            "PCG": float(CNDTest(self.graph, critical_nodes)),
        }


class DemoSixDSTAlgorithm(Algorithm):
    """Package-local quickstart algorithm used by `python -m gapa demo`."""

    def __init__(
        self,
        *,
        pop_size: int = 16,
        crossover_rate: float = 0.6,
        mutate_rate: float = 0.2,
    ):
        super().__init__()
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutate_rate = mutate_rate
        self.side = "min"

    def create_evaluator(self, data_loader: DataLoader) -> nn.Module:
        return DemoSixDSTEvaluator(
            pop_size=self.pop_size,
            adj=data_loader.A,
            device=data_loader.device,
        )

    def create_controller(self, data_loader: DataLoader, mode: str, device: torch.device) -> CustomController:
        return DemoSixDSTController(
            budget=data_loader.k,
            pop_size=self.pop_size,
            mode=mode,
            device=device,
            crossover_rate=self.crossover_rate,
            mutate_rate=self.mutate_rate,
        )

    def create_body(self, data_loader: DataLoader, device: torch.device) -> Body:
        return Body(
            critical_num=data_loader.nodes_num,
            budget=data_loader.k,
            pop_size=self.pop_size,
            fit_side=self.side,
            device=device,
        )


@contextmanager
def _temporary_results_dir(path: Path | None) -> Iterator[None]:
    if path is None:
        yield
        return
    previous = os.environ.get("GAPA_RESULTS_DIR")
    os.environ["GAPA_RESULTS_DIR"] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("GAPA_RESULTS_DIR", None)
        else:
            os.environ["GAPA_RESULTS_DIR"] = previous


def run_demo(
    *,
    graph_name: str = "karate",
    generations: int = 5,
    pop_size: int = 16,
    mode: str = "s",
    device: str = "auto",
    output_dir: str | os.PathLike[str] | None = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    results_dir = Path(output_dir) if output_dir else Path.cwd() / "results" / "quickstart"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_loader = build_demo_data(graph_name=graph_name, device=device)
    algorithm = DemoSixDSTAlgorithm(pop_size=pop_size)
    monitor = Monitor(opt_direction="min")
    with _temporary_results_dir(results_dir):
        workflow = Workflow(
            algorithm,
            data_loader,
            monitor=monitor,
            mode=mode,
            verbose=verbose,
        )
        workflow.run(generations)
    run_context = monitor.export_all(pretty=False)
    report_meta = {}
    run_info = run_context.get("run")
    if isinstance(run_info, dict) and isinstance(run_info.get("reports"), dict):
        report_meta = dict(run_info["reports"])
    return {
        "graph": graph_name,
        "requested_mode": mode,
        "resolved_mode": getattr(workflow, "mode", mode),
        "generations": int(generations),
        "pop_size": int(pop_size),
        "best_fitness": monitor.best_fitness,
        "results_dir": str(results_dir),
        "report": report_meta,
    }


def build_demo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the official GAPA quickstart demo.")
    parser.add_argument("--graph", choices=sorted(DEMO_GRAPHS), default="karate")
    parser.add_argument("--mode", choices=["s", "sm", "m", "m_cpu"], default="s")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--pop-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--quiet", action="store_true", help="Reduce runtime logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_demo_parser()
    args = parser.parse_args(argv)
    result = run_demo(
        graph_name=args.graph,
        generations=args.generations,
        pop_size=args.pop_size,
        mode=args.mode,
        device=args.device,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
    report = result.get("report") if isinstance(result.get("report"), dict) else {}
    print(f"[GAPA] Demo graph: {result['graph']}")
    print(f"[GAPA] Requested mode: {result['requested_mode']}")
    print(f"[GAPA] Resolved mode: {result['resolved_mode']}")
    print(f"[GAPA] Best fitness: {result['best_fitness']}")
    if report.get("summary_path"):
        print(f"[GAPA] Summary report: {report['summary_path']}")
    else:
        print(f"[GAPA] Results dir: {result['results_dir']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
