from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class GAIteration:
    generation: int
    best_fitness: float
    best_gene: torch.Tensor
    extra: Dict[str, Any] = field(default_factory=dict)


class GAStateObserver:
    """
    Lightweight observer to track GA progress.

    - record() once per generation
    - export curves or raw dicts for downstream plotting
    """

    def __init__(self) -> None:
        self.history: List[GAIteration] = []

    def record(
        self,
        generation: int,
        fitness_list: torch.Tensor,
        best_gene: torch.Tensor,
        extra: Optional[Dict[str, Any]] = None,
        side: str = "max",
    ) -> None:
        best_fitness = torch.max(fitness_list) if side == "max" else torch.min(fitness_list)
        self.history.append(
            GAIteration(
                generation=generation,
                best_fitness=float(best_fitness.item()),
                best_gene=best_gene.detach().cpu(),
                extra=extra or {},
            )
        )

    def to_curves(self) -> Dict[str, List[float]]:
        gens = [item.generation for item in self.history]
        fits = [item.best_fitness for item in self.history]
        return {"generation": gens, "best_fitness": fits}

    def to_dict(self) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for item in self.history:
            payload.append(
                {
                    "generation": item.generation,
                    "best_fitness": item.best_fitness,
                    "best_gene": item.best_gene.tolist(),
                    "extra": item.extra,
                }
            )
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
