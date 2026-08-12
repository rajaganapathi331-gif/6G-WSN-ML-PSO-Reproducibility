from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .clustering import (
    leach_cluster_heads, leach_c_cluster_heads, proposed_cluster_heads, assign_to_heads
)
from .routing import AdaptiveRouter


SUPPORTED_PROTOCOLS = {
    "leach", "leach_c", "greedy", "pso_only",
    "markov_pso", "ml_pso", "proposed"
}


@dataclass
class ProtocolState:
    heads: np.ndarray
    assignment: np.ndarray
    router: AdaptiveRouter | None


class ProtocolController:
    def __init__(self, name: str, cfg, seed: int):
        if name not in SUPPORTED_PROTOCOLS:
            raise ValueError(f"Unsupported protocol: {name}")
        self.name = name
        self.cfg = cfg
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed + 991)
        self.state = ProtocolState(
            heads=np.array([], dtype=int),
            assignment=np.array([], dtype=int),
            router=None if name in {"leach", "leach_c"} else AdaptiveRouter(cfg, self.seed + 777),
        )

    @property
    def uses_markov(self):
        return self.name in {"markov_pso", "proposed"}

    @property
    def uses_ml(self):
        return self.name in {"ml_pso", "proposed"}

    @property
    def uses_pso(self):
        return self.name in {"pso_only", "markov_pso", "ml_pso", "proposed"}

    def refresh_clusters(
        self, round_idx, positions, energy, alive, predicted_drain,
        link_quality, traffic, sink
    ):
        fraction = float(self.cfg["network"]["cluster_head_fraction"])
        initial = float(self.cfg["network"]["initial_energy_j"])

        if self.name == "leach":
            heads = leach_cluster_heads(alive, self.rng, fraction)
        elif self.name == "leach_c":
            heads = leach_c_cluster_heads(
                positions, energy, alive, fraction, self.seed + int(round_idx)
            )
        elif self.name in {"greedy", "pso_only"}:
            zeros = np.zeros_like(predicted_drain)
            heads = proposed_cluster_heads(
                positions, energy, initial, alive, zeros, link_quality, traffic,
                sink, fraction, self.seed + int(round_idx), self.cfg["proposed_weights"]
            )
        else:
            heads = proposed_cluster_heads(
                positions, energy, initial, alive, predicted_drain, link_quality, traffic,
                sink, fraction, self.seed + int(round_idx), self.cfg["proposed_weights"]
            )

        self.state.heads = heads
        self.state.assignment = assign_to_heads(positions, alive, heads)

    def refresh_router(
        self, positions, energy, alive, predicted_drain, link_quality, traffic, sink
    ):
        if self.state.router is None:
            return

        if self.name == "greedy":
            # Fixed interpretable weights, no PSO.
            self.state.router.weights = np.array([0.30, 0.25, 0.0, 0.20, 0.10, 0.15], dtype=float)
            self.state.router.weights /= self.state.router.weights.sum()
            return

        if self.name == "pso_only":
            self.state.router.optimize(
                positions, energy, alive, predicted_drain * 0.0,
                link_quality, traffic, sink,
                use_energy=False, use_prediction=False
            )
        elif self.name == "markov_pso":
            self.state.router.optimize(
                positions, energy, alive, predicted_drain,
                link_quality, traffic, sink,
                use_energy=True, use_prediction=True
            )
        elif self.name == "ml_pso":
            self.state.router.optimize(
                positions, energy, alive, predicted_drain,
                link_quality, traffic, sink,
                use_energy=True, use_prediction=True
            )
        elif self.name == "proposed":
            self.state.router.optimize(
                positions, energy, alive, predicted_drain,
                link_quality, traffic, sink,
                use_energy=True, use_prediction=True
            )
