from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PSOResult:
    best_position: np.ndarray
    best_fitness: float
    history: list[float]
    iterations: int


class ParticleSwarmOptimizer:
    def __init__(self, dim: int, cfg, seed: int):
        p = cfg["pso"]
        self.dim = int(dim)
        self.swarm_size = int(p["swarm_size"])
        self.max_iterations = int(p["max_iterations"])
        self.inertia = float(p["inertia"])
        self.c1 = float(p["cognitive"])
        self.c2 = float(p["social"])
        self.lb = float(p["lower_bound"])
        self.ub = float(p["upper_bound"])
        self.patience = int(p["early_stop_patience"])
        self.tol = float(p["early_stop_tolerance"])
        self.rng = np.random.default_rng(int(seed))

    def minimize(self, objective) -> PSOResult:
        x = self.rng.uniform(self.lb, self.ub, size=(self.swarm_size, self.dim))
        v = self.rng.normal(0, 0.2, size=(self.swarm_size, self.dim))

        fx = np.array([float(objective(row)) for row in x])
        pbest = x.copy()
        pbest_f = fx.copy()

        g_idx = int(np.argmin(pbest_f))
        gbest = pbest[g_idx].copy()
        gbest_f = float(pbest_f[g_idx])

        history = [gbest_f]
        no_improve = 0

        for it in range(1, self.max_iterations + 1):
            r1 = self.rng.random((self.swarm_size, self.dim))
            r2 = self.rng.random((self.swarm_size, self.dim))
            v = (
                self.inertia * v
                + self.c1 * r1 * (pbest - x)
                + self.c2 * r2 * (gbest - x)
            )
            x = np.clip(x + v, self.lb, self.ub)
            fx = np.array([float(objective(row)) for row in x])

            improved = fx < pbest_f
            pbest[improved] = x[improved]
            pbest_f[improved] = fx[improved]

            new_idx = int(np.argmin(pbest_f))
            new_best = float(pbest_f[new_idx])
            if gbest_f - new_best > self.tol:
                gbest_f = new_best
                gbest = pbest[new_idx].copy()
                no_improve = 0
            else:
                no_improve += 1

            history.append(gbest_f)
            if no_improve >= self.patience:
                return PSOResult(gbest, gbest_f, history, it)

        return PSOResult(gbest, gbest_f, history, self.max_iterations)
