from __future__ import annotations

import numpy as np


class MarkovEnergyPredictor:
    STATES = ("sleep", "idle", "rx", "tx")

    def __init__(self, state_costs: dict[str, float], alpha: float = 1.0):
        self.state_costs = dict(state_costs)
        self.alpha = float(alpha)
        self.counts = np.zeros((len(self.STATES), len(self.STATES)), dtype=float)
        self.index = {s: i for i, s in enumerate(self.STATES)}

    def update(self, previous: str, current: str):
        if previous not in self.index or current not in self.index:
            raise ValueError("Unknown Markov state")
        self.counts[self.index[previous], self.index[current]] += 1.0

    def transition_matrix(self) -> np.ndarray:
        smoothed = self.counts + self.alpha
        return smoothed / smoothed.sum(axis=1, keepdims=True)

    def predict_drain(self, current_state: str) -> float:
        idx = self.index[current_state]
        p = self.transition_matrix()[idx]
        costs = np.array([self.state_costs[s] for s in self.STATES], dtype=float)
        return float(p @ costs)

    def predict_batch(self, states) -> np.ndarray:
        return np.array([self.predict_drain(str(s)) for s in states], dtype=float)
