from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .pso import ParticleSwarmOptimizer


def softmax(x):
    x = np.asarray(x, dtype=float)
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


@dataclass
class RouteResult:
    path: list[int]
    success: bool
    estimated_latency_ms: float
    mean_reliability: float


class AdaptiveRouter:
    """
    PSO optimizes six interpretable routing-policy weights:
    distance, residual-energy risk, predicted drain, unreliability,
    latency, and insufficient sink progress.
    """

    DIM = 6

    def __init__(self, cfg, seed: int):
        self.cfg = cfg
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.weights = np.ones(self.DIM) / self.DIM
        self.last_history = []
        self.last_fitness = np.nan

    def _neighbors(self, source, positions, alive, sink, allow_sink=True):
        radius = float(self.cfg["network"]["communication_radius_m"])
        src = positions[source]
        idx = np.flatnonzero(alive)
        idx = idx[idx != source]
        d = np.linalg.norm(positions[idx] - src, axis=1)
        neighbors = idx[d <= radius].tolist()

        sink_dist = float(np.linalg.norm(sink - src))
        if allow_sink and sink_dist <= radius:
            neighbors.append(-1)
        return neighbors

    def _edge_features(
        self, u, v, positions, energy, initial_energy, predicted_drain,
        link_quality, traffic, sink
    ):
        radius = float(self.cfg["network"]["communication_radius_m"])
        qos = self.cfg["qos"]

        p_u = positions[u]
        p_v = sink if v == -1 else positions[v]
        d_uv = float(np.linalg.norm(p_v - p_u))
        d_curr = max(float(np.linalg.norm(sink - p_u)), 1e-9)
        d_next = 0.0 if v == -1 else float(np.linalg.norm(sink - p_v))

        distance = min(d_uv / max(radius, 1e-9), 3.0)
        if v == -1:
            energy_risk = 0.0
            drain = 0.0
            lq = 1.0
            tr = 0.0
        else:
            energy_risk = 1.0 - np.clip(energy[v] / initial_energy, 0.0, 1.0)
            max_drain = max(float(np.max(predicted_drain)), 1e-12)
            drain = float(predicted_drain[v] / max_drain)
            lq = float(np.clip(link_quality[v], 0.0, 1.0))
            tr = float(np.clip(traffic[v], 0.0, 1.0))

        # Reliability is degraded by distance and local traffic.
        reliability = np.clip(lq * np.exp(-0.35 * distance) * (1.0 - 0.15 * tr), 0.01, 1.0)
        unreliability = 1.0 - reliability

        tx_ms = 1000.0 * int(self.cfg["network"]["packet_bits"]) / float(qos["nominal_bitrate_bps"])
        prop_ms = 1000.0 * d_uv / float(qos["propagation_speed_m_per_s"])
        edge_ms = tx_ms + prop_ms + float(qos["edge_processing_delay_ms"])
        latency = edge_ms / max(float(qos["latency_budget_ms"]), 1e-9)

        progress_ratio = d_next / d_curr
        progress_penalty = max(0.0, progress_ratio - 0.92)

        return np.array(
            [distance, energy_risk, drain, unreliability, latency, progress_penalty],
            dtype=float
        ), float(edge_ms), float(reliability)

    def optimize(
        self, positions, energy, alive, predicted_drain, link_quality,
        traffic, sink, use_energy=True, use_prediction=True
    ):
        alive_idx = np.flatnonzero(alive)
        if len(alive_idx) <= 1:
            return self.weights

        sample_n = min(40, len(alive_idx))
        sources = self.rng.choice(alive_idx, size=sample_n, replace=False)
        initial_energy = float(self.cfg["network"]["initial_energy_j"])
        min_rel = float(self.cfg["qos"]["min_link_reliability"])

        def objective(raw):
            w = softmax(raw)
            if not use_energy:
                w = w.copy()
                w[1] = 0.0
                w = w / max(w.sum(), 1e-12)
            if not use_prediction:
                w = w.copy()
                w[2] = 0.0
                w = w / max(w.sum(), 1e-12)

            total = 0.0
            failures = 0
            qos_penalty = 0.0

            for u in sources:
                nbrs = self._neighbors(int(u), positions, alive, sink)
                if not nbrs:
                    failures += 1
                    continue
                scores = []
                reliabilities = []
                for v in nbrs:
                    phi, _, rel = self._edge_features(
                        int(u), int(v), positions, energy, initial_energy,
                        predicted_drain, link_quality, traffic, sink
                    )
                    scores.append(float(w @ phi))
                    reliabilities.append(rel)
                j = int(np.argmin(scores))
                total += scores[j]
                if reliabilities[j] < min_rel:
                    qos_penalty += (min_rel - reliabilities[j]) ** 2

            mean = total / max(1, len(sources) - failures)
            return mean + 1.5 * failures / len(sources) + 2.0 * qos_penalty / len(sources)

        pso = ParticleSwarmOptimizer(self.DIM, self.cfg, self.seed + int(np.sum(alive)))
        res = pso.minimize(objective)
        self.weights = softmax(res.best_position)
        if not use_energy:
            self.weights[1] = 0.0
            self.weights /= self.weights.sum()
        if not use_prediction:
            self.weights[2] = 0.0
            self.weights /= self.weights.sum()
        self.last_history = res.history
        self.last_fitness = res.best_fitness
        return self.weights

    def route(
        self, source, positions, energy, alive, predicted_drain,
        link_quality, traffic, sink, max_hops=40
    ) -> RouteResult:
        initial_energy = float(self.cfg["network"]["initial_energy_j"])
        current = int(source)
        visited = {current}
        path = [current]
        total_latency = 0.0
        reliabilities = []

        for _ in range(max_hops):
            if float(np.linalg.norm(positions[current] - sink)) <= float(self.cfg["network"]["communication_radius_m"]):
                phi, edge_ms, rel = self._edge_features(
                    current, -1, positions, energy, initial_energy,
                    predicted_drain, link_quality, traffic, sink
                )
                path.append(-1)
                total_latency += edge_ms
                reliabilities.append(rel)
                return RouteResult(path, True, total_latency, float(np.mean(reliabilities)))

            nbrs = [n for n in self._neighbors(current, positions, alive, sink) if n == -1 or n not in visited]
            if not nbrs:
                return RouteResult(path, False, total_latency, float(np.mean(reliabilities)) if reliabilities else 0.0)

            scored = []
            for v in nbrs:
                phi, edge_ms, rel = self._edge_features(
                    current, int(v), positions, energy, initial_energy,
                    predicted_drain, link_quality, traffic, sink
                )
                # Reject clear backward motion unless no better neighbor exists.
                scored.append((float(self.weights @ phi), int(v), edge_ms, rel))

            scored.sort(key=lambda x: x[0])
            _, nxt, edge_ms, rel = scored[0]
            total_latency += edge_ms
            reliabilities.append(rel)

            if nxt == -1:
                path.append(-1)
                return RouteResult(path, True, total_latency, float(np.mean(reliabilities)))

            current = nxt
            visited.add(current)
            path.append(current)

        return RouteResult(path, False, total_latency, float(np.mean(reliabilities)) if reliabilities else 0.0)
