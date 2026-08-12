from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .energy import RadioEnergyModel
from .markov import MarkovEnergyPredictor
from .ml import EnergyMLPredictor, FEATURES, generate_telemetry
from .protocols import ProtocolController
from .metrics import jain_fairness, run_summary


@dataclass
class SimulationOutput:
    round_metrics: pd.DataFrame
    run_summary: dict
    ml_validation: dict | None
    pso_convergence: pd.DataFrame


class WSNSimulator:
    def __init__(self, cfg, protocol_name: str, seed: int):
        self.cfg = cfg
        self.protocol_name = protocol_name
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        ncfg = cfg["network"]
        self.n = int(ncfg["n_nodes"])
        self.width = float(ncfg["width_m"])
        self.height = float(ncfg["height_m"])
        self.sink = np.array([float(ncfg["sink_x_m"]), float(ncfg["sink_y_m"])])
        self.initial_energy = float(ncfg["initial_energy_j"])
        self.packet_bits = int(ncfg["packet_bits"])

        self.positions = np.column_stack([
            self.rng.uniform(0, self.width, self.n),
            self.rng.uniform(0, self.height, self.n)
        ])
        self.energy = np.full(self.n, self.initial_energy, dtype=float)
        self.alive = np.ones(self.n, dtype=bool)
        self.state = np.array(["idle"] * self.n, dtype=object)
        self.previous_state = self.state.copy()
        self.rolling_drain = np.zeros((self.n, 5), dtype=float)

        self.energy_model = RadioEnergyModel.from_config(cfg)
        self.markov = MarkovEnergyPredictor(
            self.energy_model.state_costs(self.packet_bits),
            alpha=float(cfg["markov"]["laplace_alpha"]),
        )
        self.controller = ProtocolController(protocol_name, cfg, seed)

        self.ml_model = None
        self.ml_validation = None
        if self.controller.uses_ml:
            telemetry_markov = MarkovEnergyPredictor(
                self.energy_model.state_costs(self.packet_bits),
                alpha=float(cfg["markov"]["laplace_alpha"]),
            )
            telemetry = generate_telemetry(
                cfg, seed + 17, self.energy_model, telemetry_markov
            )
            self.ml_model = EnergyMLPredictor(cfg, seed + 19)
            val = self.ml_model.fit(telemetry)
            self.ml_validation = {
                "mae": val.mae,
                "rmse": val.rmse,
                "r2": val.r2,
                "train_size": val.train_size,
                "validation_size": val.validation_size,
            }

        self._distance_matrix = np.linalg.norm(
            self.positions[:, None, :] - self.positions[None, :, :], axis=2
        )
        radius = float(ncfg["communication_radius_m"])
        self.degree = ((self._distance_matrix <= radius) & (self._distance_matrix > 0)).sum(axis=1)

    def _environment(self, round_idx):
        env = self.cfg["environment"]
        rounds = max(int(self.cfg["network"]["rounds"]), 2)
        phase = 2 * np.pi * round_idx / rounds
        temperature = float(env["base_temperature_c"]) + float(env["temperature_amplitude_c"]) * np.sin(phase)
        traffic = self.rng.uniform(float(env["traffic_min"]), float(env["traffic_max"]), self.n)

        d_sink = np.linalg.norm(self.positions - self.sink, axis=1)
        link_quality = np.clip(
            1.0
            - 0.30 * np.minimum(d_sink / max(self.width, self.height), 1.0)
            + self.rng.normal(0, float(env["link_noise_std"]), self.n),
            0.05, 1.0
        )
        return temperature, traffic, link_quality

    def _update_states(self, traffic):
        new_state = np.empty(self.n, dtype=object)
        for i in range(self.n):
            if not self.alive[i]:
                new_state[i] = "sleep"
                continue
            p_tx = np.clip(0.15 + 0.45 * traffic[i], 0.10, 0.65)
            p_sleep = np.clip(0.30 - 0.20 * traffic[i], 0.05, 0.30)
            p_idle = 0.25
            p_rx = max(0.05, 1.0 - p_tx - p_sleep - p_idle)
            p = np.array([p_sleep, p_idle, p_rx, p_tx], dtype=float)
            p /= p.sum()
            new_state[i] = self.rng.choice(["sleep", "idle", "rx", "tx"], p=p)

        for i in range(self.n):
            self.markov.update(str(self.previous_state[i]), str(new_state[i]))
        self.previous_state = self.state.copy()
        self.state = new_state

    def _feature_frame(self, temperature, traffic, link_quality, markov_drain):
        d_sink = np.linalg.norm(self.positions - self.sink, axis=1)
        roll_mean = self.rolling_drain.mean(axis=1)
        roll_std = self.rolling_drain.std(axis=1)
        state_code = np.array([{"sleep": 0, "idle": 1, "rx": 2, "tx": 3}[str(s)] for s in self.state])

        return pd.DataFrame({
            "residual_energy": np.clip(self.energy / self.initial_energy, 0.0, 1.0),
            "distance_to_sink": d_sink / max(self.width, self.height),
            "degree": self.degree / max(1, self.n - 1),
            "traffic": traffic,
            "temperature": temperature,
            "link_quality": link_quality,
            "rolling_drain_mean": roll_mean,
            "rolling_drain_std": roll_std,
            "markov_predicted_drain": markov_drain,
            "state_code": state_code,
        })

    def _predicted_drain(self, temperature, traffic, link_quality):
        markov_drain = self.markov.predict_batch(self.state)
        if self.controller.uses_ml:
            frame = self._feature_frame(temperature, traffic, link_quality, markov_drain)
            pred = np.maximum(self.ml_model.predict(frame), 1e-12)
            if self.protocol_name == "proposed":
                # Hybrid Markov-ML prediction reduces sensitivity to one estimator.
                return 0.35 * markov_drain + 0.65 * pred
            return pred
        if self.controller.uses_markov:
            return markov_drain
        return np.zeros(self.n, dtype=float)

    def _spend(self, node, amount):
        if node < 0 or not self.alive[node]:
            return False
        amount = max(0.0, float(amount))
        if self.energy[node] <= amount:
            self.energy[node] = 0.0
            self.alive[node] = False
            return False
        self.energy[node] -= amount
        return True

    def _member_to_head(self, assignment, traffic):
        """
        Return member-generated packets and the number successfully received
        by each cluster head. End-to-end delivery is counted only after the
        corresponding cluster head reaches the sink.
        """
        generated = 0
        received_by_head = {}
        energy_spent = np.zeros(self.n, dtype=float)

        heads = set(int(x) for x in self.controller.state.heads.tolist())
        for i in np.flatnonzero(self.alive):
            if i in heads:
                continue
            if self.rng.random() > traffic[i]:
                continue
            generated += 1
            h = int(assignment[i])
            if h < 0 or not self.alive[h]:
                continue
            d = float(np.linalg.norm(self.positions[i] - self.positions[h]))
            tx = self.energy_model.tx(self.packet_bits, d)
            rx = self.energy_model.rx(self.packet_bits)
            ok_tx = self._spend(int(i), tx)
            if ok_tx and self._spend(h, rx):
                energy_spent[i] += tx
                energy_spent[h] += rx
                received_by_head[h] = received_by_head.get(h, 0) + 1

        return generated, received_by_head, energy_spent

    def _route_head_packet(self, head, predicted_drain, link_quality, traffic):
        if not self.alive[head]:
            return False, 0.0, 0, 0.0, np.zeros(self.n, dtype=float)

        # Conventional LEACH/LEACH-C send cluster-head data directly to sink.
        if self.protocol_name in {"leach", "leach_c"}:
            d = float(np.linalg.norm(self.positions[head] - self.sink))
            e = self.energy_model.tx(self.packet_bits, d) + self.energy_model.aggregate(self.packet_bits)
            ok = self._spend(head, e)
            qos = self.cfg["qos"]
            latency = (
                1000.0 * self.packet_bits / float(qos["nominal_bitrate_bps"])
                + 1000.0 * d / float(qos["propagation_speed_m_per_s"])
                + float(qos["edge_processing_delay_ms"])
            )
            spent = np.zeros(self.n, dtype=float)
            if ok:
                spent[head] += e
            return ok, latency, 1, 1.0, spent

        router = self.controller.state.router
        rr = router.route(
            head, self.positions, self.energy, self.alive, predicted_drain,
            link_quality, traffic, self.sink
        )
        spent = np.zeros(self.n, dtype=float)

        # If the nominal-radius relay graph is disconnected, use a high-power
        # direct-to-sink fallback. This keeps the 100 m nominal neighbor radius
        # while allowing the radio model to represent emergency long-range
        # transmission at the corresponding d^4 energy cost.
        if not rr.success or len(rr.path) < 2:
            d = float(np.linalg.norm(self.positions[head] - self.sink))
            e = self.energy_model.tx(self.packet_bits, d) + self.energy_model.aggregate(self.packet_bits)
            ok = self._spend(head, e)
            qos = self.cfg["qos"]
            latency = (
                1000.0 * self.packet_bits / float(qos["nominal_bitrate_bps"])
                + 1000.0 * d / float(qos["propagation_speed_m_per_s"])
                + float(qos["edge_processing_delay_ms"])
            )
            if ok:
                spent[head] += e
            reliability = float(np.exp(-d / max(float(self.cfg["network"]["communication_radius_m"]), 1e-9)))
            return ok, latency, 1, reliability, spent

        success = rr.success
        for u, v in zip(rr.path[:-1], rr.path[1:]):
            if u == -1:
                break
            p_u = self.positions[u]
            p_v = self.sink if v == -1 else self.positions[v]
            d = float(np.linalg.norm(p_u - p_v))
            tx = self.energy_model.tx(self.packet_bits, d)
            if u == head:
                tx += self.energy_model.aggregate(self.packet_bits)
            if not self._spend(int(u), tx):
                success = False
                break
            spent[int(u)] += tx
            if v != -1:
                rx = self.energy_model.rx(self.packet_bits)
                if not self._spend(int(v), rx):
                    success = False
                    break
                spent[int(v)] += rx

        return success, rr.estimated_latency_ms, max(1, len(rr.path) - 1), rr.mean_reliability, spent

    def _harvest(self, round_idx):
        env = self.cfg["environment"]
        if not bool(env["energy_harvesting_enabled"]):
            return
        # Reproducible solar-like daily curve with node-specific occlusion.
        rounds = max(int(self.cfg["network"]["rounds"]), 2)
        sunlight = max(0.0, np.sin(np.pi * round_idx / rounds))
        occlusion = self.rng.uniform(0.65, 1.0, self.n)
        harvested = float(env["max_harvest_j_per_round"]) * sunlight * occlusion
        self.energy[self.alive] = np.minimum(
            self.initial_energy,
            self.energy[self.alive] + harvested[self.alive]
        )

    def run(self) -> SimulationOutput:
        rounds = int(self.cfg["network"]["rounds"])
        cluster_interval = int(self.cfg["network"]["cluster_refresh_interval"])
        route_interval = int(self.cfg["network"]["route_refresh_interval"])

        records = []
        pso_records = []

        for r in range(rounds):
            if not np.any(self.alive):
                break

            temperature, traffic, link_quality = self._environment(r)
            self._update_states(traffic)
            predicted_drain = self._predicted_drain(temperature, traffic, link_quality)

            if r == 0 or r % cluster_interval == 0 or len(self.controller.state.heads) == 0:
                self.controller.refresh_clusters(
                    r, self.positions, self.energy, self.alive, predicted_drain,
                    link_quality, traffic, self.sink
                )

            if (
                self.controller.state.router is not None
                and (r == 0 or r % route_interval == 0)
            ):
                self.controller.refresh_router(
                    self.positions, self.energy, self.alive, predicted_drain,
                    link_quality, traffic, self.sink
                )
                if self.controller.state.router.last_history:
                    for it, val in enumerate(self.controller.state.router.last_history):
                        pso_records.append({
                            "protocol": self.protocol_name,
                            "seed": self.seed,
                            "round": r,
                            "iteration": it,
                            "best_fitness": val,
                        })

            # Sensing energy is charged to active nodes.
            sensing_cost = self.energy_model.sense(self.packet_bits)
            pre_energy = self.energy.copy()
            for i in np.flatnonzero(self.alive):
                if self.state[i] != "sleep":
                    self._spend(int(i), sensing_cost * (0.25 + 0.50 * traffic[i]))

            generated_members, received_by_head, member_spent = self._member_to_head(
                self.controller.state.assignment, traffic
            )

            # Every alive cluster head contributes one local measurement packet.
            generated = int(generated_members)
            delivered = 0
            latencies = []
            hops = []
            reliabilities = []
            route_attempts = 0
            route_successes = 0
            route_spent = np.zeros(self.n, dtype=float)

            for h in self.controller.state.heads:
                h = int(h)
                if not self.alive[h]:
                    continue
                route_attempts += 1
                generated += 1
                represented_packets = int(received_by_head.get(h, 0)) + 1
                ok, lat, hop, rel, spent = self._route_head_packet(
                    h, predicted_drain, link_quality, traffic
                )
                route_spent += spent
                if ok:
                    delivered += represented_packets
                    route_successes += 1
                latencies.append(lat)
                hops.append(hop)
                reliabilities.append(rel)

            self._harvest(r)
            self.alive = self.energy > 0.0

            drain = np.maximum(0.0, pre_energy - self.energy)
            self.rolling_drain = np.roll(self.rolling_drain, 1, axis=1)
            self.rolling_drain[:, 0] = drain

            records.append({
                "protocol": self.protocol_name,
                "seed": self.seed,
                "round": r,
                "alive_nodes": int(np.sum(self.alive)),
                "total_residual_energy_j": float(np.sum(self.energy)),
                "mean_residual_energy_j": float(np.mean(self.energy)),
                "packets_generated": int(generated),
                "packets_delivered": int(delivered),
                "packet_delivery_ratio": float(delivered / max(generated, 1)),
                "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
                "mean_hops": float(np.mean(hops)) if hops else 0.0,
                "routing_success": float(route_successes / max(route_attempts, 1)),
                "mean_route_reliability": float(np.mean(reliabilities)) if reliabilities else 0.0,
                "energy_fairness": jain_fairness(self.energy[self.alive]) if np.any(self.alive) else 0.0,
                "temperature_c": temperature,
                "mean_traffic": float(np.mean(traffic)),
                "cluster_heads": int(len(self.controller.state.heads)),
            })

        round_df = pd.DataFrame(records)
        summary = run_summary(
            round_df,
            initial_total_energy=self.n * self.initial_energy,
            packet_bits=self.packet_bits,
        )
        summary.update({
            "protocol": self.protocol_name,
            "seed": self.seed,
            "rounds_executed": int(len(round_df)),
        })

        pso_df = pd.DataFrame(
            pso_records,
            columns=["protocol", "seed", "round", "iteration", "best_fitness"]
        )
        return SimulationOutput(round_df, summary, self.ml_validation, pso_df)
