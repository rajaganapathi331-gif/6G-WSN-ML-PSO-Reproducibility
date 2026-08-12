from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURES = [
    "residual_energy",
    "distance_to_sink",
    "degree",
    "traffic",
    "temperature",
    "link_quality",
    "rolling_drain_mean",
    "rolling_drain_std",
    "markov_predicted_drain",
    "state_code",
]


@dataclass
class MLValidation:
    mae: float
    rmse: float
    r2: float
    train_size: int
    validation_size: int


class EnergyMLPredictor:
    def __init__(self, cfg, seed: int):
        ml = cfg["ml"]
        self.seed = int(seed)
        self.model = RandomForestRegressor(
            n_estimators=int(ml["n_estimators"]),
            max_depth=int(ml["max_depth"]),
            min_samples_leaf=int(ml["min_samples_leaf"]),
            random_state=self.seed,
            n_jobs=-1,
        )
        self.validation: MLValidation | None = None
        self.feature_importance_: dict[str, float] = {}

    def fit(self, df: pd.DataFrame):
        missing = [c for c in FEATURES + ["target_drain", "node_id"] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing ML columns: {missing}")

        groups = df["node_id"].to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=self.seed)
        tr_idx, va_idx = next(splitter.split(df[FEATURES], df["target_drain"], groups=groups))

        Xtr = df.iloc[tr_idx][FEATURES]
        ytr = df.iloc[tr_idx]["target_drain"]
        Xva = df.iloc[va_idx][FEATURES]
        yva = df.iloc[va_idx]["target_drain"]

        self.model.fit(Xtr, ytr)
        pred = self.model.predict(Xva)

        self.validation = MLValidation(
            mae=float(mean_absolute_error(yva, pred)),
            rmse=float(mean_squared_error(yva, pred) ** 0.5),
            r2=float(r2_score(yva, pred)),
            train_size=int(len(tr_idx)),
            validation_size=int(len(va_idx)),
        )
        self.feature_importance_ = {
            name: float(value)
            for name, value in zip(FEATURES, self.model.feature_importances_)
        }
        return self.validation

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(frame[FEATURES])


def generate_telemetry(cfg, seed: int, energy_model, markov_predictor) -> pd.DataFrame:
    """
    Generate deterministic node telemetry for explicit ML training/validation.

    This is a simulation-based telemetry generator, not real hardware data.
    Node-group holdout prevents the same node identity from being present in
    both training and validation subsets.
    """
    rng = np.random.default_rng(seed + 100_003)
    n = int(cfg["network"]["n_nodes"])
    steps = int(cfg["ml"]["telemetry_steps"])
    width = float(cfg["network"]["width_m"])
    height = float(cfg["network"]["height_m"])
    sink = np.array([float(cfg["network"]["sink_x_m"]), float(cfg["network"]["sink_y_m"])])
    initial = float(cfg["network"]["initial_energy_j"])
    radius = float(cfg["network"]["communication_radius_m"])
    packet_bits = int(cfg["network"]["packet_bits"])
    env = cfg["environment"]

    pos = np.column_stack([rng.uniform(0, width, n), rng.uniform(0, height, n)])
    d_sink = np.linalg.norm(pos - sink, axis=1)
    energy = np.full(n, initial, dtype=float)
    rolling = np.zeros((n, 5), dtype=float)
    prev_state = np.array(["idle"] * n, dtype=object)

    rows = []
    state_names = np.array(["sleep", "idle", "rx", "tx"], dtype=object)
    state_probs = np.array([0.20, 0.25, 0.20, 0.35])

    dist_matrix = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    degree = ((dist_matrix <= radius) & (dist_matrix > 0)).sum(axis=1)

    for t in range(steps):
        phase = 2 * np.pi * t / max(steps, 2)
        temperature = float(env["base_temperature_c"]) + float(env["temperature_amplitude_c"]) * np.sin(phase)
        traffic = rng.uniform(float(env["traffic_min"]), float(env["traffic_max"]), n)
        link_quality = np.clip(
            1.0 - 0.35 * np.minimum(d_sink / max(width, height), 1.0)
            + rng.normal(0, float(env["link_noise_std"]), n),
            0.05, 1.0
        )
        states = rng.choice(state_names, size=n, p=state_probs)

        for i in range(n):
            markov_predictor.update(str(prev_state[i]), str(states[i]))
        markov_drain = markov_predictor.predict_batch(states)

        # Physically interpretable simulation target with environmental and traffic effects.
        ref_d = np.clip(0.25 * d_sink + 20.0, 10.0, 250.0)
        tx = np.array([energy_model.tx(packet_bits, d) for d in ref_d])
        rx = energy_model.rx(packet_bits)
        state_factor = np.choose(
            np.array([{"sleep": 0, "idle": 1, "rx": 2, "tx": 3}[s] for s in states]),
            [0.02, 0.10, 0.65, 1.00]
        )
        temp_factor = 1.0 + 0.004 * np.abs(temperature - float(env["base_temperature_c"]))
        drain = (0.35 * tx + 0.20 * rx) * (0.4 + traffic) * state_factor * temp_factor
        drain *= (1.0 + 0.10 * (1.0 - link_quality))
        drain += 0.25 * markov_drain
        drain += rng.normal(0.0, np.maximum(1e-8, drain * 0.03))
        drain = np.maximum(drain, 1e-10)

        rolling = np.roll(rolling, 1, axis=1)
        rolling[:, 0] = drain
        roll_mean = rolling.mean(axis=1)
        roll_std = rolling.std(axis=1)

        state_code = np.array([{"sleep": 0, "idle": 1, "rx": 2, "tx": 3}[s] for s in states])

        for i in range(n):
            rows.append({
                "node_id": i,
                "residual_energy": energy[i] / initial,
                "distance_to_sink": d_sink[i] / max(width, height),
                "degree": degree[i] / max(1, n - 1),
                "traffic": traffic[i],
                "temperature": temperature,
                "link_quality": link_quality[i],
                "rolling_drain_mean": roll_mean[i],
                "rolling_drain_std": roll_std[i],
                "markov_predicted_drain": markov_drain[i],
                "state_code": state_code[i],
                "target_drain": drain[i],
            })

        energy = np.maximum(0.0, energy - drain)
        prev_state = states

    return pd.DataFrame(rows)
