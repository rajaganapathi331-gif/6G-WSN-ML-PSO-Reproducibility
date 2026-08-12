from __future__ import annotations

import numpy as np


def jain_fairness(values) -> float:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return float("nan")
    denom = x.size * float(np.sum(x * x))
    if denom <= 0:
        return 1.0
    return float(np.sum(x) ** 2 / denom)


def safe_round_of_event(alive_series, threshold_alive):
    arr = np.asarray(alive_series)
    hits = np.flatnonzero(arr <= threshold_alive)
    return int(hits[0]) if len(hits) else None


def run_summary(round_df, initial_total_energy: float, packet_bits: int):
    alive = round_df["alive_nodes"].to_numpy()
    n0 = int(alive[0]) if len(alive) else 0
    fnd = safe_round_of_event(alive, n0 - 1) if n0 else None
    hnd = safe_round_of_event(alive, n0 // 2) if n0 else None

    delivered = float(round_df["packets_delivered"].sum())
    generated = float(round_df["packets_generated"].sum())
    final_energy = float(round_df["total_residual_energy_j"].iloc[-1])
    consumed = max(initial_total_energy - final_energy, 1e-12)

    return {
        "fnd_round": fnd if fnd is not None else int(round_df["round"].iloc[-1]) + 1,
        "hnd_round": hnd if hnd is not None else int(round_df["round"].iloc[-1]) + 1,
        "final_alive_nodes": int(alive[-1]),
        "total_energy_consumed_j": consumed,
        "packet_delivery_ratio": delivered / max(generated, 1.0),
        "mean_latency_ms": float(round_df["mean_latency_ms"].replace([np.inf], np.nan).mean()),
        "mean_hops": float(round_df["mean_hops"].mean()),
        "mean_routing_success": float(round_df["routing_success"].mean()),
        "mean_energy_fairness": float(round_df["energy_fairness"].mean()),
        "delivered_bits_per_joule": float(packet_bits * delivered / consumed),
    }
