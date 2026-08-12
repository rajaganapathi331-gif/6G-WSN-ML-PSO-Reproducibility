from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save(fig, out, stem):
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def create_all_figures(output_dir: str | Path):
    out = Path(output_dir)
    rounds = pd.read_csv(out / "round_metrics.csv")

    # 1. Alive nodes
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for protocol, g in rounds.groupby("protocol"):
        mean = g.groupby("round")["alive_nodes"].mean()
        ax.plot(mean.index, mean.values, label=protocol)
    ax.set_xlabel("Simulation round")
    ax.set_ylabel("Mean alive nodes")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    _save(fig, out, "fig_alive_nodes")

    # 2. Residual energy
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for protocol, g in rounds.groupby("protocol"):
        mean = g.groupby("round")["total_residual_energy_j"].mean()
        ax.plot(mean.index, mean.values, label=protocol)
    ax.set_xlabel("Simulation round")
    ax.set_ylabel("Total residual energy (J)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    _save(fig, out, "fig_residual_energy")

    # 3. PDR
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for protocol, g in rounds.groupby("protocol"):
        mean = g.groupby("round")["packet_delivery_ratio"].mean()
        ax.plot(mean.index, mean.values, label=protocol)
    ax.set_xlabel("Simulation round")
    ax.set_ylabel("Packet-delivery ratio")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    _save(fig, out, "fig_packet_delivery_ratio")

    # 4. Energy fairness
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for protocol, g in rounds.groupby("protocol"):
        mean = g.groupby("round")["energy_fairness"].mean()
        ax.plot(mean.index, mean.values, label=protocol)
    ax.set_xlabel("Simulation round")
    ax.set_ylabel("Jain energy-fairness index")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    _save(fig, out, "fig_energy_fairness")

    pso_file = out / "pso_convergence.csv"
    if pso_file.exists():
        pso = pd.read_csv(pso_file)
        if len(pso):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            first = (
                pso.sort_values(["protocol", "seed", "round", "iteration"])
                .groupby(["protocol", "seed"], as_index=False)
                .apply(lambda x: x[x["round"] == x["round"].min()])
                .reset_index(drop=True)
            )
            for protocol, g in first.groupby("protocol"):
                mean = g.groupby("iteration")["best_fitness"].mean()
                ax.plot(mean.index, mean.values, label=protocol)
            ax.set_xlabel("PSO iteration")
            ax.set_ylabel("Best objective value")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.25)
            _save(fig, out, "fig_pso_convergence")
