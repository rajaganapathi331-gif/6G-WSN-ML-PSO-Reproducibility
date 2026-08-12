from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.experiment import run_experiment_suite
from src.statistics import create_summary_and_pairwise_stats
from src.plotting import create_all_figures


def parse_args():
    p = argparse.ArgumentParser(description="Reproduce all experiments, statistics and figures.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--output", default="results")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.quick:
        # Representative smoke workflow intended to finish quickly on a laptop.
        cfg["experiment"]["seeds"] = cfg["experiment"]["seeds"][:1]
        cfg["experiment"]["protocols"] = ["leach", "leach_c", "proposed"]
        cfg["network"]["n_nodes"] = min(60, int(cfg["network"]["n_nodes"]))
        cfg["network"]["rounds"] = min(50, int(cfg["network"]["rounds"]))
        cfg["network"]["cluster_refresh_interval"] = min(
            10, int(cfg["network"]["cluster_refresh_interval"])
        )
        cfg["network"]["route_refresh_interval"] = min(
            15, int(cfg["network"]["route_refresh_interval"])
        )
        cfg["ml"]["telemetry_steps"] = min(10, int(cfg["ml"]["telemetry_steps"]))
        cfg["ml"]["n_estimators"] = min(30, int(cfg["ml"]["n_estimators"]))
        cfg["pso"]["max_iterations"] = min(8, int(cfg["pso"]["max_iterations"]))
        cfg["pso"]["swarm_size"] = min(8, int(cfg["pso"]["swarm_size"]))
        cfg["pso"]["early_stop_patience"] = min(
            5, int(cfg["pso"]["early_stop_patience"])
        )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    run_experiment_suite(cfg, out)
    create_summary_and_pairwise_stats(cfg, out)
    create_all_figures(out)

    print(f"Reproducibility workflow completed: {out.resolve()}")


if __name__ == "__main__":
    main()
