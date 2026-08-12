from __future__ import annotations

from pathlib import Path
import json
import platform
import sys
import importlib.metadata as md

import pandas as pd

from .simulator import WSNSimulator
from .protocols import SUPPORTED_PROTOCOLS


def _package_version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "not-installed"


def _manifest(cfg):
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scipy": _package_version("scipy"),
            "scikit-learn": _package_version("scikit-learn"),
            "matplotlib": _package_version("matplotlib"),
            "PyYAML": _package_version("PyYAML"),
        },
        "config": cfg,
    }


def run_experiment_suite(cfg, output_dir: str | Path):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    protocols = list(cfg["experiment"]["protocols"])
    unknown = [p for p in protocols if p not in SUPPORTED_PROTOCOLS]
    if unknown:
        raise ValueError(f"Unknown protocol(s): {unknown}")

    seeds = [int(s) for s in cfg["experiment"]["seeds"]]
    run_rows = []
    round_frames = []
    ml_rows = []
    pso_frames = []

    for protocol in protocols:
        for seed in seeds:
            print(f"[run] protocol={protocol} seed={seed}")
            sim = WSNSimulator(cfg, protocol, seed)
            result = sim.run()
            run_rows.append(result.run_summary)
            round_frames.append(result.round_metrics)

            if result.ml_validation is not None:
                row = {"protocol": protocol, "seed": seed, **result.ml_validation}
                ml_rows.append(row)

            if len(result.pso_convergence):
                pso_frames.append(result.pso_convergence)

    pd.DataFrame(run_rows).to_csv(out / "runs.csv", index=False)
    pd.concat(round_frames, ignore_index=True).to_csv(out / "round_metrics.csv", index=False)

    pd.DataFrame(
        ml_rows,
        columns=["protocol", "seed", "mae", "rmse", "r2", "train_size", "validation_size"]
    ).to_csv(out / "ml_validation.csv", index=False)

    if pso_frames:
        pd.concat(pso_frames, ignore_index=True).to_csv(out / "pso_convergence.csv", index=False)
    else:
        pd.DataFrame(
            columns=["protocol", "seed", "round", "iteration", "best_fitness"]
        ).to_csv(out / "pso_convergence.csv", index=False)

    with (out / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(_manifest(cfg), f, indent=2)
