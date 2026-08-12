from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required_sections = [
        "experiment", "network", "radio", "environment", "markov",
        "ml", "pso", "qos", "proposed_weights", "statistics"
    ]
    missing = [s for s in required_sections if s not in cfg]
    if missing:
        raise ValueError(f"Missing configuration section(s): {missing}")

    n = cfg["network"]
    if int(n["n_nodes"]) <= 1:
        raise ValueError("network.n_nodes must be > 1")
    if float(n["initial_energy_j"]) <= 0:
        raise ValueError("network.initial_energy_j must be > 0")
    if int(n["packet_bits"]) <= 0:
        raise ValueError("network.packet_bits must be > 0")
    if int(n["rounds"]) <= 0:
        raise ValueError("network.rounds must be > 0")

    return cfg
