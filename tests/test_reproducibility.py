from copy import deepcopy

from src.config import load_config
from src.simulator import WSNSimulator


def test_same_seed_reproduces_same_short_run():
    cfg = load_config("config.yaml")
    cfg = deepcopy(cfg)
    cfg["network"]["n_nodes"] = 40
    cfg["network"]["rounds"] = 8
    cfg["network"]["cluster_refresh_interval"] = 4
    cfg["network"]["route_refresh_interval"] = 4
    cfg["pso"]["swarm_size"] = 6
    cfg["pso"]["max_iterations"] = 5
    cfg["pso"]["early_stop_patience"] = 3
    cfg["ml"]["telemetry_steps"] = 8
    cfg["ml"]["n_estimators"] = 20

    a = WSNSimulator(cfg, "leach", 123).run().round_metrics
    b = WSNSimulator(cfg, "leach", 123).run().round_metrics
    assert a.equals(b)
