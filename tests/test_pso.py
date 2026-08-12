from src.pso import ParticleSwarmOptimizer


def test_pso_reduces_simple_objective():
    cfg = {
        "pso": {
            "swarm_size": 10, "max_iterations": 25, "inertia": 0.72,
            "cognitive": 1.49, "social": 1.49, "lower_bound": -3.0,
            "upper_bound": 3.0, "early_stop_patience": 8,
            "early_stop_tolerance": 1e-8
        }
    }
    opt = ParticleSwarmOptimizer(3, cfg, seed=1)
    res = opt.minimize(lambda x: float((x * x).sum()))
    assert res.best_fitness >= 0.0
    assert res.best_fitness < 1.0
    assert len(res.history) >= 2
