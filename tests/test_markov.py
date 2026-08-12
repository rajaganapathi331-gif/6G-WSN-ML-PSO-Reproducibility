import numpy as np

from src.markov import MarkovEnergyPredictor


def test_transition_rows_sum_to_one():
    p = MarkovEnergyPredictor(
        {"sleep": 1.0, "idle": 2.0, "rx": 3.0, "tx": 4.0}, alpha=1.0
    )
    p.update("idle", "tx")
    p.update("idle", "tx")
    m = p.transition_matrix()
    assert np.allclose(m.sum(axis=1), 1.0)
    assert p.predict_drain("idle") > 0
