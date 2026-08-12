from src.protocols import SUPPORTED_PROTOCOLS


def test_required_protocols_exist():
    expected = {
        "leach", "leach_c", "greedy", "pso_only",
        "markov_pso", "ml_pso", "proposed"
    }
    assert expected.issubset(SUPPORTED_PROTOCOLS)
