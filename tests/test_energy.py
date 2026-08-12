from src.energy import RadioEnergyModel


def test_radio_energy_is_positive_and_long_distance_costs_more():
    model = RadioEnergyModel(5e-8, 1e-11, 1.3e-15, 5e-9, 5e-9)
    near = model.tx(4000, 20.0)
    far = model.tx(4000, 200.0)
    assert near > 0
    assert far > near
    assert model.rx(4000) > 0
    assert model.aggregate(4000) > 0
