from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RadioEnergyModel:
    e_elec: float
    e_fs: float
    e_mp: float
    e_da: float
    e_sense: float

    @classmethod
    def from_config(cls, cfg):
        r = cfg["radio"]
        return cls(
            e_elec=float(r["e_elec_j_per_bit"]),
            e_fs=float(r["e_fs_j_per_bit_m2"]),
            e_mp=float(r["e_mp_j_per_bit_m4"]),
            e_da=float(r["e_da_j_per_bit"]),
            e_sense=float(r["e_sense_j_per_bit"]),
        )

    @property
    def d0(self) -> float:
        return math.sqrt(self.e_fs / self.e_mp)

    def tx(self, bits: int, distance_m: float) -> float:
        bits = int(bits)
        d = max(0.0, float(distance_m))
        amplifier = self.e_fs * d * d if d < self.d0 else self.e_mp * d**4
        return bits * (self.e_elec + amplifier)

    def rx(self, bits: int) -> float:
        return int(bits) * self.e_elec

    def aggregate(self, bits: int) -> float:
        return int(bits) * self.e_da

    def sense(self, bits: int) -> float:
        return int(bits) * self.e_sense

    def state_costs(self, bits: int, reference_distance_m: float = 50.0):
        # Nominal one-step costs used only by the Markov predictor.
        sensing = self.sense(bits)
        return {
            "sleep": 0.02 * sensing,
            "idle": 0.10 * sensing,
            "rx": self.rx(bits),
            "tx": self.tx(bits, reference_distance_m),
        }
