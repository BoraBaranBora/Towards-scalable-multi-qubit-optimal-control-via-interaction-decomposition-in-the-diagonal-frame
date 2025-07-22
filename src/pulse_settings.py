from dataclasses import dataclass

@dataclass
class PulseSettings:
    basis_type: str
    basis_size: int
    maximal_pulse: float
    maximal_amplitude: float
    maximal_frequency: float
    minimal_frequency: float
    maximal_phase: float



import numpy as np
import torch
from typing import List


def get_random_pulse_parameters(pulse_settings: PulseSettings) -> torch.Tensor:
    bt = pulse_settings.basis_type
    bs = pulse_settings.basis_size
    ma = pulse_settings.maximal_amplitude
    mf = pulse_settings.maximal_frequency
    minf = pulse_settings.minimal_frequency
    mp = pulse_settings.maximal_phase

    if bt == "Carrier":
        freq_shape = np.array([(mf - minf) / 2 + minf for _ in range(bs)])
        freq_range = np.array([(mf - minf) / 2] * bs)
    else:
        freq_shape = np.array([(mf / bs) / 2 + (mf / bs) * i for i in range(bs)])
        freq_range = np.array([(mf / bs)] * bs)

    shaping = np.concatenate([
        np.zeros(bs),        # amplitude base
        freq_shape,          # frequency base
        np.zeros(bs)         # phase base
    ])
    shift_ranges = np.concatenate([
        np.full(bs, ma),
        freq_range,
        np.full(bs, mp)
    ])

    random_factors = np.concatenate([
        np.random.rand(bs),
        np.random.rand(bs) * np.random.choice([1, -1], bs) * 0.5,
        np.random.rand(bs) * np.random.choice([1, -1], bs)
    ])

    new_parameters = shaping + random_factors * shift_ranges
    return torch.tensor(new_parameters, dtype=torch.float64)


def get_random_parameter_set(pulse_settings_list: List[PulseSettings]) -> torch.Tensor:
    param_list = [
        get_random_pulse_parameters(ps) for ps in pulse_settings_list
    ]
    return torch.cat(param_list)


def get_initial_guess(sample_size: int, goal_function, pulse_settings_list: List[PulseSettings]):
    x_ensemble = [get_random_parameter_set(pulse_settings_list) for _ in range(sample_size)]
    values = [goal_function(x) for x in x_ensemble]

    best_index = int(np.argmin(values))
    best_params = x_ensemble[best_index]
    best_value = values[best_index]

    return best_params, best_value
