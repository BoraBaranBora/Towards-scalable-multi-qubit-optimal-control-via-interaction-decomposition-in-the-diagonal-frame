import torch
import math


def get_envelope_qb(time_grid, parameter_subset):
    n = len(parameter_subset) // 3
    amplitude = parameter_subset[:n]
    zero_shift = 0.2

    def mu(l):
        return (l + zero_shift) * (2 * math.pi / time_grid[-1].item())

    harmonics = torch.tensor([mu(i + 1) for i in range(n)], dtype=torch.float64)

    pulse = torch.zeros_like(time_grid, dtype=torch.float64)

    for i in range(n):
        f = harmonics[i]
        oscillation = torch.real(torch.exp(-1j * (-f * time_grid + (time_grid[-1] / 2) * f)))
        pulse += amplitude[i] * oscillation

    return pulse


def get_envelope_custom(time_grid, parameter_subset):
    n = len(parameter_subset) // 3
    amplitude = parameter_subset[:n]
    frequency = parameter_subset[n:2 * n]
    phases = parameter_subset[2 * n:3 * n]

    pulse = torch.zeros_like(time_grid, dtype=torch.float64)

    for i in range(n):
        pulse += amplitude[i] * torch.cos(-frequency[i] * time_grid + phases[i])

    return pulse


def get_envelope_gaussian(time_grid, parameter_subset):
    n = len(parameter_subset) // 3
    amplitude = parameter_subset[:n] / n
    stauch = parameter_subset[n:2 * n]
    shift = parameter_subset[2 * n:3 * n]

    pulse = torch.zeros_like(time_grid, dtype=torch.float64)

    for i in range(n):
        pulse += amplitude[i] * torch.exp(-((time_grid - (time_grid[-1] / 2)) ** 2) * stauch[i]) - shift[i]

    return pulse


def get_carrier_pulse(time_grid, parameter_subset):
    n = len(parameter_subset) // 3
    if n > 1:
        raise ValueError("Carrier basis must have size 1")

    frequency = parameter_subset[n:2 * n]
    phase = parameter_subset[2 * n:3 * n]

    pulse = torch.cos(time_grid * frequency[0] + phase[0])
    return pulse


# Dictionary mapping basis type to function
basis_function_map = {
    "QB_Basis": get_envelope_qb,
    "Custom": get_envelope_custom,
    "Gaussian": get_envelope_gaussian,
    "Carrier": get_carrier_pulse
}


def call_basis_function(time_grid, parameter_subset, basis_type):
    if basis_type not in basis_function_map:
        raise ValueError(f"Unknown basis type: {basis_type}")
    return basis_function_map[basis_type](time_grid, parameter_subset)


def get_pulse(time_grid, parameter_subset, pulse_settings):
    return call_basis_function(time_grid, parameter_subset, pulse_settings.basis_type)


def get_drive(time_grid, parameter_set, pulse_settings_list):
    bss = [ps.basis_size for ps in pulse_settings_list]
    indices = [0] + list(torch.cumsum(torch.tensor([3 * b for b in bss]), dim=0).numpy())
    parameter_subsets = [
        parameter_set[indices[i]:indices[i + 1]] for i in range(len(bss))
    ]

    pulse_list = [
        get_pulse(time_grid, p, ps) for p, ps in zip(parameter_subsets, pulse_settings_list)
    ]
    return pulse_list


def get_drive(time_grid, parameter_set, pulse_settings_list):
    bss = [ps.basis_size for ps in pulse_settings_list]
    indices = [0] + list(torch.cumsum(torch.tensor([3 * b for b in bss]), dim=0).numpy())
    parameter_subsets = [
        parameter_set[indices[i]:indices[i + 1]] for i in range(len(bss))
    ]

    # Classify by channel type
    mw_pulse = None
    rf_pulse = None

    for params, ps in zip(parameter_subsets, pulse_settings_list):
        pulse = get_pulse(time_grid, params, ps)
        if ps.channel_type == "MW":
            mw_pulse = pulse
        elif ps.channel_type == "RF":
            rf_pulse = pulse
        else:
            raise ValueError(f"Unsupported channel_type: {ps.channel_type}")

    # Ensure both MW and RF exist; pad with zeros if needed
    shape = time_grid.shape
    if mw_pulse is None:
        mw_pulse = torch.zeros(shape, dtype=torch.float64)
    if rf_pulse is None:
        rf_pulse = torch.zeros(shape, dtype=torch.float64)

    return [mw_pulse, rf_pulse]
