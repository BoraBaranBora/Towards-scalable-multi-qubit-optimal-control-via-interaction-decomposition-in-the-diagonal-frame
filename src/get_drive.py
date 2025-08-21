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






def get_envelope_custom(time_grid, parameter_subset, alpha: float = 0.15):
    """
    Build a sum-of-cosines pulse and multiply by a Tukey window.

    Args:
        time_grid (torch.Tensor): 1-D tensor of times (assumed sorted).
        parameter_subset (torch.Tensor): length 3*n vector [amplitudes, frequencies, phases].
        alpha (float): Tukey taper fraction in [0, 1]. 0 -> rectangular, 1 -> Hann.

    Returns:
        torch.Tensor: windowed pulse (same shape as time_grid).
    """
    # unpack parameters
    n = len(parameter_subset) // 3
    amplitude = parameter_subset[:n]
    frequency = parameter_subset[n:2 * n]
    phases    = parameter_subset[2 * n:3 * n]

    # build the raw sum-of-cosines pulse
    pulse = torch.zeros_like(time_grid, dtype=torch.float64)
    for i in range(n):
        pulse = pulse + amplitude[i] * torch.cos(-frequency[i] * time_grid + phases[i])

    # build normalized fractional time from 0 -> 1
    t0, t1 = time_grid[0], time_grid[-1]
    frac = (time_grid - t0) / (t1 - t0)
    frac = torch.clamp(frac, 0.0, 1.0)

    # Tukey window
    if alpha <= 0.0:
        window = torch.ones_like(frac)
    elif alpha >= 1.0:
        # alpha == 1 is identical to a Hann window
        window = 0.5 * (1.0 - torch.cos(2.0 * math.pi * frac))
    else:
        window = torch.ones_like(frac)
        left_end = alpha / 2.0
        right_start = 1.0 - left_end

        # left tapered region: 0 <= frac < alpha/2
        left_mask = frac < left_end
        if left_mask.any():
            # arg runs from -pi -> 0
            arg = (2.0 * math.pi / alpha) * (frac[left_mask] - left_end)
            window[left_mask] = 0.5 * (1.0 + torch.cos(arg))

        # right tapered region: 1-alpha/2 < frac <= 1
        right_mask = frac > right_start
        if right_mask.any():
            # shift so arg runs from 0 -> pi
            arg = (2.0 * math.pi / alpha) * (frac[right_mask] - right_start)
            window[right_mask] = 0.5 * (1.0 + torch.cos(arg))

    return pulse * window


def get_envelope_custom(time_grid, parameter_subset):
    n = len(parameter_subset) // 3
    amplitude = parameter_subset[:n]
    frequency = parameter_subset[n:2 * n]
    phases = parameter_subset[2 * n:3 * n]

    pulse = torch.zeros_like(time_grid, dtype=torch.float64)

    for i in range(n):
        pulse += amplitude[i] * torch.cos(-frequency[i] * time_grid + phases[i])

    return pulse

def get_envelope_custom(time_grid, parameter_subset):
    # unpack parameters
    n = len(parameter_subset) // 3
    amplitude = parameter_subset[:n]
    frequency = parameter_subset[n:2 * n]
    phases    = parameter_subset[2 * n:3 * n]

    # build the raw sum‐of‐cosines pulse
    pulse = torch.zeros_like(time_grid, dtype=torch.float64)
    for i in range(n):
        pulse += amplitude[i] * torch.cos(-frequency[i] * time_grid + phases[i])

    # now build a Hann window that is 0 at time_grid[0] and time_grid[-1]
    t0, t1 = time_grid[0], time_grid[-1]
    # (time_grid - t0) / (t1 - t0) runs from 0→1
    frac = (time_grid - t0) / (t1 - t0)
    window = 0.5 * (1.0 - torch.cos(2.0 * math.pi * frac))

    # multiply your pulse by the window
    return pulse * window

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
