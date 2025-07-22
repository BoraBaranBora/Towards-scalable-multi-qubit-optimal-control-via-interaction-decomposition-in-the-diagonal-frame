import torch
import numpy as np
from evolution import get_propagator
from typing import Callable, List, Dict


def calculate_primal(pulse, parameter_subset, pulse_settings):
    bs = pulse_settings.basis_size
    ma = pulse_settings.maximal_amplitude
    mf = pulse_settings.maximal_frequency
    minf = pulse_settings.minimal_frequency
    mp = pulse_settings.maximal_phase
    mpu = pulse_settings.maximal_pulse
    basis_type = pulse_settings.basis_type

    amplitude_parameters = parameter_subset[:bs]
    frequency_parameters = parameter_subset[bs:2 * bs]
    phase_parameters = parameter_subset[2 * bs:3 * bs]

    b = 0 if basis_type == "Carrier" else 1

    primal = (
        b * max(0, torch.abs(torch.max(amplitude_parameters)).item() - ma) +
        b * max(0, torch.max(torch.abs(pulse)).item() - mpu) +
        0.00 * abs(pulse[-1].item())
    )

    if basis_type != "QB_Basis":
        primal += (
            max(0, torch.max(frequency_parameters).item() - mf) -
            min(0, torch.min(frequency_parameters).item() - minf) +
            max(0, torch.max(torch.abs(phase_parameters)).item() - mp)
        )

    return primal


def get_fidelity(current_state, target_state):
    return torch.abs(torch.dot(current_state.conj(), target_state)).item()


def FoM_state_preparation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    starting_state,
    target_state,
    get_drive_fn
):
    bss = [ps.basis_size for ps in pulse_settings_list]
    indices = np.cumsum([0] + [3 * bs for bs in bss])

    parameter_subsets = [
        parameter_set[indices[i]:indices[i + 1]] for i in range(len(bss))
    ]

    drive = get_drive_fn(time_grid, parameter_set, pulse_settings_list)

    primal_value = sum(
        calculate_primal(drive[i], parameter_subsets[i], pulse_settings_list[i])
        for i in range(len(drive))
    )

    propagator = get_propagator(get_u, time_grid, drive)
    final_state = propagator @ starting_state
    fidelity = get_fidelity(final_state, target_state)

    return (1 - fidelity) + primal_value


def FoM_gate_transformation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    target_gate,
    get_drive_fn
):
    bss = [ps.basis_size for ps in pulse_settings_list]
    indices = np.cumsum([0] + [3 * bs for bs in bss])

    parameter_subsets = [
        parameter_set[indices[i]:indices[i + 1]] for i in range(len(bss))
    ]

    drive = get_drive_fn(time_grid, parameter_set, pulse_settings_list)

    primal_value = sum(
        calculate_primal(drive[i], parameter_subsets[i], pulse_settings_list[i])
        for i in range(len(drive))
    )

    propagator = get_propagator(get_u, time_grid, drive)[9:12, 9:12]
    N = target_gate.shape[0]

    fidelity = (1 / N ** 2) * abs(torch.trace(propagator.conj().T @ target_gate)) ** 2
    unitarity = 1 - abs(torch.det(propagator))

    return (1 - fidelity.item()) + primal_value + unitarity.item()


objective_dictionary: Dict[str, Callable] = {
    "State Preparation": FoM_state_preparation,
    "Gate Transformation": FoM_gate_transformation
}


def call_optimization_objective(objective_type: str):
    if objective_type not in objective_dictionary:
        raise ValueError(f"Unknown objective type: {objective_type}")
    return objective_dictionary[objective_type]


def get_goal_function(
    get_u,
    objective_type: str,
    time_grid,
    pulse_settings_list,
    get_drive_fn,
    starting_state=None,
    target_state=None,
    target_gate=None
):
    fom_function = call_optimization_objective(objective_type)

    if objective_type == "State Preparation":
        def objective_fn(x):
            return fom_function(
                get_u,
                time_grid,
                x,
                pulse_settings_list,
                starting_state,
                target_state,
                get_drive_fn
            )
        return objective_fn

    elif objective_type == "Gate Transformation":
        def objective_fn(x):
            return fom_function(
                get_u,
                time_grid,
                x,
                pulse_settings_list,
                target_gate,
                get_drive_fn
            )
        return objective_fn

    else:
        raise ValueError(f"Unsupported objective type: {objective_type}")
