import torch
import numpy as np
from evolution import get_propagator
from typing import Callable, List, Dict


def calculate_primal(pulse, parameter_subset, pulse_settings):
    
    if isinstance(parameter_subset, np.ndarray):
        parameter_subset = torch.tensor(parameter_subset, dtype=torch.float64)
    if isinstance(pulse, np.ndarray):
        pulse = torch.tensor(pulse, dtype=torch.float64)
        
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
        
    if primal > 1.1:
        print("High primal penalty:")
        print(f"  max_amp  = {torch.max(torch.abs(amplitude_parameters)).item():.2e} (limit: {ma})")
        print(f"  max_pulse = {torch.max(torch.abs(pulse)).item():.2e} (limit: {mpu})")
        print(f"  max_freq = {torch.max(frequency_parameters).item():.2e} (limit: {mf})")
        print(f"  min_freq = {torch.min(frequency_parameters).item():.2e} (limit: {minf})")
        print(f"  max_phase = {torch.max(torch.abs(phase_parameters)).item():.2e} (limit: {mp})")

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
    if isinstance(parameter_set, np.ndarray):
        parameter_set = torch.tensor(parameter_set, dtype=torch.float64)
        
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

    full_propagator = get_propagator(get_u, time_grid, drive)

    # Extract qubit subspace (assumed indices 8 to 11)
    sub_start = 8
    sub_end = 12
    prop_sub = full_propagator[sub_start:sub_end, sub_start:sub_end]
    gate_sub = target_gate[sub_start:sub_end, sub_start:sub_end]

    N = gate_sub.shape[0]

    fidelity = (1 / N ** 2) * abs(torch.trace(prop_sub.conj().T @ gate_sub)) ** 2
    unitarity = 1 - abs(torch.det(prop_sub))

    return (1 - fidelity.item()) + primal_value + unitarity.item()


def FoM_gate_transformation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    target_gate,  # Not used but kept for API consistency
    get_drive_fn
):
    # Step 1: Split parameters by channel
    bss = [ps.basis_size for ps in pulse_settings_list]
    indices = np.cumsum([0] + [3 * bs for bs in bss])
    parameter_subsets = [
        parameter_set[indices[i]:indices[i + 1]] for i in range(len(bss))
    ]

    # Step 2: Get pulses
    drive = get_drive_fn(time_grid, parameter_set, pulse_settings_list)

    # Step 3: Primal penalty
    primal_value = sum(
        calculate_primal(drive[i], parameter_subsets[i], pulse_settings_list[i])
        for i in range(len(drive))
    )

    # Step 4: Extract propagator on qubit subspace (indices 6:10 = 7:10 in Julia)
    full_propagator = get_propagator(get_u, time_grid, drive)
    prop_sub = full_propagator[6:9, 6:9]

    # Step 5: Target: diag(1, 1, 1, -1)
    target_gate_diag = torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.complex128))

    # Step 6: Fidelity (standard Hilbert-Schmidt)
    N = target_gate_diag.shape[0]
    fidelity = (1 / N**2) * abs(torch.trace(prop_sub.conj().T @ target_gate_diag)) ** 2

    # Step 7: Unitarity penalty
    unitarity = 1 - abs(torch.det(prop_sub))

    # Step 8: Combine
    return abs(1.0 - fidelity.item()) + primal_value + unitarity.item()


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
