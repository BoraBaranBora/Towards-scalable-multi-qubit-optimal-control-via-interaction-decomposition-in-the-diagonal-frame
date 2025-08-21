import torch
import numpy as np
import math
from typing import Callable, List, Dict
from evolution import get_propagator




from quantum_operators import pauli_operator_on_qubit

# -------------------------
# --- Constraint Penalty ---
# -------------------------


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

    amps = parameter_subset[:bs]
    freqs = parameter_subset[bs:2 * bs]
    phases = parameter_subset[2 * bs:3 * bs]

    b = 0 if basis_type == "Carrier" else 1

    def soft_penalty(x, limit):
        excess = torch.clamp(torch.abs(x) - limit, min=0.0)
        return (excess / limit) ** 2  # softer and smoother

    def soft_penalty_raw(x, lower, upper):
        lower_violation = torch.clamp(lower - x, min=0.0)
        upper_violation = torch.clamp(x - upper, min=0.0)
        return ((lower_violation / (upper - lower)) ** 2 +
                (upper_violation / (upper - lower)) ** 2)

    total_penalty = 0.0
    total_terms = 0

    if b:
        total_penalty += torch.sum(soft_penalty(amps, ma))
        total_terms += len(amps)

        total_penalty += soft_penalty(torch.max(torch.abs(pulse)), mpu)
        total_terms += 1

    if basis_type != "QB_Basis":
        total_penalty += torch.sum(soft_penalty_raw(freqs, minf, mf))
        total_terms += len(freqs)

        total_penalty += torch.sum(soft_penalty(phases, mp))
        total_terms += len(phases)

    # --- Normalize total penalty ---
    if total_terms > 0:
        normalized_primal = total_penalty / total_terms
    else:
        normalized_primal = torch.tensor(0.0)

    return normalized_primal.item()


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

    amps = parameter_subset[:bs]
    freqs = parameter_subset[bs:2 * bs]
    phases = parameter_subset[2 * bs:3 * bs]

    b = 0 if basis_type == "Carrier" else 1

    def soft_penalty(x, limit):
        excess = torch.clamp(torch.abs(x) - limit, min=0.0)
        return (excess / limit) ** 2  # softer and smoother

    def soft_penalty_raw(x, lower, upper):
        lower_violation = torch.clamp(lower - x, min=0.0)
        upper_violation = torch.clamp(x - upper, min=0.0)
        return ((lower_violation / (upper - lower)) ** 2 +
                (upper_violation / (upper - lower)) ** 2)

    total_penalty = 0.0
    total_terms = 0

    if b:
        # Amplitude check
        if torch.any(torch.abs(amps) >= ma):
            print(f"[BOUNDARY HIT] Amplitude limit {ma} reached! Values: {amps[torch.abs(amps) >= ma]}")

        total_penalty += torch.sum(soft_penalty(amps, ma))
        total_terms += len(amps)

        # Pulse check
        max_abs_pulse = torch.max(torch.abs(pulse))
        if max_abs_pulse >= mpu:
            print(f"[BOUNDARY HIT] Pulse limit {mpu} reached! Max abs pulse: {max_abs_pulse.item()}")

        total_penalty += soft_penalty(max_abs_pulse, mpu)
        total_terms += 1

    if basis_type != "QB_Basis":
        # Frequency check
        if torch.any(freqs <= minf) or torch.any(freqs >= mf):
            print(f"[BOUNDARY HIT] Frequency bounds [{minf}, {mf}] reached! Values: {freqs[(freqs <= minf) | (freqs >= mf)]}")

        total_penalty += torch.sum(soft_penalty_raw(freqs, minf, mf))
        total_terms += len(freqs)

        # Phase check
        if torch.any(torch.abs(phases) >= mp):
            print(f"[BOUNDARY HIT] Phase limit {mp} reached! Values: {phases[torch.abs(phases) >= mp]}")

        total_penalty += torch.sum(soft_penalty(phases, mp))
        total_terms += len(phases)

    # --- Normalize total penalty ---
    if total_terms > 0:
        normalized_primal = total_penalty / total_terms
    else:
        normalized_primal = torch.tensor(0.0)

    return normalized_primal.item()


def primal_endpoints_only(pulse, parameter_subset=None, pulse_settings=None):
    """
    Penalizes non-zero values at the first and last samples of pulse.
    Returns a Python float (soft squared penalty, optionally normalized by maximal_pulse).
    """
    # Keep drop-in dtype handling for compatibility
    if isinstance(parameter_subset, np.ndarray):
        _ = torch.tensor(parameter_subset, dtype=torch.float64)  # unused
    if isinstance(pulse, np.ndarray):
        pulse = torch.tensor(pulse, dtype=torch.float64)
    elif not isinstance(pulse, torch.Tensor):
        raise TypeError("pulse must be a numpy array or a torch.Tensor")

    if pulse.numel() == 0:
        return 0.0

    start = pulse[0]
    end = pulse[-1]

    # Optional normalization for scale invariance
    mpu = getattr(pulse_settings, "maximal_pulse", None) if pulse_settings is not None else None
    scale = float(mpu) if (mpu is not None and float(mpu) > 0.0) else 1.0

    penalty = (torch.abs(start)/scale + torch.abs(end)/scale) / 2.0
    return torch.mean(penalty).item()
# --------------------------
# --- Fidelity Utilities ---
# --------------------------

def get_fidelity(current_state, target_state):
    return torch.abs(torch.dot(current_state.conj(), target_state)).item()


# ---------------------------
# --- State Preparation ---
# ---------------------------

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

    return (1 - fidelity) + 1e-3 * primal_value


# -------------------------------
# --- Multi-State Preparation ---
# -------------------------------

def FoM_multi_state_preparation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    initial_target_pairs: List[tuple],
    get_drive_fn: Callable
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
        for i in range(len(pulse_settings_list))
    )

    propagator = get_propagator(get_u, time_grid, drive)

    total_infidelity = 0.0
    for init_idx, target_idx in initial_target_pairs:
        ψ0 = torch.zeros(12, dtype=torch.complex128)
        ψ0[init_idx] = 1.0

        ψ_target = torch.zeros(12, dtype=torch.complex128)
        ψ_target[target_idx] = 1.0

        ψ_final = propagator @ ψ0

        fidelity = torch.abs(torch.dot(ψ_final.conj(), ψ_target)) ** 2
        total_infidelity += (1 - fidelity)

    avg_infidelity = total_infidelity / len(initial_target_pairs)
    return avg_infidelity + 1e-2 * primal_value


def FoM_multi_state_preparation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    initial_target_pairs: List[tuple],
    get_drive_fn: Callable
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
        for i in range(len(pulse_settings_list))
    )

    propagator = get_propagator(get_u, time_grid, drive)

    total_infidelity = 0.0
    phases = []
    for init_idx, target_idx in initial_target_pairs:
        ψ0 = torch.zeros(12, dtype=torch.complex128)
        ψ0[init_idx] = 1.0

        ψ_target = torch.zeros(12, dtype=torch.complex128)
        ψ_target[target_idx] = 1.0

        ψ_final = propagator @ ψ0
        inner_product = torch.dot(ψ_target.conj(), ψ_final)

        fidelity = torch.abs(inner_product) ** 2
        phase_diff = torch.angle(inner_product)
        phases.append(phase_diff.item())

        total_infidelity += (1 - fidelity)

    avg_infidelity = total_infidelity / len(initial_target_pairs)

    return avg_infidelity + 1e-2 * primal_value

def FoM_multi_state_preparation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    initial_target_pairs: List[tuple],
    get_drive_fn: Callable
):
    if isinstance(parameter_set, np.ndarray):
        parameter_set = torch.tensor(parameter_set, dtype=torch.float64)

    # Partition parameters
    bss = [ps.basis_size for ps in pulse_settings_list]
    indices = np.cumsum([0] + [3 * bs for bs in bss])
    parameter_subsets = [
        parameter_set[indices[i]:indices[i + 1]] for i in range(len(bss))
    ]

    # Compute control drive and propagator
    drive = get_drive_fn(time_grid, parameter_set, pulse_settings_list)
    propagator = get_propagator(get_u, time_grid, drive)

    # Compute infidelity and relative phase
    total_infidelity = 0.0
    phases = []

    for init_idx, target_idx in initial_target_pairs:
        ψ0 = torch.zeros(12, dtype=torch.complex128)
        ψ0[init_idx] = 1.0

        ψ_target = torch.zeros(12, dtype=torch.complex128)
        ψ_target[target_idx] = 1.0

        ψ_final = propagator @ ψ0
        inner_product = torch.dot(ψ_target.conj(), ψ_final)

        fidelity = torch.abs(inner_product) ** 2
        phase_diff = torch.angle(inner_product)
        phases.append(phase_diff)

        total_infidelity += (1 - fidelity)

    avg_infidelity = total_infidelity / len(initial_target_pairs)

    # Phase consistency penalty: measure spread
    phases = torch.stack(phases)
    phase_std = torch.std(phases)  # or use range: torch.max - torch.min

    # Primal (hardware penalty)
    primal_value = sum(
        calculate_primal(drive[i], parameter_subsets[i], pulse_settings_list[i])
        for i in range(len(pulse_settings_list))
    )

    return avg_infidelity + 1e-2 * primal_value + 1e-2 * phase_std.item()


def FoM_multi_state_preparation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    initial_target_pairs: List[tuple],
    get_drive_fn: Callable
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
        for i in range(len(pulse_settings_list))
    )

    propagator = get_propagator(get_u, time_grid, drive)

    total_infidelity = 0.0

    # Standard transitions
    for init_idx, target_idx in initial_target_pairs:
        ψ0 = torch.zeros(12, dtype=torch.complex128)
        ψ0[init_idx] = 1.0

        ψ_target = torch.zeros(12, dtype=torch.complex128)
        ψ_target[target_idx] = 1.0

        ψ_final = propagator @ ψ0
        fidelity = torch.abs(torch.dot(ψ_final.conj(), ψ_target)) ** 2
        total_infidelity += (1 - fidelity)

    # Add |+++⟩ → |+++⟩ transition
    basis_indices = [0, 1, 2, 3, 6, 7, 8, 9]
    ψ_plus = torch.zeros(12, dtype=torch.complex128)
    for idx in basis_indices:
        ψ_plus[idx] = 1.0 / np.sqrt(len(basis_indices))  # equal superposition

    ψ_plus_final = propagator @ ψ_plus
    fidelity_plus = torch.abs(torch.dot(ψ_plus_final.conj(), ψ_plus)) ** 2
    total_infidelity += (1 - fidelity_plus)

    num_total_transitions = len(initial_target_pairs) + 1
    avg_infidelity = total_infidelity / num_total_transitions

    return avg_infidelity + 1e-2 * primal_value

def FoM_multi_state_preparation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    initial_target_pairs: List[tuple],
    get_drive_fn: Callable
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
        for i in range(len(pulse_settings_list))
    )

    propagator = get_propagator(get_u, time_grid, drive)

    total_infidelity = 0.0

    # Standard |i⟩ → |j⟩ transitions
    for init_idx, target_idx in initial_target_pairs:
        ψ0 = torch.zeros(12, dtype=torch.complex128)
        ψ0[init_idx] = 1.0
        ψ_target = torch.zeros(12, dtype=torch.complex128)
        ψ_target[target_idx] = 1.0

        ψ_final = propagator @ ψ0
        fidelity = torch.abs(torch.dot(ψ_final.conj(), ψ_target)) ** 2
        total_infidelity += (1 - fidelity)

    # Add Bloch phase consistency check for |+++⟩
    basis_indices = [0, 1, 2, 3, 6, 7, 8, 9]

    # Build |+++⟩ = |+⟩⊗|+⟩⊗|+⟩
    plus = (1 / np.sqrt(2)) * torch.tensor([1.0, 1.0], dtype=torch.complex128)
    ψ_logical_plus = torch.kron(torch.kron(plus, plus), plus)  # shape (8,)

    ψ_plus = torch.zeros(12, dtype=torch.complex128)
    for i, idx in enumerate(basis_indices):
        ψ_plus[idx] = ψ_logical_plus[i]

    ψ_plus_final = propagator @ ψ_plus
    ρ_final = ψ_plus_final[:, None] @ ψ_plus_final[None, :].conj()

    # Project Pauli X and Y for each qubit
    from quantum_operators import pauli_operator_on_qubit
    P = torch.zeros((len(basis_indices), 12), dtype=torch.complex128)
    for i, idx in enumerate(basis_indices):
        P[i, idx] = 1.0

    phase_penalty = 0.0
    for q in range(3):  # Qubits 0, 1, 2
        X_op = P.T @ pauli_operator_on_qubit("X", q) @ P
        Y_op = P.T @ pauli_operator_on_qubit("Y", q) @ P
        x = torch.real(torch.trace(X_op @ ρ_final))
        y = torch.real(torch.trace(Y_op @ ρ_final))
        φ = torch.atan2(y, x)  # phase in XY plane
        phase_penalty += (1 - torch.cos(φ))  # 0 if phase = 0

    avg_infidelity = total_infidelity / (len(initial_target_pairs) + 1)
    return avg_infidelity + 1e-2 * primal_value + 1e-2 * phase_penalty

def FoM_multi_state_preparation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    initial_target_pairs: List[tuple],
    get_drive_fn: Callable
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
        for i in range(len(pulse_settings_list))
    )

    propagator = get_propagator(get_u, time_grid, drive)

    total_infidelity = 0.0

    # Standard |i⟩ → |j⟩ transitions
    for init_idx, target_idx in initial_target_pairs:
        ψ0 = torch.zeros(12, dtype=torch.complex128)
        ψ0[init_idx] = 1.0
        ψ_target = torch.zeros(12, dtype=torch.complex128)
        ψ_target[target_idx] = 1.0

        ψ_final = propagator @ ψ0
        fidelity = torch.abs(torch.dot(ψ_final.conj(), ψ_target)) ** 2
        total_infidelity += (1 - fidelity)

    # Add Bloch phase consistency check for |+++⟩
    basis_indices = [0, 1, 2, 3, 6, 7, 8, 9]

    # Build |+++⟩ = |+⟩⊗|+⟩⊗|+⟩
    plus = (1 / np.sqrt(2)) * torch.tensor([1.0, 1.0], dtype=torch.complex128)
    ψ_logical_plus = torch.kron(torch.kron(plus, plus), plus)  # shape (8,)

    ψ_plus = torch.zeros(12, dtype=torch.complex128)
    for i, idx in enumerate(basis_indices):
        ψ_plus[idx] = ψ_logical_plus[i]

    ψ_plus_final = propagator @ ψ_plus
    ρ_final = ψ_plus_final[:, None] @ ψ_plus_final[None, :].conj()

    # Project Pauli X and Y for each qubit
    from quantum_operators import pauli_operator_on_qubit
    P = torch.zeros((len(basis_indices), 12), dtype=torch.complex128)
    for i, idx in enumerate(basis_indices):
        P[i, idx] = 1.0

    phase_penalty = 0.0
    for q in range(3):  # Qubits 0, 1, 2
        X_op = P.T @ pauli_operator_on_qubit("X", q) @ P
        Y_op = P.T @ pauli_operator_on_qubit("Y", q) @ P
        x = torch.real(torch.trace(X_op @ ρ_final))
        y = torch.real(torch.trace(Y_op @ ρ_final))
        φ = torch.atan2(y, x)  # phase in XY plane
        phase_penalty += (1 - torch.cos(φ))  # 0 if phase = 0

    avg_infidelity = total_infidelity #/ (len(initial_target_pairs) + 1)
    return avg_infidelity + 1e-2 * primal_value + 1e-1* phase_penalty

def FoM_multi_state_preparation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    initial_target_pairs: List[tuple],
    get_drive_fn: Callable
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
        for i in range(len(pulse_settings_list))
    )

    # Use the new endpoint-only penalty
    primal_endpoints_value = sum(
        primal_endpoints_only(drive[i], parameter_subsets[i], pulse_settings_list[i])
        for i in range(len(pulse_settings_list))
    )

    propagator = get_propagator(get_u, time_grid, drive)

    total_infidelity = 0.0

    # Standard |i⟩ → |j⟩ transitions
    for init_idx, target_idx in initial_target_pairs:
        ψ0 = torch.zeros(12, dtype=torch.complex128)
        ψ0[init_idx] = 1.0
        ψ_target = torch.zeros(12, dtype=torch.complex128)
        ψ_target[target_idx] = 1.0

        ψ_final = propagator @ ψ0
        fidelity = torch.abs(torch.dot(ψ_final.conj(), ψ_target)) ** 2
        total_infidelity += (1 - fidelity)

    # Per-qubit phase stabilization in +X direction
    basis_indices = [0, 1, 2, 3, 6, 7, 8, 9]
    P = torch.zeros((len(basis_indices), 12), dtype=torch.complex128)
    for i, idx in enumerate(basis_indices):
        P[i, idx] = 1.0

    # Prepare |+++⟩ state
    plus = (1 / np.sqrt(2)) * torch.tensor([1.0, 1.0], dtype=torch.complex128)
    ψ_logical_plus = torch.kron(torch.kron(plus, plus), plus)
    ψ_plus = torch.zeros(12, dtype=torch.complex128)
    for i, idx in enumerate(basis_indices):
        ψ_plus[idx] = ψ_logical_plus[i]

    ψ_plus_final = propagator @ ψ_plus
    ρ_final = ψ_plus_final[:, None] @ ψ_plus_final[None, :].conj()

    # Encourage ⟨X⟩ ≈ 1 for each qubit
    phase_penalty = 0.0
    for q in range(3):
        X_op = P.T @ pauli_operator_on_qubit("X", q) @ P
        x = torch.real(torch.trace(X_op @ ρ_final))
        Y_op = P.T @ pauli_operator_on_qubit("Y", q) @ P
        y = torch.real(torch.trace(Y_op @ ρ_final))
        phase_penalty += (1.0 - x) * 2 + y * 2 # penalty if qubit not on +X direction
        #phase_penalty += (1.0 - x) ** 2  # penalty if qubit not in +X direction

    # Combine objectives
    avg_infidelity = total_infidelity / 3#(len(initial_target_pairs) )
    return avg_infidelity + 1e-3 * primal_value + 1e-3 * primal_endpoints_value #+ 1e-2 * phase_penalty



def FoM_multi_state_preparation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    initial_target_pairs: List[tuple],
    get_drive_fn: Callable
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
        for i in range(len(pulse_settings_list))
    )

    # Use the new endpoint-only penalty
    primal_endpoints_value = sum(
        primal_endpoints_only(drive[i], parameter_subsets[i], pulse_settings_list[i])
        for i in range(len(pulse_settings_list))
    )

    propagator = get_propagator(get_u, time_grid, drive)

    total_infidelity = 0.0

    # Standard |i⟩ → |j⟩ transitions
    for init_obj, targ_obj in initial_target_pairs:
        # allow either ints (basis indices) or full state vectors
        if isinstance(init_obj, int):
            ψ0 = torch.zeros(12, dtype=torch.complex128, device=propagator.device)
            ψ0[init_obj] = 1.0
        else:
            ψ0 = init_obj  # assume 12-dim normalized vector

        if isinstance(targ_obj, int):
            ψt = torch.zeros(12, dtype=torch.complex128, device=propagator.device)
            ψt[targ_obj] = 1.0
        else:
            ψt = targ_obj  # 12-dim (phase-insensitive overlap below)

        ψf = propagator @ ψ0
        total_infidelity += (1 - torch.abs(torch.vdot(ψt, ψf))**2)

    # Per-qubit phase stabilization in +X direction
    basis_indices = [0, 1, 2, 3, 6, 7, 8, 9]
    P = torch.zeros((len(basis_indices), 12), dtype=torch.complex128)
    for i, idx in enumerate(basis_indices):
        P[i, idx] = 1.0

    # Prepare |+++⟩ state
    plus = (1 / np.sqrt(2)) * torch.tensor([1.0, 1.0], dtype=torch.complex128)
    ψ_logical_plus = torch.kron(torch.kron(plus, plus), plus)
    ψ_plus = torch.zeros(12, dtype=torch.complex128)
    for i, idx in enumerate(basis_indices):
        ψ_plus[idx] = ψ_logical_plus[i]

    ψ_plus_final = propagator @ ψ_plus
    ρ_final = ψ_plus_final[:, None] @ ψ_plus_final[None, :].conj()

    # Encourage ⟨X⟩ ≈ 1 for each qubit
    phase_penalty = 0.0
    for q in range(3):
        X_op = P.T @ pauli_operator_on_qubit("X", q) @ P
        x = torch.real(torch.trace(X_op @ ρ_final))
        Y_op = P.T @ pauli_operator_on_qubit("Y", q) @ P
        y = torch.real(torch.trace(Y_op @ ρ_final))
        phase_penalty += (1.0 - x) * 2 + y * 2 # penalty if qubit not on +X direction
        #phase_penalty += (1.0 - x) ** 2  # penalty if qubit not in +X direction

    # Combine objectives
    avg_infidelity = total_infidelity / (len(initial_target_pairs) )
    return avg_infidelity + 1e-3 * primal_value + 0 * primal_endpoints_value #+ 1e-2 * phase_penalty

# -------------------------
# --- Gate Transformation ---
# -------------------------

def FoM_gate_transformation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    get_drive_fn: Callable,
    target_unitary: torch.Tensor,
    basis_indices: list = [0,1,2,3,6,7,8,9]
):
    if isinstance(parameter_set, np.ndarray):
        parameter_set = torch.tensor(parameter_set, dtype=torch.float64)

    # Partition parameters
    bss = [ps.basis_size for ps in pulse_settings_list]
    indices = np.cumsum([0] + [3 * bs for bs in bss])
    parameter_subsets = [
        parameter_set[indices[i]:indices[i + 1]] for i in range(len(bss))
    ]

    # Compute drives
    drive = get_drive_fn(time_grid, parameter_set, pulse_settings_list)

    # Primal value: pulse energy/cost
    primal_value = sum(
        calculate_primal(drive[i], parameter_subsets[i], pulse_settings_list[i])
        for i in range(len(pulse_settings_list))
    )

    # Compute full propagator
    propagator = get_propagator(get_u, time_grid, drive)  # shape (12x12)

    # Project target unitary into 12D
    P = torch.zeros((len(basis_indices), 12), dtype=torch.complex128)
    for i, idx in enumerate(basis_indices):
        P[i, idx] = 1.0
    U_target_12 = P.T @ target_unitary @ P

    # Fidelity between propagator and target
    overlap = torch.trace(U_target_12.conj().T @ propagator)
    dim = U_target_12.shape[0]
    fidelity = torch.abs(overlap) / dim

    # Return infidelity + regularization
    return (1.0 - fidelity) + 1e-2 * primal_value


def FoM_gate_transformation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    get_drive_fn: Callable,
    target_unitary: torch.Tensor,
    basis_indices: list = [0, 1, 2, 3, 6, 7, 8, 9]
):
    if isinstance(parameter_set, np.ndarray):
        parameter_set = torch.tensor(parameter_set, dtype=torch.float64)

    # Partition parameters
    bss = [ps.basis_size for ps in pulse_settings_list]
    indices = np.cumsum([0] + [3 * bs for bs in bss])
    parameter_subsets = [
        parameter_set[indices[i]:indices[i + 1]] for i in range(len(bss))
    ]

    # Compute drives
    drive = get_drive_fn(time_grid, parameter_set, pulse_settings_list)

    # Primal value: pulse energy/cost
    primal_value = sum(
        calculate_primal(drive[i], parameter_subsets[i], pulse_settings_list[i])
        for i in range(len(pulse_settings_list))
    )

    # Compute full propagator (12x12)
    propagator_full = get_propagator(get_u, time_grid, drive)

    # Slice the propagator down to the computational subspace (8x8)
    U_actual = propagator_full[basis_indices][:, basis_indices]

    # Fidelity between actual evolution and target gate
    #overlap = torch.trace(target_unitary.conj().T @ U_actual)
    #dim = target_unitary.shape[0]
    #fidelity = torch.abs(overlap) / dim

    dim = target_unitary.shape[0]
    overlap = torch.trace(target_unitary.conj().T @ U_actual)
    fidelity = (torch.abs(overlap) ** 2) / (dim ** 2) # from matthias decade QOC review paper

    # Return infidelity + regularization
    return (1.0 - fidelity) + 1e-2 * primal_value


#### Non locals

def FoM_gate_transformation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    get_drive_fn: Callable,
    target_unitary: torch.Tensor,
    basis_indices: list = [0, 1, 2, 3, 6, 7, 8, 9]
):
    if isinstance(parameter_set, np.ndarray):
        parameter_set = torch.tensor(parameter_set, dtype=torch.float64)

    # Partition parameters
    bss = [ps.basis_size for ps in pulse_settings_list]
    indices = np.cumsum([0] + [3 * bs for bs in bss])
    parameter_subsets = [
        parameter_set[indices[i]:indices[i + 1]] for i in range(len(bss))
    ]

    # Compute drives
    drive = get_drive_fn(time_grid, parameter_set, pulse_settings_list)

    # Primal value: pulse energy/cost
    primal_value = sum(
        calculate_primal(drive[i], parameter_subsets[i], pulse_settings_list[i])
        for i in range(len(pulse_settings_list))
    )

    # Compute full propagator (12x12)
    propagator_full = get_propagator(get_u, time_grid, drive)

    # Slice the propagator down to the computational subspace (8x8)
    U_actual = propagator_full[basis_indices][:, basis_indices]

    # Extract diagonal phases (assumes diagonal target + actual)
    phases_U = torch.angle(torch.diagonal(U_actual))
    phases_T = torch.angle(torch.diagonal(target_unitary))
    delta_phi = phases_U - phases_T

    # Compute nonlocal invariant phase difference:
    # ∆φ = ∆φ₁ - ∆φ₂ - ∆φ₃ + ∆φ₄ (example: 4-dim subset of computational space)
    delta_phi_nl = delta_phi[0] - delta_phi[1] - delta_phi[2] + delta_phi[3]

    # Apply optimal local compensation:
    # ∆φ₁ = -∆φ₂ = -∆φ₃ = ∆φ₄ = ∆φ / 4
    optimal_deltas = torch.tensor([
        +delta_phi_nl / 4,
        -delta_phi_nl / 4,
        -delta_phi_nl / 4,
        +delta_phi_nl / 4
    ], dtype=torch.float64)

    # Nonlocal fidelity
    fidelity = torch.mean(torch.cos(optimal_deltas))

    # Return infidelity + regularization
    return (1.0 - fidelity) + 1e-2 * primal_value

def FoM_gate_transformation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    get_drive_fn: Callable,
    target_unitary: torch.Tensor,
    basis_indices: list = [0, 1, 2, 3, 6, 7, 8, 9]
):
    if isinstance(parameter_set, np.ndarray):
        parameter_set = torch.tensor(parameter_set, dtype=torch.float64)

    # Partition parameters
    bss = [ps.basis_size for ps in pulse_settings_list]
    indices = np.cumsum([0] + [3 * bs for bs in bss])
    parameter_subsets = [
        parameter_set[indices[i]:indices[i + 1]] for i in range(len(bss))
    ]

    # Compute drives
    drive = get_drive_fn(time_grid, parameter_set, pulse_settings_list)

    # Primal value: pulse energy/cost
    primal_value = sum(
        calculate_primal(drive[i], parameter_subsets[i], pulse_settings_list[i])
        for i in range(len(pulse_settings_list))
    )

    # Compute full propagator (12x12)
    propagator_full = get_propagator(get_u, time_grid, drive)

    # Slice to computational subspace (8x8)
    U_actual = propagator_full[basis_indices][:, basis_indices]

    # Diagonal phases (assumes diagonal gates)
    phases_U = torch.angle(torch.diagonal(U_actual))
    phases_T = torch.angle(torch.diagonal(target_unitary))
    delta_phi = phases_U - phases_T

    # Nonlocal invariant phase difference
    delta_phi_nl = delta_phi[0] - delta_phi[1] - delta_phi[2] + delta_phi[3]

    # Optimal local compensation
    optimal_deltas = torch.tensor([
        +delta_phi_nl / 4,
        -delta_phi_nl / 4,
        -delta_phi_nl / 4,
        +delta_phi_nl / 4
    ], dtype=torch.float64)

    # Nonlocal fidelity
    #fidelity = torch.mean(torch.cos(optimal_deltas))
    delta_phi_nl = delta_phi[0] - delta_phi[1] - delta_phi[2] + delta_phi[3]
    phase_penalty = torch.abs(delta_phi_nl - np.pi)
    cost = torch.cos(phase_penalty)

    # --- Unitarity check (penalty if not close to unitary) ---
    identity = torch.eye(U_actual.shape[0], dtype=torch.complex128)
    U_dagger_U = U_actual.conj().T @ U_actual
    unitarity_deviation = torch.norm(U_dagger_U - identity, p='fro')**2  # Frobenius norm

    # Weight for unitarity penalty (tunable)
    unitarity_weight = 1e-2

    # Final cost: 1 - fidelity + primal + unitarity penalty
    return  1-cost + 1e-2 * primal_value + unitarity_weight * unitarity_deviation
    # this needs that the third qubit is kept identity

def FoM_gate_transformation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    get_drive_fn: Callable,
    target_unitary: torch.Tensor,
    basis_indices: list = [0, 1, 2, 3, 6, 7, 8, 9]
):
    if isinstance(parameter_set, np.ndarray):
        parameter_set = torch.tensor(parameter_set, dtype=torch.float64)

    # Partition parameters
    bss = [ps.basis_size for ps in pulse_settings_list]
    indices = np.cumsum([0] + [3 * bs for bs in bss])
    parameter_subsets = [
        parameter_set[indices[i]:indices[i + 1]] for i in range(len(bss))
    ]

    # Compute drives
    drive = get_drive_fn(time_grid, parameter_set, pulse_settings_list)

    # Primal value: pulse energy/cost
    primal_value = sum(
        calculate_primal(drive[i], parameter_subsets[i], pulse_settings_list[i])
        for i in range(len(pulse_settings_list))
    )

    # Compute full propagator (12x12)
    propagator_full = get_propagator(get_u, time_grid, drive)

    # Slice to 3-qubit computational subspace (8x8)
    U_actual = propagator_full[basis_indices][:, basis_indices]

    # Diagonal phase differences (assumes diagonal gates)
    phases_U = torch.angle(torch.diagonal(U_actual))
    phases_T = torch.angle(torch.diagonal(target_unitary))
    delta_phi = phases_U - phases_T

    # --- Full 3-qubit nonlocal phase invariant ---
    delta_phi_nl = (
        delta_phi[0] - delta_phi[1] - delta_phi[2] + delta_phi[3]
        - delta_phi[4] + delta_phi[5] + delta_phi[6] - delta_phi[7]
    )

    # Penalize deviation from Δφ_nl = 2π
    target_phase = 2 * np.pi
    #phase_penalty = (delta_phi_nl - target_phase) ** 2
    phase_penalty = 1 - torch.cos((delta_phi_nl - target_phase) / 2)

    # --- Unitarity penalty ---
    identity = torch.eye(U_actual.shape[0], dtype=torch.complex128)
    U_dagger_U = U_actual.conj().T @ U_actual
    unitarity_deviation = torch.norm(U_dagger_U - identity, p='fro') ** 2

    # Regularization weights
    primal_weight = 1e-2
    unitarity_weight = 1e-2
    phase_weight = 1.0

    # Final cost
    cost = (
        phase_weight * phase_penalty
        + primal_weight * primal_value
        + unitarity_weight * unitarity_deviation
    )
    return cost



# NL between NV and C13
def FoM_gate_transformation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    get_drive_fn: Callable,
    target_unitary: torch.Tensor,
    basis_indices: list = [0, 1, 2, 3, 6, 7, 8, 9]
):
    if isinstance(parameter_set, np.ndarray):
        parameter_set = torch.tensor(parameter_set, dtype=torch.float64)

    # Partition parameters
    bss = [ps.basis_size for ps in pulse_settings_list]
    indices = np.cumsum([0] + [3 * bs for bs in bss])
    parameter_subsets = [
        parameter_set[indices[i]:indices[i + 1]] for i in range(len(bss))
    ]

    # Compute drive
    drive = get_drive_fn(time_grid, parameter_set, pulse_settings_list)

    # Primal cost
    primal_value = sum(
        calculate_primal(drive[i], parameter_subsets[i], pulse_settings_list[i])
        for i in range(len(pulse_settings_list))
    )

    # Full unitary
    propagator_full = get_propagator(get_u, time_grid, drive)

    # Project to computational subspace
    U_actual = propagator_full[basis_indices][:, basis_indices]

    # Phases of the diagonal
    phases_U = torch.angle(torch.diagonal(U_actual))

    # --- Nonlocal invariant for qubits 0 and 2 ---
    delta_phi_nl = phases_U[0] - phases_U[1] - phases_U[6] + phases_U[7]

    # --- Penalty: force Q1 (middle qubit) to act like identity ---
    # Q1 index pattern: 0/1 ↔ 2/3, 6/7 ↔ 8/9
    identity_penalty = (
        (phases_U[0] - phases_U[2])**2 +  # Q1: 0-2
        (phases_U[1] - phases_U[3])**2 +  # Q1: 1-3
        (phases_U[4] - phases_U[6])**2 +  # Q1: 6-8 (mapped to 4-6)
        (phases_U[5] - phases_U[7])**2    # Q1: 7-9 (mapped to 5-7)
    )

    # --- Phase penalty (target = π for Z⊗Z) ---
    target_phase = np.pi
    phase_penalty = 1 - torch.abs(torch.cos((delta_phi_nl - target_phase) / 2))

    # --- Unitarity + diagonalization ---
    identity = torch.eye(U_actual.shape[0], dtype=torch.complex128)
    unitarity_dev = torch.norm(U_actual.conj().T @ U_actual - identity, p='fro')**2
    offdiag_penalty = torch.norm(U_actual - torch.diag(torch.diagonal(U_actual)), p='fro') ** 2

    # Final cost
    return (
        phase_penalty +
        1e-2 * primal_value +
        1e-2 * unitarity_dev +
        1e-2 * identity_penalty +
        1e-2 * offdiag_penalty
    )



# NL between NV and N14
def FoM_gate_transformation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    get_drive_fn: Callable,
    target_unitary: torch.Tensor,
    basis_indices: list = [0, 1, 2, 3, 6, 7, 8, 9]
):
    if isinstance(parameter_set, np.ndarray):
        parameter_set = torch.tensor(parameter_set, dtype=torch.float64)

    # Partition parameters
    bss = [ps.basis_size for ps in pulse_settings_list]
    indices = np.cumsum([0] + [3 * bs for bs in bss])
    parameter_subsets = [
        parameter_set[indices[i]:indices[i + 1]] for i in range(len(bss))
    ]

    # Compute drive
    drive = get_drive_fn(time_grid, parameter_set, pulse_settings_list)

    # Primal (e.g. pulse energy)
    primal_value = sum(
        calculate_primal(drive[i], parameter_subsets[i], pulse_settings_list[i])
        for i in range(len(pulse_settings_list))
    )

    # Full propagator (12x12)
    propagator_full = get_propagator(get_u, time_grid, drive)

    # Slice to 3-qubit computational subspace
    U_actual = propagator_full[basis_indices][:, basis_indices]

    # Extract diagonal phases
    phases_U = torch.angle(torch.diagonal(U_actual))
    delta_phi = phases_U  # if target is identity (or you can subtract phases_T)

    # --- Nonlocal invariant for qubit 0 and 1 ---
    delta_phi_nl = delta_phi[0] - delta_phi[1] - delta_phi[2] + delta_phi[3]

    # --- Identity condition for qubit 2 ---
    # That qubit 2 is untouched means:
    # delta_phi[0] == delta_phi[4]
    # delta_phi[1] == delta_phi[5]
    # delta_phi[2] == delta_phi[6]
    # delta_phi[3] == delta_phi[7]

    #identity_penalty = sum(
    #    (delta_phi[i] - delta_phi[i + 4])**2 for i in range(4)
    #)

    identity_penalty = sum(
        (phases_U[i] - phases_U[i + 1])**2
        for i in range(0, 8, 2)
    )

    # --- Phase penalty: favor delta_phi_nl == π (e.g., ZZ gate) ---
    target_phase = np.pi
    phase_penalty = 1 - abs(torch.cos((delta_phi_nl - target_phase) / 2))

    # --- Unitarity penalty ---
    identity = torch.eye(U_actual.shape[0], dtype=torch.complex128)
    unitarity_dev = torch.norm(U_actual.conj().T @ U_actual - identity, p='fro')**2

    offdiag_penalty = torch.norm(U_actual - torch.diag(torch.diagonal(U_actual)), p='fro') ** 2


    # Final cost
    return (
        phase_penalty +
        1e-2 * primal_value +
        1e-2 * unitarity_dev +
        1e-2 * identity_penalty +
        5e-2 * offdiag_penalty
    )


#three qubit NL
def FoM_gate_transformation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    get_drive_fn: Callable,
    target_unitary: torch.Tensor,
    basis_indices: list = [0, 1, 2, 3, 6, 7, 8, 9]
):
    if isinstance(parameter_set, np.ndarray):
        parameter_set = torch.tensor(parameter_set, dtype=torch.float64)

    # Partition parameters
    bss = [ps.basis_size for ps in pulse_settings_list]
    indices = np.cumsum([0] + [3 * bs for bs in bss])
    parameter_subsets = [
        parameter_set[indices[i]:indices[i + 1]] for i in range(len(bss))
    ]

    # Compute drives
    drive = get_drive_fn(time_grid, parameter_set, pulse_settings_list)

    # Primal value (e.g., pulse energy/cost)
    primal_value = sum(
        calculate_primal(drive[i], parameter_subsets[i], pulse_settings_list[i])
        for i in range(len(pulse_settings_list))
    )

    # Full propagator (12x12)
    propagator_full = get_propagator(get_u, time_grid, drive)

    # Restrict to 3-qubit subspace (8x8)
    U_actual = propagator_full[basis_indices][:, basis_indices]

    # Extract diagonal phases (logical space only)
    phases_U = torch.angle(torch.diagonal(U_actual))
    phases_T = torch.angle(torch.diagonal(target_unitary))
    delta_phi = phases_U - phases_T

    # --- 3-qubit nonlocal phase invariant ---
    delta_phi_nl = (
        delta_phi[0] - delta_phi[1] - delta_phi[2] + delta_phi[3]
        - delta_phi[4] + delta_phi[5] + delta_phi[6] - delta_phi[7]
    )

    # --- Smooth cosine-based penalty (minimum at target phase) ---
    target_phase = 2 * np.pi
    phase_penalty = 1 - abs(torch.cos((delta_phi_nl - target_phase) / 2))

    # --- Unitarity regularization ---
    identity = torch.eye(U_actual.shape[0], dtype=torch.complex128)
    U_dagger_U = U_actual.conj().T @ U_actual
    unitarity_deviation = torch.norm(U_dagger_U - identity, p='fro') ** 2

    # Weights (tune if needed)
    phase_weight = 1.0
    primal_weight = 1e-2
    unitarity_weight = 1e-2

    offdiag_penalty = torch.norm(U_actual - torch.diag(torch.diagonal(U_actual)), p='fro') ** 2

    # Final objective (infidelity-type)
    cost = (
        phase_weight * phase_penalty
        + primal_weight * primal_value
        + unitarity_weight * unitarity_deviation
        + offdiag_penalty * 1e-2
    )

    return cost #three qubit NL



# modified gate transformation



#three qubit NL
def FoM_gate_transformation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    get_drive_fn: Callable,
    target_unitary: torch.Tensor,
    basis_indices: list = [0, 1, 2, 3, 6, 7, 8, 9]
):
    if isinstance(parameter_set, np.ndarray):
        parameter_set = torch.tensor(parameter_set, dtype=torch.float64)

    # Partition parameters
    bss = [ps.basis_size for ps in pulse_settings_list]
    indices = np.cumsum([0] + [3 * bs for bs in bss])
    parameter_subsets = [
        parameter_set[indices[i]:indices[i + 1]] for i in range(len(bss))
    ]

    # Compute drives
    drive = get_drive_fn(time_grid, parameter_set, pulse_settings_list)

    # Primal value (e.g., pulse energy/cost)
    primal_value = sum(
        calculate_primal(drive[i], parameter_subsets[i], pulse_settings_list[i])
        for i in range(len(pulse_settings_list))
    )

    # Full propagator (12x12)
    propagator_full = get_propagator(get_u, time_grid, drive)

    # Restrict to 3-qubit subspace (8x8)
    U_actual = propagator_full[basis_indices][:, basis_indices]

    # Extract diagonal phases (logical space only)
    phases_U = torch.angle(torch.diagonal(U_actual))
    phases_T = torch.angle(torch.diagonal(target_unitary))
    delta_phi = phases_U - phases_T

    # --- 3-qubit nonlocal phase invariant ---
    delta_phi_nl = (
        delta_phi[0] - delta_phi[1] - delta_phi[2] + delta_phi[3]
        - delta_phi[4] + delta_phi[5] + delta_phi[6] - delta_phi[7]
    )

    # --- Smooth cosine-based penalty (minimum at target phase) ---
    target_phase = 2 * np.pi
    phase_penalty = 1 - abs(torch.cos((delta_phi_nl - target_phase) / 2))

    # --- Unitarity regularization ---
    identity = torch.eye(U_actual.shape[0], dtype=torch.complex128)
    U_dagger_U = U_actual.conj().T @ U_actual
    unitarity_deviation = torch.norm(U_dagger_U - identity, p='fro') ** 2

    # Weights (tune if needed)
    phase_weight = 1.0
    primal_weight = 1e-2
    unitarity_weight = 1e-2

    offdiag_penalty = torch.norm(U_actual - torch.diag(torch.diagonal(U_actual)), p='fro') ** 2

    # Final objective (infidelity-type)
    cost = (
        phase_weight * phase_penalty
        + primal_weight * primal_value
        + unitarity_weight * unitarity_deviation
        + offdiag_penalty * 1e-2
    )

    return cost #three qubit NL


####



def FoM_gate_transformation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    get_drive_fn: Callable,
    target_unitary: torch.Tensor,
    basis_indices: list = [0, 1, 2, 3, 6, 7, 8, 9]
):
    if isinstance(parameter_set, np.ndarray):
        parameter_set = torch.tensor(parameter_set, dtype=torch.float64)

    # Partition parameters
    bss = [ps.basis_size for ps in pulse_settings_list]
    indices = np.cumsum([0] + [3 * bs for bs in bss])
    parameter_subsets = [
        parameter_set[indices[i]:indices[i + 1]] for i in range(len(bss))
    ]

    # Compute drives
    drive = get_drive_fn(time_grid, parameter_set, pulse_settings_list)

    # Primal value: pulse energy/cost
    primal_value = sum(
        calculate_primal(drive[i], parameter_subsets[i], pulse_settings_list[i])
        for i in range(len(pulse_settings_list))
    )

    # Compute full propagator (12x12)
    propagator_full = get_propagator(get_u, time_grid, drive)

    # Slice the propagator down to the computational subspace (8x8)
    U_actual = propagator_full[basis_indices][:, basis_indices]

    # Fidelity between actual evolution and target gate
    #overlap = torch.trace(target_unitary.conj().T @ U_actual)
    #dim = target_unitary.shape[0]
    #fidelity = torch.abs(overlap) / dim

    dim = target_unitary.shape[0]
    overlap = torch.trace(target_unitary.conj().T @ U_actual)
    fidelity = (torch.abs(overlap) ** 2) / (dim ** 2) # from matthias decade QOC review paper

    # Return infidelity + regularization
    return (1.0 - fidelity) + 0*1e-2 * primal_value


def FoM_gate_transformation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    get_drive_fn: Callable,
    target_unitary: torch.Tensor,
    basis_indices: list = [0, 1, 2, 3, 6, 7, 8, 9]
):
    if isinstance(parameter_set, np.ndarray):
        parameter_set = torch.tensor(parameter_set, dtype=torch.float64)

    # Partition parameters
    bss = [ps.basis_size for ps in pulse_settings_list]
    indices = np.cumsum([0] + [3 * bs for bs in bss])
    parameter_subsets = [
        parameter_set[indices[i]:indices[i + 1]] for i in range(len(bss))
    ]

    # Compute drives
    drive = get_drive_fn(time_grid, parameter_set, pulse_settings_list)

    # Primal cost (can keep off for now or tune later)
    primal_value = sum(
        calculate_primal(drive[i], parameter_subsets[i], pulse_settings_list[i])
        for i in range(len(pulse_settings_list))
    )

    # Propagator and computational subspace
    propagator_full = get_propagator(get_u, time_grid, drive)
    U_actual = propagator_full[basis_indices][:, basis_indices]

    # --- 1. Fidelity with target ---
    overlap = torch.trace(target_unitary.conj().T @ U_actual)
    dim = target_unitary.shape[0]
    fidelity = (torch.abs(overlap) * 2) / (dim * 2)

    # --- 2. Nonlocal phase penalty ---
    # For diagonal gates: extract logical phases
    phases_U = torch.angle(torch.diagonal(U_actual))
    # Choose a nonlocal invariant, e.g.:
    delta_phi_nl = (
        phases_U[0] - phases_U[1] - phases_U[2] + phases_U[3]
        - phases_U[4] + phases_U[5] + phases_U[6] - phases_U[7]
    )
    # Encourage delta_phi_nl = 2π (or π for other gates)
    target_phase = 2 * np.pi
    phase_penalty = 1 - abs(torch.cos((delta_phi_nl - target_phase) / 2))

    # --- 3. (Optional) Unitarity penalty ---
    identity = torch.eye(U_actual.shape[0], dtype=torch.complex128)
    U_dagger_U = U_actual.conj().T @ U_actual
    unitarity_deviation = torch.norm(U_dagger_U - identity, p='fro') ** 2

    # --- Final cost ---
    return (
        0*(1.0 - fidelity) +
        1.0 * phase_penalty +              # critical to break identity degeneracy
        1e-2 * primal_value +               # optional: pulse energy
        0 * unitarity_deviation         # regularize if needed
    )

def FoM_gate_transformation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    get_drive_fn: Callable,
    target_unitary: torch.Tensor,
    basis_indices: list = [0, 1, 2, 3, 6, 7, 8, 9],
    ψ_init: torch.Tensor = None,
    population_weight: float = 1.0
):
    if isinstance(parameter_set, np.ndarray):
        parameter_set = torch.tensor(parameter_set, dtype=torch.float64)

    # Default initial state: |000⟩ in the 8D computational subspace
    if ψ_init is None:
        ψ_init = torch.zeros(8, dtype=torch.complex128)
        ψ_init[0] = 1.0

    # Partition parameters
    bss = [ps.basis_size for ps in pulse_settings_list]
    indices = np.cumsum([0] + [3 * bs for bs in bss])
    parameter_subsets = [
        parameter_set[indices[i]:indices[i + 1]] for i in range(len(bss))
    ]

    # Compute drive
    drive = get_drive_fn(time_grid, parameter_set, pulse_settings_list)

    # Pulse cost
    primal_value = sum(
        calculate_primal(drive[i], parameter_subsets[i], pulse_settings_list[i])
        for i in range(len(pulse_settings_list))
    )

    # Full propagator
    propagator_full = get_propagator(get_u, time_grid, drive)

    # Computational subspace
    U_actual = propagator_full[basis_indices][:, basis_indices]

    # Fidelity to target
    dim = target_unitary.shape[0]
    overlap = torch.trace(target_unitary.conj().T @ U_actual)
    fidelity = (torch.abs(overlap) ** 2) / (dim * dim)

    # --- Block-off-diagonal penalty in B (kills B-mixing) ---
    # In your 2×3×2 lex ordering and after selecting basis_indices=[0,1,2,3,6,7,8,9],
    # the 8×8 subspace is ordered: [|000>,|001>,|010>,|011>,|100>,|101>,|110>,|111>].
    # So within this 8×8, B=0 rows/cols are [0,1,4,5] and B=1 rows/cols are [2,3,6,7].
    B0 = torch.tensor([0, 1, 4, 5], dtype=torch.long, device=U_actual.device)
    B1 = torch.tensor([2, 3, 6, 7], dtype=torch.long, device=U_actual.device)

    U01 = U_actual[B0][:, B1]
    U10 = U_actual[B1][:, B0]
    L_block = torch.linalg.matrix_norm(U01)**2 + torch.linalg.matrix_norm(U10)**2


    # --- Global identity-likeness penalty (phase-invariant) ---
    # F_id = |Tr(U_actual)|^2 / dim^2 is the process fidelity to identity.
    # Adding + w_id * F_id makes "U ≈ I" unattractive when minimizing.
    overlap_id = torch.trace(U_actual)
    F_id = (torch.abs(overlap_id) ** 2) / (dim * dim)

    # Final cost
    return (
        (1.0 - fidelity)
        + 1e-3 * primal_value
        + 0.0 * L_block      # <-- tiny penalty to enforce control-on-B structure
        + 1e-2 * F_id
    )


def FoM_custom_phase_structure(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    get_drive_fn: Callable,
    target_unitary: torch.Tensor,
    basis_indices: list = [0,1,2,3,6,7,8,9]
):
    import math

    if isinstance(parameter_set, np.ndarray):
        parameter_set = torch.tensor(parameter_set, dtype=torch.float64)

    # Partition parameters
    bss = [ps.basis_size for ps in pulse_settings_list]
    indices = np.cumsum([0] + [3 * bs for bs in bss])
    parameter_subsets = [
        parameter_set[indices[i]:indices[i + 1]] for i in range(len(bss))
    ]

    # Compute control drive and propagator
    drive = get_drive_fn(time_grid, parameter_set, pulse_settings_list)
    propagator = get_propagator(get_u, time_grid, drive)  # 12x12

    # Project to computational subspace
    P = torch.zeros((len(basis_indices), 12), dtype=torch.complex128)
    for i, idx in enumerate(basis_indices):
        P[i, idx] = 1.0
    U = P @ propagator @ P.T  # Now U is (8x8)

    # Extract diagonal phases from first 4 basis states
    diag = torch.diagonal(U)[:4]
    phases = torch.angle(diag)  # φ₁, φ₂, φ₃, φ₄

    # Compute phase-based FoM
    φ1, φ2, φ3, φ4 = phases
    phase_term = torch.cos(φ1 - φ2 - φ3 + φ4 - math.pi/4)

    # Compute diagonality error: sum of squared magnitude of off-diagonal entries
    off_diag = U - torch.diag(torch.diagonal(U))
    diagonality_error = torch.sum(torch.abs(off_diag) ** 2).item()

    # Primal penalty
    primal_value = sum(
        calculate_primal(drive[i], parameter_subsets[i], pulse_settings_list[i])
        for i in range(len(pulse_settings_list))
    )

    # Final FoM: want phase_term close to 1, diagonality_error close to 0
    fom = (1 - phase_term.item()) + 1e-2 * diagonality_error + 1e-2 * primal_value
    return fom
# --------------------------
# --- Objective Selector ---
# --------------------------

objective_dictionary: Dict[str, Callable] = {
    "State Preparation": FoM_state_preparation,
    "Gate Transformation": FoM_gate_transformation,
    "Multi-State Preparation": FoM_multi_state_preparation,
    "Custom Phase Structure": FoM_custom_phase_structure
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
    target_gate=None,
    initial_target_pairs=None
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
            ).item()
        return objective_fn

    elif objective_type == "Gate Transformation":
        def objective_fn(x):
            return fom_function(
                get_u,
                time_grid,
                x,
                pulse_settings_list,
                get_drive_fn,
                target_gate
            ).item()
        return objective_fn
    
    elif objective_type == "Custom Phase Structure":
        def objective_fn(x):
            return fom_function(
                get_u,
                time_grid,
                x,
                pulse_settings_list,
                get_drive_fn,
                target_gate,
            ).item()
        return objective_fn

    elif objective_type == "Multi-State Preparation":
        def objective_fn(x):
            return fom_function(
                get_u,
                time_grid,
                x,
                pulse_settings_list,
                initial_target_pairs,
                get_drive_fn
            ).item()
        return objective_fn
    else:
        raise ValueError(f"Unsupported objective type: {objective_type}")



