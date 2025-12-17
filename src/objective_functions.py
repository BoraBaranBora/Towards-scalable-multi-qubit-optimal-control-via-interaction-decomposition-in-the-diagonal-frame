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
        #if torch.any(torch.abs(amps) >= ma):
        #    print(f"[BOUNDARY HIT] Amplitude limit {ma} reached! Values: {amps[torch.abs(amps) >= ma]}")

        total_penalty += torch.sum(soft_penalty(amps, ma))
        total_terms += len(amps)

        # Pulse check
        max_abs_pulse = torch.max(torch.abs(pulse))
        #if max_abs_pulse >= mpu:
        #    print(f"[BOUNDARY HIT] Pulse limit {mpu} reached! Max abs pulse: {max_abs_pulse.item()}")

        total_penalty += soft_penalty(max_abs_pulse, mpu)
        total_terms += 1

    if basis_type != "QB_Basis":
        # Frequency check
        #if torch.any(freqs <= minf) or torch.any(freqs >= mf):
        #    print(f"[BOUNDARY HIT] Frequency bounds [{minf}, {mf}] reached! Values: {freqs[(freqs <= minf) | (freqs >= mf)]}")

        total_penalty += torch.sum(soft_penalty_raw(freqs, minf, mf))
        total_terms += len(freqs)

        # Phase check
        #if torch.any(torch.abs(phases) >= mp):
        #    print(f"[BOUNDARY HIT] Phase limit {mp} reached! Values: {phases[torch.abs(phases) >= mp]}")

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



################################################################################################
#############################GOOD CZ WITH PI FLIP ON NV####################################################
################################################################################################

def cz_invariant_BC(
    U_full: torch.Tensor,
    basis_idx = [0,1,2,3,6,7,8,9],
    A_value: int = 0
):
    """
    Compute the CZ-like nonlocal invariant between B and C:
        Δ_diag = φ00 - φ01 - φ10 + φ11
    using diagonal phases of the A-fixed subspace.

    Args:
        U_full: 12x12 propagator (torch.complex128).
        basis_idx: computational subspace indices in 12D (order [000,001,010,011,100,101,110,111]).
        A_value: 0 or 1 -> choose which A-manifold to use.
                 A=0 uses states [000,001,010,011]  -> indices [0,1,2,3]
                 A=1 uses states [100,101,110,111]  -> indices [6,7,8,9]

    Returns:
        Delta (wrapped to (-π, π]) and delta = Delta/4 (effective CZ angle).
    """
    idx = torch.tensor(basis_idx, dtype=torch.long, device=U_full.device)
    U8  = U_full[idx][:, idx]

    if A_value == 0:
        sel = [0,1,2,3]  # |A=0, B,C = 00,01,10,11|
    elif A_value == 1:
        sel = [6,7,8,9]  # |A=1, B,C = 00,01,10,11|
    else:
        raise ValueError("A_value must be 0 or 1")

    # Diagonal phases φ_BC = arg(<ABC|U|ABC>) within chosen A-manifold
    phases = [torch.angle(U8[i,i]).item() for i in sel]
    phi00, phi01, phi10, phi11 = phases

    Delta = phi00 - phi01 - phi10 + phi11
    # wrap to (-π, π]
    Delta = (Delta + np.pi) % (2*np.pi) - np.pi
    #delta = 0.25 * Delta
    return Delta#, delta


import torch
import numpy as np

def cz_invariant_BC(
    U_full: torch.Tensor,
    basis_idx = [0,1,2,3,6,7,8,9],
    A_value: int = 0  # kept for compatibility
):
    """
    CZ-like invariant between B and C allowing A to flip.
    Uses 2x2 A-subblocks at fixed BC and half the det(polar) phase:
        Delta = phi00 - phi01 - phi10 + phi11  in (-pi, pi]
    """
    # 8x8 computational subspace [000,001,010,011,100,101,110,111]
    idx = torch.tensor(basis_idx, dtype=torch.long, device=U_full.device)
    U8  = U_full.index_select(0, idx).index_select(1, idx)

    # Group by BC: (000,100)|(001,101)|(010,110)|(011,111)
    order = torch.tensor([0,4, 1,5, 2,6, 3,7], dtype=torch.long, device=U8.device)
    Ubc   = U8.index_select(0, order).index_select(1, order)

    phis = []
    for k in range(4):
        B = Ubc[2*k:2*k+2, 2*k:2*k+2]        # 2x2 over A_out × A_in @ fixed BC
        U,S,Vh = torch.linalg.svd(B)         # polar via SVD
        Q = U @ Vh                            # unitary polar factor
        z = torch.linalg.det(Q)               # |z|≈1
        phis.append(0.5* torch.angle(z))     # <-- halve the determinant phase

    phis = torch.stack(phis)
    Delta = phis[0] - phis[1] - phis[2] + phis[3]
    # wrap to (-π, π]
    Delta = (Delta + torch.pi) % (2*torch.pi) - torch.pi
    return Delta

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


    Delta = cz_invariant_BC(propagator_full)

    # Fidelity to target
    dim = target_unitary.shape[0]
    overlap = torch.trace(target_unitary.conj().T @ U_actual)
    fidelity = (torch.abs(overlap) ** 2) / (dim * dim)

    #offdiag_penalty = torch.norm(U_actual - torch.diag(torch.diagonal(U_actual)), p='fro') ** 2


    # With this (BC-only off-diagonal penalty; A flips allowed within each BC):
    order = torch.tensor([0,4, 1,5, 2,6, 3,7], dtype=torch.long, device=U_actual.device)
    Ubc = U_actual.index_select(0, order).index_select(1, order)

    bc_off = 0.0
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            Bij = Ubc[2*i:2*i+2, 2*j:2*j+2]  # 2x2 block: BC_out=i, BC_in=j (A_out×A_in)
            bc_off = bc_off + torch.linalg.matrix_norm(Bij, ord='fro')**2

    # Optional normalization to keep it in [0,1]:
    offdiag_penalty = bc_off / 8.0

    # prefer Δ = π (full CZ). Works with wrapped Δ in (-π, π].
    #L_cz = 0.5 * (1 + torch.cos(torch.tensor(Delta)))      # == (1 - cos(Δ - π))/2

    # Delta already wrapped to (-pi, pi]
    # distance on circle to π, scaled to [0,1]
    d = ((Delta - np.pi + np.pi) % (2*np.pi)) - np.pi
    L_cz = torch.abs(torch.tensor(d)) / np.pi

    w_cz, w_off = 1.0, 2.0

    # Final cost
    return (
        #(1-torch.tensor(abs(np.cos(abs(torch.pi + Delta)))))
        w_cz * L_cz
        + 1e-3 * primal_value
        + w_off * offdiag_penalty
    )


################################################################################################
################################CNOT#######################################################
################################################################################################

# index groupings for the 8×8 basis [A B C] = [000,001,010,011,100,101,110,111]
_PAIR_ORDER = {
    # 2×2 blocks over A, grouped by (B,C)  <-- use this for CNOT B->A
    "BC": [0,4, 1,5, 2,6, 3,7],
    # 2×2 blocks over B, grouped by (A,C)
    "AC": [0,2, 1,3, 4,6, 5,7],
    # 2×2 blocks over C, grouped by (A, B)
    "AB": [0,1, 2,3, 4,5, 6,7],
}

def cnot_loss_pair(
    U_full: torch.Tensor,
    pair: str = "BC",
    control: str = "B",
    basis_idx = [0,1,2,3,6,7,8,9],
) -> torch.Tensor:
    """
    Phase-robust CNOT loss allowing the third qubit (spectator) to do anything.

    Args
    ----
    U_full : 12x12 (complex) full propagator
    pair   : which two qubits label the 4 blocks (must be one of 'AB','AC','BC').
             The remaining qubit (not in pair) is the TARGET; 2×2 blocks are taken over it.
             Example: pair='BC' -> target=A (same layout as your B->A version).
    control: which qubit (one of the two letters in pair) is the control.
             Blocks with control=0 should look like I on target; control=1 like X on target.
    basis_idx : computational 8-dim subspace indices in the 12-dim basis.

    Returns
    -------
    L_cnot in [0,1], where 0 is a perfect CNOT.
    """
    assert pair in {"AB","AC","BC"}, "pair must be one of {'AB','AC','BC'}"
    assert control in pair, "control must be one of the two letters in pair"

    device = U_full.device
    idx = torch.tensor(basis_idx, dtype=torch.long, device=device)
    U8  = U_full.index_select(0, idx).index_select(1, idx)

    # Reorder so we get 4 blocks labeled by (control, spectator), each block 2×2 over TARGET.
    order = torch.tensor(_PAIR_ORDER[pair], dtype=torch.long, device=device)
    Uord  = U8.index_select(0, order).index_select(1, order)

    I2 = torch.eye(2, dtype=U8.dtype, device=device)
    X2 = torch.tensor([[0,1],[1,0]], dtype=U8.dtype, device=device)

    # Determine whether the control qubit is the first or second letter of pair
    ctrl_first = (pair[0] == control)

    scores = []
    for k in range(4):
        # Blocks enumerate (control, spectator) = (0,0),(0,1),(1,0),(1,1)
        Bk = Uord[2*k:2*k+2, 2*k:2*k+2]          # 2×2 over TARGET (phase-robust via polar)
        U,S,Vh = torch.linalg.svd(Bk)
        Q = U @ Vh

        # Map block index -> control bit depending on whether control is first or second in pair
        ctrl_bit = (k // 2) if ctrl_first else (k % 2)

        T = I2 if ctrl_bit == 0 else X2
        score = torch.abs(torch.trace(T.conj().T @ Q)) / 2.0   # ∈[0,1]
        scores.append(score)

    score_mean = torch.stack(scores).mean()
    L_cnot = 1.0 - score_mean
    return L_cnot


def block_offdiag_penalty(U8: torch.Tensor, pair: str) -> torch.Tensor:
    """Penalize cross-block coupling for the chosen grouping; keeps that pair 'classical'.
       For CNOT B->A we use pair='BC' so C is spectator and A-flips stay inside blocks."""
    order = torch.tensor(_PAIR_ORDER[pair], dtype=torch.long, device=U8.device)
    Uord  = U8.index_select(0, order).index_select(1, order)
    acc = torch.zeros((), dtype=U8.real.dtype, device=U8.device)
    for i in range(4):
        for j in range(4):
            if i == j: continue
            Bij = Uord[2*i:2*i+2, 2*j:2*j+2]
            acc = acc + torch.linalg.matrix_norm(Bij, ord='fro')**2
    return acc / 8.0  # ~[0,1]


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

    if ψ_init is None:
        ψ_init = torch.zeros(8, dtype=torch.complex128, device=parameter_set.device if isinstance(parameter_set, torch.Tensor) else None)
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
    idx = torch.tensor(basis_indices, dtype=torch.long, device=propagator_full.device)
    U_actual = propagator_full.index_select(0, idx).index_select(1, idx)

    # ---------- CNOT(B->A) objective ----------
    L_cnot = cnot_loss_pair(propagator_full,pair='BC', control='B' ,basis_idx=basis_indices)

    # keep BC blocks 'classical' (no mixing between (B,C)); A flips allowed inside
    offdiag_pen = block_offdiag_penalty(U_actual, pair="BC")

    # weights
    w_cnot, w_off = 1.0, 2.0

    cost = w_cnot * L_cnot + w_off * offdiag_pen + 1e-2 * primal_value
    return cost
################################################################################################
                    #NONLOCAL CZ 
################################################################################################
################################################################################################

import torch

# index groupings for the 8×8 basis [A B C] = [000,001,010,011,100,101,110,111]
_PAIR_ORDER = {
    # 2×2 blocks over A, grouped by (B,C)
    "BC": [0,4, 1,5, 2,6, 3,7],
    # 2×2 blocks over B, grouped by (A,C)
    "AC": [0,2, 1,3, 4,6, 5,7],
    # 2×2 blocks over C, grouped by (A, B)
    "AB": [0,1, 2,3, 4,5, 6,7],
}

def cz_invariant_pair(
    U_full: torch.Tensor,
    basis_idx = [0,1,2,3,6,7,8,9],
    pair: str = "BC",
):
    """CZ-like invariant for a chosen qubit pair, allowing the third qubit to flip.
       Uses 2×2 subblocks over the third qubit and half the det(polar) phase."""
    idx = torch.tensor(basis_idx, dtype=torch.long, device=U_full.device)
    U8  = U_full.index_select(0, idx).index_select(1, idx)

    order = torch.tensor(_PAIR_ORDER[pair], dtype=torch.long, device=U8.device)
    Uord  = U8.index_select(0, order).index_select(1, order)

    phis = []
    for k in range(4):
        B = Uord[2*k:2*k+2, 2*k:2*k+2]              # 2×2 over the free qubit
        U,S,Vh = torch.linalg.svd(B)                # polar via SVD
        Q = U @ Vh                                  # unitary polar factor
        z = torch.linalg.det(Q)                     # |z|=1
        phis.append(0.5 * torch.atan2(z.imag, z.real))  # <-- halve the phase

    phis  = torch.stack(phis)
    Delta = phis[0] - phis[1] - phis[2] + phis[3]
    # wrap to (-π, π]
    Delta = torch.atan2(torch.sin(Delta), torch.cos(Delta))
    return Delta

def block_offdiag_penalty(U8: torch.Tensor, pair: str) -> torch.Tensor:
    """Penalize cross-block coupling for the chosen pair (allows flips of the third qubit)."""
    order = torch.tensor(_PAIR_ORDER[pair], dtype=torch.long, device=U8.device)
    Uord  = U8.index_select(0, order).index_select(1, order)
    acc = torch.zeros((), dtype=U8.real.dtype, device=U8.device)
    for i in range(4):
        for j in range(4):
            if i == j: continue
            Bij = Uord[2*i:2*i+2, 2*j:2*j+2]
            acc = acc + torch.linalg.matrix_norm(Bij, ord='fro')**2
    return acc / 8.0  # ~[0,1]



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

    pair = "AC"  # << your Hamiltonian entangles A–C

    Delta = cz_invariant_pair(propagator_full, basis_idx=basis_indices, pair=pair)

    idx = torch.tensor(basis_indices, dtype=torch.long, device=propagator_full.device)
    U_actual = propagator_full.index_select(0, idx).index_select(1, idx)
    offdiag_penalty = block_offdiag_penalty(U_actual, pair=pair)

    # CZ loss (torch-only)
    d = torch.atan2(torch.sin(Delta - torch.pi), torch.cos(Delta - torch.pi))
    L_cz = torch.abs(d) / torch.pi

    w_cz, w_off = 1.0, 2.0
    cost = w_cz * L_cz + w_off * offdiag_penalty + 1e-4 * primal_value

    return cost


################################################################################################
# NONLOCAL CZ WITHOUT SVD
################################################################################################
# --- helpers ---

_PAIR_ORDER = {
    "BC": [0,4, 1,5, 2,6, 3,7],
    "AC": [0,2, 1,3, 4,6, 5,7],
    "AB": [0,1, 2,3, 4,5, 6,7],
}

def reorder_blocks(U8: torch.Tensor, pair: str) -> torch.Tensor:
    order = torch.tensor(_PAIR_ORDER[pair], dtype=torch.long, device=U8.device)
    return U8.index_select(0, order).index_select(1, order)

def block_list(Uord: torch.Tensor):
    # returns [B00,B01,B10,B11] each 2x2 over spectator qubit
    return [Uord[2*k:2*k+2, 2*k:2*k+2] for k in range(4)]

def wrap_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))

def cz_phase_from_dets(Bs):
    # direct (no SVD): z_k = det(B_k) / |det(B_k)|
    phis = []
    for B in Bs:
        z = torch.linalg.det(B)
        # robust unit-modulus projection (avoid NaNs if z ~ 0)
        z_unit = z / (torch.abs(z) + 1e-12)
        phi = 0.5 * torch.atan2(z_unit.imag, z_unit.real)
        phis.append(phi)
    phi00, phi01, phi10, phi11 = phis
    Delta = phi00 - phi01 - phi10 + phi11
    return wrap_pi(Delta)

def cz_phase_from_dets(Bs):
    """
    Direct single-entry phases (no det, no SVD).
    For each 2×2 block B_k (over the spectator qubit), take the phase of B_k[0,0].
    Assumes penalties drive each block ~ diagonal/unitary so this entry reflects the block phase.
    """
    eps = 1e-12
    phis = []
    for B in Bs:
        a = B[0, 0]                            # single entry
        z = a / (torch.abs(a) + eps)           # project to unit circle
        phi = torch.atan2(z.imag, z.real)      # arg(a)
        phis.append(phi)
    phi00, phi01, phi10, phi11 = phis
    Delta = phi00 - phi01 - phi10 + phi11
    return wrap_pi(Delta)

def offdiag_block_penalty(U8: torch.Tensor, pair: str) -> torch.Tensor:
    Uord = reorder_blocks(U8, pair)
    acc = torch.zeros((), dtype=U8.real.dtype, device=U8.device)
    for i in range(4):
        for j in range(4):
            if i == j: continue
            Bij = Uord[2*i:2*i+2, 2*j:2*j+2]
            acc = acc + torch.linalg.matrix_norm(Bij, ord='fro')**2
    return acc / 8.0

def block_unitarity_penalty(U8: torch.Tensor, pair: str) -> torch.Tensor:
    Uord = reorder_blocks(U8, pair)
    I2 = torch.eye(2, dtype=U8.dtype, device=U8.device)
    acc = torch.zeros((), dtype=U8.real.dtype, device=U8.device)
    for k in range(4):
        B = Uord[2*k:2*k+2, 2*k:2*k+2]
        A = B.conj().T @ B - I2
        acc = acc + torch.linalg.matrix_norm(A, ord='fro')**2
    return acc / 4.0

def global_unitarity_penalty(U8: torch.Tensor) -> torch.Tensor:
    I8 = torch.eye(8, dtype=U8.dtype, device=U8.device)
    A = U8.conj().T @ U8 - I8
    return torch.linalg.matrix_norm(A, ord='fro')**2 / 8.0


def spectator_identity_penalty(U8: torch.Tensor, pair: str) -> torch.Tensor:
    Uord = reorder_blocks(U8, pair)
    I2   = torch.eye(2, dtype=U8.dtype, device=U8.device)
    eps  = 1e-12
    acc  = torch.zeros((), dtype=U8.real.dtype, device=U8.device)
    for k in range(4):
        Bk = Uord[2*k:2*k+2, 2*k:2*k+2]
        # phase from trace (robust average)
        t  = torch.trace(Bk)
        z  = t / (torch.abs(t) + eps)          # unit-modulus
        eip = (z.real + 1j*z.imag)             # e^{iφ_k}
        acc = acc + torch.linalg.matrix_norm(Bk - eip * I2, ord='fro')**2
    return acc / 4.0


def spectator_Zonly_commutator_penalty(U8: torch.Tensor, pair: str) -> torch.Tensor:
    """
    Enforce each 2x2 block B_k to be a Z-only unitary:
    (i) unitary: B_k^\dagger B_k ≈ I
    (ii) commutes with Z: [B_k, Z] ≈ 0  ⇒ diagonal in Z basis
    """
    Uord = reorder_blocks(U8, pair)
    I2   = torch.eye(2, dtype=U8.dtype, device=U8.device)
    Z    = torch.tensor([[1., 0.],
                         [0., -1.]], dtype=U8.real.dtype, device=U8.device).type_as(U8)

    acc = torch.zeros((), dtype=U8.real.dtype, device=U8.device)
    for k in range(4):
        Bk = Uord[2*k:2*k+2, 2*k:2*k+2]
        # Unitarity
        acc = acc + torch.linalg.matrix_norm(Bk.conj().T @ Bk - I2, ord='fro')**2
        # Z-only (diagonal in Z basis)
        comm = Bk @ Z - Z @ Bk
        acc = acc + torch.linalg.matrix_norm(comm, ord='fro')**2
    return acc / 4.0

# --- main FoM ---


def FoM_gate_transformation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    get_drive_fn: Callable,
    target_unitary: torch.Tensor,
    basis_indices: list = [0,1,2,3,6,7,8,9],
    ψ_init: torch.Tensor = None,
    population_weight: float = 1.0,
    pair: str = "AC",              # entangle NV(A) – 13C(C), spectator B=14N
    w_cz: float = 1.0,
    w_off: float = 2.0,
    w_bu: float = 0.0,             # block unitarity
    w_gu: float = 1.0,             # global 8×8 unitarity
    w_spec: float = 1.0,
    lam_primal: float = 1e-4
):
    # parameters split
    if isinstance(parameter_set, np.ndarray):
        parameter_set = torch.tensor(parameter_set, dtype=torch.float64)
    bss = [ps.basis_size for ps in pulse_settings_list]
    indices = np.cumsum([0] + [3 * bs for bs in bss])
    parameter_subsets = [parameter_set[indices[i]:indices[i+1]] for i in range(len(bss))]

    # drives and pulse regularizer
    drive = get_drive_fn(time_grid, parameter_set, pulse_settings_list)
    primal_value = sum(
        calculate_primal(drive[i], parameter_subsets[i], pulse_settings_list[i])
        for i in range(len(pulse_settings_list))
    )

    # full propagator and 8×8 computational subspace
    U_full = get_propagator(get_u, time_grid, drive)
    idx = torch.tensor(basis_indices, dtype=torch.long, device=U_full.device)
    U8  =  U_full[basis_indices][:, basis_indices]

    # CZ invariant from direct block determinants (no SVD)
    Uord = reorder_blocks(U8, pair)
    Bs   = block_list(Uord)
    Delta = cz_phase_from_dets(Bs)
    L_cz = torch.abs(wrap_pi(Delta - torch.pi)) / torch.pi

    # penalties: cross-block, block unitarity, global unitarity
    P_off = offdiag_block_penalty(U8, pair)
    P_bu  = block_unitarity_penalty(U8, pair)
    P_gu  = global_unitarity_penalty(U8)
    P_spec = spectator_Zonly_commutator_penalty(U8,pair)

    cost = (w_cz * L_cz
            + w_off * P_off
            + w_bu  * P_bu
            + w_gu  * P_gu
            + w_spec * P_spec
            + lam_primal * primal_value)
    return cost


################################################################################################


import torch

# index groupings for the 8×8 basis [A B C] = [000,001,010,011,100,101,110,111]
_PAIR_ORDER = {
    # 2×2 blocks over A, grouped by (B,C)
    "BC": [0,4, 1,5, 2,6, 3,7],
    # 2×2 blocks over B, grouped by (A,C)
    "AC": [0,2, 1,3, 4,6, 5,7],
    # 2×2 blocks over C, grouped by (A, B)
    "AB": [0,1, 2,3, 4,5, 6,7],
}

def _wrap_pm_pi(x: torch.Tensor) -> torch.Tensor:
    """Angle wrap to (-pi, pi]."""
    return torch.atan2(torch.sin(x), torch.cos(x))

def cz_invariant_pair(
    U_full: torch.Tensor,
    basis_idx = [0,1,2,3,6,7,8,9],
    pair: str = "BC",
):
    """CZ-like invariant for a chosen qubit pair, allowing the third qubit to flip.
       Uses 2×2 subblocks over the third qubit and half the det(polar) phase."""
    idx = torch.tensor(basis_idx, dtype=torch.long, device=U_full.device)
    U8  = U_full.index_select(0, idx).index_select(1, idx)

    order = torch.tensor(_PAIR_ORDER[pair], dtype=torch.long, device=U8.device)
    Uord  = U8.index_select(0, order).index_select(1, order)

    phis = []
    for k in range(4):
        B = Uord[2*k:2*k+2, 2*k:2*k+2]              # 2×2 over the free qubit
        U,S,Vh = torch.linalg.svd(B)                # polar via SVD
        Q = U @ Vh                                  # unitary polar factor
        z = torch.linalg.det(Q)                     # |z|=1
        phis.append(0.5 * torch.atan2(z.imag, z.real))  # <-- halve the phase

    phis  = torch.stack(phis)
    Delta = phis[0] - phis[1] - phis[2] + phis[3]
    # wrap to (-π, π]
    Delta = torch.atan2(torch.sin(Delta), torch.cos(Delta))
    return Delta

def block_offdiag_penalty(U8: torch.Tensor, pair: str) -> torch.Tensor:
    """Penalize cross-block coupling for the chosen pair (allows flips of the third qubit)."""
    order = torch.tensor(_PAIR_ORDER[pair], dtype=torch.long, device=U8.device)
    Uord  = U8.index_select(0, order).index_select(1, order)
    acc = torch.zeros((), dtype=U8.real.dtype, device=U8.device)
    for i in range(4):
        for j in range(4):
            if i == j: continue
            Bij = Uord[2*i:2*i+2, 2*j:2*j+2]
            acc = acc + torch.linalg.matrix_norm(Bij, ord='fro')**2
    return acc / 8.0  # ~[0,1]

################################################################################################
################################################################################################
#  three-qubit nonlocal invariants from diagonal phases 
##

def _phases_from_U8_diag(U8: torch.Tensor) -> torch.Tensor:
    """
    Get the 8 computational-basis phases (|000>..|111|) from an 8x8 block.
    Returns phases wrapped to (-pi, pi], referenced to |000>.
    """
    phi = torch.angle(torch.diag(U8))
    phi = _wrap_pm_pi(phi - phi[0])  # fix global phase reference
    return phi  # shape (8,)

def invariants_3q_nonlocal_from_diag_phases(phi: torch.Tensor):
    """
    Given 8 phases (order: 000,001,010,011,100,101,110,111),
    compute:
      - c1, c2, c3  (1-body)
      - c12, c13, c23 (2-body)
      - c123 (3-body)
    All wrapped to (-pi, pi].
    """
    p000,p001,p010,p011,p100,p101,p110,p111 = phi.tolist()

    # 1-body invariants
    #c1 = (-p000 - p001 - p010 - p011 + p100 + p101 + p110 + p111) / 8.0
    #c2 = (-p000 - p001 + p010 + p011 - p100 - p101 + p110 + p111) / 8.0
    #c3 = (-p000 + p001 - p010 + p011 - p100 + p101 - p110 + p111) / 8.0

    # 2-body invariants (your originals)
    c12  = ( p000 + p001 - p010 - p011 - p100 - p101 + p110 + p111 ) / 8.0
    c13  = ( p000 - p001 + p010 - p011 - p100 + p101 - p110 + p111 ) / 8.0
    c23  = ( p000 - p001 - p010 + p011 + p100 - p101 - p110 + p111 ) / 8.0

    # 3-body invariant
    c123 = ( p000 - p001 - p010 + p011 - p100 + p101 + p110 - p111 ) / 8.0

    return {
        #"c1":   _wrap_pm_pi(torch.tensor(c1,   dtype=phi.dtype, device=phi.device)),
        #"c2":   _wrap_pm_pi(torch.tensor(c2,   dtype=phi.dtype, device=phi.device)),
        #"c3":   _wrap_pm_pi(torch.tensor(c3,   dtype=phi.dtype, device=phi.device)),
        "c12":  _wrap_pm_pi(torch.tensor(c12,  dtype=phi.dtype, device=phi.device)),
        "c13":  _wrap_pm_pi(torch.tensor(c13,  dtype=phi.dtype, device=phi.device)),
        "c23":  _wrap_pm_pi(torch.tensor(c23,  dtype=phi.dtype, device=phi.device)),
        "c123": _wrap_pm_pi(torch.tensor(c123, dtype=phi.dtype, device=phi.device)),

        #"c12":  torch.tensor(c12,  dtype=phi.dtype, device=phi.device),
        #"c13":  torch.tensor(c13,  dtype=phi.dtype, device=phi.device),
        #"c23":  torch.tensor(c23,  dtype=phi.dtype, device=phi.device),
        #"c123": torch.tensor(c123, dtype=phi.dtype, device=phi.device),
    }


def cz_invariants_three(
    U_full: torch.Tensor,
    basis_idx = [0,1,2,3,6,7,8,9],
):
    """
    Extract (c12,c13,c23,c123) from the computational 8×8 block of U_full,
    ignoring all local/global single-qubit Z phases.
    Assumes U is (approximately) diagonal in the computational basis
    (your existing offdiag penalty helps enforce this).
    """
    idx = torch.tensor(basis_idx, dtype=torch.long, device=U_full.device)
    U8  = U_full.index_select(0, idx).index_select(1, idx)
    phi = _phases_from_U8_diag(U8)
    return invariants_3q_nonlocal_from_diag_phases(phi)



def block_offdiag_penalty_simple(
    U_full: torch.Tensor,
    basis_idx = [0,1,2,3,6,7,8,9],
) -> torch.Tensor:
    """
    Penalize all off-diagonal amplitudes in the 8×8 computational block.
    Scale so 0 = perfectly diagonal, ~1 = maximally off-diagonal (no diagonal weight).
    """
    idx = torch.tensor(basis_idx, dtype=torch.long, device=U_full.device)
    U8  = U_full.index_select(0, idx).index_select(1, idx)
    # zero out diagonal
    off = U8 - torch.diag(torch.diag(U8))
    # For a unitary, sum(|U|^2) = 8, so dividing by 8 normalizes to ~[0,1]
    return (off.abs()**2).sum().real / 8.0

def unitarity_penalty(
    U_full: torch.Tensor,
    basis_idx = [0,1,2,3,6,7,8,9],
) -> torch.Tensor:
    """
    Penalize deviation from unitarity in the 8×8 computational block.
    Returns 0 if perfectly unitary, >0 otherwise.
    Normalized so ~1 would mean very non-unitary.
    """
    idx = torch.tensor(basis_idx, dtype=torch.long, device=U_full.device)
    U8  = U_full.index_select(0, idx).index_select(1, idx)

    # Compute U†U - I
    eye = torch.eye(U8.shape[0], dtype=U8.dtype, device=U8.device)
    diff = U8.conj().T @ U8 - eye

    # Frobenius norm squared, normalized by dimension
    return (diff.abs()**2).sum().real / U8.shape[0]

def FoM_gate_transformation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    get_drive_fn: Callable,
    target_unitary: torch.Tensor,
    basis_indices: list = None,
    ψ_init: torch.Tensor = None,
    population_weight: float = 1.0,
    # NEW: optional three-qubit target dict (radians) and weights
    target_3q: dict = {"c13": 0.0, "c12": 0.0, "c23": 0.0, "c123": np.pi/4},
    weights_3q: dict = None,
    # optionally keep your pair settings
    pair: str = "AC",
    w_cz: float = 0.0,
    w_off: float = 1.0,
    w_3q: float = 1.0,
    w_u: float = 1.0,
    w_pairclose: float = 1.0,     # weight for closeness penalty
    pairclose: tuple = ("c12", "c13"),  # which two invariants to tie together
):
    if isinstance(parameter_set, np.ndarray):
        parameter_set = torch.tensor(parameter_set, dtype=torch.float64)
    #parameter_set = parameter_set.to(torch.float64)

    if ψ_init is None:
        ψ_init = torch.zeros(8, dtype=torch.complex128)
        ψ_init[0] = 1.0

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

    propagator_full = get_propagator(get_u, time_grid, drive)

    idx = torch.tensor(basis_indices, dtype=torch.long, device=propagator_full.device)
    U_actual = propagator_full.index_select(0, idx).index_select(1, idx)

    # (optional) fidelity to target
    dim = target_unitary.shape[0]
    overlap = torch.trace(target_unitary.conj().T @ U_actual)
    fidelity = (torch.abs(overlap) ** 2) / (dim * dim)

    # pairwise CZ invariant (your existing piece)
    #Delta = cz_invariant_pair(propagator_full, basis_idx=basis_indices, pair=pair)
    offdiag_penalty = block_offdiag_penalty_simple(propagator_full, basis_idx=basis_indices)
    #d = _wrap_pm_pi(Delta - torch.pi)      # target CZ angle = π
    #L_cz = torch.abs(d) / torch.pi

    unitarity_loss = unitarity_penalty(propagator_full, basis_idx=basis_indices)

    # NEW: 3-qubit nonlocal target loss (only if provided)
    L_3q = torch.tensor(0.0, dtype=torch.float64, device=U_actual.device)
    if target_3q is not None:
        inv3 = cz_invariants_three(propagator_full, basis_idx=basis_indices)
        if weights_3q is None:
            weights_3q = {k: 1.0 for k in target_3q.keys()}
        for k, t in target_3q.items():
            if k not in inv3: continue
            diff = _wrap_pm_pi(inv3[k] - torch.as_tensor(t, dtype=torch.float64, device=U_actual.device))
            #diff = torch.abs(inv3[k] - torch.as_tensor(t, dtype=torch.float64, device=U_actual.device))
            w = torch.as_tensor(weights_3q.get(k, 1.0), dtype=torch.float64, device=U_actual.device)
            # scale like your pair loss: normalized by pi for magnitude
            L_3q = L_3q + w *(1.0 - torch.cos(diff))

    # NEW: encourage two 2-body phases to be close (periodic)
    L_pairclose = torch.tensor(0.0, dtype=torch.float64, device=U_actual.device)
    if w_pairclose != 0.0:
        k1, k2 = pairclose
        if (k1 in inv3) and (k2 in inv3):
            dclose = _wrap_pm_pi(inv3[k1] - inv3[k2])
            L_pairclose = 1.0 - torch.cos(dclose)

    cost =   w_off * offdiag_penalty + w_3q * L_3q + w_u * unitarity_loss + 2e-1 * primal_value #+ w_pairclose* L_pairclose
    return cost


################################################################################################

import math, torch
from quantum_model_3C import get_precomp

def make_three_qubit_basis_indices(
    carbon_pair=(1, 2),    # pick any two active 13C indices from pc['c_indices']
    mI_block=0,            # 14N manifold: 0=+1, 1=0, 2=-1
    electron_map=('0','m1')  # mapping A-bit: ('m1','0') means A=0→|-1_e>, A=1→|0_e>
):
    """
    Build the 8 indices (ordered as [000,001,010,011,100,101,110,111]) inside the full Hilbert space
    for a 3-qubit block: A=electron, B=carbon_pair[0], C=carbon_pair[1].
    Other carbons are frozen to ↑. 14N fixed to mI_block.
    Returns: list[int] of length 8.
    """
    pc = get_precomp()
    c_active = list(pc['c_indices'])       # e.g. [1,2,3,4]
    N_C      = pc['N_C']                   # number of active carbons
    nconf    = 2**N_C                      # number of carbon configurations per 14N block
    dim_nuc  = 3 * nconf                   # nuclear space per electron manifold

    # sanity
    if mI_block not in (0,1,2):
        raise ValueError("mI_block must be 0(+1),1(0),2(-1).")
    for ck in carbon_pair:
        if ck not in c_active:
            raise ValueError(f"Chosen carbon {ck} not in active set {c_active}. Call set_active_carbons([...]) first.")

    # positions of the chosen carbons in the bitstring (LSB=position 0)
    posB = c_active.index(carbon_pair[0])
    posC = c_active.index(carbon_pair[1])

    # helper: make carbon bitstring integer for given (B,C) bits; others forced to 0 (↑)
    def carbon_bits_int(B_bit, C_bit):
        x = 0
        if B_bit == 1: x |= (1 << posB)  # 1 means ↓
        if C_bit == 1: x |= (1 << posC)
        return x  # 0..(2^N_C - 1)

    # electron manifold offsets: |0_e> comes first, then |-1_e>
    def electron_offset(A_bit):
        if electron_map == ('m1','0'):      # A=0→|-1_e>, A=1→|0_e>
            return 0 if A_bit==1 else dim_nuc
        elif electron_map == ('0','m1'):    # A=0→|0_e>,  A=1→|-1_e>
            return 0 if A_bit==0 else dim_nuc
        else:
            raise ValueError("electron_map must be ('m1','0') or ('0','m1').")

    # index builder
    def idx(A,B,C):
        conf = carbon_bits_int(B,C)
        i_nuc = mI_block * nconf + conf              # within one electron manifold
        return electron_offset(A) + i_nuc

    # order: [A B C] in binary counting → 000,001,010,011,100,101,110,111
    basis_indices = [
        idx(0,0,0), idx(0,0,1), idx(0,1,0), idx(0,1,1),
        idx(1,0,0), idx(1,0,1), idx(1,1,0), idx(1,1,1),
    ]
    return basis_indices


#### with grad
def _phases_from_U8_diag(U8: torch.Tensor) -> torch.Tensor:
    """
    Get the 8 computational-basis phases (|000>..|111|) from an 8x8 block.
    Returns phases (you can reference them to |000> if you like).
    """
    phi = torch.angle(torch.diag(U8))
    # optional: reference to |000>
    # phi = phi - phi[0]
    return phi  # shape (8,)

def invariants_3q_nonlocal_from_diag_phases(phi: torch.Tensor):
    """
    Given 8 phases (order: 000,001,010,011,100,101,110,111),
    compute 2-body (c12,c13,c23) and 3-body (c123) invariants.

    All operations stay in torch → gradients preserved.
    """
    # matrix of coefficients
    M = torch.tensor([
        [ 1,  1, -1, -1, -1, -1,  1,  1],  # c12
        [ 1, -1,  1, -1, -1,  1, -1,  1],  # c13
        [ 1, -1, -1,  1,  1, -1, -1,  1],  # c23
        [ 1, -1, -1,  1, -1,  1,  1, -1],  # c123
    ], dtype=phi.dtype, device=phi.device)

    c = (M @ phi) / 8.0  # shape (4,)

    return {
        "c12":  c[0],
        "c13":  c[1],
        "c23":  c[2],
        "c123": c[3],
    }

def cz_invariants_three(
    U_full: torch.Tensor,
    basis_idx = [0,1,2,3,6,7,8,9],
):
    idx = torch.tensor(basis_idx, dtype=torch.long, device=U_full.device)
    U8  = U_full.index_select(0, idx).index_select(1, idx)
    phi = _phases_from_U8_diag(U8)
    return invariants_3q_nonlocal_from_diag_phases(phi)



def FoM_gate_transformation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    get_drive_fn: Callable,
    target_unitary: torch.Tensor,
    basis_indices: list = None,
    ψ_init: torch.Tensor = None,
    population_weight: float = 1.0,
    # NEW: optional three-qubit target dict (radians) and weights
    target_3q: dict = {"c13": 0.0, "c12": 0.0, "c23": 0.0, "c123": np.pi/4},
    weights_3q: dict = None,
    # optionally keep your pair settings
    pair: str = "AC",
    w_cz: float = 0.0,
    w_off: float = 1.0,
    w_3q: float = 1.0,
    w_u: float = 1.0,
    w_pairclose: float = 5.0,     # weight for closeness penalty
    pairclose: tuple = ("c12", "c13"),  # which two invariants to tie together
):
    #if isinstance(parameter_set, np.ndarray):
    #    parameter_set = torch.tensor(parameter_set, dtype=torch.float64)
    parameter_set = parameter_set.to(torch.float64)

    #parameter_set = torch.as_tensor(parameter_set, dtype=torch.float64)

    if ψ_init is None:
        ψ_init = torch.zeros(8, dtype=torch.complex128)
        ψ_init[0] = 1.0

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

    propagator_full = get_propagator(get_u, time_grid, drive)

    idx = torch.tensor(basis_indices, dtype=torch.long, device=propagator_full.device)
    U_actual = propagator_full.index_select(0, idx).index_select(1, idx)

    # (optional) fidelity to target
    #dim = target_unitary.shape[0]
    #overlap = torch.trace(target_unitary.conj().T @ U_actual)
    #fidelity = (torch.abs(overlap) ** 2) / (dim * dim)

    # pairwise CZ invariant (your existing piece)
    #Delta = cz_invariant_pair(propagator_full, basis_idx=basis_indices, pair=pair)
    offdiag_penalty = block_offdiag_penalty_simple(propagator_full, basis_idx=basis_indices)
    #d = _wrap_pm_pi(Delta - torch.pi)      # target CZ angle = π
    #L_cz = torch.abs(d) / torch.pi

    unitarity_loss = unitarity_penalty(propagator_full, basis_idx=basis_indices)

    # NEW: 3-qubit nonlocal target loss (only if provided)
    L_3q = torch.tensor(0.0, dtype=torch.float64, device=U_actual.device)
    if target_3q is not None:
        inv3 = cz_invariants_three(propagator_full, basis_idx=basis_indices)
        if weights_3q is None:
            weights_3q = {k: 1.0 for k in target_3q.keys()}
        for k, t in target_3q.items():
            if k not in inv3:
                continue

            # inv3[k] is already a tensor with grad
            target_k = torch.as_tensor(t, dtype=torch.float64, device=U_actual.device)
            w_k = torch.as_tensor(weights_3q.get(k, 1.0), dtype=torch.float64, device=U_actual.device)

            # cos is 2π-periodic anyway, no need for abs
            diff = inv3[k] - target_k
            #diff = _wrap_pm_pi(inv3[k] - torch.as_tensor(t, dtype=torch.float64, device=U_actual.device))

            L_3q = L_3q + w_k * (1.0 - torch.cos(1*diff))

            #diff = _wrap_pm_pi(inv3[k] - torch.as_tensor(t, dtype=torch.float64, device=U_actual.device))
            #L_3q2 = L_3q + w_k * (1.0 + (diff / torch.pi)**2)

           # NEW: encourage two 2-body phases to be close (periodic)
    #L_pairclose = torch.tensor(0.0, dtype=torch.float64, device=U_actual.device)
    #if w_pairclose != 0.0:
    #    k1, k2 = pairclose
    #    if (k1 in inv3) and (k2 in inv3):
    #        dclose = inv3[k1] - inv3[k2]
    #        L_pairclose = 1.0 - torch.cos(dclose)


        cost =   w_off * offdiag_penalty + w_3q * L_3q + w_u * unitarity_loss + 1e-1 * primal_value #+ 0.4* L_3q2 #+ w_pairclose*L_pairclose
    return cost

# here ends the gradient based

####################################################
#### ccx
####################################################

import torch
import math

# --- helpers to build local 3-qubit sandwiches --------------------------------

def _I2(dtype, device):
    return torch.eye(2, dtype=dtype, device=device)

def _H2(dtype, device):
    return (1.0 / math.sqrt(2.0)) * torch.tensor(
        [[1.0, 1.0],
         [1.0, -1.0]], dtype=dtype, device=device
    )

def _kron3(A, B, C):
    return torch.kron(torch.kron(A, B), C)

def _apply_local_sandwich_3q(U8: torch.Tensor,
                             Ls=None, Rs=None) -> torch.Tensor:
    """
    Apply a local 3-qubit sandwich U' = (L_A⊗L_B⊗L_C) · U8 · (R_A⊗R_B⊗R_C)†
    If Ls/Rs None, use identity on each qubit.
    """
    dtype, device = U8.dtype, U8.device
    if Ls is None: Ls = [_I2(dtype, device)] * 3
    if Rs is None: Rs = [_I2(dtype, device)] * 3
    L = _kron3(Ls[0], Ls[1], Ls[2])
    R = _kron3(Rs[0], Rs[1], Rs[2])
    return L @ U8 @ R.conj().T  # note † on the right

# --- unchanged utility to get wrapped diagonal phases --------------------------

def _phases_from_U8_diag(U8: torch.Tensor) -> torch.Tensor:
    """
    Get the 8 computational-basis phases (|000>..|111|) from an 8x8 block.
    Returns phases wrapped to (-pi, pi], referenced to |000>.
    """
    phi = torch.angle(torch.diag(U8))
    phi = _wrap_pm_pi(phi - phi[0])  # fix global phase reference
    return phi  # shape (8,)



# --- NEW: invariants after optional local sandwich (e.g., H on qubit A) --------

def cz_invariants_three_nondiag(
    U_full: torch.Tensor,
    basis_idx = [0,1,2,3,6,7,8,9],
    # NEW: choose a diagonalizing local frame before extracting phases:
    # e.g., hadamard_on = ['A'] applies H on logical qubit A only.
    hadamard_on: list = None,   # any subset of ['A','B','C']
    Ls: list = None,            # alternatively pass explicit 2x2 Ls [L_A,L_B,L_C]
    Rs: list = None,            # and Rs [R_A,R_B,R_C]; ignored if hadamard_on is used
):
    """
    Extract (c12,c13,c23,c123) from the computational 8×8 block of U_full,
    optionally in a locally rotated (diagonalizing) frame.

    If 'hadamard_on' includes a qubit label, we sandwich with H on that qubit
    before taking diagonal phases, so X-like factors become Z-like in that frame.
    """
    idx = torch.tensor(basis_idx, dtype=torch.long, device=U_full.device)
    U8  = U_full.index_select(0, idx).index_select(1, idx)

    # Build optional local frame
    if hadamard_on is not None:
        A_H = 'A' in hadamard_on
        B_H = 'B' in hadamard_on
        C_H = 'C' in hadamard_on
        H = _H2(U8.dtype, U8.device)
        I = _I2(U8.dtype, U8.device)
        Ls = [H if A_H else I, H if B_H else I, H if C_H else I]
        Rs = Ls  # H = H^\dagger
        U8 = _apply_local_sandwich_3q(U8, Ls=Ls, Rs=Rs)
    elif Ls is not None or Rs is not None:
        # Use explicit local transforms if provided
        dtype, device = U8.dtype, U8.device
        def _ensure(m): 
            return m if m is not None else _I2(dtype, device)
        Ls = [_ensure(Ls[0]), _ensure(Ls[1]), _ensure(Ls[2])] if Ls is not None else None
        Rs = [_ensure(Rs[0]), _ensure(Rs[1]), _ensure(Rs[2])] if Rs is not None else None
        U8 = _apply_local_sandwich_3q(U8, Ls=Ls, Rs=Rs)

    # Extract phases in the chosen (possibly rotated) frame
    phi = _phases_from_U8_diag(U8)
    return invariants_3q_nonlocal_from_diag_phases(phi)

def block_offdiag_penalty_simple_nondiag(
    U_full: torch.Tensor,
    basis_idx = [0,1,2,3,6,7,8,9],
    hadamard_on: list = ['A'],   # any subset of ['A','B','C']
) -> torch.Tensor:
    """
    Penalize all off-diagonal amplitudes in the 8×8 computational block.
    Scale so 0 = perfectly diagonal, ~1 = maximally off-diagonal (no diagonal weight).
    """

    idx = torch.tensor(basis_idx, dtype=torch.long, device=U_full.device)
    U8  = U_full.index_select(0, idx).index_select(1, idx)
    # apply hadamard
    A_H = 'A' in hadamard_on
    B_H = 'B' in hadamard_on
    C_H = 'C' in hadamard_on
    H = _H2(U8.dtype, U8.device)
    I = _I2(U8.dtype, U8.device)
    Ls = [H if A_H else I, H if B_H else I, H if C_H else I]
    Rs = Ls  # H = H^\dagger
    U8 = _apply_local_sandwich_3q(U8, Ls=Ls, Rs=Rs)
    # zero out diagonal
    off = U8 - torch.diag(torch.diag(U8))
    # For a unitary, sum(|U|^2) = 8, so dividing by 8 normalizes to ~[0,1]
    return (off.abs()**2).sum().real / 8.0 # check if this makes XZZ become a fully diagonal gate on the U8



def FoM_gate_transformationXZZ(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    get_drive_fn: Callable,
    target_unitary: torch.Tensor,
    basis_indices: list = None,
    ψ_init: torch.Tensor = None,
    population_weight: float = 1.0,
    # NEW: optional three-qubit target dict (radians) and weights
    target_3q: dict = {"c13": 0.0, "c12": 0.0, "c23": 0.0, "c123": np.pi/4},
    weights_3q: dict = None,
    # optionally keep your pair settings
    pair: str = "AC",
    w_cz: float = 0.0,
    w_off: float = 1.0,
    w_3q: float = 3.0,
    w_u: float = 0.2,
):
    if isinstance(parameter_set, np.ndarray):
        parameter_set = torch.tensor(parameter_set, dtype=torch.float64)

    if ψ_init is None:
        ψ_init = torch.zeros(8, dtype=torch.complex128)
        ψ_init[0] = 1.0

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

    propagator_full = get_propagator(get_u, time_grid, drive)

    idx = torch.tensor(basis_indices, dtype=torch.long, device=propagator_full.device)
    U_actual = propagator_full.index_select(0, idx).index_select(1, idx)

    # (optional) fidelity to target
    dim = target_unitary.shape[0]
    overlap = torch.trace(target_unitary.conj().T @ U_actual)
    fidelity = (torch.abs(overlap) ** 2) / (dim * dim)

    # pairwise CZ invariant (your existing piece)
    #Delta = cz_invariant_pair(propagator_full, basis_idx=basis_indices, pair=pair)
    offdiag_penalty = block_offdiag_penalty_simple_nondiag(propagator_full, basis_idx=basis_indices, hadamard_on=['A'])
    #d = _wrap_pm_pi(Delta - torch.pi)      # target CZ angle = π
    #L_cz = torch.abs(d) / torch.pi

    unitarity_loss = unitarity_penalty(propagator_full, basis_idx=basis_indices)

    # NEW: 3-qubit nonlocal target loss (only if provided)
    L_3q = torch.tensor(0.0, dtype=torch.float64, device=U_actual.device)
    if target_3q is not None:
        #inv3 = cz_invariants_three(propagator_full, basis_idx=basis_indices)
        inv3 = cz_invariants_three_nondiag(propagator_full,
                           basis_idx=basis_indices,
                           hadamard_on=['A'])
        if weights_3q is None:
            weights_3q = {k: 1.0 for k in target_3q.keys()}
        for k, t in target_3q.items():
            if k not in inv3: continue
            diff = _wrap_pm_pi(inv3[k] - torch.as_tensor(t, dtype=torch.float64, device=U_actual.device))
            w = torch.as_tensor(weights_3q.get(k, 1.0), dtype=torch.float64, device=U_actual.device)
            # scale like your pair loss: normalized by pi for magnitude
            L_3q = L_3q + w * (1.0 - torch.cos(diff))#(torch.abs(diff) / math.pi) 

    cost = w_off * offdiag_penalty + w_3q * L_3q + w_u * unitarity_loss + 1e-4 * primal_value #+ w_cz * L_cz 
    return cost


## grad adatpted version of the sandwiching: 

import math
import torch
import numpy as np
from typing import Callable

# ----------------------------
# periodic wrap helper (torch)
# ----------------------------
def _wrap_pm_pi(x: torch.Tensor) -> torch.Tensor:
    # maps to (-pi, pi]
    two_pi = 2.0 * torch.pi
    return (x + torch.pi) % two_pi - torch.pi


# --- helpers to build local 3-qubit sandwiches --------------------------------

def _I2(dtype, device):
    return torch.eye(2, dtype=dtype, device=device)

def _H2(dtype, device):
    # IMPORTANT: construct as torch tensor with correct dtype/device
    # Works for real or complex dtype.
    H = torch.tensor([[1.0,  1.0],
                      [1.0, -1.0]], device=device)
    H = H.to(dtype=dtype)
    return H / math.sqrt(2.0)

def _kron3(A, B, C):
    return torch.kron(torch.kron(A, B), C)

def _apply_local_sandwich_3q(U8: torch.Tensor, Ls=None, Rs=None) -> torch.Tensor:
    """
    U' = (L_A⊗L_B⊗L_C) · U8 · (R_A⊗R_B⊗R_C)†
    """
    dtype, device = U8.dtype, U8.device
    if Ls is None:
        Ls = [_I2(dtype, device) for _ in range(3)]
    if Rs is None:
        Rs = [_I2(dtype, device) for _ in range(3)]
    L = _kron3(Ls[0], Ls[1], Ls[2])
    R = _kron3(Rs[0], Rs[1], Rs[2])
    return L @ U8 @ R.conj().T


# --- diagonal phases (grad-safe) ----------------------------------------------

def _phases_from_U8_diag(U8: torch.Tensor) -> torch.Tensor:
    """
    Get diag phases of 8x8 block in computational order.
    Wrapped to (-pi, pi], referenced to |000>.
    """
    phi = torch.angle(torch.diag(U8))
    phi = _wrap_pm_pi(phi - phi[0])
    return phi


# --- invariants from diag phases (grad-safe constants) -------------------------

def invariants_3q_nonlocal_from_diag_phases(phi: torch.Tensor):
    """
    Given 8 phases [000..111], compute (c12,c13,c23,c123).
    Everything stays in torch => gradients preserved.
    """
    # build M on the fly with correct dtype/device
    M = torch.tensor([
        [ 1,  1, -1, -1, -1, -1,  1,  1],  # c12
        [ 1, -1,  1, -1, -1,  1, -1,  1],  # c13
        [ 1, -1, -1,  1,  1, -1, -1,  1],  # c23
        [ 1, -1, -1,  1, -1,  1,  1, -1],  # c123
    ], dtype=phi.dtype, device=phi.device)

    c = (M @ phi) / 8.0
    return {"c12": c[0], "c13": c[1], "c23": c[2], "c123": c[3]}


# --- invariants after optional local sandwich ---------------------------------

def cz_invariants_three_nondiag(
    U_full: torch.Tensor,
    basis_idx=(0,1,2,3,6,7,8,9),
    hadamard_on=None,   # subset of ['A','B','C']
    Ls=None,            # [L_A,L_B,L_C] each 2x2
    Rs=None,            # [R_A,R_B,R_C] each 2x2
):
    """
    Extract (c12,c13,c23,c123) from the 8x8 computational block,
    optionally in a locally rotated frame (e.g. H on qubit A).
    """
    idx = torch.as_tensor(basis_idx, dtype=torch.long, device=U_full.device)
    U8  = U_full.index_select(0, idx).index_select(1, idx)

    dtype, device = U8.dtype, U8.device

    if hadamard_on is not None:
        H = _H2(dtype, device)
        I = _I2(dtype, device)
        Ls2 = [H if ('A' in hadamard_on) else I,
               H if ('B' in hadamard_on) else I,
               H if ('C' in hadamard_on) else I]
        # H is Hermitian, so R = L is fine
        U8 = _apply_local_sandwich_3q(U8, Ls=Ls2, Rs=Ls2)

    elif (Ls is not None) or (Rs is not None):
        def _ensure(m):
            return m if m is not None else _I2(dtype, device)

        if Ls is None:
            Ls = [_I2(dtype, device) for _ in range(3)]
        else:
            Ls = [_ensure(Ls[0]), _ensure(Ls[1]), _ensure(Ls[2])]

        if Rs is None:
            Rs = [_I2(dtype, device) for _ in range(3)]
        else:
            Rs = [_ensure(Rs[0]), _ensure(Rs[1]), _ensure(Rs[2])]

        U8 = _apply_local_sandwich_3q(U8, Ls=Ls, Rs=Rs)

    phi = _phases_from_U8_diag(U8)
    return invariants_3q_nonlocal_from_diag_phases(phi)


def block_offdiag_penalty_simple_nondiag(
    U_full: torch.Tensor,
    basis_idx=(0,1,2,3,6,7,8,9),
    hadamard_on=('A',),   # subset of ['A','B','C']
) -> torch.Tensor:
    """
    Penalize off-diagonal weight of the 8x8 block in the chosen local frame.
    For a unitary 8x8 block, sum(|U|^2)=8, so /8 normalizes roughly into [0,1].
    """
    idx = torch.as_tensor(basis_idx, dtype=torch.long, device=U_full.device)
    U8  = U_full.index_select(0, idx).index_select(1, idx)

    dtype, device = U8.dtype, U8.device
    H = _H2(dtype, device)
    I = _I2(dtype, device)

    Ls = [H if ('A' in hadamard_on) else I,
          H if ('B' in hadamard_on) else I,
          H if ('C' in hadamard_on) else I]

    U8 = _apply_local_sandwich_3q(U8, Ls=Ls, Rs=Ls)

    off = U8 - torch.diag(torch.diag(U8))
    return (off.abs()**2).sum().real / 8.0


# --- FoM for XZZ-like target in rotated frame ---------------------------------

def FoM_gate_transformation(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    get_drive_fn: Callable,
    target_unitary: torch.Tensor,
    basis_indices=None,
    ψ_init: torch.Tensor=None,
    population_weight: float=1.0,
    target_3q: dict=None,     # e.g. {"c12":0,"c13":0,"c23":0,"c123":pi/4}
    weights_3q: dict=None,
    w_off: float=1.0,
    w_3q: float=3.0,
    w_u: float=1.0,
    w_fid: float=0.0,         # optional
    hadamard_on=('A',),       # use ('A',) for XZZ diagonalization
):
    # keep grads if parameter_set already torch w/ requires_grad
    if isinstance(parameter_set, np.ndarray):
        parameter_set = torch.as_tensor(parameter_set, dtype=torch.float64)
    else:
        parameter_set = parameter_set.to(torch.float64)

    if ψ_init is None:
        ψ_init = torch.zeros(8, dtype=torch.complex128, device=parameter_set.device)
        ψ_init[0] = 1.0

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

    propagator_full = get_propagator(get_u, time_grid, drive)

    idx = torch.as_tensor(basis_indices, dtype=torch.long, device=propagator_full.device)
    U_actual = propagator_full.index_select(0, idx).index_select(1, idx)


    offdiag_penalty = block_offdiag_penalty_simple_nondiag(
        propagator_full, basis_idx=basis_indices, hadamard_on=hadamard_on
    )

    unitarity_loss = unitarity_penalty(propagator_full, basis_idx=basis_indices)

    # 3q invariant loss in the rotated frame
    L_3q = torch.tensor(0.0, dtype=torch.float64, device=U_actual.device)
    if target_3q is not None:
        inv3 = cz_invariants_three_nondiag(
            propagator_full, basis_idx=basis_indices, hadamard_on=hadamard_on
        )
        if weights_3q is None:
            weights_3q = {k: 1.0 for k in target_3q.keys()}

        for k, t in target_3q.items():
            if k not in inv3:
                continue
            target_k = torch.as_tensor(t, dtype=torch.float64, device=U_actual.device)
            w_k = torch.as_tensor(weights_3q.get(k, 1.0), dtype=torch.float64, device=U_actual.device)

            diff = inv3[k] - target_k
            # periodic loss (smooth)
            L_3q = L_3q + w_k * (1.0 - torch.cos(diff))

    cost = (
        w_off * offdiag_penalty
        + w_3q * L_3q
        + w_u * unitarity_loss
        + 1e-4 * primal_value
    )
    return cost
# here end the grad version of XXZ 

################################################################################################
# CCCZ 4 Qubit gate
################################################################################################


# --- add next to your make_three_qubit_basis_indices ---
from quantum_model_3C import get_precomp

def make_four_qubit_basis_indices(
    carbon_triple=(1,2,3),   # 3 active carbons to participate (subset of set_active_carbons(...))
    mI_block=0,              # 14N manifold: 0=+1, 1=0, 2=-1
    electron_map=('0','m1'), # A=1→|-1_e>, A=0→|0_e> (same convention as your 3q helper)
):
    """
    Return the 16 indices (ordered 0000..1111) for A=electron, B=carbon_triple[0], C=..., D=...
    Others carbons are frozen to ↑; 14N fixed to mI_block.
    """
    pc = get_precomp()
    c_active = list(pc['c_indices'])
    N_C      = pc['N_C']
    nconf    = 2**N_C
    dim_nuc  = 3 * nconf

    for ck in carbon_triple:
        if ck not in c_active:
            raise ValueError(f"Chosen carbon {ck} not in active set {c_active}. Call set_active_carbons([...]) first.")

    posB = c_active.index(carbon_triple[0])
    posC = c_active.index(carbon_triple[1])
    posD = c_active.index(carbon_triple[2])

    def carbon_bits_int(B,C,D):
        x = 0
        if B==1: x |= (1<<posB)
        if C==1: x |= (1<<posC)
        if D==1: x |= (1<<posD)
        return x

    def electron_offset(A):
        if electron_map == ('m1','0'):
            return 0 if A==1 else dim_nuc
        elif electron_map == ('0','m1'):
            return 0 if A==0 else dim_nuc
        else:
            raise ValueError("electron_map must be ('m1','0') or ('0','m1').")

    def idx(A,B,C,D):
        conf  = carbon_bits_int(B,C,D)
        i_nuc = mI_block * nconf + conf
        return electron_offset(A) + i_nuc

    # binary order A B C D: 0000..1111
    basis_indices = [idx(A,B,C,D) for A in (0,1) for B in (0,1) for C in (0,1) for D in (0,1)]
    return basis_indices


# --- add next to your invariants helpers ---

def _phases_from_U16_diag(U16: torch.Tensor) -> torch.Tensor:
    """Return 16 phases (wrapped, relative to |0000>) from the 16x16 computational block."""
    phi = torch.angle(torch.diag(U16))
    phi = _wrap_pm_pi(phi - phi[0])
    return phi  # shape (16,)

def _sum_weighted(phi16, weight_fn):
    # phi16 is a length-16 tensor ordered (A,B,C,D) binary counting
    # weight_fn(a,b,c,d) returns +1/-1
    s = 0.0
    k = 0
    for A in (0,1):
        for B in (0,1):
            for C in (0,1):
                for D in (0,1):
                    idx = (A<<3)|(B<<2)|(C<<1)|D
                    s += weight_fn(A,B,C,D) * phi16[idx]
                    k += 1
    return s

def invariants_4q_nonlocal_from_diag_phases(phi16: torch.Tensor):
    """
    Compute all ZZ-type nonlocal coefficients for 4 qubits:
      pairs:   c12,c13,c14,c23,c24,c34
      triples: c123,c124,c134,c234
      quad:    c1234
    using 1/16 sum with alternating signs.
    """
    def w_pair(*Q):
        return lambda a,b,c,d: (-1)**sum([ (a if q==1 else b if q==2 else c if q==3 else d) for q in Q ])

    # pairs (1=A,2=B,3=C,4=D)
    s12 = _sum_weighted(phi16, w_pair(1,2)) / 16.0
    s13 = _sum_weighted(phi16, w_pair(1,3)) / 16.0
    s14 = _sum_weighted(phi16, w_pair(1,4)) / 16.0
    s23 = _sum_weighted(phi16, w_pair(2,3)) / 16.0
    s24 = _sum_weighted(phi16, w_pair(2,4)) / 16.0
    s34 = _sum_weighted(phi16, w_pair(3,4)) / 16.0

    # triples
    s123 = _sum_weighted(phi16, w_pair(1,2,3)) / 16.0
    s124 = _sum_weighted(phi16, w_pair(1,2,4)) / 16.0
    s134 = _sum_weighted(phi16, w_pair(1,3,4)) / 16.0
    s234 = _sum_weighted(phi16, w_pair(2,3,4)) / 16.0

    # genuine 4-body
    s1234 = _sum_weighted(phi16, w_pair(1,2,3,4)) / 16.0

    wrap = lambda x: _wrap_pm_pi(torch.as_tensor(x, dtype=phi16.dtype, device=phi16.device))
    return {
        "c12": wrap(s12), "c13": wrap(s13), "c14": wrap(s14),
        "c23": wrap(s23), "c24": wrap(s24), "c34": wrap(s34),
        "c123": wrap(s123), "c124": wrap(s124), "c134": wrap(s134), "c234": wrap(s234),
        "c1234": wrap(s1234),
    }

def invariants_4q_from_full(
    U_full: torch.Tensor,
    basis_idx = None,  # length 16 indices for 4q block
):
    idx = torch.tensor(basis_idx, dtype=torch.long, device=U_full.device)
    U16 = U_full.index_select(0, idx).index_select(1, idx)
    phi16 = _phases_from_U16_diag(U16)
    return invariants_4q_nonlocal_from_diag_phases(phi16)

def block_offdiag_penalty_4q(U_full: torch.Tensor, basis_idx) -> torch.Tensor:
    idx = torch.tensor(basis_idx, dtype=torch.long, device=U_full.device)
    U16 = U_full.index_select(0, idx).index_select(1, idx)
    off = U16 - torch.diag(torch.diag(U16))
    return (off.abs()**2).sum().real / 16.0

def unitarity_penalty_4q(U_full: torch.Tensor, basis_idx) -> torch.Tensor:
    idx = torch.tensor(basis_idx, dtype=torch.long, device=U_full.device)
    U16 = U_full.index_select(0, idx).index_select(1, idx)
    eye = torch.eye(16, dtype=U16.dtype, device=U16.device)
    diff = U16.conj().T @ U16 - eye
    return (diff.abs()**2).sum().real / 16.0


def FoM_gate_transformation4q(
    get_u: Callable,
    time_grid,
    parameter_set,
    pulse_settings_list,
    get_drive_fn: Callable,
    target_unitary: torch.Tensor,
    basis_indices: list,
    ψ_init: torch.Tensor = None,
    population_weight: float = 1.0,
    # NEW: optional three-qubit target dict (radians) and weights
    target_4q: dict = {"c13": 0.0, "c12": 0.0, "c23": 0.0, "c123": np.pi/8},
    weights_4q: dict = None,
    # optionally keep your pair settings
    pair: str = "AC",
    w_cz: float = 0.0,
    w_off: float = 1.0,
    w_4q: float = 3.0,
    w_u: float = 0.5,
):
    if isinstance(parameter_set, np.ndarray):
        parameter_set = torch.tensor(parameter_set, dtype=torch.float64)

    if ψ_init is None:
        ψ_init = torch.zeros(8, dtype=torch.complex128)
        ψ_init[0] = 1.0

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

    propagator_full = get_propagator(get_u, time_grid, drive)

    idx = torch.tensor(basis_indices, dtype=torch.long, device=propagator_full.device)
    U_actual = propagator_full.index_select(0, idx).index_select(1, idx)

    # (optional) fidelity to target
    dim = target_unitary.shape[0]
    overlap = torch.trace(target_unitary.conj().T @ U_actual)
    fidelity = (torch.abs(overlap) ** 2) / (dim * dim)

    # pairwise CZ invariant (your existing piece)
    #Delta = cz_invariant_pair(propagator_full, basis_idx=basis_indices, pair=pair)
    #offdiag_penalty = block_offdiag_penalty_4q(propagator_full, basis_idx=basis_indices)
    #d = _wrap_pm_pi(Delta - torch.pi)      # target CZ angle = π
    #L_cz = torch.abs(d) / torch.pi

    unitarity_loss = unitarity_penalty_4q(propagator_full, basis_idx=basis_indices)

    # --- modify your FoM_gate_transformation signature to include: ---
    # target_4q: dict = None,
    # weights_4q: dict = None,
    # w_4q: float = 3.0,   # weight for 4q-invariant loss

    # ... inside FoM_gate_transformation, after you compute propagator_full and U_actual:

    # (still keep your 3q piece if you want to run mixed modes)
    L_4q = torch.tensor(0.0, dtype=torch.float64, device=U_actual.device)
    if target_4q is not None:
        inv4 = invariants_4q_from_full(propagator_full, basis_idx=basis_indices)
        if weights_4q is None:
            weights_4q = {k: 1.0 for k in target_4q.keys()}
        for k, t in target_4q.items():
            if k not in inv4: continue
            diff = _wrap_pm_pi(inv4[k] - torch.as_tensor(t, dtype=torch.float64, device=U_actual.device))
            w = torch.as_tensor(weights_4q.get(k, 1.0), dtype=torch.float64, device=U_actual.device)
            L_4q = L_4q + w * (1.0 - torch.cos(diff))  # smooth periodic loss

    # swap in the 4q block penalties in the 4q mode:
    offdiag_penalty = block_offdiag_penalty_4q(propagator_full, basis_idx=basis_indices)
    unitarity_loss  = unitarity_penalty_4q(propagator_full, basis_idx=basis_indices)

    cost = (
        #w_cz * L_cz
        + w_off * offdiag_penalty
        + w_4q * L_4q
        + w_u * unitarity_loss
        + 1e-4 * primal_value
    )

    return cost


################################################################################################
# CNOT by transforming unitary into CZ as if it was a CNOT already and then checking for cz phase invariance
################################################################################################


def _cz_invariant_pair_from_U8(U8: torch.Tensor, pair: str = "AB") -> torch.Tensor:
    """CZ-like nonlocal invariant on a chosen pair, from an 8x8 already in the comp subspace."""
    order = torch.tensor(_PAIR_ORDER[pair], dtype=torch.long, device=U8.device)
    Uord  = U8.index_select(0, order).index_select(1, order)
    phis = []
    for k in range(4):
        B = Uord[2*k:2*k+2, 2*k:2*k+2]         # 2x2 over the spectator
        U,S,Vh = torch.linalg.svd(B)
        Q = U @ Vh
        z = torch.linalg.det(Q)
        phis.append(0.5 * torch.atan2(z.imag, z.real))  # crucial 1/2
    phis = torch.stack(phis)
    Delta = phis[0] - phis[1] - phis[2] + phis[3]
    return torch.atan2(torch.sin(Delta), torch.cos(Delta))  # wrap (-π,π]

def cz_invariant_pair_up_to_H(
    U_full: torch.Tensor,
    basis_idx,
    pair: str = "AB",
    H_on: str = "A"   # <-- set this to the TARGET in the pair
):
    # build 8x8 comp subspace
    idx = torch.tensor(basis_idx, dtype=torch.long, device=U_full.device)
    U8  = U_full.index_select(0, idx).index_select(1, idx)

    # local H on one qubit (others identity)
    rt2 = torch.sqrt(torch.tensor(2.0, dtype=U_full.real.dtype, device=U_full.device))
    H2  = torch.tensor([[1., 1.],[1., -1.]], dtype=U_full.real.dtype, device=U_full.device) / rt2
    H2  = H2.to(dtype=U_full.dtype)  # match complex dtype

    I2 = torch.eye(2, dtype=U_full.dtype, device=U_full.device)
    H_A = torch.kron(torch.kron(H2, I2), I2)
    H_B = torch.kron(torch.kron(I2, H2), I2)
    H_C = torch.kron(torch.kron(I2, I2), H2)
    U_local = {"A": H_A, "B": H_B, "C": H_C}[H_on]

    # Conjugate by H on the target (H is Hermitian & self-inverse)
    Uprime = U_local @ U8 @ U_local

    # Compute Δ on the conjugated 8x8
    return _cz_invariant_pair_from_U8(Uprime, pair=pair)

def FoM_gate_transformation2(
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

    pair = "AB"  # << your Hamiltonian entangles A–C

    H_on = "A"

    Delta = cz_invariant_pair_up_to_H(propagator_full, basis_idx=basis_indices, pair=pair)

    idx = torch.tensor(basis_indices, dtype=torch.long, device=propagator_full.device)
    U_actual = propagator_full.index_select(0, idx).index_select(1, idx)
    offdiag_penalty = block_offdiag_penalty(U_actual, pair=pair)

    # CZ loss (torch-only)
    d = torch.atan2(torch.sin(Delta - torch.pi), torch.cos(Delta - torch.pi))
    L_cz = torch.abs(d) / torch.pi

    w_cz, w_off = 1.0, 0
    cost = w_cz * L_cz + w_off * offdiag_penalty + 1e-2 * primal_value

    return cost



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

def get_goal_function(
    get_u,
    objective_type: str,
    time_grid,
    pulse_settings_list,
    get_drive_fn,
    starting_state=None,
    target_state=None,
    target_gate=None,
    initial_target_pairs=None,
    **fom_kwargs,   # <— NEW: pass-through for extras (e.g., basis_indices)
):
    fom_function = call_optimization_objective(objective_type)

    if objective_type == "State Preparation":
        def objective_fn(x):
            return fom_function(
                get_u, time_grid, x, pulse_settings_list,
                starting_state, target_state, get_drive_fn,
                **fom_kwargs
            ).item()
        return objective_fn

    elif objective_type == "Gate Transformation":
        def objective_fn(x):
            return fom_function(
                get_u, time_grid, x, pulse_settings_list,
                get_drive_fn, target_gate,
                **fom_kwargs
            ).item()
        return objective_fn

    elif objective_type == "Custom Phase Structure":
        def objective_fn(x):
            return fom_function(
                get_u, time_grid, x, pulse_settings_list,
                get_drive_fn, target_gate,
                **fom_kwargs
            ).item()
        return objective_fn

    elif objective_type == "Multi-State Preparation":
        def objective_fn(x):
            return fom_function(
                get_u, time_grid, x, pulse_settings_list,
                initial_target_pairs, get_drive_fn,
                **fom_kwargs
            ).item()
        return objective_fn

    else:
        raise ValueError(f"Unsupported objective type: {objective_type}")



def get_goal_function(
    get_u,
    objective_type: str,
    time_grid,
    pulse_settings_list,
    get_drive_fn,
    starting_state=None,
    target_state=None,
    target_gate=None,
    initial_target_pairs=None,
    use_autograd: bool = False,   # <--- NEW
    **fom_kwargs,
):
    fom_function = call_optimization_objective(objective_type)

    if objective_type == "State Preparation":
        if use_autograd:
            def objective_fn(x: torch.Tensor):
                return fom_function(
                    get_u, time_grid, x, pulse_settings_list,
                    starting_state, target_state, get_drive_fn,
                    **fom_kwargs
                )           # return TENSOR
        else:
            def objective_fn(x):
                return fom_function(
                    get_u, time_grid, x, pulse_settings_list,
                    starting_state, target_state, get_drive_fn,
                    **fom_kwargs
                ).item()    # return FLOAT

    elif objective_type == "Gate Transformation":
        if use_autograd:
            def objective_fn(x: torch.Tensor):
                return fom_function(
                    get_u, time_grid, x, pulse_settings_list,
                    get_drive_fn, target_gate,
                    **fom_kwargs
                )
        else:
            def objective_fn(x):
                return fom_function(
                    get_u, time_grid, x, pulse_settings_list,
                    get_drive_fn, target_gate,
                    **fom_kwargs
                ).item()

    elif objective_type == "Custom Phase Structure":
        if use_autograd:
            def objective_fn(x: torch.Tensor):
                return fom_function(
                    get_u, time_grid, x, pulse_settings_list,
                    get_drive_fn, target_gate,
                    **fom_kwargs
                )
        else:
            def objective_fn(x):
                return fom_function(
                    get_u, time_grid, x, pulse_settings_list,
                    get_drive_fn, target_gate,
                    **fom_kwargs
                ).item()

    elif objective_type == "Multi-State Preparation":
        if use_autograd:
            def objective_fn(x: torch.Tensor):
                return fom_function(
                    get_u, time_grid, x, pulse_settings_list,
                    initial_target_pairs, get_drive_fn,
                    **fom_kwargs
                )
        else:
            def objective_fn(x):
                return fom_function(
                    get_u, time_grid, x, pulse_settings_list,
                    initial_target_pairs, get_drive_fn,
                    **fom_kwargs
                ).item()

    else:
        raise ValueError(f"Unsupported objective type: {objective_type}")

    return objective_fn