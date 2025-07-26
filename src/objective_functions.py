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

    # -------------------------------
    # Per-qubit phase stabilization in +X direction
    # -------------------------------

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
    avg_infidelity = total_infidelity / (len(initial_target_pairs) + 1)
    return avg_infidelity + 1e-2 * primal_value + 1e-1 * phase_penalty
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
