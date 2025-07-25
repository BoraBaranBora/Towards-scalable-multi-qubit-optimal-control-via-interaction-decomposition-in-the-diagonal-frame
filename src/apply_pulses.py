import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from get_drive import get_drive
from quantum_model import get_U


def apply_pulse(get_U, time_grid, drive, ψ_init, Δ):
    states = [ψ_init]
    for i in range(len(time_grid)):
        dt = time_grid[1] - time_grid[0]
        Ω = [d[i] for d in drive]
        U = get_U(Ω, dt.item(), time_grid[i].item(), Δ)
        ψ_next = U @ states[-1]
        states.append(ψ_next)
    return torch.stack(states)


def apply_sequence(get_U, time_grid, drive_list, ψ_init, Δ):
    ψ = ψ_init
    all_states = []
    for drive in drive_list:
        states = apply_pulse(get_U, time_grid, drive, ψ, Δ)
        all_states.append(states)
        ψ = states[-1]
    return all_states


def plot_population_transfers_for_pairs(
    get_U,
    time_grid,
    drive,
    Δ,
    initial_target_pairs,
    save_path=None,
    title="Multi‑State Population Transfer"
):
    time_ns = np.linspace(0, time_grid[-1].item() * 1e9, len(time_grid) + 1)
    plt.figure(figsize=(10, 6))

    for init_idx, target_idx in initial_target_pairs:
        ψ0 = torch.zeros(12, dtype=torch.complex128)
        ψ0[init_idx] = 1.0
        states = [ψ0]
        for i in range(len(time_grid)):
            dt = time_grid[1] - time_grid[0]
            Ω = [d[i] for d in drive]
            U = get_U(Ω, dt.item(), time_grid[i].item(), Δ)
            ψ_next = U @ states[-1]
            states.append(ψ_next)
        pop = torch.stack([torch.abs(s) ** 2 for s in states]).numpy()
        plt.plot(time_ns, pop[:, target_idx], label=f"{init_idx} → {target_idx}")

    plt.xlabel("Time (ns)")
    plt.ylabel("Target Population")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path / "population_transfers.png")
    plt.show()


# 🆕 New helper to plot grouped populations
def plot_grouped_populations(states, time_grid, groups, title, save_path=None):
    pop = torch.stack([torch.abs(s) ** 2 for s in states]).numpy()
    time_ns = np.linspace(0, time_grid[-1].item() * 1e9, len(states))

    plt.figure(figsize=(10, 6))
    for label, indices in groups.items():
        pop_sum = pop[:, indices].sum(axis=1)
        plt.plot(time_ns, pop_sum, label=label)

    plt.xlabel("Time (ns)")
    plt.ylabel("Population")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path and save_path.is_dir():
        save_file = save_path / "population_transfer.png"
        plt.savefig(save_file)
    plt.show()


# -----------------------------
# Load pulse from a result folder
# -----------------------------
result_dir = Path("results") / "pulse_2025-07-25_10-36-00"  # <<--- EDIT THIS
checkpoint_path = result_dir / "pulse_solution.pt"

if not checkpoint_path.exists():
    raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, weights_only=False)

x_opt = checkpoint["params"]
f_opt = checkpoint["fom"]
time_grid = checkpoint["time_grid"]
pulse_settings_list = checkpoint["pulse_settings"]
initial_target_pairs = checkpoint["initial_target_pairs"]
Δ = checkpoint["Δ"]
optimized_drive = checkpoint.get("drive", get_drive(time_grid, x_opt, pulse_settings_list))

print(f"Loaded pulse with FoM: {f_opt:.6e}")
print(f"Target pairs: {initial_target_pairs}")

# -----------------------------
# Plot transfer populations
# -----------------------------
plot_population_transfers_for_pairs(
    get_U=get_U,
    time_grid=time_grid,
    drive=optimized_drive,
    Δ=Δ,
    initial_target_pairs=initial_target_pairs,
    save_path=result_dir
)


from quantum_operators import pauli_operator_on_qubit

def compute_bloch_projections(states, qubit_idx):
    """Compute ⟨Z⟩ and ⟨X⟩ for a given qubit across time."""
    # Pauli operators embedded into full system
    Z_op = pauli_operator_on_qubit("Z", qubit_idx)
    X_op = pauli_operator_on_qubit("X", qubit_idx)

    Z_vals = []
    X_vals = []
    for ψ in states:
        ρ = ψ[:, None] @ ψ[None, :].conj()
        Z_vals.append(torch.real(torch.trace(Z_op @ ρ)).item())
        X_vals.append(torch.real(torch.trace(X_op @ ρ)).item())
    return Z_vals, X_vals


def compute_bloch_projections(states, qubit_idx, basis_indices=[0, 1, 2, 3, 6, 7, 8, 9]):
    from quantum_operators import pauli_operator_on_qubit

    # Build embedding projector P: 8x12
    P = torch.zeros(len(basis_indices), 12, dtype=torch.complex128)
    for i, idx in enumerate(basis_indices):
        P[i, idx] = 1.0

    # Pauli operators in 8D subspace
    Z_op_small = pauli_operator_on_qubit("Z", qubit_idx)
    X_op_small = pauli_operator_on_qubit("X", qubit_idx)

    # Embed into 12D full space
    Z_op = P.T @ Z_op_small @ P
    X_op = P.T @ X_op_small @ P

    Z_vals = []
    X_vals = []

    for ψ in states:
        ρ = ψ.view(-1, 1) @ ψ.view(1, -1).conj()
        Z_vals.append(torch.real(torch.trace(Z_op @ ρ)).item())
        X_vals.append(torch.real(torch.trace(X_op @ ρ)).item())

    return Z_vals, X_vals

# Apply pulse once (use init 000 = level 0)
ψ0 = torch.zeros(12, dtype=torch.complex128)
ψ0[0] = 1.0
states = apply_pulse(get_U, time_grid, optimized_drive, ψ0, Δ)

time_ns = np.linspace(0, time_grid[-1].item() * 1e9, len(states))

# Plot Bloch projections
plt.figure(figsize=(12, 6))
for q in range(3):
    z_vals, x_vals = compute_bloch_projections(states, q)
    plt.plot(time_ns, z_vals, label=f"Q{q} ⟨Z⟩")
    plt.plot(time_ns, x_vals, "--", label=f"Q{q} ⟨X⟩")
plt.xlabel("Time (ns)")
plt.ylabel("Bloch Projection")
plt.title("Qubit Bloch Projections Over Time (Init: 000)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(result_dir / "bloch_projections_q0q1q2.png")
plt.show()
# -----------------------------
# Plot per-qubit and leakage population
# -----------------------------

# Choose one ψ0 for analysis
ψ0 = torch.zeros(12, dtype=torch.complex128)
ψ0[0] = 1.0  # Starting from |000⟩

states = apply_pulse(get_U, time_grid, optimized_drive, ψ0, Δ)

# Group definitions
qubit0_groups = {
    "Q0 = 0": [0, 1, 2, 3],
    "Q0 = 1": [6, 7, 8, 9]
}
qubit1_groups = {
    "Q1 = 0": [0, 6],
    "Q1 = 1": [1, 7],
    "Q1 = 2": [2, 8],
    "Q1 = 3": [3, 9]
}
leakage_group = {
    "Leakage": [4, 5, 10, 11]
}

# Plot grouped dynamics
#plot_grouped_populations(states, time_grid, qubit0_groups,
#                         "Qubit Q0 Population", result_dir / "q0_populations.png")
#plot_grouped_populations(states, time_grid, qubit1_groups,
#                         "Qubit Q1 Population", result_dir / "q1_populations.png")
plot_grouped_populations(states, time_grid, leakage_group,
                         "Leakage Population", result_dir / "leakage_population.png")

# -----------------------------
# Optional: Apply pulse sequence
# -----------------------------
# pulse2 = torch.load(...path...)["drive"]
# sequence = [optimized_drive, pulse2]
# ψ_start = torch.zeros(12, dtype=torch.complex128); ψ_start[0] = 1.0
# full_states = apply_sequence(get_U, time_grid, sequence, ψ_start, Δ)
