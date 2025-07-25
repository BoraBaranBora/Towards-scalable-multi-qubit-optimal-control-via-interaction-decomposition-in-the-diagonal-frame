import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from get_drive import get_drive
from quantum_model import get_U
from quantum_operators import pauli_operator_on_qubit

# -----------------------------
# Utility functions
# -----------------------------

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

def compute_bloch_projections(states, qubit_idx, basis_indices=[0,1,2,3,6,7,8,9]):
    P = torch.zeros(len(basis_indices), 12, dtype=torch.complex128)
    for i, idx in enumerate(basis_indices):
        P[i, idx] = 1.0

    Z_op_small = pauli_operator_on_qubit("Z", qubit_idx)
    X_op_small = pauli_operator_on_qubit("X", qubit_idx)
    Z_op = P.T @ Z_op_small @ P
    X_op = P.T @ X_op_small @ P

    Z_vals, X_vals = [], []
    for ψ in states:
        ρ = ψ[:, None] @ ψ[None, :].conj()
        Z_vals.append(torch.real(torch.trace(Z_op @ ρ)).item())
        X_vals.append(torch.real(torch.trace(X_op @ ρ)).item())
    return Z_vals, X_vals

def plot_bloch_projections(states, time_axis, save_path):
    plt.figure(figsize=(12, 6))
    for q in range(3):
        z_vals, x_vals = compute_bloch_projections(states, q)
        plt.plot(time_axis, z_vals, label=f"Q{q} ⟨Z⟩")
        plt.plot(time_axis, x_vals, '--', label=f"Q{q} ⟨X⟩")
    plt.xlabel("Time (ns)")
    plt.ylabel("Projection")
    plt.title("Qubit Bloch Projections Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path / "bloch_projections.png")
    plt.show()

def plot_grouped_populations(states, time_axis, groups, title, save_path):
    pop = torch.stack([torch.abs(s)**2 for s in states]).numpy()
    plt.figure(figsize=(10, 6))
    for label, idxs in groups.items():
        plt.plot(time_axis, pop[:, idxs].sum(axis=1), label=label)
    plt.xlabel("Time (ns)")
    plt.ylabel("Population")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path / f"{title.replace(' ', '_').lower()}.png")
    plt.show()

# -----------------------------
# Load multiple pulses
# -----------------------------
pulse_dirs = [
    Path("results/pulse_2025-07-25_10-36-00"),
    Path("results/pulse_2025-07-25_10-36-00"),
    # Add more if needed
]

pulse_sequence = []
for path in pulse_dirs:
    ckpt = torch.load(path / "pulse_solution.pt", weights_only=False)
    drive = ckpt.get("drive", get_drive(ckpt["time_grid"], ckpt["params"], ckpt["pulse_settings"]))
    pulse_sequence.append(drive)

# Use Δ and time grid from first pulse
ckpt_main = torch.load(pulse_dirs[0] / "pulse_solution.pt", weights_only=False)
Δ = ckpt_main["Δ"]
time_grid = ckpt_main["time_grid"]
dt = (time_grid[1] - time_grid[0]).item()
total_steps = (len(time_grid) + 1) * len(pulse_sequence)
time_axis = np.linspace(0, dt * (total_steps - 1), total_steps) * 1e9

# -----------------------------
# Initial state |000⟩
# -----------------------------
ψ0 = torch.zeros(12, dtype=torch.complex128)
ψ0[0] = 1.0
states_sequence = apply_sequence(get_U, time_grid, pulse_sequence, ψ0, Δ)
states_concat = torch.cat(states_sequence, dim=0)

# -----------------------------
# Output folder
# -----------------------------
output_dir = pulse_dirs[-1]  # Save plots in last pulse folder
print(f"Saving analysis to: {output_dir}")

# -----------------------------
# Plot: Bloch projections
# -----------------------------
plot_bloch_projections(states_concat, time_axis, output_dir)

# -----------------------------
# Plot: Leakage and Qubit populations
# -----------------------------
qubit0_groups = {
    "Q0 = 0": [0,1,2,3],
    "Q0 = 1": [6,7,8,9]
}
qubit1_groups = {
    "Q1 = 0": [0,6],
    "Q1 = 1": [1,7],
    "Q1 = 2": [2,8],
    "Q1 = 3": [3,9]
}
leakage_group = {
    "Leakage": [4,5,10,11]
}

plot_grouped_populations(states_concat, time_axis, qubit0_groups, "Qubit Q0 Population", output_dir)
plot_grouped_populations(states_concat, time_axis, qubit1_groups, "Qubit Q1 Population", output_dir)
plot_grouped_populations(states_concat, time_axis, leakage_group, "Leakage Population", output_dir)
