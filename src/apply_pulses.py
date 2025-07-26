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

def apply_sequence(get_U, time_grids, drive_list, ψ_init, Δ):
    ψ = ψ_init
    all_states = []
    for drive, tg in zip(drive_list, time_grids):
        states = apply_pulse(get_U, tg, drive, ψ, Δ)
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
    #Path("results/pulse_2025-07-25_11-59-35"),
    #Path("results/pulse_2025-07-25_13-34-44"),
    #Path("results/pulse_2025-07-25_18-15-24"),
    #Path("results/pulse_2025-07-25_18-25-28"),
    #Path("results/pulse_2025-07-25_18-49-16"),
    #Path("results/pulse_2025-07-25_20-56-15"),
    #Path("results/pulse_2025-07-25_22-19-36"),
    #Path("results/pulse_2025-07-26_10-40-10"),
    Path("results/pulse_2025-07-26_11-57-48"),







    # Add more if needed
]

#pulse_sequence = []
#for path in pulse_dirs:
#    ckpt = torch.load(path / "pulse_solution.pt", weights_only=False)
#    drive = ckpt.get("drive", get_drive(ckpt["time_grid"], ckpt["params"], ckpt["pulse_settings"]))
#    pulse_sequence.append(drive)
    
pulse_sequence = []
time_grids = []
for path in pulse_dirs:
    ckpt = torch.load(path / "pulse_solution.pt", weights_only=False)
    drive = ckpt.get("drive", get_drive(ckpt["time_grid"], ckpt["params"], ckpt["pulse_settings"]))
    pulse_sequence.append(drive)
    time_grids.append(ckpt["time_grid"])

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
#states_sequence = apply_sequence(get_U, time_grid, pulse_sequence, ψ0, Δ)
states_sequence = apply_sequence(get_U, time_grids, pulse_sequence, ψ0, Δ)
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


def compute_qubit_phases(states, qubit_idx, basis_indices=[0,1,2,3,6,7,8,9]):
    """
    Computes phase φ = atan2(⟨Y⟩, ⟨X⟩) for a single qubit over time.
    """
    P = torch.zeros(len(basis_indices), 12, dtype=torch.complex128)
    for i, idx in enumerate(basis_indices):
        P[i, idx] = 1.0

    X_op_small = pauli_operator_on_qubit("X", qubit_idx)
    Y_op_small = pauli_operator_on_qubit("Y", qubit_idx)
    X_op = P.T @ X_op_small @ P
    Y_op = P.T @ Y_op_small @ P

    phases = []
    for ψ in states:
        ρ = ψ[:, None] @ ψ[None, :].conj()
        x = torch.real(torch.trace(X_op @ ρ)).item()
        y = torch.real(torch.trace(Y_op @ ρ)).item()
        φ = np.arctan2(y, x)
        phases.append(φ)
    return phases


def plot_qubit_phases(states, time_axis, save_path):
    plt.figure(figsize=(10, 6))
    for q in range(3):  # change if you have more or fewer qubits
        phases = compute_qubit_phases(states, qubit_idx=q)
        phases = np.unwrap(phases)  # Optional: unwrap to avoid jumps
        plt.plot(time_axis, phases, label=f"Q{q} Phase (X-Y plane)")

    plt.xlabel("Time (ns)")
    plt.ylabel("Phase (radians)")
    plt.title("Qubit Phases in X-Y Plane")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path / "qubit_phases.png")
    plt.show()

plot_qubit_phases(states_concat, time_axis, output_dir)



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_bloch_vector_trajectory(states, qubit_idx, basis_indices=[0,1,2,3,6,7,8,9]):
    """Computes ⟨X⟩, ⟨Y⟩, ⟨Z⟩ over time for a single qubit."""
    P = torch.zeros(len(basis_indices), 12, dtype=torch.complex128)
    for i, idx in enumerate(basis_indices):
        P[i, idx] = 1.0

    X_op = P.T @ pauli_operator_on_qubit("X", qubit_idx) @ P
    Y_op = P.T @ pauli_operator_on_qubit("Y", qubit_idx) @ P
    Z_op = P.T @ pauli_operator_on_qubit("Z", qubit_idx) @ P

    bloch_coords = []
    for ψ in states:
        ρ = ψ[:, None] @ ψ[None, :].conj()
        x = torch.real(torch.trace(X_op @ ρ)).item()
        y = torch.real(torch.trace(Y_op @ ρ)).item()
        z = torch.real(torch.trace(Z_op @ ρ)).item()
        bloch_coords.append((x, y, z))

    return np.array(bloch_coords)  # shape: (T, 3)


from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import cm, colors

def plot_bloch_trajectories_3d(states, save_path=None):
    fig = plt.figure(figsize=(14, 4))
    qubit_indices = [0, 1, 2]

    for i, q in enumerate(qubit_indices):
        coords = compute_bloch_vector_trajectory(states, q)  # shape (T, 3)
        ax = fig.add_subplot(1, 3, i+1, projection='3d')

        ax.set_title(f"Qubit {q} Bloch Trajectory")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel("⟨X⟩")
        ax.set_ylabel("⟨Y⟩")
        ax.set_zlabel("⟨Z⟩")

        # Draw Bloch sphere
        u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.1, linewidth=0)

        # Coordinate axes
        ax.quiver(0, 0, 0, 1, 0, 0, color='r', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', arrow_length_ratio=0.1)

        # Trajectory with light-to-dark color gradient
        points = coords.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Define a custom normalization from light to dark
        cmap = cm.get_cmap("viridis")
        norm = colors.Normalize(vmin=0, vmax=len(segments))
        color_vals = cmap(norm(np.arange(len(segments))))

        # Adjust alpha/lightness — fade in
        min_alpha = 0.1
        max_alpha = 1.0
        alphas = np.linspace(min_alpha, max_alpha, len(segments))
        color_vals[:, 3] = alphas  # Set alpha channel

        lc = Line3DCollection(segments, colors=color_vals, linewidth=2)
        ax.add_collection3d(lc)

        # Final state marker
        ax.scatter(*coords[-1], color='black', s=50)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path / "bloch_trajectories_3d_fade.png")
    plt.show()

plot_bloch_trajectories_3d(states_concat, save_path=output_dir)



def create_plus_state_in_12d(basis_indices=[0,1,2,3,6,7,8,9]):
    """
    Creates a |+++⟩-like state projected into the 12D basis.
    Assumes indices [0,1,2,3,6,7,8,9] correspond to valid computational basis states.
    """
    from itertools import product

    qubit_basis = {
        '0': torch.tensor([1.0, 0.0], dtype=torch.complex128),
        '1': torch.tensor([0.0, 1.0], dtype=torch.complex128),
        '+': (1/np.sqrt(2)) * torch.tensor([1.0, 1.0], dtype=torch.complex128)
    }

    ψ_full = torch.zeros(12, dtype=torch.complex128)
    basis_map = {idx: i for i, idx in enumerate(basis_indices)}

    for idx in basis_indices:
        # Decode the qubit configuration of this index
        # e.g., if idx=6 represents |101⟩, then config = '101'
        config = format(idx, '04b')  # adjust if 3 qubits encoded differently
        config = config[-3:]  # truncate to last 3 bits if needed

        # Build amplitude as product of ⟨config|+⟩
        amp = 1.0
        for bit in config:
            if bit == '0':
                amp *= 1/np.sqrt(2)  # ⟨0|+⟩
            elif bit == '1':
                amp *= 1/np.sqrt(2)  # ⟨1|+⟩
            else:
                amp *= 0.0
        ψ_full[idx] = amp

    # Normalize (just in case)
    ψ_full = ψ_full / torch.norm(ψ_full)
    return ψ_full

ψ0 = create_plus_state_in_12d()

states_sequence = apply_sequence(get_U, time_grids, pulse_sequence, ψ0, Δ)
states_concat = torch.cat(states_sequence, dim=0)
plot_bloch_trajectories_3d(states_concat, save_path=output_dir)


import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm, colors

def compute_xy_trajectory(states, qubit_idx, basis_indices=[0,1,2,3,6,7,8,9]):
    """Compute ⟨X⟩, ⟨Y⟩ over time for one qubit."""
    P = torch.zeros(len(basis_indices), 12, dtype=torch.complex128)
    for i, idx in enumerate(basis_indices):
        P[i, idx] = 1.0

    X_op = P.T @ pauli_operator_on_qubit("X", qubit_idx) @ P
    Y_op = P.T @ pauli_operator_on_qubit("Y", qubit_idx) @ P

    xy = []
    for ψ in states:
        ρ = ψ[:, None] @ ψ[None, :].conj()
        x = torch.real(torch.trace(X_op @ ρ)).item()
        y = torch.real(torch.trace(Y_op @ ρ)).item()
        xy.append((x, y))
    return np.array(xy)  # shape (T, 2)

def plot_xy_phase_trajectories(states, save_path=None):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    qubit_indices = [0, 1, 2]

    for i, q in enumerate(qubit_indices):
        coords = compute_xy_trajectory(states, q)  # shape (T, 2)
        ax = axs[i]

        ax.set_title(f"Qubit {q} Phase Trajectory (XY plane)")
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_aspect('equal')
        ax.set_xlabel("⟨X⟩")
        ax.set_ylabel("⟨Y⟩")

        # Draw unit circle
        ax.add_patch(plt.Circle((0, 0), 1.0, color='lightgray', fill=False, linestyle='--'))

        # Create fading color segments
        points = coords.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        #cmap = cm.colormaps["viridis"]
        #norm = colors.Normalize(vmin=0, vmax=len(segments))
        #color_vals = cmap(norm(np.arange(len(segments))))

        cmap = cm.get_cmap("viridis")
        norm = colors.Normalize(vmin=0, vmax=len(segments))
        color_vals = cmap(norm(np.arange(len(segments))))


        min_alpha = 0.1
        max_alpha = 1.0
        alphas = np.linspace(min_alpha, max_alpha, len(segments))
        color_vals[:, 3] = alphas  # Apply fading alpha

        lc = LineCollection(segments, colors=color_vals, linewidth=2)
        ax.add_collection(lc)

        # Final state marker
        ax.plot(*coords[-1], 'ko', markersize=6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path / "xy_phase_trajectories.png")
    plt.show()

plot_xy_phase_trajectories(states_concat, save_path=output_dir)


def compute_single_qubit_entropy(states, qubit_idx, basis_indices=[0,1,2,3,6,7,8,9]):
    """
    Computes the von Neumann entropy S(ρ_A) of a single qubit reduced density matrix over time.
    This quantifies entanglement with the rest of the system.
    """
    import scipy.linalg

    entropy_vals = []

    for ψ in states:
        ρ = ψ[:, None] @ ψ[None, :].conj()  # full density matrix (12x12)

        # Project into computational subspace
        P = torch.zeros(len(basis_indices), 12, dtype=torch.complex128)
        for i, idx in enumerate(basis_indices):
            P[i, idx] = 1.0
        ρ_small = P @ ρ @ P.T  # (8x8)

        # Reshape to 3-qubit tensor
        ρ_tensor = ρ_small.view(2, 2, 2, 2, 2, 2)  # shape: (2,2,2)x(2,2,2)

        # Partial trace over other qubits
        if qubit_idx == 0:
            ρA = torch.einsum("abcdef->bf", ρ_tensor)  # trace over 0th qubit
        elif qubit_idx == 1:
            ρA = torch.einsum("abcdef->df", ρ_tensor)  # trace over 1st qubit
        elif qubit_idx == 2:
            ρA = torch.einsum("abcdef->af", ρ_tensor)  # trace over 2nd qubit
        else:
            raise ValueError("Only qubit indices 0, 1, 2 are supported.")

        # Convert to NumPy and compute eigenvalues
        ρA = ρA.cpu().numpy()
        eigvals = np.linalg.eigvalsh(ρA)
        eigvals = np.clip(eigvals, 1e-12, 1.0)
        entropy = -np.sum(eigvals * np.log2(eigvals))
        entropy_vals.append(entropy)

    return entropy_vals


def plot_qubit_entropies(states, time_axis, save_path=None):
    plt.figure(figsize=(10, 6))
    for q in range(3):
        ent = compute_single_qubit_entropy(states, qubit_idx=q)
        plt.plot(time_axis, ent, label=f"Qubit {q}")

    plt.xlabel("Time (ns)")
    plt.ylabel("Entropy S(ρ_A)")
    plt.title("Single-Qubit Entanglement Entropy Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path / "qubit_entropies.png")
    plt.show()

plot_qubit_entropies(states_concat, time_axis, save_path=output_dir)