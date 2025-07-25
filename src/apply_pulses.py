import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from get_drive import get_drive
from quantum_model import get_U


def apply_pulse(get_U, time_grid, drive, ψ_init, Δ):
    """Apply a single pulse to an initial state and return the state trajectory."""
    states = [ψ_init]
    for i in range(len(time_grid)):
        dt = time_grid[1] - time_grid[0]
        Ω = [d[i] for d in drive]  # List[float] at time i
        U = get_U(Ω, dt.item(), time_grid[i].item(), Δ)
        ψ_next = U @ states[-1]
        states.append(ψ_next)
    return torch.stack(states)


def apply_sequence(get_U, time_grid, drive_list, ψ_init, Δ):
    """Apply a sequence of pulses to an initial state."""
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
    """
    Plots all initial→target transitions in one plot.
    """
    time_ns = np.linspace(0, time_grid[-1].item() * 1e9, len(time_grid) + 1)

    plt.figure(figsize=(10, 6))

    for init_idx, target_idx in initial_target_pairs:
        # Prepare initial state
        ψ0 = torch.zeros(12, dtype=torch.complex128)
        ψ0[init_idx] = 1.0

        # Evolve the state
        states = [ψ0]
        for i in range(len(time_grid)):
            dt = time_grid[1] - time_grid[0]
            Ω = [d[i] for d in drive]
            U = get_U(Ω, dt.item(), time_grid[i].item(), Δ)
            ψ_next = U @ states[-1]
            states.append(ψ_next)

        # Compute population in the target state
        pop = torch.stack([torch.abs(s) ** 2 for s in states]).numpy()
        plt.plot(time_ns, pop[:, target_idx], label=f"{init_idx} → {target_idx}")

    plt.xlabel("Time (ns)")
    plt.ylabel("Target Population")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# -----------------------------
# Load pulse from a result folder
# -----------------------------
# Change this to the result folder you want to load
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
# Apply pulse to each init state
# -----------------------------

plot_population_transfers_for_pairs(
    get_U=get_U,
    time_grid=time_grid,
    drive=optimized_drive,
    Δ=Δ,
    initial_target_pairs=[(0, 6), (1, 7), (2, 8), (3, 9)],
    save_path=result_dir
)

# -----------------------------
# Optional: Apply pulse sequence
# -----------------------------
# Example for chaining multiple pulses
# pulse2 = torch.load(...)["drive"]
# pulse_sequence = [optimized_drive, pulse2]
# ψ_start = torch.zeros(12, dtype=torch.complex128)
# ψ_start[0] = 1.0
# states_seq = apply_sequence(get_U, time_grid, pulse_sequence, ψ_start, Δ)
