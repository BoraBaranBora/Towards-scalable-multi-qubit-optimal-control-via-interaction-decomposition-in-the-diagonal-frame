import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from pulse_settings import PulseSettings, get_initial_guess
from get_drive import get_drive
from objective_functions import get_goal_function
from evolution import get_time_grid, get_evolution_vector
from quantum_model import get_U, Λ_s, Λ00, Λ01, Λ10, Λ11, Λm10, Λm11, γ_e

from custom_gates import ccz_gate  # or wherever you defined it


from cma_runner import (
    initialize_cmaes,
    cmaes_iteration_step,
    unnormalize_params
)
from nelder_mead import call_initialization as nm_init, call_algo_step as nm_step


# -----------------------------
# Step 1: Define pulse settings
# -----------------------------
# "What field amplitude gives me a 5 MHz Rabi frequency if each Tesla gives 175,929.1886 MHz of rotation?"
#  in the case of the electron?"

Ω_rabi = 2 * np.pi * 15e6 # in(rad/s) because the simulation time is in seconds
B_max = Ω_rabi / γ_e  # Units: (rad/s) / (rad/μs/T) = μT
print(B_max)
pulse_settings_list = [
    PulseSettings(
        basis_type="Custom",
        basis_size=5,
        maximal_pulse=B_max,               # Or total integral limit
        maximal_amplitude=B_max/2,             # Normalized: optimizer outputs b(t) ∈ [-1, 1]
        maximal_frequency=35 * 2*np.pi * 1e6,
        minimal_frequency=-35 * 2*np.pi * 1e6,
        maximal_phase=10*np.pi,
        channel_type="MW"
    )
]

# -----------------------------
# Step 2: Simulation parameters
# -----------------------------
duration_ns = 330
steps_per_ns = 10 # show mathematicallz why this is fine
time_grid = get_time_grid(duration_ns, steps_per_ns)

# -----------------------------
# Step 3: Detuning Δ
# -----------------------------
#Λ_dict = {
#    0: Λ00, 1: Λ01, 2: Λ00, 3: Λ01, 4: Λ00, 5: Λ01,
#    6: Λ10, 7: Λ11, 8: Λ10, 9: Λ11, 10: Λ10, 11: Λ11
#}

Λ_dict = {
    0: Λ10,  1: Λ11,
    2: Λ00,  3: Λ01,
    4: Λm10, 5: Λm11,
    6: Λ10,  7: Λ11,
    8: Λ00,  9: Λ01,
    10: Λm10, 11: Λm11
}

initial_target_pairs = [(0, 6), (1, 7), (2, 8), (3, 9)]
# Use the first pair to compute global Δ
#Δ = (Λ_dict[initial_target_pairs[0][1]] - Λ_s).item()
Δ = (Λ_dict[2] - Λ_s).item()


# -----------------------------
# Step 4: Multi-State Preparation objective
# -----------------------------
objective_type = "Multi-State Preparation"  # or "Multi-State Preparation"


if objective_type == "Gate Transformation":
    X = torch.tensor([[0,1],[1,0]], dtype=torch.complex128)
    Y = torch.tensor([[0,-1j],[1j,0]], dtype=torch.complex128)
    Z = torch.tensor([[1,0],[0,-1]], dtype=torch.complex128)
    I = torch.eye(2, dtype=torch.complex128)

    def kron3(a,b,c): return torch.kron(torch.kron(a,b), c)

    # Projectors on B
    P0 = (I + Z)/2  # |0><0|
    P1 = (I - Z)/2  # |1><1|
    P0_full = kron3(I, P0, I)
    P1_full = kron3(I, P1, I)

    # iSWAP(θ): exp(-i θ/2 (XX + YY)) on A–C.  θ=π/2 → iSWAP
    theta = np.pi/2
    H_XY_AC_full = kron3(X, I, X) + kron3(Y, I, Y)  # acts on A and C (B is spectator)
    U_AC_full = torch.matrix_exp(-1j * (theta/2) * H_XY_AC_full)  # I_B ⊗ iSWAP_AC

    # Controlled on B: |0><0|_B ⊗ I + |1><1|_B ⊗ iSWAP_AC
    target_gate = P0_full + P1_full @ U_AC_full
    objective_config = {"target_gate": target_gate}
    
if objective_type == "Gate Transformation":
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    I = torch.eye(2, dtype=torch.complex128)
    #target_gate = torch.kron(torch.kron(X, I), I)
    #objective_config = {"target_gate": target_gate}


    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    XZZ = torch.kron(torch.kron(X, Z), Z)
    target_gate = torch.matrix_exp(-1j * (np.pi / 4) * XZZ) 
    objective_config = {"target_gate": target_gate}

    #target_gate = ccz_gate()
    #objective_config = {"target_gate": target_gate}


elif objective_type == "Multi-State Preparation":
    initial_target_pairs = [(0, 6), (1, 7), (2, 8), (3, 9)]
    #initial_target_pairs = [ (0, 0), (1, 7), (2, 2), (3, 9)]

    #initial_target_pairs = [(0, 6), (2, 8)]
    objective_config = {"initial_target_pairs": initial_target_pairs}


elif objective_type == "Custom Phase Structure":
    objective_config = {"target_gate": None}  # no additional inputs needed

else:
    raise ValueError(f"Unsupported objective type: {objective_type}")

goal_fn = get_goal_function(
    get_u=lambda Ω, dt, t: get_U(Ω, dt, t, Δ),
    objective_type=objective_type,
    time_grid=time_grid,
    pulse_settings_list=pulse_settings_list,
    get_drive_fn=get_drive,
    **objective_config  # Use config to avoid duplication
)

# -----------------------------
# Step 5: Load or Generate x0
# -----------------------------
use_previous = True
resume_from = "results/pulse_2025-07-26_20-33-48"  # Path to previous result

if use_previous:
    try:
        # Load from previous checkpoint
        ckpt = torch.load(Path(resume_from) / "pulse_solution.pt", map_location="cpu", weights_only=False)
        x0 = ckpt["params"]

        # Optionally recompute FoM with current settings
        f0 = goal_fn(x0)
        print(f"Resumed from previous checkpoint: FoM = {f0:.6e}")

    except Exception as e:
        print(f"[WARN] Failed to load from {resume_from}: {e}")
        print("Falling back to random initial guess...")
        x0, f0 = get_initial_guess(
            sample_size=50,
            goal_function=goal_fn,
            pulse_settings_list=pulse_settings_list
        )
        print(f"Generated new initial guess: FoM = {f0:.6e}")
else:
    # Generate a new random initial guess
    x0, f0 = get_initial_guess(
        sample_size=50,
        goal_function=goal_fn,
        pulse_settings_list=pulse_settings_list
    )
    print(f"Generated new initial guess: FoM = {f0:.6e}")

# -----------------------------
# Step 6: CMA-ES optimization
# -----------------------------
algo_type = "CMA-ES"  # or "Nelder Mead"
iterations = 15
superiterations = 1
log = True
verbose = True

if algo_type == "CMA-ES":
    es, solutions_norm, values, scale = initialize_cmaes(
        goal_fn, x0, pulse_settings_list, sigma_init=0.01
    )
    for _ in range(superiterations):
        for j in range(iterations):
            es, solutions_norm, values = cmaes_iteration_step(
                goal_fn, es, solutions_norm, values, scale
            )
            best_idx = int(np.argmin(values))
            if verbose:
                print(f"CMA‑ES Iter {j+1}/{iterations}: FoM = {values[best_idx]:.6e}")
            if log:
                with open("fom_log.txt", "a") as f:
                    f.write(f"{j},{values[best_idx]:.6e}\n")

    best_idx = int(np.argmin(values))
    best_params = solutions_norm[best_idx]
    x_opt = torch.tensor(unnormalize_params(best_params, scale), dtype=torch.float64)
    f_opt = values[best_idx]


elif algo_type == "Nelder Mead":
    samples, values = nm_init("Nelder Mead", goal_fn, x0)

    for i in range(superiterations):
        best_idx = int(np.argmin(values))
        best_guess = samples[best_idx]
        samples, values = nm_init("Nelder Mead", goal_fn, best_guess)

        for j in range(iterations):
            samples, values = nm_step("Nelder Mead", goal_fn, samples, values)
            best_idx = int(np.argmin(values))
            best_value = values[best_idx]

            if verbose:
                print(f"NM Iteration {j+1}/{iterations}: FoM = {best_value:.6e}")
            if log:
                with open("fom_log.txt", "a") as f:
                    f.write(f"{i},{j},{best_value}\n")

    x_opt = torch.tensor(samples[best_idx], dtype=torch.float64)
    f_opt = values[best_idx]

else:
    raise ValueError(f"Unsupported algorithm: {algo_type}")
# -----------------------------
# Step 7–8: Save and visualize
# -----------------------------


# Generate timestamped result folder
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
result_dir = Path("results") / f"pulse_{timestamp}"
result_dir.mkdir(parents=True, exist_ok=True)

# Compute optimized drive
optimized_drive = get_drive(time_grid, x_opt, pulse_settings_list)

# Save checkpoint
checkpoint = {
    "params": x_opt,
    "fom": f_opt,
    "time_grid": time_grid,
    "pulse_settings": pulse_settings_list,
    "Δ": Δ,
    "drive": optimized_drive,
    "timestamp": timestamp,
    "objective_type": objective_type,
    **objective_config  # Save relevant gate or state config
}
torch.save(checkpoint, result_dir / "pulse_solution.pt")

# Save extras
np.savetxt(result_dir / "best_params.txt", x_opt.numpy())
with open(result_dir / "best_fom.txt", "w") as f:
    f.write(str(f_opt))

# Compute final propagator and save it
from evolution import get_propagator
propagator = get_propagator(lambda Ω, dt, t: get_U(Ω, dt, t, Δ), time_grid, optimized_drive)
torch.save(propagator, result_dir / "optimized_propagator.pt")

# Project into computational subspace
basis_indices = [0, 1, 2, 3, 6, 7, 8, 9]
P = torch.zeros((len(basis_indices), 12), dtype=torch.complex128)
for i, idx in enumerate(basis_indices):
    P[i, idx] = 1.0

propagator_projected = P @ propagator @ P.T
torch.save(propagator_projected, result_dir / "propagator_projected.pt")

# -----------------------------
# Plot: Control Drives
# -----------------------------
plt.figure(figsize=(10, 4))
for i, d in enumerate(optimized_drive):
    plt.plot(time_grid.numpy() * 1e9, d.numpy(), label=f"Drive {i+1}")
plt.xlabel("Time (ns)")
plt.ylabel("Drive Amplitude")
plt.title("Optimized Control Drives")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(result_dir / "optimized_drive.png")
plt.close()

# -----------------------------
# Plot: Population Transfers
# -----------------------------
initial_target_pairs = [(0, 6), (1, 7), (2, 8), (3, 9)]

plt.figure(figsize=(8, 6))
for init_idx, tgt_idx in initial_target_pairs:
    ψ0 = torch.zeros(12, dtype=torch.complex128)
    ψ0[init_idx] = 1.0

    states = get_evolution_vector(
        lambda Ω, dt, t: get_U(Ω, dt, t, Δ),
        time_grid,
        optimized_drive,
        ψ0
    )
    pop = torch.stack([torch.abs(s) ** 2 for s in states]).numpy()
    plt.plot(time_grid.numpy() * 1e9, pop[:, tgt_idx], label=f"{init_idx} → {tgt_idx}")

plt.xlabel("Time (ns)")
plt.ylabel("Target Population")
plt.title("Multi‑State Population Transfer")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(result_dir / "multi_state_populations.png")
plt.close()
