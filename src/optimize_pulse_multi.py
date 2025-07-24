import numpy as np
import torch
import matplotlib.pyplot as plt

from pulse_settings import PulseSettings, get_initial_guess
from get_drive import get_drive
from objective_functions import get_goal_function
from evolution import get_time_grid, get_evolution_vector
from quantum_model import get_U, Λ_s, Λ00, Λ01, Λ10, Λ11, Λm10, Λm11

from cma_runner import (
    initialize_cmaes,
    cmaes_iteration_step,
    get_scaling_from_pulse_settings,
    unnormalize_params
)
from nelder_mead import call_initialization as nm_init, call_algo_step as nm_step


# -----------------------------
# Step 1: Define pulse settings
# -----------------------------
pulse_settings_list = [
    PulseSettings(
        basis_type="Custom",
        basis_size=8,
        maximal_pulse=2*np.pi * 5e6,
        maximal_amplitude=(1/4) * 2*np.pi * 5e6,
        maximal_frequency=20e6 * 2 * np.pi,
        minimal_frequency=0,
        maximal_phase=100 * np.pi
    )
]

# -----------------------------
# Step 2: Simulation parameters
# -----------------------------
duration_ns = 200
steps_per_ns = 10
time_grid = get_time_grid(duration_ns, steps_per_ns)

# -----------------------------
# Step 3: Detuning Δ
# -----------------------------
Λ_dict = {
    0: Λ00, 1: Λ01, 2: Λ00, 3: Λ01, 4: Λ00, 5: Λ01,
    6: Λ10, 7: Λ11, 8: Λ10, 9: Λ11, 10: Λ10, 11: Λ11
}
initial_target_pairs = [(0, 6), (1, 7), (2, 8), (3, 9)]
# Use the first pair to compute global Δ
Δ = (Λ_dict[initial_target_pairs[0][1]] - Λ_s).item()

# -----------------------------
# Step 4: Multi-State Preparation objective
# -----------------------------
goal_fn = get_goal_function(
    get_u=lambda Ω, dt, t: get_U(Ω, dt, t, Δ),
    objective_type="Multi-State Preparation",
    time_grid=time_grid,
    pulse_settings_list=pulse_settings_list,
    get_drive_fn=get_drive,
    initial_target_pairs=initial_target_pairs
)

# -----------------------------
# Step 5: Initial guess
# -----------------------------
x0, f0 = get_initial_guess(
    sample_size=30,
    goal_function=goal_fn,
    pulse_settings_list=pulse_settings_list
)
print(f"Initial guess FoM: {f0:.6e}")

# Optional: custom initial guess (commented)
# def generate_custom_initial_guess(...):
#     ...
# x0 = generate_custom_initial_guess(pulse_settings_list)
# f0 = goal_fn(x0)
# print(f"Custom initial guess FoM: {f0:.6e}")



# -----------------------------
# Step 6: CMA-ES optimization
# -----------------------------
algo_type = "CMA-ES"  # or "Nelder Mead"
iterations = 250
superiterations = 1
log = True
verbose = True

if algo_type == "CMA-ES":
    es, solutions_norm, values, scale = initialize_cmaes(
        goal_fn, x0, pulse_settings_list, sigma_init=0.25
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
# Step 7: Save and report
# -----------------------------
print("\nOptimization complete.")
print(f"Best FoM: {f_opt:.6e}")
print("Best parameters:", x_opt.numpy())
np.savetxt("best_params.txt", x_opt.numpy())
with open("best_fom.txt", "w") as f:
    f.write(str(f_opt))

# -----------------------------
# Step 8: Visualize
# -----------------------------
optimized_drive = get_drive(time_grid, x_opt, pulse_settings_list)

# Plot control drives
plt.figure(figsize=(10, 4))
for i, d in enumerate(optimized_drive):
    plt.plot(time_grid.numpy() * 1e9, d.numpy(), label=f"Drive {i+1}")
plt.xlabel("Time (ns)")
plt.ylabel("Drive Amplitude")
plt.title("Optimized Control Drives")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("optimized_drive.png")
plt.show()

# Plot population transfers for each pair
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
plt.savefig("multi_state_populations.png")
plt.show()
