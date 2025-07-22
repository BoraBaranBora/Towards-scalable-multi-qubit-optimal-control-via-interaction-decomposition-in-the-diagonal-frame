import numpy as np
import torch
import math

from pulse_settings import PulseSettings, get_initial_guess
from get_drive import get_drive
from objective_functions import get_goal_function
from evolution import get_time_grid
from quantum_model import get_U

from nelder_mead import call_initialization as nm_init, call_algo_step as nm_step
from cma_runner import initialize_cmaes, cmaes_iteration_step


# -----------------------------
# Step 1: Define pulse settings
# -----------------------------
pulse_settings_list = [
    PulseSettings(
        basis_type="Custom",
        basis_size=4,
        maximal_pulse=2*np.pi*5e6,
        maximal_amplitude=2e6,
        maximal_frequency=20e6,
        minimal_frequency=0.0,
        maximal_phase=np.pi
    )
]

# -----------------------------
# Step 2: Simulation parameters
# -----------------------------
duration_ns = 1000
steps_per_ns = 10
time_grid = get_time_grid(duration_ns, steps_per_ns)

# -----------------------------
# Step 3: Quantum states
# -----------------------------
dtype = torch.complex128
ψ0 = torch.zeros(12, dtype=dtype)
ψ0[0] = 1.0
ψ_target = torch.zeros(12, dtype=dtype)
ψ_target[6] = 1.0

from quantum_model import Λ_s, Λ00, Λ01, Λ10, Λ11, Λm10, Λm11

Λ_dict = {
    0: Λ00, 1: Λ01, 2: Λ00, 3: Λ01, 4: Λ00, 5: Λ01,
    6: Λ10, 7: Λ11, 8: Λ10, 9: Λ11, 10: Λ10, 11: Λ11
}

initial_index = 0
target_index = 6

Δ = (Λ_dict[target_index] - Λ_s).item()


# -----------------------------
# Step 4: Choose objective
# -----------------------------
objective_type = "Gate Transformation"  # or "Gate Transformation"

if objective_type == "State Preparation":
    goal_fn = get_goal_function(
        get_u=lambda Ω, dt, t: get_U(Ω, dt, t, Δ),  # wrapped with detuning
        objective_type=objective_type,
        time_grid=time_grid,
        pulse_settings_list=pulse_settings_list,
        get_drive_fn=get_drive,
        starting_state=ψ0,
        target_state=ψ_target,
        target_gate=None
    )

# Define target gate for phase gate objective
target_gate = None
if objective_type == "Gate Transformation":
    # Create diag(1, 1, 1, -1) embedded in bottom-right 4x4 block
    target_gate = torch.diag(torch.tensor([1, 1, 1, -1], dtype=dtype))
    #target_gate = torch.eye(12, dtype=dtype)
    #target_gate[8:12, 8:12] = phase_block

    goal_fn = get_goal_function(
        get_u=lambda Ω, dt, t: get_U(Ω, dt, t, Δ),
        objective_type=objective_type,
        time_grid=time_grid,
        pulse_settings_list=pulse_settings_list,
        get_drive_fn=get_drive,
        starting_state=ψ0,
        target_state=ψ_target,
        target_gate=target_gate
    )


# -----------------------------
# Step 5: Generate initial guess
# -----------------------------
x0, f0 = get_initial_guess(
    sample_size=2,
    goal_function=goal_fn,
    pulse_settings_list=pulse_settings_list
)

print(f"Initial guess FoM: {f0:.6e}")

# -----------------------------
# Step 6: Run optimization
# -----------------------------
algo_type = "CMA-ES"  # or   "Nelder Mead"
iterations = 100
superiterations = 1
log = True
verbose = True

if algo_type == "CMA-ES":
    es, solutions, values = initialize_cmaes(goal_fn, x0)

    for i in range(superiterations):
        for j in range(iterations):
            es, solutions, values = cmaes_iteration_step(goal_fn, es, solutions, values)
            best_idx = int(np.argmin(values))
            best_value = values[best_idx]
            best_params = solutions[best_idx]

            if verbose:
                print(f"CMA-ES Iteration {j+1}/{iterations}: FoM = {best_value:.6e}")
            if log:
                with open("fom_log.txt", "a") as f:
                    f.write(f"{i},{j},{best_value}\n")

    x_opt = torch.tensor(best_params, dtype=torch.float64)
    f_opt = best_value

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
# Step 7: Report and save
# -----------------------------
print("\nOptimization complete.")
print(f"Best FoM: {f_opt:.6e}")
print("Best parameters:", x_opt.numpy())

np.savetxt("best_params.txt", x_opt.numpy())
with open("best_fom.txt", "w") as f:
    f.write(str(f_opt))
