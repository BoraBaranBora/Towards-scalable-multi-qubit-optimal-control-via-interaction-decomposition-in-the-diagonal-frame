import numpy as np
import torch
import math

from pulse_settings import PulseSettings, get_initial_guess
from get_drive import get_drive
from objective_functions import get_goal_function
from evolution import get_time_grid
from quantum_model import get_U

from nelder_mead import call_initialization as nm_init, call_algo_step as nm_step
from cma_runner import initialize_cmaes, cmaes_iteration_step, get_scaling_from_pulse_settings, unnormalize_params


# -----------------------------
# Step 1: Define pulse settings
# -----------------------------
pulse_settings_list = [
    PulseSettings(
        basis_type="Custom",
        basis_size=8,
        maximal_pulse=1*np.pi*5e6,
        maximal_amplitude=1/3 *np.pi*5e6,
        maximal_frequency=20e6*2*np.pi,
        minimal_frequency=0,#-5e6*2*np.pi,
        maximal_phase=100*np.pi
    )
]

# -----------------------------
# Step 2: Simulation parameters
# -----------------------------
duration_ns = 300
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
    sample_size=30,
    goal_function=goal_fn,
    pulse_settings_list=pulse_settings_list
)
print(f"Initial guess FoM: {f0:.6e}")


def generate_custom_initial_guess(pulse_settings_list):
    pulse_params = []
    for ps in pulse_settings_list:
        bs = ps.basis_size

        # --- Define parameter groups ---
        amps = [0.8 * ps.maximal_amplitude] + [0.01] * (bs - 1)
        freqs = [1e3] * bs
        phases = [0.0] + [0.1 * ps.maximal_phase] * (bs - 1)

        # Wrap all phases safely
        phases = [p % (2 * np.pi) for p in phases]

        # Concatenate in correct order: amp | freq | phase
        pulse_params.extend(amps + freqs + phases)

    return np.array(pulse_params, dtype=np.float64)

#x0 = generate_custom_initial_guess(pulse_settings_list)
#f0 = goal_fn(x0)

#print(f'custom x0:{x0}')
#print(f"Custom initial guess FoM: {f0:.6e}")



# -----------------------------
# Step 6: Run optimization
# -----------------------------
algo_type = "CMA-ES"  # or   "Nelder Mead"
iterations = 100
superiterations = 1
log = True
verbose = True

if algo_type == "CMA-ES":
    #es, solutions, values = initialize_cmaes(goal_fn, x0, sigma_init=10)
    es, solutions_norm, values, scale = initialize_cmaes(goal_fn, x0, pulse_settings_list, sigma_init=0.001)

    for i in range(superiterations):
        for j in range(iterations):
            #es, solutions, values = cmaes_iteration_step(goal_fn, es, solutions, values)
            es, solutions_norm, values = cmaes_iteration_step(goal_fn, es, solutions_norm, values, scale)

            best_idx = int(np.argmin(values))
            best_value = values[best_idx]
            best_params = solutions_norm[best_idx]

            if verbose:
                print(f"CMA-ES Iteration {j+1}/{iterations}: FoM = {best_value:.6e}")
            if log:
                with open("fom_log.txt", "a") as f:
                    f.write(f"{i},{j},{best_value}\n")

    scale = get_scaling_from_pulse_settings(pulse_settings_list)
    x_opt = torch.tensor(unnormalize_params(best_params,scale), dtype=torch.float64)
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



import matplotlib.pyplot as plt
from evolution import get_evolution_vector

# Get optimized drive
optimized_drive = get_drive(time_grid, x_opt, pulse_settings_list)

# Plot the drive(s)
plt.figure(figsize=(10, 4))
for i, d in enumerate(optimized_drive):
    plt.plot(time_grid.numpy() * 1e9, d.numpy(), label=f"Drive {i+1}")
plt.xlabel("Time (ns)")
plt.ylabel("Drive Amplitude")
plt.title("Optimized Control Drive")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("optimized_drive.png")
plt.show()


# Simulate evolution with optimized parameters
ψ0 = torch.zeros(12, dtype=dtype)
ψ0[initial_index] = 1.0

states = get_evolution_vector(
    lambda Ω, dt, t: get_U(Ω, dt, t, Δ),
    time_grid,
    optimized_drive,
    ψ0
)

# Compute population over time
populations = torch.stack([torch.abs(state) ** 2 for state in states]).numpy()

# Plot population dynamics for selected states
plt.figure(figsize=(10, 6))
for i in [initial_index, target_index]:
    plt.plot(time_grid.numpy() * 1e9, populations[:, i], label=f"Population of state {i}")
plt.xlabel("Time (ns)")
plt.ylabel("Population")
plt.title("Quantum State Population Dynamics")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("population_dynamics.png")
plt.show()