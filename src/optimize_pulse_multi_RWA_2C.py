import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from pulse_settings import PulseSettings, get_initial_guess
from get_drive import get_drive
from objective_functions import get_goal_function, make_three_qubit_basis_indices
from evolution import get_time_grid, get_evolution_vector
#from quantum_model import get_U_RWA, Λ_s, Λ00, Λ01, Λ10, Λ11, Λm10, Λm11, ω1, ω2, γ_e

from quantum_model_3C import (
    get_U_RWA, dtype, Λ_s, γ_e, ω1, B_0, γ_n, Azz_n, A_ort_n,
    set_active_carbons, get_active_carbons, get_precomp, detuning_for_target_all_up
)

from custom_gates import ccz_gate  # or wherever you defined it


from cma_runner import (
    initialize_cmaes,
    cmaes_iteration_step,
    unnormalize_params
)
from nelder_mead import call_initialization as nm_init, call_algo_step as nm_step



from quantum_model_3C import set_active_carbons, get_precomp
# choose how many carbons participate in the model
set_active_carbons([1,4])    # the Hamiltonian uses these 4; others ignored


print("Active carbons:", get_active_carbons())
pc = get_precomp()
print("dim_nuc per e-manifold =", pc["dim_nuc"], " (should be 3 * 2^N_C )")

# pick which two carbons form your logical qubits B and C
basis_indices = make_three_qubit_basis_indices(
    carbon_pair=(1,4),   # e.g., use carbons with indices 1 and 3 from your parameter arrays
    mI_block=0,          # fix 14N to +1
    electron_map=('m1','0')  # A=0→|-1_e>, A=1→|0_e>
)



def nconf_from_pc(pc=pc):
    # number of carbon configurations = 2^N_C
    return 2 ** int(pc['N_C'])

def dim_from_pc(pc=pc):
    return 2 * 3 * nconf_from_pc(pc)  # 2 (electron) * 3 (14N) * 2^N_C

def basis_index(e_manifold: int, mI_block: int, c_bits: int, pc=pc) -> int:
    """
    e_manifold: 0 for |0_e> (aux), 1 for |-1_e> (comp)
    mI_block:   0:+1, 1:0, 2:-1
    c_bits:     integer 0..(2^N_C-1), bitstring of all carbons (0=↑, 1=↓)
    """
    nconf = nconf_from_pc(pc)
    dim_nuc = 3 * nconf
    offset_e = 0 if e_manifold == 0 else dim_nuc
    return offset_e + mI_block * nconf + c_bits

def Lambda_target(mI_block: int, c_bits: int, pc=pc):
    """
    Return Λ_target (rad/s) for a nuclear config (14N block, carbon bitstring).
    This is the Λ used inside H_e^(RWA) for the addressed nuclear state.
    """
    nconf = nconf_from_pc(pc)
    idx_nuc = mI_block * nconf + c_bits
    return pc['Λ_nuc'][idx_nuc]  # (rad/s)

def Delta_e_for(mI_block: int, c_bits: int, pc=pc):
    """Δ_e = Λ_target - Λ_s for the selected nuclear configuration."""
    return float(Lambda_target(mI_block, c_bits, pc) - Λ_s)


# -----------------------------
# Step 1: Define pulse settings
# -----------------------------
# "What field amplitude gives me a 5 MHz Rabi frequency if each Tesla gives 175,929.1886 MHz of rotation?"
#  in the case of the electron?"

Ω_rabi = 2 * np.pi * 5e6 # in(rad/s) because the simulation time is in seconds
B_max = Ω_rabi / γ_e  # Units: (rad/s) / (rad/μs/T) = μT
print(B_max)
pulse_settings_list = [
    PulseSettings(
        basis_type="Custom",
        basis_size=11,
        maximal_pulse=B_max,               # Or total integral limit
        maximal_amplitude=B_max/8,             # Normalized: optimizer outputs b(t) ∈ [-1, 1]
        maximal_frequency=5 * 2*np.pi * 1e6,
        minimal_frequency=-5 * 2*np.pi * 1e6,
        maximal_phase=11*np.pi,
        channel_type="MW"
    )
]

#kohlenstoff carbon 50khZ

# -----------------------------
# Step 2: Simulation parameters
# -----------------------------
duration_ns =  1250#750
steps_per_ns = 1.0 # show mathematicallz why this is fine
time_grid = get_time_grid(duration_ns, steps_per_ns)


# choose the addressed nuclear configuration
mI_block = 0         # 14N = +1
c_bits   = 0         # all carbons ↑ (bitstring 000…0)

# initial/target (flip only the electron, same nuclear config)
initial_index = basis_index(e_manifold=1, mI_block=mI_block, c_bits=c_bits, pc=pc)  # |-1_e, mI=+1, C=↑…↑
target_index  = basis_index(e_manifold=0, mI_block=mI_block, c_bits=c_bits, pc=pc)  # |0_e,  mI=+1, C=↑…↑

# global electron detuning to park the MW drive on that transition
Δ_e  = Delta_e_for(mI_block, c_bits, pc=pc)   # rad/s
ω_RF = ω1                                       # your single RF carrier (still fine)

# same nuclear block, but sweep a few carbon configurations if you want:
nconf = nconf_from_pc(pc)
# example: first four carbon configs 0..3 (↑↑↑…, ↑↑…↓, …)
initial_target_pairs = [
    (basis_index(1, mI_block, cb, pc), basis_index(0, mI_block, cb, pc))
    for cb in range(min(4, nconf))
]

# -----------------------------
# Step 4: Multi-State Preparation objective
# -----------------------------
objective_type = "Gate Transformation"  # or "Multi-State Preparation"


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
    objective_config = {"target_gate": target_gate, "basis_indices": basis_indices}
    
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
    objective_config = {"target_gate": target_gate,
                         "basis_indices": basis_indices,
                         "target_3q": {"c13": torch.pi* 0, "c12": torch.pi* 0, "c23": torch.pi* 0, "c123": torch.pi* (1/4)},
                         "weights_3q": {"c123": 0.5, "c13": 0.5, "c12": 0.5, "c23": 0.2},
    }

    #objective_config = {"target_gate": target_gate,
    #                     "basis_indices": basis_indices,
    #                     "target_3q": {"c1":-np.pi/8,"c2":-np.pi/8,"c3":-np.pi/8,"c13": np.pi/8, "c12": np.pi/8, "c23": np.pi/8, "c123": -np.pi/8},
    #                     "weights_3q": {"c123": 0.4, "c13": 0.5, "c12": 0.5, "c23": 0.5,"c1": 0.5, "c2": 0.5, "c3": 0.5},
    #}

    #objective_config = {"target_gate": target_gate,
    #                     "basis_indices": basis_indices,
    #                     "target_3q": {"c13": +np.pi/8, "c12": +np.pi/8, "c23": +np.pi/8, "c123": np.pi/8},
    #                     "weights_3q": {"c123": 0.4, "c13": 0.5, "c12": 0.5, "c23": 0.5},
    #}

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



# -----------------------------
# Step 5: Load or Generate x0
# -----------------------------
use_previous = True
resume_from = "results/pulse_2025-12-17_14-29-57"#"results/pulse_2025-12-15_21-28-07"#"results/pulse_2025-09-02_10-30-53"  # Path to previous result
sample_size = 50

def generate_initial_state():
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
                sample_size=sample_size,
                goal_function=goal_fn,
                pulse_settings_list=pulse_settings_list
            )
            print(f"Generated new initial guess: FoM = {f0:.6e}")
    else:
        # Generate a new random initial guess
        x0, f0 = get_initial_guess(
            sample_size=sample_size,
            goal_function=goal_fn,
            pulse_settings_list=pulse_settings_list
        )
        print(f"Generated new initial guess: FoM = {f0:.6e}")

    return x0, f0


# quick probe: make sure get_u returns a square matrix and print its size
#dt0 = (time_grid[1] - time_grid[0]).item()
#t0  = time_grid[0].item()
#U0  = (lambda Ω, dt, t: get_U_RWA(Ω, dt, t, Δ_e=Δ_e, ω_RF=ω1))(
#        [d[0] for d in get_drive(time_grid, x0, pulse_settings_list)], dt0, t0)
#print("Model dim:", U0.shape)  # expect torch.Size([96,96]) for N_C=4

# -----------------------------
# Step 6: CMA-ES optimization
# -----------------------------
algo_type = "BFGS"  # or "Nelder Mead"
iterations = 200
superiterations = 1
log = True
verbose = True

if algo_type == "CMA-ES":
    goal_fn = get_goal_function(
                get_u=lambda Ω, dt, t: get_U_RWA(Ω, dt, t, Δ_e=Δ_e, ω_RF=ω1),
                objective_type=objective_type,
                time_grid=time_grid,
                pulse_settings_list=pulse_settings_list,
                get_drive_fn=get_drive,
                use_autograd=False,
                **objective_config  # Use config to avoid duplication
            )
    x0,f0 = generate_initial_state()

    es, solutions_norm, values, scale = initialize_cmaes(
        goal_fn, x0, pulse_settings_list, sigma_init=0.001
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
    goal_fn = get_goal_function(
            get_u=lambda Ω, dt, t: get_U_RWA(Ω, dt, t, Δ_e=Δ_e, ω_RF=ω1),
            objective_type=objective_type,
            time_grid=time_grid,
            pulse_settings_list=pulse_settings_list,
            get_drive_fn=get_drive,
            use_autograd=False,
            **objective_config  # Use config to avoid duplication
        )
    x0,f0 = generate_initial_state()

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

elif algo_type == "BFGS":
    goal_fn = get_goal_function(
        get_u=lambda Ω, dt, t: get_U_RWA(Ω, dt, t, Δ_e=Δ_e, ω_RF=ω1),
        objective_type=objective_type,
        time_grid=time_grid,
        pulse_settings_list=pulse_settings_list,
        get_drive_fn=get_drive,
        use_autograd=True,
        **objective_config  # Use config to avoid duplication
    )

    x0, f0 = generate_initial_state()

    # If x0 is already a tensor, clone it properly; otherwise create from numpy
    if isinstance(x0, torch.Tensor):
        x = x0.detach().clone().to(torch.float64).requires_grad_(True)
    else:
        x = torch.tensor(x0, dtype=torch.float64, requires_grad=True)

    optimizer = torch.optim.LBFGS(
        [x],
        max_iter=iterations,
        #tolerance_grad=1e-8,
        #tolerance_change= 1e-8,
        line_search_fn="strong_wolfe",
    )

    iter_counter = [0]  # mutable wrapper

    def closure():
        optimizer.zero_grad()

        f = goal_fn(x)   # must return a scalar tensor (no .item() inside goal_fn)

        f.backward()

        if verbose:
            print(f"BFGS Iter {iter_counter[0]+1}/{iterations}: FoM = {f.item():.6e}")
        if log:
            with open("fom_log.txt", "a") as fp:
                fp.write(f"{iter_counter[0]},{f.item():.6e}\n")

        iter_counter[0] += 1
        return f

    optimizer.step(closure)

    x_opt = x.detach().clone()
    f_opt = goal_fn(x_opt).item()

elif algo_type == "Adam":
    goal_fn = get_goal_function(
        get_u=lambda Ω, dt, t: get_U_RWA(Ω, dt, t, Δ_e=Δ_e, ω_RF=ω1),
        objective_type=objective_type,
        time_grid=time_grid,
        pulse_settings_list=pulse_settings_list,
        get_drive_fn=get_drive,
        use_autograd=True,
        **objective_config  # Use config to avoid duplication
    )

    x0, f0 = generate_initial_state()

    # If x0 is already a tensor, clone it properly; otherwise create from numpy
    if isinstance(x0, torch.Tensor):
        x = x0.detach().clone().to(torch.float64).requires_grad_(True)
    else:
        x = torch.tensor(x0, dtype=torch.float64, requires_grad=True)

    # Adam optimizer (no closure needed)
    optimizer = torch.optim.Adam(
        [x],
        lr=1e-2,  # you can tune this (1e-3 .. 1e-2 is a good starting range)
    )

    iter_counter = 0

    for j in range(iterations):
        optimizer.zero_grad()

        # goal_fn should return a scalar tensor (no .item() inside goal_fn)
        f = goal_fn(x)

        f.backward()
        optimizer.step()

        if verbose:
            print(f"Adam Iter {j+1}/{iterations}: FoM = {f.item():.6e}")
        if log:
            with open("fom_log.txt", "a") as fp:
                fp.write(f"{iter_counter},{f.item():.6e}\n")

        iter_counter += 1

    x_opt = x.detach().clone()
    f_opt = goal_fn(x_opt).item()

else:
    raise ValueError(f"Unknown algo_type: {algo_type}")

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
    "Δ": Δ_e,
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
propagator = get_propagator(lambda Ω, dt, t: get_U_RWA(Ω, dt, t, Δ_e), time_grid, optimized_drive)
torch.save(propagator, result_dir / "optimized_propagator.pt")

# Project into computational subspace
#basis_indices = [0, 1, 2, 3, 6, 7, 8, 9]
P = torch.zeros((len(basis_indices), 2*3*2**(len(get_active_carbons()))), dtype=torch.complex128)
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
plt.title("Optimized Control Pulses")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(result_dir / "optimized_drive.png")
plt.close()

# -----------------------------
# Plot: Population Transfers
# -----------------------------
#initial_target_pairs = [(6, 0), (7, 1), (8, 2), (9, 3)]

plt.figure(figsize=(8, 6))
for init_idx, tgt_idx in initial_target_pairs:
    ψ0 = torch.zeros(2*3*2**(len(get_active_carbons())), dtype=torch.complex128)
    ψ0[init_idx] = 1.0

    states = get_evolution_vector(
        lambda Ω, dt, t: get_U_RWA(Ω, dt, t, Δ_e),
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



