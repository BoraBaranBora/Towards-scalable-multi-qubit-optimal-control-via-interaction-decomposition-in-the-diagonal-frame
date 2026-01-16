import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- import the full multi-carbon RWA model & constants from your quantum model ---
#from quantum_model_3C import (
#    get_U_RWA, dtype, Λ_s, γ_e, ω1,  # ω1 used as the single RF carrier for pruning
#    B_0, γ_n, Azz_n, A_ort_n        # needed to compute Δ_e for the full system
#)

from evolution import get_time_grid, get_evolution_vector


from quantum_model_3C import (
    get_U_RWA, dtype, Λ_s, γ_e, ω1, B_0, γ_n, Azz_n, A_ort_n,
    set_active_carbons, get_active_carbons, get_precomp, detuning_for_target_all_up
)

# ---- choose the carbon subset here ----
# Examples:
#   0 carbons is not supported by this multi-C code (needs at least one),
#   1 carbon  -> [1]
#   2 carbons -> [1,2]
#   up to the number available in your parameter arrays (here up to 4)
set_active_carbons([1,3,4])   # <-- change this list to pick N_C

print("Active carbons:", get_active_carbons())
pc = get_precomp()
print("dim_nuc per e-manifold =", pc["dim_nuc"], " (should be 3 * 2^N_C )")

# -----------------------------
# Helper: nuclear frequencies (MHz) consistent with your model
# -----------------------------
def ω_n(i):
    # √[(Azz + γ B0)^2 + A_ort^2]  in MHz
    term = Azz_n[i] + γ_n[i] * B_0
    return torch.sqrt(torch.abs(term)*2 + torch.abs(A_ort_n[i])*2)

def ω_na(i):
    # “aux” (no transverse HF), in MHz
    return γ_n[i] * B_0

# -----------------------------
# Compute Δ_e for a chosen nuclear config in the full multi-C system
# Addressed target: (e = |0_e>, m_I(14N)=+1, all carbons ↑)
# -----------------------------
def electron_detuning_for_target_all_up():
    """
    Uses your full-system convention:
    Λ_target = Λ_s - [ Δ_N + 0.5 * Σ_k Δ_Ck ] * 2π*1e6 (rad/s),
    where Δ_N = γ_n[0] B0 - ω_n(0), Δ_Ck = γ_n[ck] B0 - ω_n(ck) in MHz.
    Then Δ_e = Λ_target - Λ_s = -[ ... ] * 2π*1e6 .
    """
    two_pi_1e6 = 2 * math.pi * 1e6
    Δ_N  = (γ_n[0] * B_0 - ω_n(0))  # MHz
    # carbons are indices 1..len(γ_n)-1
    Δ_sum = 0.0
    for ck in range(1, int(len(γ_n))):
        Δ_sum = Δ_sum + (γ_n[ck] * B_0 - ω_n(ck))  # MHz
    Δ_total_MHz = Δ_N + 0.5 * Δ_sum                # MHz
    Δ_e = - float(Δ_total_MHz) * two_pi_1e6        # rad/s
    return Δ_e

# -----------------------------
# Simulation parameters
# -----------------------------
Ω_rabi_target = 2 * math.pi * 5e6  # rad/s — desired electron Rabi frequency
B1_amplitude  = Ω_rabi_target / γ_e  # Tesla — MW field amplitude (γ_e is rad/s/T)
print("B1_amplitude [T] =", float(B1_amplitude))
print("γ_e [rad/s/T]    =", float(γ_e))

steps_per_ns = 2
duration_ns  = 1000
time_grid    = get_time_grid(duration_ns, steps_per_ns)
t_ns         = time_grid.numpy() * 1e9

# -----------------------------
# Drives (two channels: MW + RF)
# RF off here; still OK under RWA
# -----------------------------
Ω1 = torch.full_like(time_grid, B1_amplitude)  # MW (Tesla)
Ω2 = torch.zeros_like(time_grid)               # RF (Tesla)

# -----------------------------
# Build a probe state to infer dimension, then set initial/target indices generically
# Ordering (per your model):
#   nuclear register per electron manifold = 3 blocks (m_I = +1, 0, −1), each of size nconf = 2^N_C
# Full dimension = 2 (e) * 3 * nconf
# We choose:
#   initial = (e = |-1_e>, m_I=+1, carbons all ↑)  → index = dim_nuc + 0
#   target  = (e = |0_e>,  m_I=+1, carbons all ↑)  → index = 0
# -----------------------------
# quick 1-step evolution to get state length
ψ_probe = torch.zeros(1, dtype=dtype)
states_probe = get_evolution_vector(
    lambda Ω, dt, t: get_U_RWA(Ω, dt, t, Δ_e=0.0, ω_RF=ω1),  # Δ_e not needed for size probe
    time_grid[:2], [Ω1[:2], Ω2[:2]], torch.zeros(2*3*2**(len(get_active_carbons())), dtype=dtype)  # seed shape; will be resized by the model
)
dim = states_probe[-1].numel()

if dim % 6 != 0:
    raise RuntimeError(f"Unexpected Hilbert dimension {dim} (not divisible by 6).")

nconf   = dim // 6            # = 2^N_C
dim_nuc = 3 * nconf           # nuclear register size per electron manifold
print(f"Full dimension = {dim}  → nconf = {nconf} (2^N_C), N_C = {int(round(math.log2(nconf)))}")

initial_index = dim_nuc + 0   # |-1_e>, m_I=+1, all carbons ↑ (first config)
target_index  = 0             # |0_e>,  m_I=+1, all carbons ↑ (first config)

# -----------------------------
# Electron detuning for addressed target (full system)
# -----------------------------
#Δ_e = electron_detuning_for_target_all_up() 
Δ_e = detuning_for_target_all_up()
ω_RF = ω1  # single RF carrier (rad/s), used only for pruning 14N Fourier components

# -----------------------------
# Initial state and full evolution
# -----------------------------
ψ0 = torch.zeros(dim, dtype=dtype)
ψ0[initial_index] = 1.0

states = get_evolution_vector(
    lambda Ω, dt, t: get_U_RWA(Ω, dt, t, Δ_e=Δ_e, ω_RF=ω_RF),
    time_grid, [Ω1, Ω2], ψ0
)

# -----------------------------
# Target population trace & Rabi fit
# -----------------------------
pop_target = np.array([torch.abs(state[target_index])**2 for state in states])

def sin2_fit(t, A, f):
    return A * np.sin(2 * math.pi * f * t * 1e-9)**2

guess = [1.0, 2e6]
popt, pcov = curve_fit(sin2_fit, t_ns, pop_target, p0=guess, maxfev=10000)
perr = np.sqrt(np.diag(pcov))
A_fit, f_fit = popt
A_err, f_err = perr
T_pi = 1 / (2 * f_fit)
T_pi_err = f_err / (2 * f_fit**2)

print("Fitted parameters:")
print(f"  A     = {A_fit:.3f} ± {A_err:.3f}")
print(f"  f     = {f_fit/1e6:.3f} ± {f_err/1e6:.3f} MHz")
print(f"  T_pi  = {T_pi*1e9:.2f} ± {T_pi_err*1e9:.2f} ns")

# -----------------------------
# Pretty labels for multi-C (show carbons as a bitstring)
# -----------------------------
def label_multiC(idx):
    # Which electron manifold?
    e = 0 if idx < dim_nuc else -1
    # index within that manifold
    i = idx if e == 0 else (idx - dim_nuc)
    # which 14N block?
    block = i // nconf  # 0:+1, 1:0, 2:-1
    mI_map = {0: "+1", 1: "0", 2: "−1"}
    m_I = mI_map[block]
    # carbon configuration (binary of length N_C)
    c_idx = i % nconf
    N_C = int(round(math.log2(nconf)))
    bits = format(c_idx, f"0{N_C}b") if N_C > 0 else ""
    # map 0→↑, 1→↓ (because we used c_idx=0 for all-↑)
    spins = "".join("↑" if b=="0" else "↓" for b in bits)
    return fr"$|m_s={e}, m_I={m_I}, \mathrm{{C}}=[{spins}] \rangle$"

fit_label = (
    r"Fit: "
    fr"$A = ({A_fit:.2f} \pm {A_err:.2f}),\; "
    fr"f = ({f_fit/1e6:.3f} \pm {f_err/1e6:.3f})\,\mathrm{{MHz}},\; "
    fr"T_\pi = ({T_pi*1e9:.1f} \pm {T_pi_err*1e9:.1f})\,\mathrm{{ns}}$"
)
transition_label = fr"{label_multiC(initial_index)} $\rightarrow$ {label_multiC(target_index)}"

# --- Plot Rabi oscillation & fit ---
plt.figure(figsize=(8, 5))
plt.plot(t_ns, pop_target, label=transition_label)
plt.plot(t_ns, sin2_fit(t_ns, *popt), '--', label=fit_label)
plt.xlabel("Time (ns)")
plt.ylabel("Population")
plt.title("Rabi Oscillation Fit (RWA, full multi-C system)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Leakage over time (m_I = −1 in both electron manifolds)
# -----------------------------
leak_start = 2 * nconf               # m_I = −1 block start within a manifold
leak_end   = 3 * nconf               # exclusive
leak_aux   = list(range(leak_start, leak_end))             # in |0_e⟩ manifold
leak_comp  = [i + dim_nuc for i in leak_aux]               # in |-1_e⟩ manifold
leakage_indices = leak_aux + leak_comp

# total norm check (should be ~1)
norm_t = np.array([float(torch.sum(torch.abs(s)**2)) for s in states])

# leakage population = sum over leakage indices
pop_leakage = np.array([
    float(torch.sum(torch.abs(state[leakage_indices])**2))
    for state in states
])

print(f"Max leakage over run: {pop_leakage.max():.3e}")
print(f"Min/Max norm: {norm_t.min():.6f} / {norm_t.max():.6f}")

# --- Plot leakage ---
plt.figure(figsize=(8, 4))
plt.plot(t_ns, pop_leakage, label="Leakage population (m_I = −1, both e-manifolds)")
plt.xlabel("Time (ns)")
plt.ylabel("Population")
plt.title("Leakage vs Time (full multi-C system)")
plt.grid(True)
plt.legend()
plt.tight_layout()
#plt.show()