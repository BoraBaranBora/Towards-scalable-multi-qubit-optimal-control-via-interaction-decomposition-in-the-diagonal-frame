import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- import the RWA model & constants from your quantum model ---
from quantum_model import (
    get_U_RWA, dtype, Λ_s, Λ00, Λ01, Λ10, Λ11, Λm10, Λm11,
    γ_e, ω1  # ω1 used as the single RF carrier for pruning
)

from evolution import get_time_grid, get_evolution_vector

# --- Index → Λ Mapping (unchanged) ---
# logical assignments
# e: 0 ↔ |-1_e>, 1 ↔ |0_e>
# N: 0 ↔ m_I=+1, 1 ↔ m_I=0   (m_I=-1 = leakage)
# C: 0 ↔ ↑,     1 ↔ ↓

Λ_dict = {
    0: Λ10 #logically:100
    ,  1: Λ11,
    2: Λ00,  3: Λ01,
    4: Λm10, 5: Λm11,
    6: Λ10#logically:000
    ,  7: Λ11,
    8: Λ00,  9: Λ01,
    10: Λm10, 11: Λm11
}

# --- Parameters ---
Ω_rabi_target = 2 * math.pi * 5e6  # rad/s — desired electron Rabi frequency
# IMPORTANT: γ_e must be in rad/s/T here (export that from quantum_model)
B1_amplitude = Ω_rabi_target / γ_e  # Tesla — MW field amplitude
print("B1_amplitude [T] =", B1_amplitude.item())
print("γ_e [rad/s/T]    =", γ_e.item())

# --- Time grid ---
# After RWA, fastest relevant rates are ~Ω_rabi_target and small detunings (few MHz).
# Rule of thumb: Δt ≤ (2π)/(N*ω_max) with N≈20 ⇒ 1–2 steps/ns is fine for ~10 MHz.
steps_per_ns = 10
duration_ns = 1000
time_grid = get_time_grid(duration_ns, steps_per_ns)
t_ns = time_grid.numpy() * 1e9

# --- Transition selection ---
initial_index = 6   # 0..11
target_index  = 0   # e.g. 6 = |-1, +1, ↑⟩

# --- Electron detuning for the chosen target ---
Δ_e = (Λ_dict[target_index] - Λ_s).item()  # rad/s
# --- RF carrier (single) for RWA pruning; here use ω1 from quantum_model ---
ω_RF = ω1  # rad/s

# --- Drives (two channels: MW + RF). RF set to zero here; still OK under RWA. ---
Ω1 = torch.full_like(time_grid, B1_amplitude)  # MW channel (Tesla)
Ω2 = torch.zeros_like(time_grid)               # RF channel (Tesla)

# --- Initial state ---
ψ0 = torch.zeros(12, dtype=dtype)
ψ0[initial_index] = 1.0

# --- Evolution with RWA Hamiltonian ---
states = get_evolution_vector(
    lambda Ω, dt, t: get_U_RWA(Ω, dt, t, Δ_e=Δ_e, ω_RF=ω_RF),
    time_grid, [Ω1, Ω2], ψ0
)

# --- Target population trace ---
pop_target = np.array([torch.abs(state[target_index])**2 for state in states])

# --- Fit a sin^2 to extract Rabi frequency ---
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

# --- Pretty labels ---
def state_label(index):
    m_s = 0 if index < 6 else -1
    sub_index = index % 6
    m_I_14N = [+1, +1, 0, 0, -1, -1][sub_index]
    spin_13C = ["↑", "↓"] * 3
    spin = spin_13C[sub_index]
    return fr"$|m_s={m_s}, m_I={m_I_14N}, {spin}\rangle$"

fit_label = (
    r"Fit: "
    fr"$A = ({A_fit:.2f} \pm {A_err:.2f}),\; "
    fr"f = ({f_fit/1e6:.3f} \pm {f_err/1e6:.3f})\,\mathrm{{MHz}},\; "
    fr"T_\pi = ({T_pi*1e9:.1f} \pm {T_pi_err*1e9:.1f})\,\mathrm{{ns}}$"
)
transition_label = fr"{state_label(initial_index)} $\rightarrow$ {state_label(target_index)}"

# --- Plot ---
plt.figure(figsize=(8, 5))
plt.plot(t_ns, pop_target, label=transition_label)
plt.plot(t_ns, sin2_fit(t_ns, *popt), '--', label=fit_label)
plt.xlabel("Time (ns)")
plt.ylabel("Population")
plt.title("Rabi Oscillation Fit (RWA)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Leakage over time (m_I = -1 manifold) ---
leakage_indices = [4, 5, 10, 11]           # m_I=-1 in both electron manifolds
comp_indices    = [0,1,2,3, 6,7,8,9]       # 8-dim computational subspace

# total norm check (should be ~1)
norm_t = np.array([float(torch.sum(torch.abs(s)**2)) for s in states])

# leakage population = sum over leakage indices
pop_leakage = np.array([
    float(torch.sum(torch.abs(state[leakage_indices])**2))
    for state in states
])

# (optional) computational-subspace population
pop_comp = 1.0 - pop_leakage

print(f"Max leakage over run: {pop_leakage.max():.3e}")
print(f"Min/Max norm: {norm_t.min():.6f} / {norm_t.max():.6f}")

# --- Plot leakage ---
plt.figure(figsize=(8, 4))
plt.plot(t_ns, pop_leakage, label="Leakage population (m_I = -1)")
plt.xlabel("Time (ns)")
plt.ylabel("Population")
plt.title("Leakage vs Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

