import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from quantum_model import get_U, dtype, Λ_s, Λ00, Λ01, Λ10, Λ11, Λm10, Λm11
from evolution import get_time_grid, get_evolution_vector

# --- Index → Λ Mapping ---
Λ_dict = {
    0: Λ00, 1: Λ01, 2: Λ00, 3: Λ01, 4: Λ00, 5: Λ01,
    6: Λ10, 7: Λ11, 8: Λ10, 9: Λ11, 10: Λ10, 11: Λ11
}

# --- Parameters ---
Ω_rabi = 5e6  # Hz
steps_per_ns = 10
duration_ns = 1000  # ns
Ω_drive = 2 * math.pi * Ω_rabi

# --- Transition Selection ---
initial_index = 0   # Choose from 0 to 11
target_index = 6    # Final state index (e.g. 6 = |-1, +1, ↑⟩)

# --- Calculate Detuning ---
Δ = (Λ_dict[target_index] - Λ_s).item()

# --- Time Grid ---
time_grid = get_time_grid(duration_ns, steps_per_ns)
t_ns = time_grid.numpy() * 1e9

# --- Drive ---
Ω1 = torch.ones_like(time_grid) * Ω_drive
Ω2 = torch.ones_like(time_grid) * 1.0

# --- Initial State ---
ψ0 = torch.zeros(12, dtype=dtype)
ψ0[initial_index] = 1.0

# --- Run Evolution ---
states = get_evolution_vector(
    lambda Ω, dt, t: get_U(Ω, dt, t, Δ),
    time_grid, [Ω1, Ω2], ψ0
)
pop_target = np.array([torch.abs(state[target_index])**2 for state in states])

# --- Sin² Fit Function ---
def sin2_fit(t, A, f):
    return A * np.sin(2 * math.pi * f * t * 1e-9)**2

# --- Fit ---
guess = [1.0, 2e6]
popt, pcov = curve_fit(sin2_fit, t_ns, pop_target, p0=guess, maxfev=10000)
perr = np.sqrt(np.diag(pcov))

A_fit, f_fit = popt
A_err, f_err = perr
T_pi = 1 / (2 * f_fit)
T_pi_err = f_err / (2 * f_fit**2)

# --- Print Results ---
print("Fitted parameters:")
print(f"  A     = {A_fit:.3f} ± {A_err:.3f}")
print(f"  f     = {f_fit/1e6:.3f} ± {f_err/1e6:.3f} MHz")
print(f"  T_pi  = {T_pi*1e9:.2f} ± {T_pi_err*1e9:.2f} ns")

# --- Index to Label Map ---
def state_label(index):
    m_s = 0 if index < 6 else -1
    sub_index = index % 6
    m_I_14N = [+1, +1, 0, 0, -1, -1][sub_index]
    spin_13C = ["↑", "↓"] * 3
    spin = spin_13C[sub_index]
    return fr"$|m_s={m_s}, m_I={m_I_14N}, {spin}\rangle$"

# --- Plot Labels ---
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
plt.title("Rabi Oscillation Fit for Selected Transition")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
