# three_qubit_gate_utils.py

import math
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

from quantum_model_NV import (
    get_U_RWA, ω1, γ_e,
    get_active_carbons
)
from evolution import get_evolution_vector

mpl.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.titlesize": 18,
})

# ============================================================
# Checkpoint loading & evolution
# ============================================================

def load_ckpt(result_dir: Path):
    ckpt_path = result_dir / "pulse_solution.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    drive = ckpt.get("drive")
    time_grid = ckpt["time_grid"]
    delta_e = ckpt["Δ"]
    basis_indices = ckpt.get("basis_indices", None)
    initial_target_pairs = ckpt.get("initial_target_pairs", None)

    return {
        "drive": drive,
        "time_grid": time_grid,
        "Δ_e": float(delta_e),
        "basis_indices": basis_indices,
        "initial_target_pairs": initial_target_pairs,
        "objective_type": ckpt.get("objective_type", "Unknown"),
        "timestamp": ckpt.get("timestamp", "Unknown"),
    }

def apply_pulse_with_3C(Δ_e, time_grid, drive, ψ_init):
    """
    Evolves a single pulse using 3C model get_U_RWA(Ω, dt, t, Δ_e, ω_RF).
    Returns Tensor of shape (T+1, D).
    """
    U_fn = (lambda Ω, dt, t: get_U_RWA(Ω, dt, t, Δ_e=Δ_e, ω_RF=ω1))
    states_list = get_evolution_vector(U_fn, time_grid, drive, ψ_init)
    states = torch.stack(states_list, dim=0)  # (T+1, D)
    return states

def apply_sequence_with_3C(Δ_e, time_grids, drives, ψ0):
    ψ = ψ0
    chunks = []
    for tg, drv in zip(time_grids, drives):
        states = apply_pulse_with_3C(Δ_e, tg, drv, ψ)
        chunks.append(states)
        ψ = states[-1]
    return chunks

def fuse_states_and_time(time_grids, states_chunks):
    """
    Make states_concat and time_axis_ns that always match in length.
    - First chunk: keep all samples
    - Later chunks: drop the first sample (boundary).
    """
    if not states_chunks:
        raise ValueError("states_chunks is empty")

    cat_states = []
    times_s = []
    t_acc = 0.0

    for i, (tg, chunk) in enumerate(zip(time_grids, states_chunks)):
        if chunk.ndim != 2:
            raise ValueError("Each states chunk must be Tensor [T_i, D].")
        if len(tg) < 2:
            raise ValueError("Each time_grid must have >= 2 points.")
        dt = float((tg[1] - tg[0]).item())
        L = int(chunk.shape[0])  # samples (steps+1)

        if i == 0:
            cat_states.append(chunk)
            times_s.extend(t_acc + np.arange(L) * dt)
            t_acc += (L - 1) * dt
        else:
            if L < 2:
                raise ValueError("Chunk too short (needs at least 2 samples).")
            cat_states.append(chunk[1:])
            times_s.extend(t_acc + np.arange(1, L) * dt)
            t_acc += (L - 1) * dt

    states_concat = torch.cat(cat_states, dim=0)
    time_axis_ns = np.asarray(times_s) * 1e9
    return states_concat, time_axis_ns

# ============================================================
# Basis & projectors
# ============================================================

def nconf_from_pc(pc):
    return 2 ** int(pc['N_C'])

def basis_index(e_manifold: int, mI_block: int, c_bits: int, pc):
    nconf = nconf_from_pc(pc)
    dim_nuc = 3 * nconf
    offset_e = 0 if e_manifold == 0 else dim_nuc
    return int(offset_e + mI_block * nconf + c_bits)

def make_multi_qubit_basis_indices(pc, n14_pair=(0,1), electron_map=('m1','0')):
    """
    Build indices for the computational subspace including:
      Q0 = Electron
      Q1 = 14N
      Q2.. = all active carbons (order of get_active_carbons()).
    """
    act = list(get_active_carbons())
    N_C = len(act)
    n_qubits = 2 + N_C

    e_log_to_manifold = {
        0: 1 if electron_map[0] == 'm1' else 0,
        1: 1 if electron_map[1] == 'm1' else 0,
    }

    indices = []
    for q_bits_int in range(1 << n_qubits):
        qe = (q_bits_int >> 0) & 1
        qn = (q_bits_int >> 1) & 1
        e_m = e_log_to_manifold[qe]
        mI  = int(n14_pair[qn])

        c_bits = 0
        for iC, _label in enumerate(act):
            b = (q_bits_int >> (2 + iC)) & 1
            c_bits |= (b << iC)

        idx = basis_index(e_m, mI, c_bits, pc)
        indices.append(idx)

    names = ["e⁻", "14N"] + [f"C{label}" for label in act]
    return indices, names

def projector_from_indices_general(basis_indices, full_dim):
    P = torch.zeros((len(basis_indices), full_dim), dtype=torch.complex128)
    for i, idx in enumerate(basis_indices):
        P[i, int(idx)] = 1.0
    return P

def build_full_000(Pn, qubit_names, idx_e, idx_N, idx_c1, idx_c2):
    """
    Build FULL-Hilbert ket corresponding to 3q = |000> (e=0,C1=0,C2=0),
    with N14=0 and all other carbons = 0.

    This makes sure the population plot starts at P(|000>) = 1.
    """
    nq = len(qubit_names)
    vec_proj = torch.zeros(2**nq, dtype=torch.complex128)
    bits = [0]*nq
    bits[idx_e]  = 0
    bits[idx_N]  = 0
    bits[idx_c1] = 0
    bits[idx_c2] = 0
    # all other carbons already 0 in bits
    idx = sum((bits[q] << q) for q in range(nq))
    vec_proj[idx] = 1.0
    return Pn.conj().T @ vec_proj

def build_full_ket_for_abc(a, b, c, qubit_names, idx_e, idx_N, idx_c1, idx_c2, Pn):
    """
    Build FULL-Hilbert ket for |a, N14=0, b, c, others=0> in projected basis then embed.
    a: e, b: C1, c: C2.
    """
    nq = len(qubit_names)
    vec_proj = torch.zeros(2**nq, dtype=torch.complex128)
    bits = [0]*nq
    bits[idx_e]  = int(a)
    bits[idx_N]  = 0
    bits[idx_c1] = int(b)
    bits[idx_c2] = int(c)
    for q in range(nq):
        if q not in (idx_e, idx_N, idx_c1, idx_c2):
            bits[q] = 0
    idx = sum((bits[q] << q) for q in range(nq))
    vec_proj[idx] = 1.0
    return Pn.conj().T @ vec_proj

def slice_3q_from_proj(vec_proj, qubit_names, idx_e, idx_c1, idx_c2, idx_N):
    """
    Extract amplitudes for (e, C1, C2) with N14=0 and all other carbons=0,
    ordered as |abc> with (a=e, b=C1, c=C2).
    """
    nq = len(qubit_names)
    sel = []
    for a in (0,1):
        for b in (0,1):
            for c in (0,1):
                bits = [0]*nq
                bits[idx_e], bits[idx_N], bits[idx_c1], bits[idx_c2] = a, 0, b, c
                for q in range(nq):
                    if q not in (idx_e, idx_N, idx_c1, idx_c2):
                        bits[q] = 0
                sel.append(sum((bits[q] << q) for q in range(nq)))
    out = vec_proj[sel].clone()
    nrm = torch.linalg.norm(out)
    return out / nrm if float(nrm) > 0 else out

# ============================================================
# Pulse plot
# ============================================================


def plot_drives(drives, time_grids, outdir: Path, title="MW Pulse",
                gamma_e_rad_per_us_per_T=γ_e):
    """
    Plot concatenated drives as Rabi frequency [Mrad/s] vs time [ns].
    """
    pieces_t = []
    pieces_y = []
    t_acc = 0.0

    for tg, drv in zip(time_grids, drives):
        dt = float((tg[1] - tg[0]).item())
        t = t_acc + np.arange(len(tg)) * dt
        for ch, d in enumerate(drv):
            pieces_t.append((ch, t.copy()))
            pieces_y.append((ch, d.detach().cpu().numpy().copy()))
        t_acc += dt * len(tg)

    plt.figure(figsize=(10, 4))
    nchan = len(drives[0])

    for ch in range(nchan):
        t_ch = np.concatenate([t for c, t in pieces_t if c == ch])
        y_ch_uT = np.concatenate([y for c, y in pieces_y if c == ch])
        B_T = y_ch_uT * 1e-6
        Omega_rad_per_us = gamma_e_rad_per_us_per_T * B_T
        plt.plot(t_ch * 1e9, Omega_rad_per_us, label=f"Drive {ch+1}")

    plt.xlabel("Time [ns]")
    plt.ylabel("Rabi frequency [Mrad/s]")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "drives_concatenated.png", dpi=200)
    plt.show()

# ============================================================
# 3-qubit populations
# ============================================================

def compute_3q_population_trajectory(states_full_over_time,
                                     Pn,
                                     qubit_names,
                                     idx_e,
                                     idx_c1,
                                     idx_c2,
                                     idx_N):
    """
    For each time t:
      - Project FULL state to projected compsub (Pn).
      - Renormalize inside that subspace.
      - Slice to 3-qubit (e, C1, C2) with N14=|0>, others=|0>.
      - Store populations |amp_{abc}|^2 (a=e,b=C1,c=C2).
    Returns: pop_3q (T,8).
    """
    T = states_full_over_time.shape[0]
    pop_3q = np.zeros((T, 8), dtype=float)

    for t in range(T):
        psi_full = states_full_over_time[t]
        psi_all = Pn @ psi_full
        w = float((psi_all.conj() @ psi_all).real)
        if w <= 1e-14:
            continue
        psi_all = psi_all / math.sqrt(w)
        psi_3q = slice_3q_from_proj(psi_all, qubit_names,
                                    idx_e, idx_c1, idx_c2, idx_N)
        pop_3q[t, :] = torch.abs(psi_3q).cpu().numpy()**2

    return pop_3q

def plot_3q_populations(time_axis_ns, pop_3q, outdir: Path,
                        filename: str,
                        title: str,
                        highlight_state="100"):
    """
    Plot populations of |000>,...,|111> vs time,
    with shaded area under a chosen basis state, default |100>.
    """
    labels = [f"|{a}{b}{c}⟩"
              for a in (0,1)
              for b in (0,1)
              for c in (0,1)]

    a_h, b_h, c_h = (int(x) for x in highlight_state)
    idx_h = (a_h << 2) | (b_h << 1) | c_h
    pop_h = pop_3q[:, idx_h]
    area_h = np.trapz(pop_h, time_axis_ns)

    plt.figure(figsize=(10, 6))
    for i, lab in enumerate(labels):
        lw = 2.5 if i == idx_h else 1.5
        plt.plot(time_axis_ns, pop_3q[:, i], label=lab, linewidth=lw)

    plt.fill_between(time_axis_ns, pop_h, alpha=0.25, zorder=0,
                     label=rf"Area under {labels[idx_h]}")

    plt.xlabel("Time (ns)")
    plt.ylabel("Population")
    plt.title(title)
    plt.legend(ncol=2)
    plt.grid(True)

    textstr = (
        rf"$\int P_{{{labels[idx_h]}}}(t)\,dt$"
        + "\n"
        + rf"$= {area_h:.3f}\,\mathrm{{ns}}$"
    )
    plt.text(
        0.98, 0.2, textstr,
        transform=plt.gca().transAxes,
        fontsize=15,
        verticalalignment='center',
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / filename
    plt.savefig(outpath, dpi=200)
    plt.show()
    print(f"Saved 3-qubit population plot to: {outpath}")
    print(f"∫ P_{labels[idx_h]}(t) dt = {area_h:.6f} ns")

# ============================================================
# Phase invariants vs time (ZZZ + XZZ via local rotation)
# ============================================================

def _I2(dtype, device):
    return torch.eye(2, dtype=dtype, device=device)

def _H2(dtype, device):
    return (1.0 / math.sqrt(2.0)) * torch.tensor(
        [[1.0, 1.0],
         [1.0, -1.0]], dtype=dtype, device=device
    )

def _kron3(A, B, C):
    return torch.kron(torch.kron(A, B), C)

def _apply_local_to_state_3q(psi8: torch.Tensor, Ls=None) -> torch.Tensor:
    dtype, device = psi8.dtype, psi8.device
    if Ls is None:
        Ls = [_I2(dtype, device)] * 3
    L = _kron3(Ls[0], Ls[1], Ls[2])
    return L @ psi8

def compute_all_phases_vs_time(
    time_grids, drives, Δ_e,
    Pn, qubit_names,
    idx_e, idx_N, idx_c1, idx_c2,
    hadamard_on=None,   # None for ZZZ, ['A'] for XZZ
):
    """
    Compute time-dependent 1-body, 2-body, and 3-body phase coefficients
    using unwrapped φ_{abc}(t) trajectories.
    Optionally applies local H on selected qubits of the 3q slice (hadamard_on).
    """
    phi_abc = {}
    time_axis_ns = None
    Ls_eff = None

    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                psi_full0 = build_full_ket_for_abc(
                    a, b, c, qubit_names, idx_e, idx_N, idx_c1, idx_c2, Pn
                )
                chunks = apply_sequence_with_3C(Δ_e, time_grids, drives, psi_full0)
                states_concat, t_ns = fuse_states_and_time(time_grids, chunks)

                if time_axis_ns is None:
                    time_axis_ns = t_ns
                else:
                    if len(time_axis_ns) != len(t_ns):
                        raise ValueError("Time axis mismatch")

                T = states_concat.shape[0]
                phi_t = np.zeros(T, dtype=np.float64)
                idx3 = (a << 2) | (b << 1) | c

                for ti in range(T):
                    psi_full_t = states_concat[ti]
                    psi_all_t  = Pn @ psi_full_t
                    w = float((psi_all_t.conj() @ psi_all_t).real)
                    if w <= 1e-14:
                        phi_t[ti] = 0.0
                        continue
                    psi_all_t = psi_all_t / math.sqrt(w)
                    psi_3q_t = slice_3q_from_proj(
                        psi_all_t, qubit_names, idx_e, idx_c1, idx_c2, idx_N
                    )

                    if Ls_eff is None:
                        dtype, device = psi_3q_t.dtype, psi_3q_t.device
                        if hadamard_on is None:
                            I = _I2(dtype, device)
                            Ls_eff = [I, I, I]
                        else:
                            H = _H2(dtype, device)
                            I = _I2(dtype, device)
                            A_H = 'A' in hadamard_on
                            B_H = 'B' in hadamard_on
                            C_H = 'C' in hadamard_on
                            Ls_eff = [H if A_H else I,
                                      H if B_H else I,
                                      H if C_H else I]

                    psi_3q_rot = _apply_local_to_state_3q(psi_3q_t, Ls=Ls_eff)
                    amp_t = psi_3q_rot[idx3]
                    phi_t[ti] = float(torch.angle(amp_t).item())

                phi_t = np.unwrap(phi_t)
                phi_abc[(a, b, c)] = phi_t

    T = len(time_axis_ns)
    alpha = np.zeros(T)
    beta  = np.zeros(T)
    chi   = np.zeros(T)
    gamma_ab = np.zeros(T)
    gamma_ac = np.zeros(T)
    gamma_bc = np.zeros(T)
    lamb   = np.zeros(T)

    for ti in range(T):
        A = B = C = Gab = Gac = Gbc = L = 0.0
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    phi = phi_abc[(a,b,c)][ti]
                    A   += (-1)**a        * phi
                    B   += (-1)**b        * phi
                    C   += (-1)**c        * phi
                    Gab += (-1)**(a + b)  * phi
                    Gac += (-1)**(a + c)  * phi
                    Gbc += (-1)**(b + c)  * phi
                    L   += (-1)**(a+b+c)  * phi

        alpha[ti]    = A/8.0
        beta[ti]     = B/8.0
        chi[ti]      = C/8.0
        gamma_ab[ti] = Gab/8.0
        gamma_ac[ti] = Gac/8.0
        gamma_bc[ti] = Gbc/8.0
        lamb[ti]     = L/8.0

    phases = {
        "alpha": alpha,
        "beta": beta,
        "chi": chi,
        "gamma_ab": gamma_ab,
        "gamma_ac": gamma_ac,
        "gamma_bc": gamma_bc,
        "lambda": lamb,
    }
    return time_axis_ns, phases

def plot_invariants(
    time_axis_ns, phases, outdir: Path, filename: str,
    title: str,
    target_line: float = None,
    target_label: str = None,
):
    plt.figure(figsize=(12, 6))

    plt.plot(time_axis_ns, phases["alpha"],    label=r"$\Delta_{\{a\}}$")
    plt.plot(time_axis_ns, phases["beta"],     label=r"$\Delta_{\{b\}}$")
    plt.plot(time_axis_ns, phases["chi"],      label=r"$\Delta_{\{c\}}$")
    plt.plot(time_axis_ns, phases["gamma_ab"], label=r"$\Delta_{\{a,b\}}$")
    plt.plot(time_axis_ns, phases["gamma_ac"], label=r"$\Delta_{\{a,c\}}$")
    plt.plot(time_axis_ns, phases["gamma_bc"], label=r"$\Delta_{\{b,c\}}$")
    plt.plot(time_axis_ns, phases["lambda"],
             linewidth=2.6, color="black",
             label=r"$\Delta_{\{a,b,c\}}$")

    if target_line is not None:
        plt.axhline(
            target_line,
            linestyle="--", color="black", alpha=0.6,
            label=target_label if target_label else f"{target_line:.3f}"
        )

    plt.xlabel("Time (ns)")
    plt.ylabel("Phase invariant value (rad)")
    plt.title(title)
    plt.grid(True)
    plt.legend(ncol=2, frameon=True, loc='upper left')

    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / filename
    plt.savefig(outpath, dpi=200)
    plt.show()
    print(f"Saved invariant plot to: {outpath}")

# ============================================================
# Gate fidelities (ZZZ & XZZ)
# ============================================================

Z = torch.tensor([[1,0],[0,-1]], dtype=torch.complex128)
X = torch.tensor([[0,1],[1,0]], dtype=torch.complex128)

def kron3(a,b,c):
    return torch.kron(torch.kron(a,b), c)

def gate_fids(Ut, Ua):
    d = Ut.shape[0]
    ov = torch.trace(Ut.conj().T @ Ua)
    F_pro = (torch.abs(ov)**2 / (d*d)).item()
    F_avg = ((d * F_pro + 1) / (d + 1))
    return F_pro, F_avg

def zzz_gate_fidelities(U_proj: torch.Tensor):
    """
    Use diagonal of U_proj to remove local Z phases, then compare to exp(± i π/4 ZZZ).
    """
    dvec = torch.diagonal(U_proj)
    dvec = dvec / torch.clamp(dvec.abs(), min=1e-15)
    phi = torch.angle(dvec)

    def idx(a,b,c): return (a<<2) | (b<<1) | c

    def signed_sum(weight):
        s = 0.0
        for a in (0,1):
            for b in (0,1):
                for c in (0,1):
                    s += weight(a,b,c) * phi[idx(a,b,c)]
        return s

    α  =  (1/8.0) * signed_sum(lambda a,b,c: (-1)**a)
    β  =  (1/8.0) * signed_sum(lambda a,b,c: (-1)**b)
    χ  =  (1/8.0) * signed_sum(lambda a,b,c: (-1)**c)

    phases_loc = torch.zeros(8, dtype=torch.complex128)
    for a in (0,1):
        for b in (0,1):
            for c in (0,1):
                z = (-a)*α + (-b)*β + (-c)*χ
                phases_loc[idx(a,b,c)] = torch.exp(-2j * torch.as_tensor(z, dtype=torch.complex128))
    Uloc = torch.diag(phases_loc)

    U_corr = Uloc @ U_proj

    ZZZ = kron3(Z,Z,Z)
    U_plus  = torch.linalg.matrix_exp( 1j * (math.pi/4) * ZZZ)

    Fpro_p, Favg_p = gate_fids(U_plus,  U_corr)

    print(f"[ZZZ] vs exp(+i π/4 ZZZ): F_pro={Fpro_p:.8f},  F_avg={Favg_p:.8f}")

def xzz_gate_fidelities(U_proj: torch.Tensor, hadamard_on=('A',)):
    """
    Apply local H on chosen qubits in propagator frame, remove local Z phases there,
    compare to exp(± i π/4 XZZ) in lab frame.
    """
    dtype, device = U_proj.dtype, U_proj.device
    H = _H2(dtype, device)
    I = _I2(dtype, device)

    A_H = 'A' in hadamard_on
    B_H = 'B' in hadamard_on
    C_H = 'C' in hadamard_on

    L = _kron3(H if A_H else I,
               H if B_H else I,
               H if C_H else I)

    U_eff = L @ U_proj @ L  # frame with X on A

    dvec = torch.diagonal(U_eff)
    dvec = dvec / torch.clamp(dvec.abs(), min=1e-15)
    phi = torch.angle(dvec)

    def idx(a,b,c): return (a<<2) | (b<<1) | c

    def signed_sum(weight):
        s = 0.0
        for a in (0,1):
            for b in (0,1):
                for c in (0,1):
                    s += weight(a,b,c) * phi[idx(a,b,c)]
        return s

    α  =  (1/8.0) * signed_sum(lambda a,b,c: (-1)**a)
    β  =  (1/8.0) * signed_sum(lambda a,b,c: (-1)**b)
    χ  =  (1/8.0) * signed_sum(lambda a,b,c: (-1)**c)

    phases_loc = torch.zeros(8, dtype=torch.complex128, device=device)
    for a in (0,1):
        for b in (0,1):
            for c in (0,1):
                z = (-a)*α + (-b)*β + (-c)*χ
                phases_loc[idx(a,b,c)] = torch.exp(-1j * z)
    Uloc_eff = torch.diag(phases_loc)

    U_corr_eff = Uloc_eff @ U_eff
    U_corr_lab = L @ U_corr_eff @ L

    XZZ = kron3(X,Z,Z)
    U_plus  = torch.linalg.matrix_exp( 1j * (math.pi/4) * XZZ)

    Fpro_p, Favg_p = gate_fids(U_plus,  U_corr_lab)

    print(f"[XZZ] vs exp(+i π/4 XZZ): F_pro={Fpro_p:.8f},  F_avg={Favg_p:.8f}")
