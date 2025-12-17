# Minimal, working script for the current MAIN
# - keeps: loading, evolution, fuse timeline, 4q slice, local-Z extraction,
#          Bloch trajectory plotting, and the main pipeline
# - drops: unused city plots, projectors not needed, extra imports

import math, itertools
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pathlib import Path

# Your model utilities ---
from quantum_model_3C import get_U_RWA, ω1, set_active_carbons, get_active_carbons, get_precomp
from evolution import get_evolution_vector

# ================
# Loading & evolve
# ================
def load_ckpt(result_dir: Path):
    ckpt = torch.load(result_dir / "pulse_solution.pt", map_location="cpu", weights_only=False)
    return {
        "drive": ckpt["drive"],
        "time_grid": ckpt["time_grid"],
        "Δ_e": float(ckpt["Δ"]),
        "basis_indices": ckpt.get("basis_indices", None),
        "objective_type": ckpt.get("objective_type", "Unknown"),
        "timestamp": ckpt.get("timestamp", "Unknown"),
    }

def apply_pulse_3C(Δ_e, time_grid, drive, ψ_init):
    U_fn = (lambda Ω, dt, t: get_U_RWA(Ω, dt, t, Δ_e=Δ_e, ω_RF=ω1))
    states = get_evolution_vector(U_fn, time_grid, drive, ψ_init)
    return torch.stack(states, dim=0)  # (T+1, D)

def apply_sequence_3C(Δ_e, time_grids, drives, ψ0):
    ψ = ψ0
    chunks = []
    for tg, drv in zip(time_grids, drives):
        chunk = apply_pulse_3C(Δ_e, tg, drv, ψ)  # (Ti+1, D)
        chunks.append(chunk)
        ψ = chunk[-1]
    return chunks

def fuse_states_and_time(time_grids, states_chunks):
    """Fuse per-pulse trajectories into one timeline (drop boundary duplicates)."""
    cat, times_s, t_acc = [], [], 0.0
    for i, (tg, chunk) in enumerate(zip(time_grids, states_chunks)):
        dt = float((tg[1] - tg[0]).item())
        L  = int(chunk.shape[0])
        if i == 0:
            cat.append(chunk); times_s.extend(t_acc + np.arange(L) * dt); t_acc += (L-1) * dt
        else:
            cat.append(chunk[1:]); times_s.extend(t_acc + np.arange(1, L) * dt); t_acc += (L-1) * dt
    return torch.cat(cat, dim=0), (np.asarray(times_s) * 1e9)

# ===============================
# Computational index & 4q slicing
# ===============================
def make_allqubits_basis_indices(pc, n14_pair=(0,1), electron_map=('m1','0')):
    """
    Build indices for [e, 14N, C<label0>, C<label1>, ...] (all active carbons).
    Returns indices (length 2^(2+N_C)) and the name list in the same order.
    """
    act = list(get_active_carbons())      # labels in Hamiltonian order
    N_C = len(act)
    n_qubits = 2 + N_C

    # electron logical map: 0/1 -> manifold index (0: |0_e>, 1: |-1_e>)
    e_log_to_m = {0: (1 if electron_map[0]=='m1' else 0),
                  1: (1 if electron_map[1]=='m1' else 0)}

    nconf = 2**int(pc['N_C']); dim_nuc = 3 * nconf
    def basis_index(e_manifold, mI_block, c_bits):
        off_e = 0 if e_manifold == 0 else dim_nuc
        return int(off_e + mI_block * nconf + c_bits)

    indices = []
    for bits_int in range(1 << n_qubits):
        e = (bits_int >> 0) & 1
        n = (bits_int >> 1) & 1
        e_m = e_log_to_m[e]
        mI  = int(n14_pair[n])

        # build c_bits in Hamiltonian’s carbon order
        c_bits = 0
        for iC in range(N_C):
            b = (bits_int >> (2 + iC)) & 1
            c_bits |= (b << iC)

        indices.append(basis_index(e_m, mI, c_bits))

    names = ["e⁻", "14N"] + [f"C{label}" for label in act]
    return indices, names

def slice_qubits(psi_full, idx_comp, selected_pos, qubit_names):
    """
    Directly slice a k-qubit subvector from the FULL state (others assumed |0>).
    Returns: (2^k,) complex tensor in little-endian order within the slice.
    """
    k = len(selected_pos)
    out = torch.zeros(1 << k, dtype=torch.complex128)
    for j, bits_sel in enumerate(itertools.product([0, 1], repeat=k)):
        j_all = sum((int(bits_sel[t]) << selected_pos[t]) for t in range(k))  # little-endian
        full_idx = int(idx_comp[j_all])
        out[j] = psi_full[full_idx]
    return out

# =========================
# Local-Z extraction (fixed)
# =========================


# ===========================
# Bloch trajectories & plots
# ===========================
σx = torch.tensor([[0,1],[1,0]], dtype=torch.complex128)
σy = torch.tensor([[0,-1j],[1j,0]], dtype=torch.complex128)
σz = torch.tensor([[1,0],[0,-1]], dtype=torch.complex128)

def _partial_trace_keep_one_qubit(rho_proj, n_qubits, keep_q):
    """Trace out all but one qubit from an n-qubit density matrix."""
    import string
    R = rho_proj.reshape(*([2]*n_qubits + [2]*n_qubits))  # (i0..i{n-1}, j0..j{n-1})
    letters = list(string.ascii_lowercase)
    if n_qubits > len(letters): raise ValueError("n_qubits too large.")
    left  = letters[:n_qubits]
    right = [ch.upper() for ch in left]
    for r in range(n_qubits):
        if r != keep_q: right[r] = left[r]  # enforce i_r == j_r
    eqn = ''.join(left + right) + '->' + left[keep_q] + right[keep_q]
    rho_1q = torch.einsum(eqn, R)
    return rho_1q

def _partial_trace_keep_one_qubit(rho_proj, n_qubits, keep_q):
    """Trace out all but one qubit from an n-qubit density matrix.
    Assumes little-endian logical ordering (qubit 0 = LSB)."""
    import string
    R = rho_proj.reshape(*([2]*n_qubits + [2]*n_qubits))  # (i..., j...)

    # logical → physical axis mapping for little-endian
    axis = n_qubits - 1 - keep_q

    letters = list(string.ascii_lowercase)
    left  = letters[:n_qubits]
    right = [ch.upper() for ch in left]

    # Trace out everything except 'axis'
    for r in range(n_qubits):
        if r != axis:
            right[r] = left[r]  # i_r == j_r (trace)

    eqn = ''.join(left + right) + '->' + left[axis] + right[axis]
    rho_1q = torch.einsum(eqn, R)
    return rho_1q

def compute_bloch_traj_any_idx(states_full, time_ns, idx_comp, qubit_names,
                               positions=None, max_points=600, renorm_in_proj=True):
    """Gather computational register each step, optionally renormalize, then Bloch coords for chosen qubits."""
    n = len(qubit_names)
    if positions is None:
        positions = list(range(n))
    positions = [int(p) for p in positions]
    if any(p < 0 or p >= n for p in positions):
        raise ValueError(f"positions must be in 0..{n-1}; got {positions}")

    T = states_full.shape[0]
    if T == 0: raise ValueError("states_full has zero time steps.")

    stride = max(1, T // max_points)
    idxs = np.arange(0, T, stride, dtype=int)
    if idxs[-1] != T - 1: idxs = np.append(idxs, T - 1)
    times_sel = time_ns[idxs]

    traj = {p: [] for p in positions}
    for t in idxs:
        psi_proj = states_full[t][idx_comp].clone()
        if renorm_in_proj:
            nrm2 = float((psi_proj.conj() @ psi_proj).real)
            if nrm2 > 1e-14: psi_proj /= math.sqrt(nrm2)

        if float((psi_proj.conj() @ psi_proj).real) <= 1e-14:
            for p in positions: traj[p].append((np.nan, np.nan, np.nan))
            continue

        rho = psi_proj[:, None] @ psi_proj[None, :].conj()
        for p in positions:
            rho_p = _partial_trace_keep_one_qubit(rho, n_qubits=n, keep_q=p)
            x = torch.real(torch.trace(σx @ rho_p)).item()
            y = torch.real(torch.trace(σy @ rho_p)).item()
            z = torch.real(torch.trace(σz @ rho_p)).item()
            traj[p].append((x, y, z))

    traj = {p: np.asarray(traj[p]) for p in positions}
    panel_names = [qubit_names[p] for p in positions]
    return traj, times_sel, panel_names

def plot_bloch_3d_any(traj, panel_names, out_path, filename="bloch_trajectories_any.png", cmap_name="plasma"):
    fig = plt.figure(figsize=(4*len(traj), 4), dpi=160)
    axes = [fig.add_subplot(1, len(traj), i+1, projection='3d') for i in range(len(traj))]
    cmap = mpl.colormaps.get_cmap(cmap_name)
    phi = np.linspace(0, 2*np.pi, 361)

    for i, (pos, coords) in enumerate(traj.items()):
        ax = axes[i]
        ax.set_title(panel_names[i])
        ax.set_xlim([-1.1,1.1]); ax.set_ylim([-1.1,1.1]); ax.set_zlim([-1.1,1.1])
        ax.set_xlabel("⟨X⟩"); ax.set_ylabel("⟨Y⟩"); ax.set_zlabel("⟨Z⟩")

        # unit sphere & great circles
        u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
        xs, ys, zs = np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)
        ax.plot_surface(xs, ys, zs, color='lightblue', alpha=0.10, linewidth=0)
        ax.plot(np.cos(phi), np.sin(phi), 0*phi, color='#5D6D7E', lw=0.9, alpha=0.85)
        ax.plot(0*phi, np.cos(phi), np.sin(phi), color='#5D6D7E', lw=0.7, alpha=0.6)
        ax.plot(np.cos(phi), 0*phi, np.sin(phi), color='#5D6D7E', lw=0.7, alpha=0.6)

        if coords.size >= 2:
            points = coords.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = colors.Normalize(vmin=0, vmax=len(segments))
            col = cmap(norm(np.arange(len(segments))))
            col[:,3] = np.linspace(0.12, 1.0, len(segments))
            lc = Line3DCollection(segments, colors=col, linewidth=2.0)
            ax.add_collection3d(lc)
            ax.scatter(*coords[-1], color='black', s=40)

    fig.subplots_adjust(wspace=0.15, left=0.04, right=0.98, top=0.90, bottom=0.12)
    out = out_path / filename
    fig.savefig(out, dpi=200); plt.show()
    print(f"Saved: {out}")

def plot_bloch_xy_any(traj, panel_names, out_path, filename="bloch_xy_any.png", cmap_name="plasma"):
    m = len(traj)
    fig, axes = plt.subplots(1, m, figsize=(4*m, 3.6), dpi=160)
    if m == 1: axes = [axes]
    cmap = mpl.colormaps.get_cmap(cmap_name)
    phi = np.linspace(0, 2*np.pi, 361)

    for i, (pos, coords) in enumerate(traj.items()):
        ax = axes[i]
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-1.1,1.1); ax.set_ylim(-1.1,1.1)
        ax.set_xlabel("⟨X⟩"); ax.set_ylabel("⟨Y⟩"); ax.set_title(panel_names[i])
        ax.plot(np.cos(phi), np.sin(phi), lw=1.0, color='#5D6D7E', alpha=0.85)
        ax.axhline(0, lw=0.8, color='#5D6D7E', alpha=0.6)
        ax.axvline(0, lw=0.8, color='#5D6D7E', alpha=0.6)

        if coords.size >= 2:
            pts = coords[:, :2]; P = pts.reshape(-1,1,2)
            segs = np.concatenate([P[:-1], P[1:]], axis=1)
            norm = colors.Normalize(vmin=0, vmax=len(segs))
            col = cmap(norm(np.arange(len(segs)))); col[:,3] = np.linspace(0.12, 1.0, len(segs))
            lc = LineCollection(segs, colors=col, linewidths=2.0)
            ax.add_collection(lc)
            ax.scatter(pts[-1,0], pts[-1,1], s=40, color='black', zorder=3)

    fig.subplots_adjust(wspace=0.35, left=0.06, right=0.98, top=0.88, bottom=0.20)
    out = out_path / filename
    fig.savefig(out, dpi=200); plt.show()
    print(f"Saved: {out}")

def plot_bloch_any_by_positions_idx(states_full, time_ns, idx_comp, qubit_names, positions,
                                    outdir, prefix="bloch_any", max_points=600, cmap_name="plasma",
                                    renorm_in_proj=True):
    traj, t_sel, panel_names = compute_bloch_traj_any_idx(
        states_full, time_ns, idx_comp, qubit_names,
        positions=positions, max_points=max_points, renorm_in_proj=renorm_in_proj
    )
    plot_bloch_3d_any(traj, panel_names, outdir, filename=f"{prefix}_3d.png", cmap_name=cmap_name)
    plot_bloch_xy_any(traj, panel_names, outdir, filename=f"{prefix}_xy.png", cmap_name=cmap_name)
    return traj, t_sel, panel_names


def evolve_basis_and_get_phi(time_grids, drives, Δ_e, *bits,
                             idx_comp, selected_pos, D):
    """
    Evolve |b> (others=0) in FULL space and return:
      - phi_meas = angle(<b|U|b>)   [measured phase, consistent with φ_meas in the notes]
      - psi_k    = normalized k-qubit slice at the end (optional)

    Args:
        bits            : tuple/list of 0/1 of length k, the slice basis |b>
        idx_comp        : length 2^n_total mapping logical index -> FULL index
        selected_pos    : positions (in logical order) of the k qubits (e.g., [0,2,3,4])
        D               : full Hilbert-space dimension
        get_full        : if True, also return the final k-qubit slice column
        renorm_in_slice : normalize the returned slice to unit norm (if its norm > 0)

    Returns:
        (phi_meas, psi_k) if get_full else phi_meas
    """
    import torch, math

    # --- 1) Build |b, others=0> in FULL space
    if len(bits) == 1 and isinstance(bits[0], (list, tuple)):
        b_tuple = tuple(int(x) for x in bits[0])
    else:
        b_tuple = tuple(int(x) for x in bits)
    k = len(selected_pos)
    if len(b_tuple) != k:
        raise ValueError(f"bits length {len(b_tuple)} != len(selected_pos) {k}")

    # logical index for |b> with others=0 (little-endian across all logical qubits)
    j_all = sum((int(b_tuple[t]) << selected_pos[t]) for t in range(k))
    full_idx = int(idx_comp[j_all])

    psi_full0 = torch.zeros(D, dtype=torch.complex128)
    psi_full0[full_idx] = 1.0

    # --- 2) Evolve under the pulse sequence in FULL space
    chunks = apply_sequence_3C(Δ_e, time_grids, drives, psi_full0)
    states_concat, _ = fuse_states_and_time(time_grids, chunks)
    psi_full_T = states_concat[-1]

    # --- 3) Measured diagonal phase for |b> (NO renorm; consistent with φ_meas)
    amp = psi_full_T[full_idx]
    phi_meas = float(torch.angle(amp))


    return phi_meas
# ================
# --MAIN -----
# ================

# 0) Pulse folders (concatenate as a sequence if multiple)
pulse_dirs = [
            #Path("results/pulse_2025-10-16_15-45-42"),
            Path("results/pulse_2025-10-17_17-45-05"),

              ]

# 1) Match active carbons to the optimization run; expect 3 actives (C1,C2,C3)
set_active_carbons([1,2,3])
pc = get_precomp()
N_C = int(pc["N_C"])
D = 2 * 3 * (2 ** N_C)

# 2) Logical computational indices and names
n14_pair = (0,1)            # 14N “qubit” levels
electron_map = ('0','m1')   # logical 1→|-1_e>, 0→|0_e>
idx_comp, qubit_names = make_allqubits_basis_indices(pc, n14_pair, electron_map)

# Prepare |00000> in FULL space (D = 2 * 3 * 2^N_C)
psi_full = torch.zeros(D, dtype=torch.complex128); psi_full[0] = 1.0

# Apply H on chosen qubits (e, N, C1, C2, C3)
H  = (1/torch.sqrt(torch.tensor(2.0))) * torch.tensor([[1, 1],[1,-1]], dtype=torch.complex128)
I2 = torch.eye(2, dtype=torch.complex128)
U_H = torch.kron(torch.kron(torch.kron(torch.kron(H, H), H), I2), H)  # order [e, N, C1, C2, C3] -> operators have to be ordered the opposite way
psi_proj = psi_full[idx_comp].clone()
psi_proj = U_H @ psi_proj


print(f"post hadamard application:{psi_proj}")

# rembed into full space
psi_full[:] = 0
psi_full[idx_comp] = psi_proj

# Load pulse(s) and evolve
first = load_ckpt(pulse_dirs[0])
Δ_e = first["Δ_e"]
drives, time_grids = [], []
for p in pulse_dirs:
    info = load_ckpt(p)
    drives.append(info["drive"])
    time_grids.append(info["time_grid"])

states_chunks = apply_sequence_3C(Δ_e, time_grids, drives, psi_full)
states_concat, time_axis_ns = fuse_states_and_time(time_grids, states_chunks)

# Plot Bloch trajectories (all 5 logicals)
_traj, _t_sel, _names = plot_bloch_any_by_positions_idx(
    states_concat, time_axis_ns, idx_comp, qubit_names,
    positions=[0, 1, 2, 3], outdir=pulse_dirs[-1], prefix="all5", max_points=800
)


#
# --- 1) Phases from the 4-qubit computational basis |a b c d>
selected_pos = [0, 2, 3, 4]
phi = {}
for a in (0,1):
    for b in (0,1):
        for c in (0,1):
            for d in (0,1):
                ph = evolve_basis_and_get_phi(
                    time_grids, drives, Δ_e, a, b, c, d,
                    idx_comp=idx_comp, selected_pos=selected_pos, D=D,
                )
                phi[(a,b,c,d)] = ph

# --- 2) Finite-difference coefficients (4 locals, 6 pairs, 4 triples, 1 four-body)
def sum_over_abcd(weight_fn):
    s = 0.0
    for a in (0,1):
        for b in (0,1):
            for c in (0,1):
                for d in (0,1):
                    s += weight_fn(a,b,c,d) * phi[(a,b,c,d)]
    return s

inv16 = 1.0/16.0

# locals
α0 = inv16 * sum_over_abcd(lambda a,b,c,d: (-1)**a)
α1 = inv16 * sum_over_abcd(lambda a,b,c,d: (-1)**b)
α2 = inv16 * sum_over_abcd(lambda a,b,c,d: (-1)**c)
α3 = inv16 * sum_over_abcd(lambda a,b,c,d: (-1)**d)

# pairs
γ01 = inv16 * sum_over_abcd(lambda a,b,c,d: (-1)**(a+b))
γ02 = inv16 * sum_over_abcd(lambda a,b,c,d: (-1)**(a+c))
γ03 = inv16 * sum_over_abcd(lambda a,b,c,d: (-1)**(a+d))
γ12 = inv16 * sum_over_abcd(lambda a,b,c,d: (-1)**(b+c))
γ13 = inv16 * sum_over_abcd(lambda a,b,c,d: (-1)**(b+d))
γ23 = inv16 * sum_over_abcd(lambda a,b,c,d: (-1)**(c+d))

# triples
λ012 = inv16 * sum_over_abcd(lambda a,b,c,d: (-1)**(a+b+c))
λ013 = inv16 * sum_over_abcd(lambda a,b,c,d: (-1)**(a+b+d))
λ023 = inv16 * sum_over_abcd(lambda a,b,c,d: (-1)**(a+c+d))
λ123 = inv16 * sum_over_abcd(lambda a,b,c,d: (-1)**(b+c+d))

# 4-body (κ). With the convention in your notes, this one has a + sign for theory φ.
κ = inv16 * sum_over_abcd(lambda a,b,c,d: (-1)**(a+b+c+d))

print("Locals (rad):", [f"{x:.6f}" for x in (α0,α1,α2,α3)])
print("Pairs  (rad):", [f"{x:.6f}" for x in (γ01,γ02,γ03,γ12,γ13,γ23)])
print("Triples(rad):", [f"{x:.6f}" for x in (λ012,λ013,λ023,λ123)])
print("4-body κ (rad):", f"{κ:.6f}")

# --- 3) Build U_loc^† on the 4-qubit slice
# Use Z eigenvalues (-1)^{bit} = 1 - 2*bit, and your earlier dagger convention.
alphas = (α0, α1, α2, α3)

phases_loc_dag = torch.empty(16, dtype=torch.complex128)
for a in (0,1):
    for b in (0,1):
        for c in (0,1):
            for d in (0,1):
                idx = (a<<3) | (b<<2) | (c<<1) | d  # little-endian within 4q slice
                s = (1 - 2*a)*α0 + (1 - 2*b)*α1 + (1 - 2*c)*α2 + (1 - 2*d)*α3
                phases_loc_dag[idx] = torch.exp(-1j * torch.tensor(s))
Uloc_dag = torch.diag(phases_loc_dag)
#


psi_full_T = states_concat[-1]

# Slice computational part down to action part and extract local-Z on action qubits [e, C1, C2, C3]
psi_comp_T = psi_full_T[idx_comp].clone()
action_positions = [0, 2, 3, 4]  # e⁻, C1, C2, C3 (14N pinned to 0 implicitly)
#alphas, Uloc_dag_proj = extract_localZ_correction_from_sequence_any_k_idx(
#    time_grids, drives, Δ_e, idx_comp, qubit_names, action_positions, D
#)
#print("k =", len(action_positions), "alphas (rad) =", [f"{a:.6f}" for a in alphas])

psi_action = slice_qubits(psi_comp_T, idx_comp, action_positions, qubit_names)
psi_action_T_corr = Uloc_dag @ psi_action

# Second H layer on [e, C1, C2, C3]
U_H2 = torch.kron(torch.kron(torch.kron(H, H), H), H)
psi_proj_final = U_H2 @ psi_action_T_corr

print(f'first:{psi_proj_final}')