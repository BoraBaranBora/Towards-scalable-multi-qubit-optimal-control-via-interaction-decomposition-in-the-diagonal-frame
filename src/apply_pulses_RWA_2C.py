import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# --- Your modules (3C model) ---
from quantum_model_3C import (
    get_U_RWA, ω1, γ_e, Λ_s,
    set_active_carbons, get_active_carbons, get_precomp
)
from evolution import get_evolution_vector

# -----------------------------
# Helper: load one checkpoint
# -----------------------------
def load_ckpt(result_dir: Path):
    ckpt_path = result_dir / "pulse_solution.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    drive = ckpt.get("drive")
    time_grid = ckpt["time_grid"]
    delta_e = ckpt["Δ"]  # saved by your optimizer script
    basis_indices = ckpt.get("basis_indices", None)  # present for Gate objectives
    initial_target_pairs = ckpt.get("initial_target_pairs", None)  # present for Multi-State objectives

    return {
        "drive": drive,
        "time_grid": time_grid,
        "Δ_e": float(delta_e),
        "basis_indices": basis_indices,
        "initial_target_pairs": initial_target_pairs,
        "objective_type": ckpt.get("objective_type", "Unknown"),
        "timestamp": ckpt.get("timestamp", "Unknown"),
    }

# -----------------------------
# Evolution utilities
# -----------------------------
def apply_pulse_with_3C(Δ_e, time_grid, drive, ψ_init):
    """
    Evolves a single pulse using your 3C model get_U_RWA(Ω, dt, t, Δ_e, ω_RF).
    Uses the same stepping as your optimizer (via get_evolution_vector).
    """
    U_fn = (lambda Ω, dt, t: get_U_RWA(Ω, dt, t, Δ_e=Δ_e, ω_RF=ω1))
    states = get_evolution_vector(U_fn, time_grid, drive, ψ_init)
    return states  # shape: (T+1, D)

def apply_sequence_with_3C(Δ_e, time_grids, drives, ψ0):
    ψ = ψ0
    all_states = []
    for tg, drv in zip(time_grids, drives):
        states = apply_pulse_with_3C(Δ_e, tg, drv, ψ)
        all_states.append(states)
        ψ = states[-1]
    return all_states


def apply_pulse_with_3C(Δ_e, time_grid, drive, ψ_init):
    """
    Evolves a single pulse using your 3C model get_U_RWA(Ω, dt, t, Δ_e, ω_RF).
    Returns a Tensor of shape (T+1, D).
    """
    U_fn = (lambda Ω, dt, t: get_U_RWA(Ω, dt, t, Δ_e=Δ_e, ω_RF=ω1))
    states_list = get_evolution_vector(U_fn, time_grid, drive, ψ_init)  # list[Tensor]
    # convert to a single Tensor so the sequence is tensors, not lists
    states = torch.stack(states_list, dim=0)  # (T+1, D)
    return states

def apply_sequence_with_3C(Δ_e, time_grids, drives, ψ0):
    ψ = ψ0
    chunks = []
    for tg, drv in zip(time_grids, drives):
        states = apply_pulse_with_3C(Δ_e, tg, drv, ψ)  # Tensor (T+1, D)
        chunks.append(states)
        ψ = states[-1]
    return chunks

# -----------------------------
# Computational subspace helpers
# -----------------------------
def projector_from_indices(basis_indices, full_dim):
    """
    Returns P (8 x D) that picks the 3-qubit computational basis states out of the full Hilbert space.
    """
    if basis_indices is None:
        raise ValueError("basis_indices not found in checkpoint. Rerun optimizer or provide them manually.")
    P = torch.zeros((len(basis_indices), full_dim), dtype=torch.complex128)
    for i, idx in enumerate(basis_indices):
        P[i, idx] = 1.0
    return P

def kron3(a, b, c):
    return torch.kron(torch.kron(a, b), c)

def pauli_ops_3q():
    X = torch.tensor([[0,1],[1,0]], dtype=torch.complex128)
    Z = torch.tensor([[1,0],[0,-1]], dtype=torch.complex128)
    I = torch.eye(2, dtype=torch.complex128)
    # For qubit ordering (A,B, C) -> indices 0,1,2
    X_ops = [
        kron3(X, I, I),  # qubit 0
        kron3(I, X, I),  # qubit 1
        kron3(I, I, X),  # qubit 2
    ]
    Z_ops = [
        kron3(Z, I, I),
        kron3(I, Z, I),
        kron3(I, I, Z),
    ]
    return X_ops, Z_ops

def compute_bloch_projections_from_full(states_full, P, normalize=True):
    """
    Project full states into 3-qubit space and compute <X>, <Z> for each qubit.
    If leakage occurs, optionally renormalize in the subspace to keep values in [-1,1].
    """
    X_ops, Z_ops = pauli_ops_3q()
    X_ops = [op for op in X_ops]
    Z_ops = [op for op in Z_ops]

    X_vals_all = [[] for _ in range(3)]
    Z_vals_all = [[] for _ in range(3)]

    for ψ in states_full:
        ψ_comp = P @ ψ  # (8,)
        norm2 = torch.real((ψ_comp.conj() @ ψ_comp))
        if normalize and norm2 > 0:
            ψ_eff = ψ_comp / torch.sqrt(norm2)
        else:
            ψ_eff = ψ_comp

        ρ = ψ_eff[:, None] @ ψ_eff[None, :].conj()  # (8,8)

        for q in range(3):
            X_vals_all[q].append(torch.real(torch.trace(X_ops[q] @ ρ)).item())
            Z_vals_all[q].append(torch.real(torch.trace(Z_ops[q] @ ρ)).item())

    return Z_vals_all, X_vals_all  # lists of length 3, each length T


#

def nconf_from_pc(pc):
    return 2 ** int(pc['N_C'])

def basis_index(e_manifold: int, mI_block: int, c_bits: int, pc):
    """
    e_manifold: 0 for |0_e>, 1 for |-1_e>
    mI_block:   0:+1, 1:0, 2:-1
    c_bits:     integer 0..(2^N_C-1), bitstring of ALL active carbons (0=↑, 1=↓)
    """
    nconf = nconf_from_pc(pc)
    dim_nuc = 3 * nconf
    offset_e = 0 if e_manifold == 0 else dim_nuc
    return int(offset_e + mI_block * nconf + c_bits)

def _carbon_label_to_bitpos(pc):
    """
    Map each active carbon label to its bit position in c_bits.
    Bit positions follow the order used by the Hamiltonian for active carbons.
    """
    act = list(get_active_carbons())  # e.g. [1,2,4] -> 3 active carbons
    return {label: i for i, label in enumerate(act)}    # label -> bitpos

def make_multi_qubit_basis_indices(pc, n14_pair=(0,1), electron_map=('m1','0')):
    """
    Build indices for the computational subspace including:
      Q0 = Electron (logical 0/1 set by electron_map)
      Q1 = 14N      (two selected levels via n14_pair, e.g. (0:+1, 1:0))
      Q2..Q(1+N_C) = all active Carbons in the order of get_active_carbons()

    Returns:
      indices: list[int] of length 2^(2+N_C)
      names:   list[str] qubit names in the same order
    """
    act = list(get_active_carbons())            # carbon labels in Hamiltonian order
    N_C = len(act)
    n_qubits = 2 + N_C

    # Electron logical mapping
    e_log_to_manifold = {
        0: 1 if electron_map[0] == 'm1' else 0,
        1: 1 if electron_map[1] == 'm1' else 0,
    }

    # Build all bitstrings for the (2 + N_C) qubits in the order [e, N14, C[label0], C[label1], ...]
    indices = []
    for q_bits_int in range(1 << n_qubits):
        # Extract bits
        # bit 0 -> electron, bit 1 -> 14N, bits 2.. -> carbons in order of act
        qe = (q_bits_int >> 0) & 1
        qn = (q_bits_int >> 1) & 1
        # Electron manifold & 14N block
        e_m = e_log_to_manifold[qe]
        mI  = int(n14_pair[qn])

        # Build c_bits integer for ALL active carbons
        c_bits = 0
        for iC, label in enumerate(act):
            b = (q_bits_int >> (2 + iC)) & 1
            c_bits |= (b << iC)  # iC is the Hamiltonian bit position for this carbon

        idx = basis_index(e_m, mI, c_bits, pc)
        indices.append(idx)

    names = ["e⁻", "14N"] + [f"C{label}" for label in act]
    return indices, names

def projector_from_indices_general(basis_indices, full_dim):
    P = torch.zeros((len(basis_indices), full_dim), dtype=torch.complex128)
    for i, idx in enumerate(basis_indices):
        P[i, int(idx)] = 1.0
    return P

def pauli_ops_n(n_qubits):
    X = torch.tensor([[0,1],[1,0]], dtype=torch.complex128)
    Z = torch.tensor([[1,0],[0,-1]], dtype=torch.complex128)
    I = torch.eye(2, dtype=torch.complex128)

    def kron_all(mats):
        out = mats[0]
        for m in mats[1:]:
            out = torch.kron(out, m)
        return out

    X_ops, Z_ops = [], []
    for q in range(n_qubits):
        mats_x = [I]*n_qubits
        mats_z = [I]*n_qubits
        mats_x[q] = X
        mats_z[q] = Z
        X_ops.append(kron_all(mats_x))
        Z_ops.append(kron_all(mats_z))
    return X_ops, Z_ops

def compute_bloch_projections_projected(states_full, P, n_qubits, normalize=True):
    X_ops, Z_ops = pauli_ops_n(n_qubits)
    X_vals = [[] for _ in range(n_qubits)]
    Z_vals = [[] for _ in range(n_qubits)]

    for ψ in states_full:
        ψ_comp = P @ ψ
        norm2 = torch.real(ψ_comp.conj() @ ψ_comp)
        ψ_comp = ψ_comp / torch.sqrt(norm2) if (normalize and norm2 > 0) else ψ_comp
        ρ = ψ_comp[:, None] @ ψ_comp[None, :].conj()

        for q in range(n_qubits):
            X_vals[q].append(torch.real(torch.trace(X_ops[q] @ ρ)).item())
            Z_vals[q].append(torch.real(torch.trace(Z_ops[q] @ ρ)).item())

    return Z_vals, X_vals

#
import math

def project_norm_state(ψ_full, Pn):
    """Project full state to computational subspace and L2-normalize."""
    ψ = Pn @ ψ_full
    n2 = torch.real(ψ.conj() @ ψ)
    return ψ / torch.sqrt(n2) if n2 > 0 else ψ

def von_neumann_entropy(rho):
    """S(ρ) in bits."""
    # Hermitize for numerical stability
    rho = 0.5*(rho + rho.conj().T)
    evals = torch.linalg.eigvalsh(rho).real.clamp(min=0)
    evals = evals/ evals.sum()
    # avoid log(0)
    nz = evals > 1e-14
    s = -(evals[nz] * torch.log2(evals[nz])).sum().item()
    return s

def partial_trace(rho, dims, keep):
    """
    Trace out all subsystems NOT in keep.
    dims: list of local dims, e.g. [2,2,2,...]
    keep: iterable of subsystem indices to keep (0-based)
    Returns density matrix on ⊗_{k in keep} H_k.
    """
    N = len(dims)
    keep = sorted(set(keep))
    trace = sorted([i for i in range(N) if i not in keep], reverse=True)

    # reshape to (i0..iN-1, j0..jN-1)
    rho_t = rho.reshape(*(dims + dims))

    # For each traced subsystem t (in DESC order):
    #   - take diagonal along (it, jt)
    #   - sum over that new diagonal axis (position t)
    for t in trace:
        rho_t = rho_t.diagonal(dim1=t, dim2=t + N).sum(dim=t)

    kept_dims = [dims[i] for i in keep]
    d_out = int(np.prod(kept_dims)) if kept_dims else 1
    return rho_t.reshape(d_out, d_out)

def partial_trace(rho, dims, keep):
    """
    Trace out all subsystems NOT in keep.

    rho: (D, D) with D = prod(dims)
    dims: list of local dims, e.g. [2,2,2,...]
    keep: iterable of subsystem indices to keep (0-based)
    """
    import numpy as np
    keep = sorted(set(keep))
    N = len(dims)
    trace = [i for i in range(N) if i not in keep]

    # Fast path: keep everything → return rho
    if not trace:
        return rho

    dims_keep = [dims[i] for i in keep]
    dims_trace = [dims[i] for i in trace]

    # Reshape to (d0..dN-1, d0..dN-1)
    R = rho.reshape([*dims, *dims])

    # Permute to [keep_rows, trace_rows, keep_cols, trace_cols]
    perm = keep + trace + [N + i for i in keep] + [N + i for i in trace]
    R = R.permute(perm)

    # Collapse keep and trace blocks
    dk = int(np.prod(dims_keep)) if dims_keep else 1
    dt = int(np.prod(dims_trace)) if dims_trace else 1
    R = R.reshape(dk, dt, dk, dt)  # (i_k, i_t, j_k, j_t)

    # Perform partial trace over the traced subsystem: sum_a R[i_k, a, j_k, a]
    R = R.diagonal(offset=0, dim1=1, dim2=3).sum(-1)  # → (dk, dk)

    return R

def partial_transpose(rho, dims, sysA):
    """
    Partial transpose over subsystems in sysA.
    dims: [d0, d1, ...]
    """
    N = len(dims)
    # reshape to (i0..iN-1, j0..jN-1)
    rho_t = rho.reshape(*(dims + dims))
    # for each system in A, swap (ik, jk) indices
    idx_i = list(range(N))
    idx_j = list(range(N, 2*N))
    for k in sysA:
        idx_i[k], idx_j[k] = idx_j[k], idx_i[k]
    perm = idx_i + idx_j
    rho_pt = rho_t.permute(perm).reshape(rho.shape)
    return rho_pt

def log_negativity(rho, dims, A):
    """
    Logarithmic negativity between partition A and B (rest).
    dims: local dims
    A: set/tuple of subsystem indices in A
    """
    rho_pt = partial_transpose(rho, dims, A)
    # trace norm = sum singular values = sum |eigs| for Hermitian PT
    eigs = torch.linalg.eigvalsh(0.5*(rho_pt+rho_pt.conj().T))
    trace_norm = eigs.abs().sum().real.item()
    # E_N = log2 ||ρ^{T_A}||_1
    return math.log2(trace_norm)

def two_qubit_concurrence(rho_2q):
    """
    Wootters concurrence for a 2-qubit mixed state.
    """
    σy = torch.tensor([[0,-1j],[1j,0]], dtype=torch.complex128)
    Y = torch.kron(σy, σy)
    ρ = rho_2q
    ρ_tilde = Y @ ρ.conj() @ Y
    # eigenvalues of ρ ρ_tilde (real non-negative)
    M = ρ @ ρ_tilde
    eigs = torch.linalg.eigvals(M).real.clamp(min=0).sqrt().sort(descending=True).values
    c = max(0.0, (eigs[0] - eigs[1] - eigs[2] - eigs[3]).item())
    return c

# -----------------------------
# Plotting
# -----------------------------
def plot_bloch(Z_vals_all, X_vals_all, time_axis_ns, save_path: Path):
    plt.figure(figsize=(12, 6))
    for q in range(3):
        plt.plot(time_axis_ns, Z_vals_all[q], label=f"Q{q} ⟨Z⟩")
        plt.plot(time_axis_ns, X_vals_all[q], '--', label=f"Q{q} ⟨X⟩")
    plt.xlabel("Time (ns)")
    plt.ylabel("Projection")
    plt.title("Qubit Bloch Projections Over Time (projected 3-qubit subspace)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    (save_path / "bloch_projections.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / "bloch_projections.png")
    plt.show()

def plot_grouped_populations_full(states_full, time_axis_ns, groups: dict, save_path: Path, title="Grouped Populations"):
    pop = torch.stack([torch.abs(s)**2 for s in states_full]).numpy()  # (T, D)
    plt.figure(figsize=(10, 6))
    for label, idxs in groups.items():
        plt.plot(time_axis_ns, pop[:, idxs].sum(axis=1), label=label)
    plt.xlabel("Time (ns)")
    plt.ylabel("Population")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path / f"{title.replace(' ', '_').lower()}.png")
    plt.show()

def plot_population_transfers(get_U_RWA_3C, time_grid, drive, Δ_e, init_tgt_pairs, save_path: Path, filename="multi_state_populations.png"):
    """
    Reuses your earlier idea but against the 3C model and full Hilbert space.
    """
    U_fn = (lambda Ω, dt, t: get_U_RWA_3C(Ω, dt, t, Δ_e=Δ_e, ω_RF=ω1))
    # Dimension from active carbons:
    D = 2 * 3 * (2 ** len(get_active_carbons()))

    plt.figure(figsize=(8, 6))
    for init_idx, tgt_idx in init_tgt_pairs:
        ψ0 = torch.zeros(D, dtype=torch.complex128)
        ψ0[init_idx] = 1.0
        states = get_evolution_vector(U_fn, time_grid, drive, ψ0)
        pop = torch.stack([torch.abs(s)**2 for s in states]).numpy()
        # Build an "extended" time in ns (states length = len(time_grid)+1)
        dt = (time_grid[1] - time_grid[0]).item()
        time_ns = np.linspace(0, dt * len(time_grid), len(time_grid) + 1) * 1e9
        plt.plot(time_ns, pop[:, tgt_idx], label=f"{init_idx} → {tgt_idx}")

    plt.xlabel("Time (ns)")
    plt.ylabel("Target Population")
    plt.title("Multi-State Population Transfer (3C model)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path / filename)
    plt.show()
    
def plot_basis_state_transfers_sequence(time_grids, drives, Δ_e, init_tgt_pairs,
                                        save_path: Path, filename="state_transfer_fused.png"):
    """
    For each (init_idx -> tgt_idx), evolve from the init basis state through the WHOLE sequence
    and plot the population on tgt_idx vs the fused time axis.
    """
    # Hilbert-space dimension from active carbons
    D = 2 * 3 * (2 ** len(get_active_carbons()))

    plt.figure(figsize=(9, 6))

    for (init_idx, tgt_idx) in init_tgt_pairs:
        # 1) unit vector |ψ0> = |init_idx>
        ψ0 = torch.zeros(D, dtype=torch.complex128)
        ψ0[init_idx] = 1.0

        # 2) evolve across the full sequence using your 3C model
        states_chunks = apply_sequence_with_3C(Δ_e, time_grids, drives, ψ0)

        # 3) fuse to get a single trajectory and matching time axis
        states_concat, time_axis_ns = fuse_states_and_time(time_grids, states_chunks)  # [T, D], [T]

        # 4) population on the target basis state over time
        #    (abs^2 of the target component)
        pop_tgt = torch.abs(states_concat[:, tgt_idx])**2  # [T]

        # 5) plot
        plt.plot(time_axis_ns, pop_tgt.detach().cpu().numpy(),
                 label=f"{init_idx} → {tgt_idx}")

    plt.xlabel("Time (ns)")
    plt.ylabel("Target population")
    plt.title("Basis-state transfers over full sequence (3C RWA)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path / filename, dpi=200)
    plt.show()

# -----------------------------
# Main: point to one or more result folders
# -----------------------------

# 1) List result folders you want to analyze (can be one or many = sequence)
pulse_dirs = [
    # Example:
    #Path("results/pulse_2025-09-29_12-50-33_report_3Q"),  # <- replace with your actual result path(s)
    #Path("results/pulse_2025-10-27_10-02-06"), #0.95
    #Path("results/pulse_2025-10-27_10-19-30"), #0.96
    #Path("results/pulse_2025-10-27_10-26-36"), #0.97
    #Path("results/pulse_2025-10-27_17-49-40"),  #0.975
    #Path("results/pulse_2025-10-29_14-06-22"),  #0.975

    #Path("results/pulse_2025-11-05_03-35-20")
    #Path("results/pulse_2025-11-07_12-53-57") #exp(i pi/4 XZZ) 0.99 # report
    #Path("results/pulse_2025-11-11_15-40-22") #exp(i pi/4 XZZ) 0.99
    #Path("results/pulse_2025-11-12_12-28-32") #exp(i pi/4 XZZ) 0.99

    #Path("results/pulse_2025-11-14_05-57-40")

    #Path("results/pulse_2025-11-20_12-50-47"),  #neu ZZZ # 10 basis
    #Path("results/pulse_2025-11-20_15-01-58"),  #neu ZZZ
    #Path("results/pulse_2025-11-21_14-50-58"),  #neu ZZZ
    #Path("results/pulse_2025-11-24_14-11-22") # 1.75 mus

    #Path("results/pulse_2025-11-24_16-15-09") # 2 mus #16 basis
    #Path("results/pulse_2025-11-25_10-58-43") # 2 mus
    #Path("results/pulse_2025-11-25_14-55-13") # 2 mus good one, smooth
    #Path("results/pulse_2025-11-26_11-19-10") # 2 mus good one, smooth
    #Path("results/pulse_2025-11-27_10-06-29")

    #Path("results/pulse_2025-11-28_09-26-31") # 8 basis
    #Path("results/pulse_2025-11-28_20-47-51")
    #Path("results/pulse_2025-11-29_13-32-29")

    #Path("results/pulse_2025-12-01_13-05-12")

    #Path("results/pulse_2025-12-04_16-15-51")
    #Path("results/pulse_2025-12-05_10-20-14")

    #Path("results/pulse_2025-12-05_13-08-34")
    #Path("results/pulse_2025-12-05_22-11-44")

    #Path("results/pulse_2025-12-06_16-18-21")

    #Path("results/pulse_2025-12-08_09-35-53")

    #Path("results/pulse_2025-12-08_17-31-18")
    
    #Path("results/pulse_2025-12-08_19-43-30")

    #Path("results/pulse_2025-12-09_11-30-58") # 1200

    #Path("results/pulse_2025-12-10_11-22-43") # 1200


    #Path("results/pulse_2025-12-10_14-52-49") # gradient 8 basis
    #Path("results/pulse_2025-12-10_14-59-11") # gradient 8 basis
    #Path("results/pulse_2025-12-10_15-08-39") # gradient 8 basis
    #Path("results/pulse_2025-12-10_15-49-04") # gradient 8 basis
    #Path("results/pulse_2025-12-10_15-53-31") # gradient 10Basis
    #Path("results/pulse_2025-12-10_16-02-02") # gradient 11Basis 1400mus 6e-3

    #Path("results/pulse_2025-12-10_16-13-51") # gradient 10Basis 1450mus
    #Path("results/pulse_2025-12-10_16-19-35") # gradient 10Basis 1500mus

    #Path("results/pulse_2025-12-10_16-29-12") # gradient 10Basis 1600mus

    #Path("results/pulse_2025-12-10_16-29-12") # gradient 10Basis 1600mus #    "target_3q": {"c13": torch.pi* 2, "c12": torch.pi* 2, "c23": torch.pi* 2, "c123": torch.pi* 3/4},


    #Path("results/pulse_2025-12-10_23-12-31") # gradient 11Basis 1500mus Haan window


    #Path("results/pulse_2025-12-11_12-46-21")  # 2600mus

    #Path("results/pulse_2025-12-11_12-51-25")  # 1600

    #Path("results/pulse_2025-12-11_16-03-57")  # big two qubit phases

    #Path("results/pulse_2025-12-11_16-08-41")  # CCZ

    #Path("results/pulse_2025-12-11_16-24-14")  # 10MHz CCZ

    #Path("results/pulse_2025-12-12_10-27-49")  # 10MHz Ultrawiggle

    #Path("results/pulse_2025-12-12_11-22-49")  # 10MHz Ultrawiggle

    #Path("results/pulse_2025-12-12_14-21-00")  # 10MHz Ultrawiggle

    #Path("results/pulse_2025-12-12_14-51-47")  # 3000mus

    #Path("results/pulse_2025-12-12_14-57-33")  # 3000mus

    #Path("results/pulse_2025-12-12_15-22-40")  # make c12 and c13 stay close to avoid local trap where improvement comes from one being closer to zero


    #Path("results/pulse_2025-12-15_11-52-11")  # 


    #Path("results/pulse_2025-12-15_12-05-12") # gradient 11Basis 1400mus 6e-3

    #Path("results/pulse_2025-12-15_14-18-53") # report #interesting #goood [1,4] carbons going to 2pi and 3/4pi and shit

    #Path("results/pulse_2025-12-15_14-49-25") # best [1,4] carbons going to 0pi and pi/4 # 2e-3
    
    #Path("results/pulse_2025-12-15_15-10-44") # goood [1,4] carbons going to 0pi and pi/4

   # Path("results/pulse_2025-12-15_15-10-44") # goood [1,4] carbons going to 0pi and pi/4

    #Path("results/pulse_2025-12-15_15-20-42") # goood [1,4] carbons going to 0pi and pi/4

    #Path("results/pulse_2025-12-15_16-05-22") # goood [1,4] carbons going to 0pi and pi/4

    #Path("results/pulse_2025-12-15_19-32-55") # 8er basis goood [1,4] carbons going to 0pi and pi/4

    #Path("results/pulse_2025-12-15_19-46-34") # 11er basis goood [1,4] carbons going to 0pi and pi/4

    #Path("results/pulse_2025-12-15_19-50-40") # 11er basis goood [1,4] carbons going to 0pi and pi/4
    #Path("results/pulse_2025-12-15_19-57-14") # interesing # 1525 1mus 1er basis goood [1,4] carbons going to 0pi and pi/4
    #hier fast 0.999


    #Path("results/pulse_2025-12-15_20-19-41") # goood [1,4] carbons going to 2pi and 3/4pi and shit

    #Path("results/pulse_2025-12-15_20-22-04") # goood [1,4] carbons going to 2pi and 3/4pi and shit

    #Path("results/pulse_2025-12-15_20-40-27") # goood [1,4] carbons going to 2pi and 3/4pi and shit

    #Path("results/pulse_2025-12-15_20-49-48") # also very good [1,4] carbons going to 2pi and 3/4pi and shit

    #Path("results/pulse_2025-12-15_20-54-08") # goood [1,4] carbons going to 2pi and 3/4pi and shit


    #Path("results/pulse_2025-12-15_21-05-02") #1700mus goood [1,4] carbons going to 2pi and 3/4pi and shit



    #Path("results/pulse_2025-12-15_21-24-16") #1700mus goood [1,4] carbons going to 2pi and 3/4pi and shit
    
    #Path("results/pulse_2025-12-15_21-31-04") # interesting

    # XZZ grad
    #Path("results/pulse_2025-12-16_15-07-43") # 0.97 maybe report
    #Path("results/pulse_2025-12-17_14-40-54") #  0.98
    #Path("results/pulse_2025-12-17_14-36-34") #  0.96

    Path("results/pulse_2025-12-17_14-44-22") #  0.998



]

# Ensure the same active carbons setting as during optimization (the optimizer script set [1,2])
set_active_carbons([1, 4])
pc = get_precomp()
N_C = int(pc["N_C"])
D = 2 * 3 * (2 ** N_C)

nconf = nconf_from_pc(pc)


# 2) Load drives & time grids and common Δ_e
drives = []
time_grids = []
basis_indices = None
initial_target_pairs = None

# Take Δ_e and basis_indices from the first checkpoint (assume consistent across sequence)
first = load_ckpt(pulse_dirs[0])
Δ_e = first["Δ_e"]
basis_indices = first["basis_indices"]  # must exist for Bloch plots
initial_target_pairs = first["initial_target_pairs"]  # only present for Multi-State objective, else None


# Build initial→target pairs across BOTH 14N qubit levels and all carbon configs
def make_e_flip_pairs_both_N14(pc, n14_pair):
    """
    Returns a list of dicts with:
      - init_idx, tgt_idx: full-Hilbert space indices
      - mI_block: 14N block (0:+1, 1:0, 2:-1)
      - cb: carbon configuration integer 0..(2^N_C - 1)
    """
    pairs = []
    nconf = nconf_from_pc(pc)  # = 2**N_C
    for mI_block in n14_pair:
        for cb in range(nconf):
            pairs.append({
                "init_idx": basis_index(1, mI_block, cb, pc),  # |-1_e, mI, cb>
                "tgt_idx":  basis_index(0, mI_block, cb, pc),  # |0_e,  mI, cb>
                "mI_block": int(mI_block),
                "cb": int(cb),
            })
    return pairs

# Choose which two 14N levels form the qubit (you already do this later too)
#n14_pair = (0, 1)  # (0:+1, 1:0). If you change this elsewhere, keep it consistent.

# If the checkpoint didn't save pairs, generate them (BOTH N14 levels, ALL carbons)
#if initial_target_pairs is None:
#    pairs_dicts = make_e_flip_pairs_both_N14(pc, n14_pair)
#    # keep a simple (init_idx, tgt_idx) list for backward compat where needed:
#    initial_target_pairs = [(d["init_idx"], d["tgt_idx"]) for d in pairs_dicts]
#else:
#    # If it did save pairs, we can still reconstruct metadata for labeling
#    # Try to infer mI, cb from indices if you want; otherwise skip labeling nicety.
#    pairs_dicts = [{"init_idx": i, "tgt_idx": j, "mI_block": None, "cb": None}
#                   for (i, j) in initial_target_pairs]

# If the optimizer didn't save pairs, build them on the fly:
#if initial_target_pairs is None:
#    # Use the same 14N block you treat as logical |0> later.
#    # You set n14_pair = (0, 1) below, so take mI=0 for the transfer pairs:
#    mI_block = 0  # or: mI_block = int(n14_pair[0]) if you’d rather tie it to that choice

#    # Create up to 4 pairs: |-1_e, mI, cb>  ->  |0_e, mI, cb>
#    initial_target_pairs = [
#        (basis_index(1, mI_block, cb, pc), basis_index(0, mI_block, cb, pc))
#        for cb in range(min(8, nconf))
#    ]

for p in pulse_dirs:
    info = load_ckpt(p)
    drives.append(info["drive"])
    time_grids.append(info["time_grid"])

# 3) Build time axis for concatenated sequence
dts = [(tg[1] - tg[0]).item() for tg in time_grids]
steps_per = [len(tg) for tg in time_grids]  # states length per pulse = steps + 1
total_steps = sum(s + 1 for s in steps_per)
# Build concatenated time in seconds (then convert to ns)
seg_times = []
t_acc = 0.0
for dt, steps in zip(dts, steps_per):
    seg_times.append(t_acc + np.linspace(0, dt * steps, steps + 1))
    t_acc += dt * steps
time_axis_ns = np.concatenate(seg_times) * 1e9

# 4) Initial state: |000> in the computational subspace corresponds to idx = basis_indices[0]
ψ0 = torch.zeros(D, dtype=torch.complex128)
# Be safe: if basis_indices provided, use its first element as |000>
if basis_indices is None:
    raise ValueError("basis_indices is required to define |000⟩ consistently. Not found in checkpoint.")
ψ0[basis_indices[0]] = 1.0

# 5) Evolve through sequence
states_chunks = apply_sequence_with_3C(Δ_e, time_grids, drives, ψ0)
#states_concat = torch.cat(states_chunks, dim=0)  # (total_T, D)

def fuse_states_and_time(time_grids, states_chunks):
    """
    Make states_concat and time_axis_ns that always match in length.
    - For the first chunk: keep all samples (L0)
    - For later chunks: drop the first sample (boundary), keep (Li-1)
    """
    if not states_chunks:
        raise ValueError("states_chunks is empty")

    cat_states = []
    times_s = []          # seconds
    t_acc = 0.0

    for i, (tg, chunk) in enumerate(zip(time_grids, states_chunks)):
        if chunk.ndim != 2:
            raise ValueError("Each states chunk must be Tensor [T_i, D].")
        if len(tg) < 2:
            raise ValueError("Each time_grid must have >= 2 points.")
        dt = float((tg[1] - tg[0]).item())
        L  = int(chunk.shape[0])               # samples in this chunk (usually steps+1)

        if i == 0:
            # keep all L samples; times: 0, dt, ..., (L-1)dt
            cat_states.append(chunk)           # [L, D]
            times_s.extend(t_acc + np.arange(L) * dt)
            t_acc += (L - 1) * dt
        else:
            # drop the boundary: keep samples 1..(L-1)
            if L < 2:
                raise ValueError("Chunk too short (needs at least 2 samples).")
            cat_states.append(chunk[1:])       # [(L-1), D]
            times_s.extend(t_acc + np.arange(1, L) * dt)  # dt..(L-1)dt
            t_acc += (L - 1) * dt

    states_concat = torch.cat(cat_states, dim=0)         # [sum(L0 + Σ(Li-1)), D]
    time_axis_ns  = np.asarray(times_s) * 1e9            # ns
    return states_concat, time_axis_ns

states_concat, time_axis_ns = fuse_states_and_time(time_grids, states_chunks)

# (optional sanity)
assert states_concat.shape[0] == len(time_axis_ns)

# 6) Output folder (use last pulse folder to save figures)
output_dir = pulse_dirs[-1]
print(f"Saving analysis to: {output_dir}")





# === Bloch projections including 14N and ALL active carbons ===
n14_pair = (0, 1)           # choose two 14N levels: (0:+1, 1:0) by default
electron_map = ('m1', '0')  # logical 0 -> |-1_e>, logical 1 -> |0_e>

basis_indices_multi, qubit_names = make_multi_qubit_basis_indices(
    pc, n14_pair=n14_pair, electron_map=electron_map
)
Pn = projector_from_indices_general(basis_indices_multi, D)
n_qubits = len(qubit_names)  # = 2 + N_C



## =========================
# === Keep only 4 plots ===
# =========================

# --- helper: plot drives across the full sequence ---
def plot_drives(drives, time_grids, outdir: Path, title="MW Pulse"):
    pieces_t = []
    pieces_y = []
    t_acc = 0.0
    for tg, drv in zip(time_grids, drives):
        dt = float((tg[1]-tg[0]).item())
        t = t_acc + np.arange(len(tg)) * dt  # len = steps
        # for plotting, align drive samples to grid points (same convention as saved)
        for ch, d in enumerate(drv):
            pieces_t.append((ch, t.copy()))
            pieces_y.append((ch, d.detach().cpu().numpy().copy()))
        t_acc += dt * len(tg)

    plt.figure(figsize=(10, 4))
    nchan = len(drives[0])
    for ch in range(nchan):
        # stitch channel ch
        t_ch = np.concatenate([t for c,t in pieces_t if c==ch])
        y_ch = np.concatenate([y for c,y in pieces_y if c==ch])
        plt.plot(t_ch*1e9, y_ch, label=f"Drive {ch+1}")
    plt.xlabel("Time [ns]")
    plt.ylabel("Amplitude [μT]")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "drives_concatenated.png", dpi=200)
    plt.show()

plot_drives(drives, time_grids, output_dir)

# --- (1) State-transfer plot (if pairs are in the checkpoint) ---

if initial_target_pairs is not None and len(initial_target_pairs) > 0:
    plot_basis_state_transfers_sequence(
        time_grids=time_grids,
        drives=drives,
        Δ_e=Δ_e,
        init_tgt_pairs=initial_target_pairs,
        save_path=output_dir,
        filename="state_transfer_fused.png",
    )
else:
    print("No 'initial_target_pairs' available; skipping basis-state transfer plot.")

# --- (2) Build the all-qubits projector (includes e⁻, 14N, and all active carbons) ---
n14_pair = (0, 1)           # use mI=+1 as |0>, mI=0 as |1>
electron_map = ( 'm1','0',)  # logical 1→|-1_e>, logical 0→|0_e>
basis_indices_multi, qubit_names = make_multi_qubit_basis_indices(pc, n14_pair, electron_map)
Pn = projector_from_indices_general(basis_indices_multi, D)

# Choose which two carbons you care about
# (use labels if you prefer: idx_c1 = qubit_names.index("C1"), etc.)
idx_e   = qubit_names.index("e⁻")
idx_N   = qubit_names.index("14N")
idx_c1, idx_c2 = 2, 3    # first two carbons in qubit_names list

# --- (3) Get the final state and slice to the 3-qubit subspace with N14=0 and others=0 ---
psi_full_T = states_concat[-1]
psi_all    = Pn @ psi_full_T               # (2^n,)
# normalize inside the all-qubits comp. subspace (ignores leakage)
w_sub      = float((psi_all.conj() @ psi_all).real.item())
if w_sub > 0:
    psi_all = psi_all / np.sqrt(w_sub)

nq = len(qubit_names)
def idx_from_bits(bits):
    return sum((bits[q] << q) for q in range(nq))

# pick amplitudes where N14=0 and every carbon except c1,c2 is 0
sel_indices = []
for a in (0,1):      # electron
    for b in (0,1):  # c1
        for c in (0,1):  # c2
            bits = [0]*nq
            bits[idx_e]  = a
            bits[idx_N]  = 0          # fix 14N to |0>
            bits[idx_c1] = b
            bits[idx_c2] = c
            # pin ALL other carbons to |0>
            for q in range(2, nq):
                if q not in (idx_c1, idx_c2):
                    bits[q] = 0
            sel_indices.append(idx_from_bits(bits))

psi_3q = psi_all[sel_indices].clone()      # (8,)
# renormalize the slice
norm = torch.linalg.norm(psi_3q)
psi_3q = psi_3q / norm if float(norm) > 0 else psi_3q


##


# --- (4) Bloch spheres for the three-qubit state (electron, C1, C2) ---
# pip install qiskit (if needed)
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.visualization import plot_bloch_multivector, plot_state_city


# Initialize: e=|+>, N14=|0>, C1=|+>, C2=|+>, all other carbons |0>
import math

nq = len(qubit_names)

def idx_from_bits(bits):
    # little-endian: bit q corresponds to qubit_names[q]
    return sum((bits[q] << q) for q in range(nq))

# Build the state in the projected (all-qubits) computational subspace
psi_all0 = torch.zeros(2**nq, dtype=torch.complex128)
amp = 1 / math.sqrt(8)  # (1/sqrt(2))^3 for e, C1, C2
for a in (0, 1):        # electron
    for b in (0, 1):    # C1
        for c in (0, 1):  # C2
            bits = [0]*nq
            bits[idx_e]  = 0          # e = superposition # set zero to have e in 0 # orthogonal
            bits[idx_N]  = 0          # N14 fixed to |0>
            bits[idx_c1] = b          # C1 = superposition
            bits[idx_c2] = c          # C2 = superposition
            # all other carbons = |0>
            for q in range(2, nq):
                if q not in (idx_c1, idx_c2):
                    bits[q] = 0
            psi_all0[idx_from_bits(bits)] = amp

# Embed to FULL Hilbert space to start evolution
psi_full0 = Pn.conj().T @ psi_all0


# Evolve
states_chunks = apply_sequence_with_3C(Δ_e, time_grids, drives, psi_full0)
states_concat, time_axis_ns = fuse_states_and_time(time_grids, states_chunks)

# Project final state back to the all-qubits computational subspace (for slicing/plots)
psi_all_T = Pn @ states_concat[-1]
w_sub = float((psi_all_T.conj() @ psi_all_T).real)
psi_all_T = psi_all_T / math.sqrt(w_sub) if w_sub > 0 else psi_all_T


# Slice to (e, C1, C2) with N14=0 and all other carbons=0
sel_indices = []
for a in (0,1):
    for b in (0,1):
        for c in (0,1):
            bits = [0]*nq
            bits[idx_e]  = a
            bits[idx_N]  = 0
            bits[idx_c1] = b
            bits[idx_c2] = c
            for q in range(2, nq):
                if q not in (idx_c1, idx_c2):
                    bits[q] = 0
            sel_indices.append(idx_from_bits(bits))

psi_3q = psi_all_T[sel_indices].clone()
psi_3q = psi_3q / torch.linalg.norm(psi_3q)

from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.visualization import plot_bloch_multivector, plot_state_city

sv_3q = Statevector(psi_3q.detach().cpu().numpy())
#plot_bloch_multivector(sv_3q, title="Bloch spheres (e, C1, C2) with N14=|0>, others=|+>").savefig(output_dir / "bloch_spheres_3q.png", dpi=200)
#plt.show()
#plot_state_city(DensityMatrix(sv_3q), title="City (e, C1, C2) with N14=|0>, others=|0>").savefig(output_dir / "city_3q.png", dpi=200)
#plt.show()


# Keep (e, N14, C1, C2); pin any additional carbons to |0⟩
sel4 = []
for a in (0,1):
    for n in (0,1):
        for b in (0,1):
            for c in (0,1):
                bits = [0]*nq
                bits[idx_e], bits[idx_N], bits[idx_c1], bits[idx_c2] = a, n, b, c
                for q in range(2, nq):
                    if q not in (idx_c1, idx_c2):
                        bits[q] = 0
                sel4.append(idx_from_bits(bits))

psi_4q = psi_all_T[sel4].clone()
psi_4q = psi_4q / torch.linalg.norm(psi_4q)

sv_4q = Statevector(psi_4q.detach().cpu().numpy())

plt.show()

import re
#fig = plot_bloch_multivector(sv_4q, title="Bloch spheres (e, N14, C1, C2)")

#for ax, lbl in zip(fig.axes[:4], [qubit_names[idx_c2], qubit_names[idx_c1], qubit_names[idx_N], qubit_names[idx_e]]):
#    ax.set_title("")  # clear default
#    for txt in list(ax.texts):
#        if re.match(r"\s*qubit\s*\d+", txt.get_text(), flags=re.IGNORECASE):
#            txt.set_visible(False)
#    ax.set_title(lbl, pad=22)  # push the title further above the axes

# add a bit of top margin in case tight_layout squashes things
#fig.subplots_adjust(top=0.88)
#fig.savefig(output_dir / "bloch_spheres_4q.png", dpi=200)
#plt.show()

#plot_state_city(DensityMatrix(sv_4q), title="City (e, N14, C1, C2)").savefig(output_dir / "city_4q.png", dpi=200)
#plt.show()





#####

# ===== 4-Bloch-sphere trajectories with gradient (NV-e, N14, C1, C2) =====
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import cm, colors

# Pauli matrices (torch)
σx = torch.tensor([[0, 1],[1, 0]], dtype=torch.complex128)
σy = torch.tensor([[0,-1j],[1j, 0]], dtype=torch.complex128)
σz = torch.tensor([[1, 0],[0,-1]], dtype=torch.complex128)

def _idx_from_bits(bits):
    return sum((bits[q] << q) for q in range(len(bits)))

def slice_4q_in_order(psi_all, qubit_names, idx_e, idx_N, idx_c1, idx_c2):
    """
    Build a 4-qubit statevector ordered as [e, N14, C1, C2] from the
    all-qubits projected state 'psi_all'. All other carbons are pinned to |0>.
    """
    nq = len(qubit_names)
    psi_4q = torch.zeros(16, dtype=torch.complex128)
    for j in range(16):
        e_bit  = (j >> 0) & 1  # q0
        N_bit  = (j >> 1) & 1  # q1
        C1_bit = (j >> 2) & 1  # q2
        C2_bit = (j >> 3) & 1  # q3

        bits = [0]*nq
        bits[idx_e]  = e_bit
        bits[idx_N]  = N_bit
        bits[idx_c1] = C1_bit
        bits[idx_c2] = C2_bit
        for q in range(nq):
            if q not in (idx_e, idx_N, idx_c1, idx_c2):
                bits[q] = 0  # pin extra carbons to |0>

        psi_4q[j] = psi_all[_idx_from_bits(bits)]

    nrm = torch.linalg.norm(psi_4q)
    if float(nrm) > 0:
        psi_4q = psi_4q / nrm
    return psi_4q

def compute_bloch_traj_4q(states_full_over_time, time_axis_ns,
                          Pn, qubit_names, idx_e, idx_N, idx_c1, idx_c2,
                          max_points=500):
    """
    Returns Bloch vectors over time for the 4-qubit slice [e, N14, C1, C2].
    Output: traj = {0: Nx3, 1: Nx3, 2: Nx3, 3: Nx3}, times_sel (N,)
    """
    T = states_full_over_time.shape[0]
    stride = max(1, T // max_points)
    idxs = np.arange(0, T, stride, dtype=int)
    if idxs[-1] != T-1:
        idxs = np.append(idxs, T-1)

    traj = {0: [], 1: [], 2: [], 3: []}
    times_sel = time_axis_ns[idxs]

    for t in idxs:
        # project to all-qubits computational subspace and renormalize inside it
        psi_all = Pn @ states_full_over_time[t]
        w = float((psi_all.conj() @ psi_all).real)
        if w <= 1e-14:
            continue
        psi_all = psi_all / np.sqrt(w)

        # 4-qubit slice [e, N14, C1, C2]
        psi_4q = slice_4q_in_order(psi_all, qubit_names, idx_e, idx_N, idx_c1, idx_c2)
        rho_4q = psi_4q[:, None] @ psi_4q[None, :].conj()  # (16,16)

        # single-qubit reductions and Bloch coords
        for q in range(4):
            rho_q = partial_trace(rho_4q, [2,2,2,2], keep=(q,))  # (2,2)
            x = torch.real(torch.trace(σx @ rho_q)).item()
            y = torch.real(torch.trace(σy @ rho_q)).item()
            z = torch.real(torch.trace(σz @ rho_q)).item()
            traj[q].append((x, y, z))

    # convert to arrays
    traj = {q: np.asarray(traj[q]) for q in traj}
    return traj, times_sel

import matplotlib as mpl
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def plot_bloch_trajectories_4q_like_ref(traj, panel_names, out_path: Path,
                                        filename="bloch_trajectories_4q_fade.png",
                                        cmap_name="viridis"):
    """
    1×4 Bloch spheres, reference style + great-circle grid:
      - light sphere (alpha=0.1)
      - XYZ quivers with arrow_length_ratio=0.1
      - equator (z=0) and the two orthogonal 'equators' (x=0, y=0)
      - trajectory as gradient line segments (no points)
      - alpha fades from light→dark along the path
      - ticks at -1,0,1 (keep/remove as you like)
    """
    fig = plt.figure(figsize=(10, 10), dpi=200)
    axes = [fig.add_subplot(2, 2, i+1, projection='3d') for i in reversed(range(4))]

    cmap = mpl.colormaps.get_cmap(cmap_name)

    for i, ax in enumerate(axes):
        coords = traj[i]  # (T, 3)
        ax.set_title(f"{panel_names[i]} Bloch Trajectory")
        ax.set_facecolor('none')
        ax.grid(False)

        # limits + ticks (optional: comment out to hide)
        ax.set_xlim([-1.1, 1.1]); ax.set_ylim([-1.1, 1.1]); ax.set_zlim([-1.1, 1.1])
        ax.set_xlabel("⟨X⟩"); ax.set_ylabel("⟨Y⟩"); ax.set_zlabel("⟨Z⟩")
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_zticks([-1, 0, 1])
        ax.tick_params(labelsize=8, pad=0)
        try: ax.set_box_aspect([1,1,1])
        except Exception: pass

        # Bloch sphere surface
        u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
        xs = np.cos(u) * np.sin(v)
        ys = np.sin(u) * np.sin(v)
        zs = np.cos(v)
        ax.plot_surface(xs, ys, zs, color='lightblue', alpha=0.1, linewidth=0)

        # --- great-circle grid (equator + two orthogonal meridians) ---
        phi = np.linspace(0, 2*np.pi, 361)
        # equator (z = 0)
        ax.plot(np.cos(phi), np.sin(phi), 0*phi, color='#5D6D7E', lw=0.9, alpha=0.85)
        # x = 0 (circle in YZ plane)
        ax.plot(0*phi, np.cos(phi), np.sin(phi), color='#5D6D7E', lw=0.8, alpha=0.65)
        # y = 0 (circle in XZ plane)
        ax.plot(np.cos(phi), 0*phi, np.sin(phi), color='#5D6D7E', lw=0.8, alpha=0.65)

        # Coordinate axes
        ax.quiver(0, 0, 0, 1, 0, 0, color='r', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', arrow_length_ratio=0.1)

        # basis labels
        ax.text(0, 0,  1.05, r'$|0\rangle$', ha='center', va='bottom', fontsize=10)
        ax.text(0, 0, -1.05, r'$|1\rangle$', ha='center', va='top',    fontsize=10)
        ax.text(1.05, 0, 0,  r'$|+\rangle$', ha='left',   va='center', fontsize=10)

        if coords.size >= 2:
            # gradient line segments (index-based)
            points = coords.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            norm = colors.Normalize(vmin=0, vmax=len(segments))
            color_vals = cmap(norm(np.arange(len(segments))))
            color_vals[:, 3] = np.linspace(0.1, 1.0, len(segments))  # alpha fade

            lc = Line3DCollection(segments, colors=color_vals, linewidth=2)
            ax.add_collection3d(lc)

            # final state marker
            ax.scatter(*coords[-1], color='black', s=50)

    # a touch more room to avoid overlap
    fig.subplots_adjust(wspace=0.1, top=0.92, bottom=0.12, left=0.04, right=0.98)
    out_file = out_path / filename
    fig.savefig(out_file, dpi=200)
    plt.show()
    print(f"Saved: {out_file}")


# ---- Compute & plot (uses your existing states_concat, time_axis_ns) ----
traj4, t_sel = compute_bloch_traj_4q(
    states_concat, time_axis_ns,
    Pn, qubit_names, idx_e, idx_N, idx_c1, idx_c2,
    max_points=600  # adjust for density
)


panel_names = [qubit_names[idx_c2],qubit_names[idx_c1],qubit_names[idx_N],qubit_names[idx_e]]
plot_bloch_trajectories_4q_like_ref(traj4, panel_names, output_dir,
                                    filename="bloch_trajectories_4q_fade.png",
                                    cmap_name="plasma")  # or "viridis"


# ========================================================================




# e = |+>, N14 = |+>, C1 = |+>, C2 = |+>, all other carbons |0>
import math
nq = len(qubit_names)

def idx_from_bits(bits):
    return sum((bits[q] << q) for q in range(nq))

psi_all0 = torch.zeros(2**nq, dtype=torch.complex128)
amp = 1 / math.sqrt(16)  # (1/√2)^4 for e, N14, C1, C2

for a in (0, 1):      # e
    for n in (0, 1):  # N14  <-- NOW in superposition
        for b in (0, 1):  # C1
            for c in (0, 1):  # C2
                bits = [0]*nq
                bits[idx_e]  = a
                bits[idx_N]  = n
                bits[idx_c1] = b
                bits[idx_c2] = c
                # all other carbons pinned to |0>
                for q in range(2, nq):
                    if q not in (idx_c1, idx_c2):
                        bits[q] = 0
                psi_all0[idx_from_bits(bits)] = amp

# Embed to FULL Hilbert space and evolve
psi_full0 = Pn.conj().T @ psi_all0
#states_chunks = apply_sequence_with_3C(Δ_e, time_grids, drives, psi_full0)
#states_concat, time_axis_ns = fuse_states_and_time(time_grids, states_chunks)

#compute trajectories
traj4, t_sel = compute_bloch_traj_4q(
    states_concat, time_axis_ns,
    Pn, qubit_names, idx_e, idx_N, idx_c1, idx_c2,
    max_points=600  # adjust for density
)


import matplotlib as mpl
from matplotlib import colors
from matplotlib.collections import LineCollection

def plot_bloch_xy_4q_like_ref(traj, panel_names, out_path: Path,
                              filename="bloch_xy_4q_fade.png",
                              cmap_name="viridis"):
    """
    1×4 XY-plane (⟨X⟩,⟨Y⟩) trajectories:
      - unit circle + cross-axes
      - gradient line segments (no points)
      - alpha fades light→dark along the path
      - tick marks at -1, 0, 1
      - |0>, |1>, |+>, |-> labels on the circle
    traj: dict {0..3 -> (T,3)} for [e, N14, C1, C2]
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    cmap = mpl.colormaps.get_cmap(cmap_name)

    for i, ax in enumerate(axes):
        coords = traj[i]  # (T,3)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel("⟨X⟩"); ax.set_ylabel("⟨Y⟩")
        ax.set_title(f"{panel_names[i]} (XY)")
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
        ax.set_facecolor('none')

        # unit circle & axes
        phi = np.linspace(0, 2*np.pi, 361)
        ax.plot(np.cos(phi), np.sin(phi), lw=1.0, color='#5D6D7E', alpha=0.85)
        ax.axhline(0, lw=0.8, color='#5D6D7E', alpha=0.6)
        ax.axvline(0, lw=0.8, color='#5D6D7E', alpha=0.6)

        # basis labels on the circle
        #ax.text( 1.05,  0.00, r'$|+\rangle$', ha='left',  va='center', fontsize=9)
        #ax.text(-1.05,  0.00, r'$|-\rangle$', ha='right', va='center', fontsize=9)
        #ax.text( 0.00,  1.05, r'$|0\rangle$', ha='center', va='bottom', fontsize=9)
        #ax.text( 0.00, -1.05, r'$|1\rangle$', ha='center', va='top',    fontsize=9)

        if coords.size >= 2:
            pts = coords[:, :2]  # (T,2)
            P = pts.reshape(-1, 1, 2)
            segs = np.concatenate([P[:-1], P[1:]], axis=1)  # (T-1, 2, 2)

            # segment colors with alpha fade
            norm = colors.Normalize(vmin=0, vmax=len(segs))
            cols = cmap(norm(np.arange(len(segs))))
            cols[:, 3] = np.linspace(0.12, 1.0, len(segs))

            lc = LineCollection(segs, colors=cols, linewidths=2.0)
            ax.add_collection(lc)

            # final point
            ax.scatter(pts[-1,0], pts[-1,1], s=40, color='black', zorder=3)

    fig.subplots_adjust(wspace=0.35, top=0.88, bottom=0.20, left=0.06, right=0.98)
    out_file = out_path / filename
    fig.savefig(out_file, dpi=200)
    plt.show()
    print(f"Saved: {out_file}")

panel_names = [qubit_names[idx_c2],qubit_names[idx_c1],qubit_names[idx_N],qubit_names[idx_e]]
plot_bloch_xy_4q_like_ref(traj4, panel_names, output_dir, filename="bloch_xy_4q_fade.png", cmap_name="plasma")




# ========= GHZ fidelity (Local-Z correction from diagonal phases) =========

def build_full_ket_for_abc(a, b, c, qubit_names, idx_e, idx_N, idx_c1, idx_c2, Pn):
    """Build FULL-Hilbert ket corresponding to |a, N14=0, b, c, others=0> in the projected basis, then embed back."""
    nq = len(qubit_names)
    vec_proj = torch.zeros(2**nq, dtype=torch.complex128)
    bits = [0]*nq
    bits[idx_e]  = int(a)
    bits[idx_N]  = 0               # fix N14 = |0>
    bits[idx_c1] = int(b)
    bits[idx_c2] = int(c)
    # pin any additional carbons to |0>
    for q in range(nq):
        if q not in (idx_e, idx_N, idx_c1, idx_c2):
            bits[q] = 0
    # little-endian index
    idx = sum((bits[q] << q) for q in range(nq))
    vec_proj[idx] = 1.0
    # embed to FULL space
    return Pn.conj().T @ vec_proj

def slice_3q_from_proj(vec_proj, qubit_names, idx_e, idx_c1, idx_c2, idx_N):
    """Extract amplitudes for (e, C1, C2) with N14=0 and all other carbons=0, ordered as |a b c>."""
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

def evolve_basis_and_get_phi(time_grids, drives, Δ_e, a,b,c, get_full=True):
    """Evolve |a b c> (with N14=0, others=0), project to all-qubits subspace and return 3q slice and phase element."""
    psi_full0 = build_full_ket_for_abc(a,b,c, qubit_names, idx_e, idx_N, idx_c1, idx_c2, Pn)
    chunks = apply_sequence_with_3C(Δ_e, time_grids, drives, psi_full0)
    states_concat, _ = fuse_states_and_time(time_grids, chunks)
    psi_all_T = Pn @ states_concat[-1]
    # renorm inside projected subspace
    w = float((psi_all_T.conj() @ psi_all_T).real)
    psi_all_T = psi_all_T / math.sqrt(w) if w > 0 else psi_all_T
    psi_3q_T  = slice_3q_from_proj(psi_all_T, qubit_names, idx_e, idx_c1, idx_c2, idx_N)
    # diagonal element amplitude ≈ <abc|U|abc>
    idx = (a<<2) | (b<<1) | (c<<0)   # order [e, C1, C2]
    amp = psi_3q_T[idx]
    phi = torch.angle(amp).item()
    return phi, psi_3q_T if get_full else phi

# --- 1) Reconstruct all diagonal phases φ_{abc} ---
phi = {}
psi_cols = {}  # optional: keep final 3q columns
for a in (0,1):
    for b in (0,1):
        for c in (0,1):
            ph, col = evolve_basis_and_get_phi(time_grids, drives, Δ_e, a,b,c, get_full=True)
            phi[(a,b,c)] = ph
            psi_cols[(a,b,c)] = col

# --- 2) Finite-difference coefficients (three locals, three pairs, one 3-body) ---
def sum_over_abc(weight_fn):
    return sum(weight_fn(a,b,c) * phi[(a,b,c)] for a in (0,1) for b in (0,1) for c in (0,1))

α =  (1/8.0) * sum_over_abc(lambda a,b,c: (-1)**a)
β =  (1/8.0) * sum_over_abc(lambda a,b,c: (-1)**b)
χ =  (1/8.0) * sum_over_abc(lambda a,b,c: (-1)**c)
γ_ab = (1/8.0) * sum_over_abc(lambda a,b,c: (-1)**(a+b))
γ_ac = (1/8.0) * sum_over_abc(lambda a,b,c: (-1)**(a+c))
γ_bc = (1/8.0) * sum_over_abc(lambda a,b,c: (-1)**(b+c))
λ = -(1/8.0) * sum_over_abc(lambda a,b,c: (-1)**(a+b+c))

print(f"Local phases:  α={α:.6f}, β={β:.6f}, χ={χ:.6f}  (rad)")
print(f"Pairwise:      γ_ab={γ_ab:.6f}, γ_ac={γ_ac:.6f}, γ_bc={γ_bc:.6f}  (rad)")
print(f"Three-body:    λ={λ:.6f}  (rad)")


phases_loc_dag = torch.zeros(8, dtype=torch.complex128)
for a in (0,1):
    for b in (0,1):
        for c in (0,1):
            idx = (a<<2) | (b<<1) | c
            # z is a real scalar phase in radians
            z = ((-1)**a)*α + ((-1)**b)*β + ((-1)**c)*χ
            z_t = torch.tensor(z, dtype=torch.complex128)
            phases_loc_dag[idx] = torch.exp(-1j * z_t)   
Uloc_dag = torch.diag(phases_loc_dag)

# --- 4) Start from |000>, apply H^⊗3 (→ |+++>), evolve pulse, remove local Z phases,
#         apply H^⊗3, fix GHZ phase, compare (with debug prints) ---

import math
import torch

# Single-qubit gates
H = (1/torch.sqrt(torch.tensor(2.0))) * torch.tensor([[1, 1],
                                                      [1,-1]], dtype=torch.complex128)
I2 = torch.eye(2, dtype=torch.complex128)

def kron3(a,b,c): 
    return torch.kron(torch.kron(a,b), c)

def amp(v, idx):
    """Return amplitude, magnitude, phase (rad) of v[idx]."""
    z = v[idx]
    return z, float(torch.abs(z)), float(torch.angle(z))

def pretty_amp(tag, v, idx):
    z, mag, ph = amp(v, idx)
    print(f"  {tag:<14} amp={z.real:+.6f}{z.imag:+.6f}i |amp|={mag:.6f} phase={ph:.6f} rad")

def check_norm(tag, v, atol=5e-10):
    n = float(torch.linalg.norm(v))
    print(f"[check] ‖{tag}‖ = {n:.12f}")
    if abs(n - 1.0) > atol:
        print(f"  WARNING: {tag} not normalized (Δ={n-1.0:+.3e}). Normalizing now.")
        v = v / torch.linalg.norm(v)
    return v

def subspace_weight(psi_all):
    """Compute weight inside projected computational subspace."""
    return float((psi_all.conj() @ psi_all).real)

def build_full_000(Pn, qubit_names, idx_e, idx_N, idx_c1, idx_c2):
    nq = len(qubit_names)
    vec_proj = torch.zeros(2**nq, dtype=torch.complex128)
    bits = [0]*nq
    bits[idx_e] = 0; bits[idx_N] = 0; bits[idx_c1] = 0; bits[idx_c2] = 0
    idx = sum((bits[q] << q) for q in range(nq))
    vec_proj[idx] = 1.0
    return Pn.conj().T @ vec_proj  # embed to FULL Hilbert space

def build_full_plusplusplus(Pn, qubit_names, idx_e, idx_N, idx_c1, idx_c2):
    nq = len(qubit_names)
    vec_proj = torch.zeros(2**nq, dtype=torch.complex128)
    amp0 = 1 / math.sqrt(8)  # (1/√2)^3 for e,C1,C2; N14 fixed to |0>, others |0>
    for a in (0,1):
        for b in (0,1):
            for c in (0,1):
                bits = [0]*nq
                bits[idx_e], bits[idx_N], bits[idx_c1], bits[idx_c2] = a, 0, b, c
                for q in range(nq):
                    if q not in (idx_e, idx_N, idx_c1, idx_c2):
                        bits[q] = 0
                idx = sum((bits[q] << q) for q in range(nq))
                vec_proj[idx] = amp0
    return Pn.conj().T @ vec_proj  # embed to FULL Hilbert space

print("\n=== GHZ pipeline (debug) ===")

# 1) |000> and H^{⊗3} (we will just start from |+++> exactly)
psi_full0 = build_full_000(Pn, qubit_names, idx_e, idx_N, idx_c1, idx_c2)
psi_full0 = check_norm("|000>_full", psi_full0)

psi_full0_ppp = build_full_plusplusplus(Pn, qubit_names, idx_e, idx_N, idx_c1, idx_c2)
psi_full0_ppp = check_norm("|+++>_full (embedded)", psi_full0_ppp)

# 2) Evolve under the pulse
chunks = apply_sequence_with_3C(Δ_e, time_grids, drives, psi_full0_ppp)
states_concat, _ = fuse_states_and_time(time_grids, chunks)

# 3) Project to all-qubits computational subspace, renormalize there
psi_all_T = Pn @ states_concat[-1]
w_comp = subspace_weight(psi_all_T)
print(f"[post-pulse] weight inside computational subspace = {w_comp:.8f}")
psi_all_T = psi_all_T / math.sqrt(w_comp) if w_comp > 0 else psi_all_T

# Slice to 3q (A=e, B=C1, C=C2) with N14=0 (others pinned to |0>)
psi3 = slice_3q_from_proj(psi_all_T, qubit_names, idx_e, idx_c1, idx_c2, idx_N)  # (8,)
psi3 = check_norm("psi3 (post-pulse, 3q slice)", psi3)

# Report some key amplitudes BEFORE any corrections
print("Amplitudes (3q slice) BEFORE local-Z removal:")
pretty_amp("|000>", psi3, 0)
pretty_amp("|111>", psi3, 7)

# 4) Remove local Z phases (U_loc†)
psi3_corr = Uloc_dag @ psi3
psi3_corr = check_norm("psi3_corr = U_loc^† psi3", psi3_corr)

print("Amplitudes AFTER local-Z removal:")
pretty_amp("|000>", psi3_corr, 0)
pretty_amp("|111>", psi3_corr, 7)

# 5) Apply H^{⊗3}
H3 = kron3(H, H, H)
psi3_H = H3 @ psi3_corr
psi3_H = check_norm("psi3_H = H^{⊗3} psi3_corr", psi3_H)

print("Amplitudes AFTER H^{⊗3}:")
pretty_amp("|000>", psi3_H, 0)
pretty_amp("|111>", psi3_H, 7)

# Optional: “best-phase” GHZ fidelity at this stage
ket000 = torch.zeros(8, dtype=torch.complex128); ket000[0] = 1.0
ket111 = torch.zeros(8, dtype=torch.complex128); ket111[7] = 1.0
a0, a7 = psi3_H[0], psi3_H[7]
phi_best = (torch.angle(a7) - torch.angle(a0)).item()
GHZ_best = (ket000 + torch.exp(1j*torch.as_tensor(phi_best)) * ket111) / math.sqrt(2)
F_best_pre_fix = torch.abs(torch.vdot(GHZ_best, psi3_H))**2
print(f"[pre-fix] best-phase GHZ fidelity = {F_best_pre_fix.item():.6f} (φ* = {phi_best:.6f} rad)")

# 6) Fix residual relative phase between |000> and |111> with a single-qubit Rz on A
#    Rz_A(θ) with θ = -2*phi shifts |1> by e^{-i phi} relative to |0> on qubit A only.
theta = torch.as_tensor(-2.0 * phi_best, dtype=torch.float64)
rz_A = torch.diag(torch.tensor([torch.exp(-0.5j*theta), torch.exp(+0.5j*theta)],
                               dtype=torch.complex128, device=psi3_H.device))
U_fix = kron3(rz_A, I2, I2)
psi3_final = U_fix @ psi3_H
psi3_final = check_norm("psi3_final", psi3_final)

print("Amplitudes AFTER single-qubit Z phase fix (on A):")
pretty_amp("|000>", psi3_final, 0)
pretty_amp("|111>", psi3_final, 7)

# 7) Compare to canonical GHZ
GHZ_std = (ket000 + ket111) / math.sqrt(2)
F_GHZ_std = torch.abs(torch.vdot(GHZ_std, psi3_final))**2
print(f"[final] Fidelity to canonical GHZ = {F_GHZ_std.item():.6f}")

# Sanity: best-phase GHZ after the fix (should ~match canonical)
phi_star = (torch.angle(psi3_final[7]) - torch.angle(psi3_final[0])).item()
GHZ_best2 = (ket000 + torch.exp(1j*torch.as_tensor(phi_star)) * ket111) / math.sqrt(2)
F_GHZ_best2 = torch.abs(torch.vdot(GHZ_best2, psi3_final))**2
print(f"[final] Best-phase GHZ fidelity (sanity) = {F_GHZ_best2.item():.6f} (φ* = {phi_star:.6f} rad)")

# Optional: small leakage proxy in 3q basis after H^{⊗3}+phase-fix
pop_000 = float(torch.abs(psi3_final[0])**2)
pop_111 = float(torch.abs(psi3_final[7])**2)
pop_rest = 1.0 - (pop_000 + pop_111)
print(f"[final] Pop(|000>)={pop_000:.6f}, Pop(|111>)={pop_111:.6f}, Pop(rest)={pop_rest:.6f}")


sv_3q = Statevector(psi3_final.detach().cpu().numpy())
#plot_bloch_multivector(sv_3q, title="Bloch spheres (e, C1, C2) with N14=|0>, others=|+>").savefig(output_dir / "bloch_spheres_3q.png", dpi=200)
#plt.show()
plot_state_city(DensityMatrix(sv_3q), title="City (e, C1, C2) with N14=|0>, others=|0>").savefig(output_dir / "city_3q.png", dpi=200)
plt.show()



##### population evolutions

def compute_3q_population_trajectory(states_full_over_time,
                                     time_axis_ns,
                                     Pn,
                                     qubit_names,
                                     idx_e,
                                     idx_c1,
                                     idx_c2,
                                     idx_N):
    """
    For each time t:
      1) Project FULL state to the all-qubits computational subspace (Pn).
      2) Renormalize inside that subspace.
      3) Slice down to the 3-qubit system (e, C1, C2) with N14 = |0>, others = |0>.
      4) Store populations |amp_{abc}|^2 in order |000>, |001>, ..., |111>.

    Returns:
      pop_3q : np.ndarray with shape (T, 8)
               columns correspond to |000>,|001>,...,|111> in order (a=e, b=C1, c=C2).
    """
    T = states_full_over_time.shape[0]
    pop_3q = np.zeros((T, 8), dtype=float)

    for t in range(T):
        psi_full = states_full_over_time[t]                # (D,)
        psi_all  = Pn @ psi_full                           # (2^nq,)

        # weight inside all-qubits comp. subspace
        w = float((psi_all.conj() @ psi_all).real)
        if w <= 1e-14:
            # no support -> leave row as zeros
            continue
        psi_all = psi_all / math.sqrt(w)

        # 3-qubit slice (e, C1, C2), with N14=0, others=0
        psi_3q = slice_3q_from_proj(psi_all, qubit_names,
                                    idx_e, idx_c1, idx_c2, idx_N)   # (8,)
        pop_3q[t, :] = torch.abs(psi_3q).cpu().numpy()**2

    return pop_3q


def plot_3q_populations(time_axis_ns, pop_3q, outdir: Path,
                        filename: str = "populations_3q_000_to_111.png",
                        title: str = "Populations of |abc⟩ (e, C1, C2)"):
    """
    Plot populations of |000>,...,|111> vs time.
    """
    labels = [f"|{a}{b}{c}⟩"
              for a in (0,1)
              for b in (0,1)
              for c in (0,1)]

    plt.figure(figsize=(10, 6))
    for i, lab in enumerate(labels):
        plt.plot(time_axis_ns, pop_3q[:, i], label=lab)

    plt.xlabel("Time (ns)")
    plt.ylabel("Population")
    plt.title(title)
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()

    outpath = outdir / filename
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.show()
    print(f"Saved 3-qubit population plot to: {outpath}")

# Initialize: e=|0>, N14=|0>, C1=|+>, C2=|+>, all other carbons |0>
import math

nq = len(qubit_names)

def idx_from_bits(bits):
    # little-endian: bit q corresponds to qubit_names[q]
    return sum((bits[q] << q) for q in range(nq))

# Build the state in the projected (all-qubits) computational subspace
psi_all0 = torch.zeros(2**nq, dtype=torch.complex128)
amp = 1 / math.sqrt(8)  # (1/sqrt(2))^3 for e, C1, C2
for a in (0, 1):        # electron
        for b in (0, 1):    # C1
            for c in (0, 1):  # C2
                bits = [0]*nq
                bits[idx_e]  = 0          # e = superposition # set zero to have e in 0 # orthogonal
                bits[idx_N]  = 0          # N14 fixed to |0>
                bits[idx_c1] = 0          # C1 = superposition
                bits[idx_c2] = 0          # C2 = superposition
                # all other carbons = |0>
                for q in range(2, nq):
                    if q not in (idx_c1, idx_c2):
                        bits[q] = 0
                psi_all0[idx_from_bits(bits)] = amp

# Embed to FULL Hilbert space to start evolution
psi_full0 = Pn.conj().T @ psi_all0


# Evolve
states_chunks = apply_sequence_with_3C(Δ_e, time_grids, drives, psi_full0)
states_concat, time_axis_ns = fuse_states_and_time(time_grids, states_chunks)


# --- Time evolution of |000>, |001>, ..., |111> populations for (e, C1, C2) ---
pop_3q = compute_3q_population_trajectory(
    states_full_over_time=states_concat,
    time_axis_ns=time_axis_ns,
    Pn=Pn,
    qubit_names=qubit_names,
    idx_e=idx_e,
    idx_c1=idx_c1,
    idx_c2=idx_c2,
    idx_N=idx_N,
)

plot_3q_populations(
    time_axis_ns,
    pop_3q,
    outdir=output_dir,
    filename="populations_e_C1_C2_000_to_111.png",
    title="Populations of |abc⟩ (e, C1, C2) during pulse",
)


### direct gate fidelities

import torch, math
from pathlib import Path

# --- load projected propagator (8x8) ---
result_dir = pulse_dirs[0]#Path("results/pulse_2025-11-14_05-57-40")  # adjust
U_proj = torch.load(result_dir / "propagator_projected.pt", map_location="cpu").to(torch.complex128)


U = Uloc_dag @ U_proj

# --- build exp(± i π/4 ZZZ) via general matrix exponential ---
Z = torch.tensor([[1,0],[0,-1]], dtype=torch.complex128)
def kron3(a,b,c): return torch.kron(torch.kron(a,b), c)
ZZZ = kron3(Z,Z,Z)  # 8x8, eigenvalues ±1

U_plus  = torch.linalg.matrix_exp( 1j * (math.pi/4) * ZZZ)  # exp(+i π/4 ZZZ)
U_minus = torch.linalg.matrix_exp(-1j * (math.pi/4) * ZZZ)  # exp(-i π/4 ZZZ)

def fidelities(U_target, U_actual):
    d = U_target.shape[0]
    overlap = torch.trace(U_target.conj().T @ U_actual)
    F_pro = (torch.abs(overlap)**2 / (d*d)).item()
    F_avg = ((d * F_pro + 1) / (d + 1))
    return F_pro, F_avg

Fpro_plus,  Favg_plus  = fidelities(U_plus,  U)
Fpro_minus, Favg_minus = fidelities(U_minus, U)

print(f"vs exp(+i π/4 ZZZ): F_pro={Fpro_plus:.8f},  F_avg={Favg_plus:.8f}")
print(f"vs exp(-i π/4 ZZZ): F_pro={Fpro_minus:.8f}, F_avg={Favg_minus:.8f}")

if Fpro_plus >= Fpro_minus:
    print("→ Better match: exp(+i π/4 ZZZ)")
else:
    print("→ Better match: exp(-i π/4 ZZZ)")






import torch, math
from pathlib import Path

# --- load projected propagator (8x8) ---
#result_dir = Path("results/pulse_2025-09-29_12-50-33_report_3Q")  # adjust
result_dir = pulse_dirs[0]#Path("results/pulse_2025-11-19_12-08-00")  # adjust
U_proj = torch.load(result_dir / "propagator_projected.pt", map_location="cpu").to(torch.complex128)


# --- 1) Read diagonal phases φ_{abc} directly from U's diagonal ---
d = torch.diagonal(U_proj)                        # complex diag entries for |abc>
d = d / torch.clamp(d.abs(), min=1e-15)      # unit-modulus (robust if tiny magnitude drift)
phi = torch.angle(d)                         # radians in (-pi, pi]

# Assume basis index order: idx = (a<<2) | (b<<1) | c  with a=e, b=C1, c=C2
def idx(a,b,c): return (a<<2) | (b<<1) | c

# --- 2) Alternating sum (mixed finite difference) to get coefficients ---
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
γ_ab = (1/8.0) * signed_sum(lambda a,b,c: (-1)**(a+b))
γ_ac = (1/8.0) * signed_sum(lambda a,b,c: (-1)**(a+c))
γ_bc = (1/8.0) * signed_sum(lambda a,b,c: (-1)**(b+c))
λ   = (1/8.0) * signed_sum(lambda a,b,c: (-1)**(a+b+c))  # minus sign by convention

print(f"Local phases:  α={α:.6f}, β={β:.6f}, χ={χ:.6f}  (rad)")
print(f"Pairwise:      γ_ab={γ_ab:.6f}, γ_ac={γ_ac:.6f}, γ_bc={γ_bc:.6f}  (rad)")
print(f"Three-body:    λ={λ:.6f}  (rad)")

# --- 3) Build U_loc^† that cancels ONLY the single-qubit Z phases ---
phases_loc_dag = torch.zeros(8, dtype=torch.complex128)
for a in (0,1):
    for b in (0,1):
        for c in (0,1):
            #z = ((-1)*a)*α + ((-1)b)*β + ((-1)*c)*χ
            z = ((-1)**a)*α + ((-1)**b)*β + ((-1)**c)*χ
            phases_loc_dag[idx(a,b,c)] = torch.exp(1j * torch.as_tensor(z, dtype=torch.complex128))
Uloc_prop = torch.diag(phases_loc_dag)


# --- 4) Compare to exp(± i π/4 ZZZ) via general matrix exponential ---
Z = torch.tensor([[1,0],[0,-1]], dtype=torch.complex128)
X = torch.tensor([[0,1],[1,0]], dtype=torch.complex128)

def kron3(a,b,c): return torch.kron(torch.kron(a,b), c)
ZZZ = kron3(Z,Z,Z)

U_plus  = torch.linalg.matrix_exp( 1j * (math.pi/4) * ZZZ)   # exp(+i π/4 ZZZ)
U_minus = torch.linalg.matrix_exp(-1j * (math.pi/4) * ZZZ)   # exp(-i π/4 ZZZ)

U_corr = Uloc_prop @ U_proj   # local-Z removed (pairwise & 3-body content remains)

def gate_fids(Ut, Ua):
    d = Ut.shape[0]
    ov = torch.trace(Ut.conj().T @ Ua)
    F_pro = (torch.abs(ov)**2 / (d*d)).item()
    F_avg = ((d * F_pro + 1) / (d + 1))
    return F_pro, F_avg

Fpro_p, Favg_p = gate_fids(U_plus,  U_corr)
Fpro_m, Favg_m = gate_fids(U_minus, U_corr)

print(f"vs exp(+i π/4 ZZZ): F_pro={Fpro_p:.8f},  F_avg={Favg_p:.8f}")
print(f"vs exp(-i π/4 ZZZ): F_pro={Fpro_m:.8f},  F_avg={Favg_m:.8f}")
print("→ Better match:", "+i" if Fpro_p >= Fpro_m else "−i")


## frame change included, hadamard sandwich considered

import torch, math
from pathlib import Path

# --- helpers (same spirit as your other file) ---
def _I2(dtype, device):
    return torch.eye(2, dtype=dtype, device=device)

def _H2(dtype, device):
    return (1.0 / math.sqrt(2.0)) * torch.tensor(
        [[1.0, 1.0],
         [1.0, -1.0]], dtype=dtype, device=device
    )

def _kron3(A, B, C):
    return torch.kron(torch.kron(A, B), C)

def idx(a,b,c):  # idx = (a<<2) | (b<<1) | c
    return (a<<2) | (b<<1) | c

# --- load projected propagator (8x8) ---
result_dir = pulse_dirs[0]
U_proj = torch.load(result_dir / "propagator_projected.pt", map_location="cpu").to(torch.complex128)

dtype, device = U_proj.dtype, U_proj.device

# --- choose the same "hadamard_on" interface as your state code ---
hadamard_on = ['A']   # 'A' = first qubit; add 'B','C' if needed

H = _H2(dtype, device)
I = _I2(dtype, device)
A_H = 'A' in hadamard_on
B_H = 'B' in hadamard_on
C_H = 'C' in hadamard_on

L = _kron3(H if A_H else I, H if B_H else I, H if C_H else I)

# --- Hadamard-sandwiched propagator (frame) ---
U_eff = L @ U_proj @ L  # since H†=H

# --- 1) Read diagonal phases φ_{abc} from U_eff diagonal ---
d = torch.diagonal(U_eff)
d = d / torch.clamp(d.abs(), min=1e-15)
phi = torch.angle(d)

# --- 2) Alternating sums to get coefficients ---
def signed_sum(weight):
    s = 0.0
    for a in (0,1):
        for b in (0,1):
            for c in (0,1):
                s += weight(a,b,c) * phi[idx(a,b,c)]
    return s

α    = (1/8.0) * signed_sum(lambda a,b,c: (-1)**a)
β    = (1/8.0) * signed_sum(lambda a,b,c: (-1)**b)
χ    = (1/8.0) * signed_sum(lambda a,b,c: (-1)**c)
γ_ab = (1/8.0) * signed_sum(lambda a,b,c: (-1)**(a+b))
γ_ac = (1/8.0) * signed_sum(lambda a,b,c: (-1)**(a+c))
γ_bc = (1/8.0) * signed_sum(lambda a,b,c: (-1)**(b+c))
λ    = (1/8.0) * signed_sum(lambda a,b,c: (-1)**(a+b+c))

print(f"Local phases:  α={α:.6f}, β={β:.6f}, χ={χ:.6f}  (rad)")
print(f"Pairwise:      γ_ab={γ_ab:.6f}, γ_ac={γ_ac:.6f}, γ_bc={γ_bc:.6f}  (rad)")
print(f"Three-body:    λ={λ:.6f}  (rad)")

# --- 3) Build U_loc^† (diagonal) that cancels ONLY the single-qubit Z phases in THIS frame ---
phases_loc_dag = torch.zeros(8, dtype=torch.complex128, device=device)
for a in (0,1):
    for b in (0,1):
        for c in (0,1):
            z = (-a)*α + (-b)*β + (-c)*χ
            phases_loc_dag[idx(a,b,c)] = torch.exp(-1j * z)

Uloc_prop = torch.diag(phases_loc_dag)

# If you want the corrected propagator in the SAME frame:
U_corr_eff = Uloc_prop @ U_eff

# If you want the corrected propagator back in the LAB frame:
U_corr_lab = L @ U_corr_eff @ L


# --- 4) Compare to exp(± i π/4 XZZ) via general matrix exponential ---
Z = torch.tensor([[1,0],[0,-1]], dtype=torch.complex128)
X = torch.tensor([[0,1],[1,0]], dtype=torch.complex128)

def kron3(a,b,c): return torch.kron(torch.kron(a,b), c)
XZZ = kron3(X,Z,Z)

U_plus  = torch.linalg.matrix_exp( 1j * (math.pi/4) * XZZ)   # exp(+i π/4 ZZZ)
U_minus = torch.linalg.matrix_exp(-1j * (math.pi/4) * XZZ)   # exp(-i π/4 ZZZ)

U_corr = U_corr_lab #Uloc_prop @ U_proj   # local-Z removed (pairwise & 3-body content remains)

def gate_fids(Ut, Ua):
    d = Ut.shape[0]
    ov = torch.trace(Ut.conj().T @ Ua)
    F_pro = (torch.abs(ov)**2 / (d*d)).item()
    F_avg = ((d * F_pro + 1) / (d + 1))
    return F_pro, F_avg

Fpro_p, Favg_p = gate_fids(U_plus,  U_corr)
Fpro_m, Favg_m = gate_fids(U_minus, U_corr)

print(f"vs exp(+i π/4 XZZ): F_pro={Fpro_p:.8f},  F_avg={Favg_p:.8f}")
print(f"vs exp(-i π/4 XZZ): F_pro={Fpro_m:.8f},  F_avg={Favg_m:.8f}")
print("→ Better match:", "+i" if Fpro_p >= Fpro_m else "−i")




#======= phase evolutions ####


import numpy as np
import math

def compute_all_phases_vs_time_principal(time_grids, drives, Δ_e,
                                         Pn, qubit_names,
                                         idx_e, idx_N, idx_c1, idx_c2):
    """
    Compute the time-dependent 1-body, 2-body, and 3-body phase coefficients
    using the SAME principal-branch convention as diag(U_proj),
    with NO per-basis unwrap.

      α(t), β(t), χ(t)               (single-qubit phases)
      γ_ab(t), γ_ac(t), γ_bc(t)     (two-qubit phases)
      λ(t)                          (three-qubit phase)

    Based on computing φ_{abc}(t) = angle(<abc|ψ(t)>), each in (-π, π].

    Returns:
      time_axis_ns : (T,)
      phases : dict of numpy arrays (each size T)
    """

    # φ_{abc}(t) stored here
    phi_abc = {}
    time_axis_ns = None

    # --- Loop over the 8 basis states |a,b,c>
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                # Build full initial ket (full Hilbert space)
                psi_full0 = build_full_ket_for_abc(
                    a, b, c, qubit_names, idx_e, idx_N, idx_c1, idx_c2, Pn
                )

                # Evolve
                chunks = apply_sequence_with_3C(Δ_e, time_grids, drives, psi_full0)
                states_concat, t_ns = fuse_states_and_time(time_grids, chunks)

                if time_axis_ns is None:
                    time_axis_ns = t_ns
                else:
                    if len(time_axis_ns) != len(t_ns):
                        raise ValueError("Time axis mismatch")

                T = states_concat.shape[0]
                phi_t = np.zeros(T, dtype=np.float64)
                idx3 = (a << 2) | (b << 1) | c   # index in 3q slice

                # Extract φ_{abc}(t)
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
                    amp_t = psi_3q_t[idx3]
                    phi_t[ti] = float(torch.angle(amp_t).item())   # PRINCIPAL BRANCH
                
                # Immediately unwrap this trajectory ONCE, and only once
                phi_t = np.unwrap(phi_t)

                # Store principal-branch φ_{abc}(t)
                phi_abc[(a, b, c)] = phi_t

    # ---- Now compute α(t), β(t), χ(t), γ_ab(t), γ_ac(t), γ_bc(t), λ(t)
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

        alpha[ti] = A/8.0
        beta[ti]  = B/8.0
        chi[ti]   = C/8.0
        gamma_ab[ti] = Gab/8.0
        gamma_ac[ti] = Gac/8.0
        gamma_bc[ti] = Gbc/8.0
        lamb[ti]     = L/8.0

    # Clean unwrapped versions (smooth plotting), but only unwrap after combination:
    alpha_u = alpha
    beta_u  = beta
    chi_u   = chi
    gab_u   = gamma_ab
    gac_u   = gamma_ac
    gbc_u   = gamma_bc
    lamb_u  = lamb

    phases = {
        "alpha": alpha,       "alpha_u": alpha_u,
        "beta": beta,         "beta_u": beta_u,
        "chi": chi,           "chi_u": chi_u,
        "gamma_ab": gamma_ab, "gamma_ab_u": gab_u,
        "gamma_ac": gamma_ac, "gamma_ac_u": gac_u,
        "gamma_bc": gamma_bc, "gamma_bc_u": gbc_u,
        "lambda": lamb,       "lambda_u": lamb_u,
    }

    return time_axis_ns, phases


import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 15,          # base font size
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
})

def plot_all_invariants_one_plot(
    pulse_dir: Path,
    time_grids, drives, Δ_e,
    Pn, qubit_names,
    idx_e, idx_N, idx_c1, idx_c2,
    outdir: Path,
    filename="all_invariants_one_plot.png",
):
    """
    Plot the 3-qubit phase invariants:
        Δ_{a}, Δ_{b}, Δ_{c},
        Δ_{a,b}, Δ_{a,c}, Δ_{b,c},
        Δ_{a,b,c}
    using unwrapped phases.
    """

    time_axis, ph = compute_all_phases_vs_time_principal(
        time_grids, drives, Δ_e,
        Pn, qubit_names,
        idx_e, idx_N, idx_c1, idx_c2
    )



    plt.figure(figsize=(12, 6), dpi=200)

    # --- 1-body invariants ---
    plt.plot(time_axis, ph["alpha_u"], label=r"$\Delta_{\{a\}}$ (single, e)")
    plt.plot(time_axis, ph["beta_u"],  label=r"$\Delta_{\{b\}}$ (single, C1)")
    plt.plot(time_axis, ph["chi_u"],   label=r"$\Delta_{\{c\}}$ (single, C2)")

    # --- 2-body invariants ---
    plt.plot(time_axis, ph["gamma_ab_u"], label=r"$\Delta_{\{a,b\}}$ (e–C1)")
    plt.plot(time_axis, ph["gamma_ac_u"], label=r"$\Delta_{\{a,c\}}$ (e–C2)")
    plt.plot(time_axis, ph["gamma_bc_u"], label=r"$\Delta_{\{b,c\}}$ (C1–C2)")

    # --- 3-body invariant ---
    plt.plot(
        time_axis, ph["lambda_u"], 
        linewidth=2.6, color="black",
        label=r"$\Delta_{\{a,b,c\}}$ (3-body)"
    )

    # Target reference line
    plt.axhline(
        -np.pi* 3/4, linestyle="--", color="black", alpha=0.6,
        label=r"$- 3/4 \pi$"
    )

    plt.axhline(
        np.pi, linestyle=":", color="black", alpha=0.6,
        label=r"$+\pi$"
    )

    plt.axhline(
        -np.pi, linestyle=":", color="black", alpha=0.6,
        label=r"$-\pi$"
    )

    plt.xlabel("Time (ns)")
    plt.ylabel("Phase invariant value (rad)")
    plt.title("Time Evolution of Three-Qubit (e,C1,C2) Phase Invariants")
    plt.grid(True)

    plt.legend(ncol=2, frameon=True)

    plt.tight_layout()
    outpath = outdir / filename
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.show()

    print(f"Saved invariant plot to: {outpath}")

plot_all_invariants_one_plot(
    pulse_dir=pulse_dirs[0],
    time_grids=time_grids,
    drives=drives,
    Δ_e=Δ_e,
    Pn=Pn,
    qubit_names=qubit_names,
    idx_e=idx_e, idx_N=idx_N, idx_c1=idx_c1, idx_c2=idx_c2,
    outdir=output_dir
)



# non-diag XZZ plotting:

import numpy as np
import math
import torch

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
    """
    psi8: shape (8,) complex state in computational basis |000>..|111>.
    returns (L_A⊗L_B⊗L_C) @ psi8
    """
    dtype, device = psi8.dtype, psi8.device
    if Ls is None:
        Ls = [_I2(dtype, device)] * 3
    L = _kron3(Ls[0], Ls[1], Ls[2])
    return L @ psi8

def compute_all_phases_vs_time_principal_nondiag(
    time_grids, drives, Δ_e,
    Pn, qubit_names,
    idx_e, idx_N, idx_c1, idx_c2,
    hadamard_on=None,   # subset of ['A','B','C']
    Ls=None,            # explicit [L_A,L_B,L_C] overrides hadamard_on if provided
):
    """
    Same as the compute_all_phases_vs_time_principal, but extracts phases
    after applying a local 3-qubit rotation to the 3q slice:
        psi' = (L_A⊗L_B⊗L_C) psi
    This is the state-analog of your U -> L U R† sandwich (with R irrelevant for states).
    """

    phi_abc = {}
    time_axis_ns = None

    # figure dtype/device once we see a state; we'll lazily init Ls_eff
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
                    )  # torch tensor shape (8,)

                    # --- build Ls_eff once (dtype/device known now) ---
                    if Ls_eff is None:
                        dtype, device = psi_3q_t.dtype, psi_3q_t.device
                        if Ls is not None:
                            Ls_eff = Ls
                        else:
                            # hadamard_on mode
                            if hadamard_on is None:
                                hadamard_on = []
                            H = _H2(dtype, device)
                            I = _I2(dtype, device)
                            A_H = 'A' in hadamard_on
                            B_H = 'B' in hadamard_on
                            C_H = 'C' in hadamard_on
                            Ls_eff = [H if A_H else I, H if B_H else I, H if C_H else I]

                    # --- apply local rotation to the STATE slice ---
                    psi_3q_rot = _apply_local_to_state_3q(psi_3q_t, Ls=Ls_eff)

                    amp_t = psi_3q_rot[idx3]
                    phi_t[ti] = float(torch.angle(amp_t).item())

                # unwrap once per trajectory (your original choice)
                phi_t = np.unwrap(phi_t)
                phi_abc[(a, b, c)] = phi_t

    # combine into invariants (unchanged math)
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

        alpha[ti] = A/8.0
        beta[ti]  = B/8.0
        chi[ti]   = C/8.0
        gamma_ab[ti] = Gab/8.0
        gamma_ac[ti] = Gac/8.0
        gamma_bc[ti] = Gbc/8.0
        lamb[ti]     = L/8.0

    phases = {
        "alpha": alpha,       "alpha_u": alpha,
        "beta": beta,         "beta_u": beta,
        "chi": chi,           "chi_u": chi,
        "gamma_ab": gamma_ab, "gamma_ab_u": gamma_ab,
        "gamma_ac": gamma_ac, "gamma_ac_u": gamma_ac,
        "gamma_bc": gamma_bc, "gamma_bc_u": gamma_bc,
        "lambda": lamb,       "lambda_u": lamb,
    }
    return time_axis_ns, phases

#time_axis, ph = compute_all_phases_vs_time_principal_nondiag(
#    time_grids, drives, Δ_e,
#    Pn, qubit_names,
#    idx_e, idx_N, idx_c1, idx_c2,
#    hadamard_on=['A'],
#)

def plot_all_invariants_one_plot(
    pulse_dir: Path,
    time_grids, drives, Δ_e,
    Pn, qubit_names,
    idx_e, idx_N, idx_c1, idx_c2,
    outdir: Path,
    filename="all_invariants_one_plot.png",
):
    """
    Plot the 3-qubit phase invariants:
        Δ_{a}, Δ_{b}, Δ_{c},
        Δ_{a,b}, Δ_{a,c}, Δ_{b,c},
        Δ_{a,b,c}
    using unwrapped phases.
    """



    time_axis, ph = compute_all_phases_vs_time_principal_nondiag(
    time_grids, drives, Δ_e,
    Pn, qubit_names,
    idx_e, idx_N, idx_c1, idx_c2,
    hadamard_on=['A'],
)

    plt.figure(figsize=(12, 6))

    # --- 1-body invariants ---
    plt.plot(time_axis, ph["alpha_u"], label=r"$\Delta_{\{a\}}$ (single, e)")
    plt.plot(time_axis, ph["beta_u"],  label=r"$\Delta_{\{b\}}$ (single, C1)")
    plt.plot(time_axis, ph["chi_u"],   label=r"$\Delta_{\{c\}}$ (single, C2)")

    # --- 2-body invariants ---
    plt.plot(time_axis, ph["gamma_ab_u"], label=r"$\Delta_{\{a,b\}}$ (e–C1)")
    plt.plot(time_axis, ph["gamma_ac_u"], label=r"$\Delta_{\{a,c\}}$ (e–C2)")
    plt.plot(time_axis, ph["gamma_bc_u"], label=r"$\Delta_{\{b,c\}}$ (C1–C2)")

    # --- 3-body invariant ---
    plt.plot(
        time_axis, ph["lambda_u"], 
        linewidth=2.6, color="black",
        label=r"$\Delta_{\{a,b,c\}}$ (3-body)"
    )

    # Target reference line
    plt.axhline(
        np.pi* 1/4, linestyle="--", color="black", alpha=0.6,
        label=r"$\pi/4$"
    )

    #plt.axhline(
    #    np.pi, linestyle=":", color="black", alpha=0.6,
    #    label=r"$+\pi$"
    #)

    #plt.axhline(
    #    -np.pi, linestyle=":", color="black", alpha=0.6,
    #    label=r"$-\pi$"
    #)

    plt.xlabel("Time (ns)")
    plt.ylabel("Phase invariant value (rad)")
    plt.title("Time Evolution of Three-Qubit (e,C1,C2) Phase Invariants")
    plt.grid(True)

    plt.legend(ncol=2, frameon=True)

    plt.tight_layout()
    outpath = outdir / filename
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.show()

    print(f"Saved invariant plot to: {outpath}")

plot_all_invariants_one_plot(
    pulse_dir=pulse_dirs[0],
    time_grids=time_grids,
    drives=drives,
    Δ_e=Δ_e,
    Pn=Pn,
    qubit_names=qubit_names,
    idx_e=idx_e, idx_N=idx_N, idx_c1=idx_c1, idx_c2=idx_c2,
    outdir=output_dir
)
