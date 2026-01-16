# reproduce_numerics.py
"""
Reproduce the numerical demonstration plots for the three-qubit gate.

This script is intended as the main entry point for the GitHub repository
accompanying the theory paper. It produces, for each gate type:

  1) Pulse shape (drive amplitudes → Rabi frequency vs time)
  2) Three-qubit populations vs time in the (e, C1, C2) computational basis
     starting from |000⟩
  3) Phase invariants vs time

and prints the gate fidelities vs exp(± i π/4 G), where
G = Z⊗Z⊗Z for the diagonal gate and G = X⊗Z⊗Z for the non-diagonal gate.

Usage
-----
From the project root:

    python -m src.reproduce_numerics --gate diagonal
    python -m src.reproduce_numerics --gate nondiagonal

or from within src/:

    python reproduce_numerics.py --gate diagonal
    python reproduce_numerics.py --gate nondiagonal
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from quantum_model_NV import set_active_carbons, get_precomp
from three_qubit_gate_utils import (
    load_ckpt,
    apply_sequence_with_3C,
    fuse_states_and_time,
    make_multi_qubit_basis_indices,
    projector_from_indices_general,
    build_full_000,
    plot_drives,
    compute_3q_population_trajectory,
    plot_3q_populations,
    compute_all_phases_vs_time,
    plot_invariants,
    zzz_gate_fidelities,
    xzz_gate_fidelities,
)

# Anchor paths at project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

GATE_CONFIGS = {
    "diagonal": {
        "label": "diagonal (ZZZ)",
        "pulse_dir": PROJECT_ROOT / "results" / "pulse_2025-12-15_14-18-53",
        "fig_subdir": PROJECT_ROOT / "figs" / "diagonal",
        "target_line": -np.pi * 3/4,
        "target_label": r"$-3\pi/4$",
        "hadamard_on_invariants": None,
        "invariants_filename": "invariants_diagonal_ZZZ.png",
        "pop_filename": "populations_diagonal_ZZZ.png",
        "pulse_filename": "pulse_diagonal_ZZZ.png",  # name only, used inside plot_drives folder
        "gate_fidelity_fn": "zzz",
    },
    "nondiagonal": {
        "label": "non-diagonal (XZZ)",
        "pulse_dir": PROJECT_ROOT / "results" / "pulse_2025-12-17_14-44-22",
        "fig_subdir": PROJECT_ROOT / "figs" / "nondiagonal",
        "target_line": np.pi * 1/4,
        "target_label": r"$\pi/4$",
        "hadamard_on_invariants": ["A"],
        "invariants_filename": "invariants_nondiagonal_XZZ.png",
        "pop_filename": "populations_nondiagonal_XZZ.png",
        "pulse_filename": "pulse_nondiagonal_XZZ.png",
        "gate_fidelity_fn": "xzz",
    },
}


def run_demo(gate_key: str):
    cfg = GATE_CONFIGS[gate_key]
    gate_label = cfg["label"]
    result_dir = cfg["pulse_dir"]
    fig_dir = cfg["fig_subdir"]

    print(f"=== Numerical demonstration: {gate_label} ===")
    print(f"Using pulse directory: {result_dir}")

    # ---------------------------------------------------------------------
    # 1) Hamiltonian setup and checkpoint loading
    # ---------------------------------------------------------------------
    set_active_carbons([1, 2])  # same as in optimization scripts
    pc = get_precomp()
    N_C = int(pc["N_C"])
    D = 2 * 3 * (2 ** N_C)

    # Load pulse checkpoint
    ckpt = load_ckpt(result_dir)
    Δ_e = ckpt["Δ_e"]
    basis_indices = ckpt["basis_indices"]
    if basis_indices is None:
        raise ValueError("basis_indices is required but missing in checkpoint.")

    drives = [ckpt["drive"]]
    time_grids = [ckpt["time_grid"]]

    print(f"Loaded Δ_e = {Δ_e:.6e} and drive with {len(drives[0])} channels.")

    # ---------------------------------------------------------------------
    # 2) Build multi-qubit projector and indices
    # ---------------------------------------------------------------------
    n14_pair = (0, 1)
    electron_map = ('m1', '0')
    basis_indices_multi, qubit_names = make_multi_qubit_basis_indices(
        pc, n14_pair=n14_pair, electron_map=electron_map
    )
    Pn = projector_from_indices_general(basis_indices_multi, D)

    idx_e   = qubit_names.index("e⁻")
    idx_N   = qubit_names.index("14N")
    idx_c1, idx_c2 = 2, 3  # first two carbons as logical B and C

    print(f"Projected qubits: {qubit_names}")
    print(f"Using (A,B,C) = (e⁻, C1, C2) = indices ({idx_e}, {idx_c1}, {idx_c2}), N14 index = {idx_N}")

    # ---------------------------------------------------------------------
    # 3) Pulse plot
    # ---------------------------------------------------------------------
    # Note: plot_drives saves into result_dir by default; for paper you might
    # want to copy or save into fig_dir instead. Easiest is: run in-place,
    # then copy the files when preparing the final manuscript.
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("Plotting pulse...")
    plot_drives(drives, time_grids, fig_dir, title=f"{gate_label} MW pulse")

    # ---------------------------------------------------------------------
    # 4) Population plot: start from pure |000> in (e,C1,C2)
    # ---------------------------------------------------------------------
    print("Computing population evolution starting from |000> (e,C1,C2)...")
    psi_full0 = build_full_000(Pn, qubit_names, idx_e, idx_N, idx_c1, idx_c2)
    chunks = apply_sequence_with_3C(Δ_e, time_grids, drives, psi_full0)
    states_concat, time_axis_ns = fuse_states_and_time(time_grids, chunks)

    pop_3q = compute_3q_population_trajectory(
        states_full_over_time=states_concat,
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
        outdir=fig_dir,
        filename=cfg["pop_filename"],
        title=f"{gate_label}: populations of |abc⟩ (e, C1, C2) during pulse",
        highlight_state="100",  # adjust if you want a different highlighted path
    )

    # ---------------------------------------------------------------------
    # 5) Phase invariants vs time
    # ---------------------------------------------------------------------
    print("Computing phase invariants vs time...")
    had_on = cfg["hadamard_on_invariants"]
    t_inv, phases = compute_all_phases_vs_time(
        time_grids, drives, Δ_e,
        Pn, qubit_names,
        idx_e, idx_N, idx_c1, idx_c2,
        hadamard_on=had_on,
    )

    plot_invariants(
        t_inv, phases,
        outdir=fig_dir,
        filename=cfg["invariants_filename"],
        title=f"{gate_label}: phase invariants for (e, C1, C2)",
        target_line=cfg["target_line"],
        target_label=cfg["target_label"],
    )

    # ---------------------------------------------------------------------
    # 6) Gate fidelities
    # ---------------------------------------------------------------------
    print("Computing gate fidelities from projected propagator...")
    try:
        U_proj = torch.load(result_dir / "propagator_projected.pt",
                            map_location="cpu").to(torch.complex128)
    except FileNotFoundError as e:
        print("ERROR: propagator_projected.pt not found; cannot compute gate fidelities.")
        print(e)
        return

    if cfg["gate_fidelity_fn"] == "zzz":
        zzz_gate_fidelities(U_proj)
    else:
        xzz_gate_fidelities(U_proj, hadamard_on=('A',))

    print("=== Done. Figures saved to:", fig_dir, "===\n")


#if __name__ == "_main_":
parser = argparse.ArgumentParser(
    description="Reproduce numerical demonstrations for diagonal and non-diagonal three-qubit gates."
)
parser.add_argument(
    "--gate",
    choices=["diagonal", "nondiagonal"],
    default="diagonal",
    help="Which gate to analyze: 'diagonal' ≡ ZZZ, 'nondiagonal' ≡ XZZ.",
)
args = parser.parse_args()
run_demo(args.gate)