 ## Reproducing the numerical demonstrations

This repository accompanies the paper *Towards scalable multi-qubit optimal control
via interaction decomposition in the diagonal frame* and provides the data and scripts used in the numerical demonstrations shown in the manuscript.

The results demonstrate that specifying multi-qubit control targets in a diagonal frame enables scalable k-body decomposable objectives, substantially reducing target specification and optimization complexity. On a simulated room-temperature NV–\(^{14}\)N–\(^{13}\)C–\(^{13}\)C register, this approach enables genuinely tripartite entangling gates to be synthesized with a single shaped microwave pulse, achieving fidelities ≈ 0.999 and yielding a 10–100× speed-up compared with previous multi-qubit entanglers beyond two qubits on this platform.

The datasets correspond to two tripartite three-qubit entangling gates:

- *Diagonal:* ≈ e^{i π/4 Z⊗Z⊗Z} (ZZZ)  
- *Non-diagonal:* ≈ e^{i π/4 X⊗Z⊗Z} (XZZ)

### Contents

The repository provides:

- the synthesized control pulses used to generate the gates,
- scripts for evolving under the control pulses,
- and plotting utilities for regenerating the figures in the manuscript,
- the resulting k-body decomposed phase contributions,
- the achieved gate fidelities.

These resources allow the plots and numerical results from the paper to be reproduced.

To regenerate the figures:

```bash
# Diagonal gate (ZZZ)
python src/reproduce_numerics --gate diagonal

# Non-diagonal gate (XZZ)
python src/reproduce_numerics --gate nondiagonal

```
### Repository Structure

├─ results/
│ ├─ pulse_diagonal/ # diagonal (ZZZ) data
│ └─ pulse_nondiagonal/ # non-diagonal (XZZ) data
├─ src/
│ ├─ three_qubit_gate_utils.py # gate synthesis and k-body analysis
│ └─ reproduce_numerics.py # thin wrapper for generating paper figures
├─ figs/
│ ├─ diagonal/
│ └─ nondiagonal/
└─ README.md


