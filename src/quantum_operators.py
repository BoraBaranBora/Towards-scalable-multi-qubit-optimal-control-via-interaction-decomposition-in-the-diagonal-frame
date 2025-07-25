import torch
from functools import reduce
import operator

# Pauli matrices
I = torch.eye(2, dtype=torch.complex128)
X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

def pauli_operator_on_qubit(pauli: str, qubit_index: int, total_qubits: int = 3):
    """Construct a full system operator with a Pauli operator on a specific qubit."""
    pauli_map = {"X": X, "Z": Z}
    if pauli not in pauli_map:
        raise ValueError("Only 'X' and 'Z' supported for now.")

    ops = [I] * total_qubits
    ops[qubit_index] = pauli_map[pauli]
    return reduce(operator.kron, ops)
