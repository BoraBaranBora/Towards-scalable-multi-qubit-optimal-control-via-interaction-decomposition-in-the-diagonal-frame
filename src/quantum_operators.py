import torch
from functools import reduce

# Pauli matrices
I = torch.eye(2, dtype=torch.complex128)
X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

def pauli_operator_on_qubit(pauli: str, qubit_index: int, total_qubits: int = 3):
    """Construct a full system operator with a Pauli operator on a specific qubit."""
    pauli_map = {"X": X, "Y": Y, "Z": Z}
    if pauli not in pauli_map:
        raise ValueError("Only 'X', 'Y', and 'Z' are supported.")

    ops = [I] * total_qubits
    ops[qubit_index] = pauli_map[pauli]
    return reduce(torch.kron, ops)

