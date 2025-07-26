import torch

def ccz_gate():
    """Returns a 3-qubit CCZ gate as an 8x8 matrix with a -1 phase on |011⟩."""
    gate = torch.eye(8, dtype=torch.complex128)
    gate[3, 3] = -1
    gate[7, 7] = -1
    return gate

def cz_on_qubits_0_and_2():
    """Returns an 8x8 matrix applying CZ between qubit 0 and qubit 2 in a 3-qubit system."""
    gate = torch.eye(8, dtype=torch.complex128)
    for i in range(8):
        b = f"{i:03b}"  # Get binary string of 3 bits
        if b[0] == '1' and b[2] == '1':
            gate[i, i] = -1
    return gate