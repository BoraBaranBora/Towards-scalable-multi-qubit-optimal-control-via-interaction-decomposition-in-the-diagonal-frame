import torch
import math
import cmath


# Use complex dtype
dtype = torch.complex128
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Constants
Gauss_to_Tesla = 0.0001
B_0 = torch.tensor(4500.0 * Gauss_to_Tesla, dtype=torch.float64)
D = torch.tensor(2.87e3, dtype=torch.float64)  # GHz
γ_e = torch.tensor(28.0e3, dtype=torch.float64)  # GHz/T

γ_n = torch.tensor([3.077, 10.71, 10.71, 10.71, 10.71], dtype=torch.float64)  # MHz/T
Azz_n = torch.tensor([-2.14, 2.281, 1.884, -1.386, -1.011], dtype=torch.float64)  # MHz
A_ort_n = torch.tensor([0.0, 0.24, 0.208, 0.13, 0.014], dtype=torch.float64)
Q_n = torch.tensor([-5.01, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
ϕ_n = torch.tensor([0.0, 0.0], dtype=torch.float64)



def ω_n(i):
    term = Azz_n[i] + γ_n[i] * B_0
    return torch.sqrt(torch.abs(term)**2 + torch.abs(A_ort_n[i])**2)

def ω_na(i):
    return γ_n[i] * B_0

def θ(n):
    return torch.atan2(A_ort_n[n], Azz_n[n] + γ_n[n] * B_0)

def ϵ(n):
    theta = θ(n)
    phi = ϕ_n[n]
    num = (1 - torch.cos(theta)) * torch.tan(phi)
    denom = 1 + torch.cos(theta) * torch.tan(phi)**2
    return torch.atan2(num, denom)

def δ(i):
    return (ω_n(i) - ω_na(i)) / 2



def Trafo3(theta, phi, epsilon):
    return torch.stack([
        torch.stack([
            0.5 * torch.exp(1j * epsilon) * (1 + torch.cos(theta)),
            1j * torch.exp(1j * (epsilon - phi)) * torch.sin(theta) / math.sqrt(2),
            0.5 * torch.exp(1j * (epsilon - 2 * phi)) * (-1 + torch.cos(theta))
        ]),
        torch.stack([
            1j * torch.exp(1j * phi) * torch.sin(theta) / math.sqrt(2),
            torch.cos(theta),
            1j * torch.exp(-1j * phi) * torch.sin(theta) / math.sqrt(2)
        ]),
        torch.stack([
            0.5 * torch.exp(-1j * (epsilon - 2 * phi)) * (-1 + torch.cos(theta)),
            1j * torch.exp(-1j * (epsilon - phi)) * torch.sin(theta) / math.sqrt(2),
            0.5 * torch.exp(-1j * epsilon) * (1 + torch.cos(theta))
        ])
    ]).to(dtype)


def Trafo2(theta, phi, epsilon):
    return torch.stack([
        torch.stack([
            torch.exp(1j * epsilon / 2) * torch.cos(theta / 2),
            1j * torch.exp(0.5j * (epsilon - 2 * phi)) * torch.sin(theta / 2)
        ]),
        torch.stack([
            1j * torch.exp(-0.5j * (epsilon - 2 * phi)) * torch.sin(theta / 2),
            torch.exp(-1j * epsilon / 2) * torch.cos(theta / 2)
        ])
    ]).to(dtype)


def spin_operators():
    qx = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
    qy = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype)
    qz = torch.tensor([[1, 0], [0, -1]], dtype=dtype)

    sx = 0.5 * qx
    sy = 0.5 * qy
    sz = 0.5 * qz

    I2 = torch.eye(2, dtype=dtype)
    I3 = torch.eye(3, dtype=dtype)

    sx3 = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=dtype) / math.sqrt(2)
    sy3 = torch.tensor([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=dtype) / math.sqrt(2)
    sz3 = torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=dtype)

    return {
        'qx': qx, 'qy': qy, 'qz': qz,
        'sx': sx, 'sy': sy, 'sz': sz,
        'I2': I2, 'I3': I3,
        'sx3': sx3, 'sy3': sy3, 'sz3': sz3
    }

ops = spin_operators()


def kronN(*args):
    result = args[0]
    for a in args[1:]:
        result = torch.einsum('ab,cd->acbd', result, a).reshape(
            result.size(0) * a.size(0),
            result.size(1) * a.size(1)
        )
    return result


e0 = torch.tensor([[1.0 + 0j, 0], [0, 0]], dtype=dtype)
e1 = torch.tensor([[0, 0], [0, 1.0 + 0j]], dtype=dtype)


# Extract operators
I2, I3 = ops['I2'], ops['I3']
sx, sy, sz = ops['sx'], ops['sy'], ops['sz']
sx3, sy3, sz3 = ops['sx3'], ops['sy3'], ops['sz3']
qx, qy, qz = ops['qx'], ops['qy'], ops['qz']

# 12x12 base operators
Ix1 = kronN(qx, I3, I2)
Ix2 = kronN(I2, sx3, I2)
Ix3 = kronN(I2, I3, sx)

Iy1 = kronN(qy, I3, I2)
Iy2 = kronN(I2, sy3, I2)
Iy3 = kronN(I2, I3, sy)

Iz1 = kronN(qz, I3, I2)
Iz2 = kronN(I2, sz3, I2)
Iz3 = kronN(I2, I3, sz)


# A-type: electron in |0⟩
Ix2A = kronN(e0, sx3, I2)
Ix3A = kronN(e0, I3, sx)
Iy2A = kronN(e0, sy3, I2)
Iy3A = kronN(e0, I3, sy)
Iz2A = kronN(e0, sz3, I2)
Iz3A = kronN(e0, I3, sz)

# B1, B2 transformations
Θ1, Θ2 = θ(0), θ(1)
ϕ1, ϕ2 = ϕ_n[0], ϕ_n[1]
ϵ1, ϵ2 = ϵ(0), ϵ(1)

B1 = Trafo3(Θ1, ϕ1, ϵ1)
B2 = Trafo2(Θ2, ϕ2, ϵ2)

# C-type: electron in |1⟩, rotated basis
Ix2C = kronN(e1, B1.conj().T @ sx3 @ B1, I2)
Ix3C = kronN(e1, I3, B2.conj().T @ sx @ B2)
Iy2C = kronN(e1, B1.conj().T @ sy3 @ B1, I2)
Iy3C = kronN(e1, I3, B2.conj().T @ sy @ B2)
Iz2C = kronN(e1, B1.conj().T @ sz3 @ B1, I2)
Iz3C = kronN(e1, I3, B2.conj().T @ sz @ B2)


Iex3 = torch.kron(I3, sx)
Iey3 = torch.kron(I3, sy)
Iez3 = torch.kron(I3, sz)


two_pi = 2 * math.pi

# Convert to Hz
γ_e = γ_e * two_pi
γ1 = γ_n[0] * two_pi * 1e6  # MHz → Hz
γ2 = γ_n[1] * two_pi * 1e6

ωa1 = ω_na(0) * two_pi * 1e6
ωa2 = ω_na(1) * two_pi * 1e6
ω1 = ω_n(0) * two_pi * 1e6
ω2 = ω_n(1) * two_pi * 1e6

Q1 = Q_n[0] * two_pi * 1e6
Q2 = Q_n[1] * two_pi * 1e6
δ1 = δ(0) * two_pi * 1e6
δ2 = δ(1) * two_pi * 1e6


Λ_s = γ_e * B_0 - D * two_pi * 1e9  # D in GHz → Hz

# Helper: GHz shift minus MHz ω_n → all scaled to Hz
def MHz(val): return val * two_pi * 1e6

Λ00 = Λ_s - (0 * (γ_n[0] * B_0 - ω_n(0)) + 0.5 * (γ_n[1] * B_0 - ω_n(1))) * two_pi * 1e6
Λ01 = Λ_s - (0 * (γ_n[0] * B_0 - ω_n(0)) + -0.5 * (γ_n[1] * B_0 - ω_n(1))) * two_pi * 1e6
Λ10 = Λ_s - (-1 * (γ_n[0] * B_0 - ω_n(0)) + 0.5 * (γ_n[1] * B_0 - ω_n(1))) * two_pi * 1e6
Λ11 = Λ_s - (-1 * (γ_n[0] * B_0 - ω_n(0)) + -0.5 * (γ_n[1] * B_0 - ω_n(1))) * two_pi * 1e6
Λm10 = Λ_s - (1 * (γ_n[0] * B_0 - ω_n(0)) + 0.5 * (γ_n[1] * B_0 - ω_n(1))) * two_pi * 1e6
Λm11 = Λ_s - (1 * (γ_n[0] * B_0 - ω_n(0)) + -0.5 * (γ_n[1] * B_0 - ω_n(1))) * two_pi * 1e6



factor = 1.0

def get_U(Ω, dt, t, Δ=0.0):
    # Control fields
    Ω_e = factor * torch.cos((Λ_s + Δ) * t) * (Ω[0]) / γ_e #2.0*math.pi*5e6 
    Ω_x = Ω_e * (γ_e / γ2)

    # Diagonal modulation
    diag_cos = torch.tensor([
        torch.cos(Λ10 * t), torch.cos(Λ11 * t), torch.cos(Λ00 * t),
        torch.cos(Λ01 * t), torch.cos(Λm10 * t), torch.cos(Λm11 * t)
    ] * 2, dtype=dtype)

    diag_sin = torch.tensor([
        torch.sin(Λ10 * t), torch.sin(Λ11 * t), torch.sin(Λ00 * t),
        torch.sin(Λ01 * t), torch.sin(Λm10 * t), torch.sin(Λm11 * t)
    ] * 2, dtype=dtype)

    H = Ix1 @ torch.diag(diag_cos)
    H2 = Iy1 @ torch.diag(diag_sin)

    H = H - H2
    H = H * (γ_e / math.sqrt(2)) * Ω_e

    # H3 term (tensor product with modulated Iex3/Iey3)
    mod1 = qx * torch.sin(Λ_s * t) + qy * torch.cos(Λ_s * t)
    mod2 = Θ2 * (Iex3 * torch.cos(δ2 * t - ϕ2) + Iey3 * torch.sin(δ2 * t - ϕ2))
    H3 = -math.sqrt(2) * γ_e * Ω_e * kronN(mod1, mod2)

    # S14A: auxiliary modulation
    s14_mat = torch.tensor([
        [0, torch.exp(1j * ((-Q1 + ωa1) * t)), 0],
        [torch.exp(1j * ((Q1 - ωa1) * t)), 0, torch.exp(1j * ((Q1 + ωa1) * t))],
        [0, torch.exp(1j * ((-Q1 - ωa1) * t)), 0]
    ], dtype=dtype) / math.sqrt(2)
    S14A = kronN(e0, s14_mat, I2)

    # S14C: control-field modulation
    s14_mat_c = torch.tensor([
        [0, torch.exp(1j * ((-Q1 + ω1) * -t)), 0],
        [torch.exp(1j * ((Q1 - ω1) * -t)), 0, torch.exp(1j * ((Q1 + ω1) * -t))],
        [0, torch.exp(1j * ((-Q1 - ω1) * -t)), 0]
    ], dtype=dtype) / math.sqrt(2)
    S14C = kronN(e1, s14_mat_c, I2)

    # H4 & H5: interactions with nuclear spins
    H4 = Ω_x * (γ1 * S14A + γ2 * (Ix3A * torch.cos(ωa2 * t) + Iy3A * torch.sin(ωa2 * t)))
    H5 = Ω_x * (γ1 * S14C + γ2 * (
        Ix3C * torch.cos(ω2 * t) +
        Iy3C * torch.sin(ω2 * t) +
        Iz3C * Θ2 * torch.sin(ϕ2)
    ))

    # Total Hamiltonian
    H_total = H + H3 + H4 + H5

    # Matrix exponential for unitary
    U = torch.linalg.matrix_exp(-1j * H_total * dt)

    return U





def get_U(Ω, dt, t, Δ=0.0):
    # Control fields
    Ω_e = factor * torch.cos((Λ_s + Δ) * t) * Ω[0]   # Microwave drive (electron) in Tesla
    Ω_n = Ω[1]  # Direct nuclear drive (e.g., RF amplitude in Hz)
    #Ω_n = Ω[1] if len(Ω) > 1 else 0.0
    # Diagonal modulation
    diag_cos = torch.tensor([
        torch.cos(Λ10 * t), torch.cos(Λ11 * t), torch.cos(Λ00 * t),
        torch.cos(Λ01 * t), torch.cos(Λm10 * t), torch.cos(Λm11 * t)
    ] * 2, dtype=dtype)
    diag_sin = torch.tensor([
        torch.sin(Λ10 * t), torch.sin(Λ11 * t), torch.sin(Λ00 * t),
        torch.sin(Λ01 * t), torch.sin(Λm10 * t), torch.sin(Λm11 * t)
    ] * 2, dtype=dtype)

    H_diag = Ix1 @ torch.diag(diag_cos) - Iy1 @ torch.diag(diag_sin)
    H_diag *= (γ_e / math.sqrt(2)) * Ω_e

    # H3: electron-nuclear modulation term
    mod1 = qx * torch.sin(Λ_s * t) + qy * torch.cos(Λ_s * t)
    mod2 = Θ2 * (Iex3 * torch.cos(δ2 * t - ϕ2) + Iey3 * torch.sin(δ2 * t - ϕ2))
    H3 = -math.sqrt(2) * γ_e * Ω_e * kronN(mod1, mod2)

    # S14A (electron in |0⟩) modulation
    s14_mat = torch.tensor([
        [0, torch.exp(1j * ((-Q1 + ωa1) * t)), 0],
        [torch.exp(1j * ((Q1 - ωa1) * t)), 0, torch.exp(1j * ((Q1 + ωa1) * t))],
        [0, torch.exp(1j * ((-Q1 - ωa1) * t)), 0]
    ], dtype=dtype) / math.sqrt(2)
    S14A = kronN(e0, s14_mat, I2)

    # S14C (electron in |1⟩) modulation
    s14_mat_c = torch.tensor([
        [0, torch.exp(1j * ((-Q1 + ω1) * -t)), 0],
        [torch.exp(1j * ((Q1 - ω1) * -t)), 0, torch.exp(1j * ((Q1 + ω1) * -t))],
        [0, torch.exp(1j * ((-Q1 - ω1) * -t)), 0]
    ], dtype=dtype) / math.sqrt(2)
    S14C = kronN(e1, s14_mat_c, I2)

    # H4 & H5 — interactions with nuclear spins (now scaled by Ω_n)
    H4 = Ω_n * (γ1 * S14A + γ2 * (Ix3A * torch.cos(ωa2 * t) + Iy3A * torch.sin(ωa2 * t)))
    H5 = Ω_n * (γ1 * S14C + γ2 * (
        Ix3C * torch.cos(ω2 * t) +
        Iy3C * torch.sin(ω2 * t) +
        Iz3C * Θ2 * torch.sin(ϕ2)
    ))

    # Total Hamiltonian
    H_total = H_diag + H3 + H4 + H5

    # Matrix exponential to get time evolution operator
    U = torch.linalg.matrix_exp(-1j * H_total * dt)

    return U
