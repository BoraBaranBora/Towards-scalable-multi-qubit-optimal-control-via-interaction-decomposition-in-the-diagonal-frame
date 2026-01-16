# ──────────────────────────────────────────────────────────────────────────────
# Multi-13C NV model (RWA), plug-compatible get_U_RWA
# ──────────────────────────────────────────────────────────────────────────────
import math
import torch

# ── global dtype/device
dtype  = torch.complex128
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── constants (your values)
Gauss_to_Tesla = 0.0001
B_0 = torch.tensor(4500.0 * Gauss_to_Tesla, dtype=torch.float64)
D   = torch.tensor(2.87e3, dtype=torch.float64)     # GHz
γ_e = torch.tensor(28.0e3, dtype=torch.float64)     # GHz/T

# index 0 = 14N, 1.. = 13C's
γ_n   = torch.tensor([3.077, 10.71, 10.71], dtype=torch.float64)  # MHz/T
Azz_n = torch.tensor([-2.14, 2.281, -1.011], dtype=torch.float64) # MHz
A_ort_n = torch.tensor([0.0, 0.24, 0.014], dtype=torch.float64)     # MHz
Q_n   = torch.tensor([-5.01, 0.0, 0.0], dtype=torch.float64)           # MHz
ϕ_n   = torch.tensor([0.0, 0.0], dtype=torch.float64)                            # rad (given)

# ── frequency helpers (return MHz unless noted)
def ω_n(i):
    # √[(Azz + γ B0)^2 + A_ort^2]  in MHz
    term = Azz_n[i] + γ_n[i] * B_0
    return torch.sqrt(torch.abs(term)*2 + torch.abs(A_ort_n[i])*2)

def ω_na(i):   # "aux" nuclear frequency (no transverse HF), in MHz
    return γ_n[i] * B_0

def θ(i):      # small misalignment angle (rad)
    return torch.atan2(A_ort_n[i], Azz_n[i] + γ_n[i] * B_0)

def ϵ(i):      # gauge phase from θ,ϕ (rad)
    th  = θ(i)
    phi = ϕ_n[i] if i < len(ϕ_n) else torch.tensor(0.0, dtype=torch.float64)
    num = (1 - torch.cos(th)) * torch.tan(phi)
    den = 1 + torch.cos(th) * torch.tan(phi)**2
    return torch.atan2(num, den)

def δ(i):      # half-splitting (MHz) difference aux vs comp
    return (ω_n(i) - ω_na(i)) / 2

# ── basis transforms
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

# ── spin operators
def spin_operators():
    qx = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
    qy = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype)
    qz = torch.tensor([[1, 0], [0, -1]], dtype=dtype)
    sx = 0.5 * qx; sy = 0.5 * qy; sz = 0.5 * qz
    I2 = torch.eye(2, dtype=dtype); I3 = torch.eye(3, dtype=dtype)
    sx3 = torch.tensor([[0, 1, 0],[1, 0, 1],[0, 1, 0]], dtype=dtype) / math.sqrt(2)
    sy3 = torch.tensor([[0,-1j,0],[1j,0,-1j],[0,1j,0]], dtype=dtype) / math.sqrt(2)
    sz3 = torch.tensor([[1,0,0],[0,0,0],[0,0,-1]], dtype=dtype)
    return {'qx':qx,'qy':qy,'qz':qz,'sx':sx,'sy':sy,'sz':sz,'I2':I2,'I3':I3,'sx3':sx3,'sy3':sy3,'sz3':sz3}

ops = spin_operators()
qx, qy, qz = ops['qx'], ops['qy'], ops['qz']
sx, sy, sz = ops['sx'], ops['sy'], ops['sz']
I2, I3     = ops['I2'], ops['I3']
sx3, sy3, sz3 = ops['sx3'], ops['sy3'], ops['sz3']

# ── convenient kron
def kronN(*args):
    out = args[0]
    for a in args[1:]:
        out = torch.einsum('ab,cd->acbd', out, a).reshape(out.size(0)*a.size(0), out.size(1)*a.size(1))
    return out

def _kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = torch.einsum('ab,cd->acbd', out, m).reshape(out.size(0)*m.size(0), out.size(1)*m.size(1))
    return out

def _embed_carbon_op(op_k, k, N_C, I2_car):
    mats = [op_k if j==k else I2_car for j in range(N_C)]
    return _kron_all(mats)

# ── electron projectors |0><0|, |-1><-1|
e0 = torch.tensor([[1.0+0j, 0],[0, 0]], dtype=dtype)
e1 = torch.tensor([[0, 0],[0, 1.0+0j]], dtype=dtype)

# ── convert to rad/s
two_pi = 2*math.pi
γ_e = γ_e * two_pi                 # rad/s/T
γ1  = γ_n[0] * two_pi * 1e6       # 14N gyromag in rad/s/T

ωa1 = ω_na(0) * two_pi * 1e6      # 14N aux  (rad/s)
ω1  = ω_n(0)  * two_pi * 1e6      # 14N comp (rad/s)
Q1  = Q_n[0]  * two_pi * 1e6      # 14N quadrupole (rad/s)

# NOTE: the remaining carbons will be handled generically below

# ── electron carrier reference
Λ_s = (γ_e * B_0 - D * two_pi * 1e9)  # D in GHz → rad/s

# ── (optional) single-carbon Λ labels for backwards checks (rad/s)
#    These follow your latest sign convention and are not used in the multi-C Λ table.
Λ00 = Λ_s - (0  * (γ_n[0]*B_0 - ω_n(0)) + 0.5 * (γ_n[1]*B_0 - ω_n(1))) * two_pi * 1e6
Λ01 = Λ_s - (0  * (γ_n[0]*B_0 - ω_n(0)) - 0.5 * (γ_n[1]*B_0 - ω_n(1))) * two_pi * 1e6
Λ10 = Λ_s - (-1 * (γ_n[0]*B_0 - ω_n(0)) + 0.5 * (γ_n[1]*B_0 - ω_n(1))) * two_pi * 1e6
Λ11 = Λ_s - (-1 * (γ_n[0]*B_0 - ω_n(0)) - 0.5 * (γ_n[1]*B_0 - ω_n(1))) * two_pi * 1e6
Λm10= Λ_s - ( 1 * (γ_n[0]*B_0 - ω_n(0)) + 0.5 * (γ_n[1]*B_0 - ω_n(1))) * two_pi * 1e6
Λm11= Λ_s - ( 1 * (γ_n[0]*B_0 - ω_n(0)) - 0.5 * (γ_n[1]*B_0 - ω_n(1))) * two_pi * 1e6

# ── amplitude scale (kept for compatibility with your code)
factor = 1.0

# ──────────────────────────────────────────────────────────────────────────────
#   Λ table for many 13C spins (state-resolved over all carbons)
# ──────────────────────────────────────────────────────────────────────────────
def build_Lambda_nuc_vec_multi(c_indices=None, dev=device):
    """
    Build Λ_nuc (rad/s) over the nuclear space 3×2^N_C (N14 mI = +1,0,−1; carbons in binary order).
    This excludes the electron factor (so length = 3*2^N_C). Use with qx⊗diag(Λ) in H_e^(RWA).
    """
    if c_indices is None:
        c_indices = list(range(1, int(len(γ_n))))  # all available carbons
    N_C = len(c_indices)
    if N_C < 1:
        raise ValueError("Need at least one carbon index (c_indices).")

    two_pi_MHz = 2*math.pi*1e6

    # Δ_N, Δ_Ck in MHz
    Δ_N  = (γ_n[0] * B_0 - ω_n(0))
    Δ_Ck = [ (γ_n[ck] * B_0 - ω_n(ck)) for ck in c_indices ]

    mI_list = [+1, 0, -1]  # order: +1, 0, −1 (matches your grouping)
    Λ_list = []
    for mI in mI_list:
        for bits in range(2**N_C):
            # bit 0 -> ↑ (+1), bit 1 -> ↓ (-1)
            s_sum = 0.0
            for k, ck in enumerate(c_indices):
                s_k = +1.0 if ((bits >> k) & 1) == 0 else -1.0
                s_sum += 0.5 * s_k * Δ_Ck[k]
            Δ_total_MHz = (mI * Δ_N) + s_sum
            Λ_here = Λ_s - (Δ_total_MHz * two_pi_MHz)
            Λ_list.append(Λ_here)

    Λ_nuc = torch.stack(Λ_list, dim=0).to(dtype=dtype, device=dev)  # length 3*2^N_C
    return Λ_nuc

# ──────────────────────────────────────────────────────────────────────────────
#   Precompute blocks for many carbons
# ──────────────────────────────────────────────────────────────────────────────
def build_multiC_precomp(c_indices=None, use_rot_comp_basis=True, dev=device):
    """
    Precompute everything needed by the RWA Hamiltonian for many 13C spins.
    Returns dict 'pc' with:
      N_C, c_indices, dim_nuc, I_e, I_N, I_Creg,
      per-carbon lists Xaux/Yaux and Xcom/Ycom/Zcom,
      γC, ωaC, ωC, δC, ΘC, ϕC,
      S14A_core(t, ω_RF, cutoff), S14C_core(t, ω_RF, cutoff),
      Λ_nuc   (length 3*2^N_C, rad/s)
    """
    if c_indices is None:
        c_indices = list(range(1, int(len(γ_n))))
    N_C = len(c_indices)
    if N_C < 1:
        raise ValueError("Need at least one carbon index.")

    _dt = dtype
    I_e  = torch.eye(2, dtype=_dt, device=dev)
    I_N  = I3.to(dtype=_dt, device=dev)
    I2c  = torch.eye(2, dtype=_dt, device=dev)
    I_Creg = torch.eye(2**N_C, dtype=_dt, device=dev)

    _sx = sx.to(dtype=_dt, device=dev)
    _sy = sy.to(dtype=_dt, device=dev)
    _sz = sz.to(dtype=_dt, device=dev)

    # per-carbon parameters (rad/s)
    γC_vals, ωaC_vals, ωC_vals, δC_vals, ΘC_vals, ϕC_vals = [], [], [], [], [], []
    for ck in c_indices:
        γC_vals .append( (γ_n[ck] * two_pi * 1e6) )
        ωaC_vals.append( (ω_na(ck) * two_pi * 1e6) )
        ωC_vals .append( (ω_n(ck)  * two_pi * 1e6) )
        δC_vals .append( (δ(ck)    * two_pi * 1e6) )
        ΘC_vals .append( θ(ck) )
        ϕC_vals .append( ϕ_n[ck] if ck < len(ϕ_n) else torch.tensor(0.0, dtype=torch.float64) )

    γC  = torch.stack([torch.as_tensor(x, dtype=torch.float64, device=dev) for x in γC_vals])
    ωaC = torch.stack([torch.as_tensor(x, dtype=torch.float64, device=dev) for x in ωaC_vals])
    ωC  = torch.stack([torch.as_tensor(x, dtype=torch.float64, device=dev) for x in ωC_vals])
    δC  = torch.stack([torch.as_tensor(x, dtype=torch.float64, device=dev) for x in δC_vals])
    ΘC  = torch.stack([torch.as_tensor(x, dtype=torch.float64, device=dev) for x in ΘC_vals])
    ϕC  = torch.stack([torch.as_tensor(x, dtype=torch.float64, device=dev) for x in ϕC_vals])

    # embeddings on carbon register
    Xaux_list, Yaux_list = [], []
    Xcom_list, Ycom_list, Zcom_list = [], [], []
    for k in range(N_C):
        # aux: bare σ on carbon k
        sx_k = _embed_carbon_op(_sx, k, N_C, I2c)
        sy_k = _embed_carbon_op(_sy, k, N_C, I2c)
        Xaux_list.append(_kron_all([I_N, sx_k]))
        Yaux_list.append(_kron_all([I_N, sy_k]))

        # comp: rotated or bare
        if use_rot_comp_basis:
            B2k = Trafo2(ΘC[k], ϕC[k], ϵ(c_indices[k])).to(dtype=_dt, device=dev)
            sx_comp = B2k.conj().T @ _sx @ B2k
            sy_comp = B2k.conj().T @ _sy @ B2k
            sz_comp = B2k.conj().T @ _sz @ B2k
        else:
            sx_comp, sy_comp, sz_comp = _sx, _sy, _sz

        sx_kc = _embed_carbon_op(sx_comp, k, N_C, I2c)
        sy_kc = _embed_carbon_op(sy_comp, k, N_C, I2c)
        sz_kc = _embed_carbon_op(sz_comp, k, N_C, I2c)
        Xcom_list.append(_kron_all([I_N, sx_kc]))
        Ycom_list.append(_kron_all([I_N, sy_kc]))
        Zcom_list.append(_kron_all([I_N, sz_kc]))

    # 14N time-dependent cores (rad/s), pruned near ω_RF
    Q1_Hz  = Q1.to(device=dev)
    ω1_Hz  = ω1.to(device=dev)
    ωa1_Hz = ωa1.to(device=dev)

    def _keep_phase(Ω_val, t, ω_RF=None, cutoff=1e3):
        if ω_RF is None:
            return torch.exp(1j * (Ω_val * t))
        δloc = Ω_val - ω_RF
        return torch.exp(1j * (δloc * t)) if abs(float(δloc)) < cutoff else \
               torch.zeros((), dtype=_dt, device=dev)

    def S14C_core(t, ω_RF=None, cutoff=1e3):
        Ωvals_C = {(0,1): -( -Q1_Hz + ω1_Hz ), (1,0): -(  Q1_Hz - ω1_Hz ),
                   (1,2): -(  Q1_Hz + ω1_Hz ), (2,1): -( -Q1_Hz - ω1_Hz )}
        s = torch.zeros((3,3), dtype=_dt, device=dev)
        for (i,j), Ωv in Ωvals_C.items():
            s[i,j] = _keep_phase(Ωv, t, ω_RF, cutoff)
        return s / math.sqrt(2)

    def S14A_core(t, ω_RF=None, cutoff=1e3):
        Ωvals_A = {(0,1): ( -Q1_Hz + ωa1_Hz ), (1,0): (  Q1_Hz - ωa1_Hz ),
                   (1,2): (  Q1_Hz + ωa1_Hz ), (2,1): ( -Q1_Hz - ωa1_Hz )}
        s = torch.zeros((3,3), dtype=_dt, device=dev)
        for (i,j), Ωv in Ωvals_A.items():
            s[i,j] = _keep_phase(Ωv, t, ω_RF, cutoff)
        return s / math.sqrt(2)

    # Λ over nuclear space only (3×2^N_C), rad/s
    Λ_nuc = build_Lambda_nuc_vec_multi(c_indices=c_indices, dev=dev)

    dim_nuc = 3 * (2**N_C)
    return dict(
        N_C=N_C, c_indices=c_indices,
        dim_nuc=dim_nuc,
        I_e=I_e, I_N=I_N, I_Creg=I_Creg,
        Xaux_list=Xaux_list, Yaux_list=Yaux_list,
        Xcom_list=Xcom_list, Ycom_list=Ycom_list, Zcom_list=Zcom_list,
        γC=γC, ωaC=ωaC, ωC=ωC, δC=δC, ΘC=ΘC, ϕC=ϕC,
        S14A_core=S14A_core, S14C_core=S14C_core,
        Λ_nuc=Λ_nuc
    )

# ──────────────────────────────────────────────────────────────────────────────
#   RWA propagator for many carbons
# ──────────────────────────────────────────────────────────────────────────────
def get_U_RWA_multi(
    Ω, dt, t,
    Δ_e=0.0,               # electron MW detuning (rad/s)
    ω_RF=None,             # single RF carrier (rad/s)
    detuning_cutoff=1e3,   # keep 14N Fourier terms with |Ω-ω_RF| < cutoff
    pc=None,               # precomp dict from build_multiC_precomp()
    include_eC_mod=True    # include MW-driven e–C modulation pathway
):
    assert pc is not None, "Pass 'pc' from build_multiC_precomp() or use module-level get_U_RWA."
    _dt = dtype; dev = device

    # controls
    Ω_e0 = Ω[0]  # MW amplitude (Tesla)
    Ω_n  = Ω[1]  # RF amplitude (Tesla)

    # (1) Electron drive H_e^(RWA) = (γ_e/√2)*(Ω_e0*factor/2) * [ qx⊗cos(Δe*t) - qy⊗sin(Δe*t) ]
    Δ_e = torch.as_tensor(Δ_e, dtype=_dt, device=dev)
    Δe_nuc = pc['Λ_nuc'] - (Λ_s.to(dtype=_dt, device=dev) + Δ_e)  # length = dim_nuc
    cos_vec = torch.cos(Δe_nuc * t)
    sin_vec = torch.sin(Δe_nuc * t)

    Ddiag = torch.diag(cos_vec)
    Sdiag = torch.diag(sin_vec)

    Hx = kronN(qx.to(dtype=_dt, device=dev), torch.eye(pc['dim_nuc'], dtype=_dt, device=dev))
    Hy = kronN(qy.to(dtype=_dt, device=dev), torch.eye(pc['dim_nuc'], dtype=_dt, device=dev))

    H_e_RWA = (Hx @ kronN(torch.eye(2, dtype=_dt, device=dev), Ddiag)) - \
              (Hy @ kronN(torch.eye(2, dtype=_dt, device=dev), Sdiag))
    H_e_RWA = ((γ_e / math.sqrt(2)) * 0.5 * factor * Ω_e0) * H_e_RWA

    # (2) MW-driven e–C modulation H_eN^(RWA)  ~  -√2 γ_e Ω_e0 * [ (qx sinΔe t + qy cosΔe t)/2 ⊗ Σ_k Θ_k (Xk cos(δk t − φC) + Yk sin(...)) ]
    if include_eC_mod:
        mod1 = 0.5 * ( qx.to(dtype=_dt, device=dev) * torch.sin(Δ_e * t) +
                       qy.to(dtype=_dt, device=dev) * torch.cos(Δ_e * t) )
        mod2 = 0.0

        for k in range(pc['N_C']): 
            Θk, φC, δk = pc['ΘC'][k], pc['φC'][k], pc['δC'][k]
            Xk, Yk = pc['Xaux_list'][k], pc['Yaux_list'][k]  # modulation uses transverse components
            mod2 = mod2 + Θk * ( Xk * torch.cos(δk*t - φC) + Yk * torch.sin(δk*t - φC) )
        H_eN_RWA = -math.sqrt(2) * γ_e * (factor * Ω_e0) * kronN(mod1, mod2)
    else:
        H_eN_RWA = 0.0 * H_e_RWA

    # (3) 14N blocks with RF transverse drives (sum over carbons)
    #S14A = kronN(e1.to(dtype=_dt, device=dev), _kron_all([pc['S14A_core'](t, ω_RF, detuning_cutoff), pc['I_Creg']]))
    #S14C = kronN(e0.to(dtype=_dt, device=dev), _kron_all([pc['S14C_core'](t, ω_RF, detuning_cutoff), pc['I_Creg']]))

    # (3) 14N blocks (NUCLEAR-ONLY). Electron embedding happens below with P_e0, P_e1.
    S14A = _kron_all([pc['S14A_core'](t, ω_RF, detuning_cutoff), pc['I_Creg']])  # shape: dim_nuc × dim_nuc
    S14C = _kron_all([pc['S14C_core'](t, ω_RF, detuning_cutoff), pc['I_Creg']])  # shape: dim_nuc × dim_nuc

    # RF drives in aux (e=|0>) and comp (e=|-1>) manifolds
    H_aux_RF = 0.0
    H_com_RF = 0.0
    for k in range(pc['N_C']):
        γk  = pc['γC'][k]
        ωak = pc['ωaC'][k]
        ωk  = pc['ωC'][k]
        δ_aux = (ωak - ω_RF) if (ω_RF is not None) else ωak
        δ_com = ( ωk - ω_RF) if (ω_RF is not None) else  ωk

        Xaux, Yaux = pc['Xaux_list'][k], pc['Yaux_list'][k]
        Xcom, Ycom, Zcom = pc['Xcom_list'][k], pc['Ycom_list'][k], pc['Zcom_list'][k]

        H_aux_RF = H_aux_RF + γk * 0.5 * ( Xaux * torch.cos(δ_aux*t) + Yaux * torch.sin(δ_aux*t) )
        # static longitudinal piece in comp manifold
        H_com_RF = H_com_RF + γk * 0.5 * ( Xcom * torch.cos(δ_com*t) + Ycom * torch.sin(δ_com*t) ) \
                             + γk * ( Zcom * pc['ΘC'][k] * torch.sin(pc['φC'][k]) )

    H_aux_RWA  = Ω_n * ( γ1 * S14A + H_aux_RF )
    H_comp_RWA = Ω_n * ( γ1 * S14C + H_com_RF )

    # place aux/comp blocks using electron projectors
    P_e0 = kronN(e0.to(dtype=_dt, device=dev), torch.eye(pc['dim_nuc'], dtype=_dt, device=dev))
    P_e1 = kronN(e1.to(dtype=_dt, device=dev), torch.eye(pc['dim_nuc'], dtype=_dt, device=dev))

    H_eff = H_e_RWA + H_eN_RWA \
          + P_e0 @ kronN(torch.eye(2, dtype=_dt, device=dev), H_aux_RWA)  @ P_e0 \
          + P_e1 @ kronN(torch.eye(2, dtype=_dt, device=dev), H_comp_RWA) @ P_e1

    U = torch.linalg.matrix_exp(-1j * H_eff * dt)
    return U

# ──────────────────────────────────────────────────────────────────────────────
#   Module-level precomp + plug-compatible wrapper
# ──────────────────────────────────────────────────────────────────────────────
# Build once (use all available carbons 1..len(γ_n)-1). Change c_indices to subset if desired.
_ACTIVE_C_INDICES = list(range(1, int(len(γ_n))))
_PC = build_multiC_precomp(c_indices=_ACTIVE_C_INDICES, use_rot_comp_basis=True, dev=device)

def get_U_RWA(Ω, dt, t, Δ_e=0.0, ω_RF=None, detuning_cutoff=1e3):
    """
    Plug-compatible wrapper so your optimization scripts can just import get_U_RWA.
    """
    return get_U_RWA_multi(Ω, dt, t, Δ_e=Δ_e, ω_RF=ω_RF, detuning_cutoff=detuning_cutoff, pc=_PC, include_eC_mod=True)


def set_active_carbons(c_indices):
    """
    Rebuilds the precomputation for the given 13C index list (subset of 1..len(γ_n)-1).
    Example: set_active_carbons([1,2]) -> use only the first two carbons from your parameter arrays.
    """
    global _ACTIVE_C_INDICES, _PC
    if not isinstance(c_indices, (list, tuple)) or not c_indices:
        raise ValueError("c_indices must be a non-empty list/tuple of valid 13C indices (>=1).")
    # basic sanity: indices are within the declared γ_n/Azz_n/etc. arrays
    for ck in c_indices:
        if ck < 1 or ck >= int(len(γ_n)):
            raise ValueError(f"Carbon index {ck} out of range; valid are 1..{int(len(γ_n))-1}.")
    _ACTIVE_C_INDICES = list(c_indices)
    _PC = build_multiC_precomp(c_indices=_ACTIVE_C_INDICES, use_rot_comp_basis=True, dev=device)

def get_active_carbons():
    """Returns the current list of active carbon indices (module-level)."""
    return list(_ACTIVE_C_INDICES)

def get_precomp():
    """Returns the current precomp dict (_PC). Useful for debugging or sizing."""
    return _PC

def detuning_for_target_all_up():
    """
    Δ_e (rad/s) for |0_e, m_I=+1, all carbons ↑>, consistent with the active carbon set.
    Λ_target = Λ_s - [ Δ_N + 0.5 * Σ_active Δ_Ck ] * 2π*1e6 ;  Δ_e = Λ_target - Λ_s.
    """
    two_pi_1e6 = 2 * math.pi * 1e6
    # Δ_N, Δ_Ck in MHz
    Δ_N  = (γ_n[0] * B_0 - ω_n(0))
    Δ_sum = 0.0
    for ck in _ACTIVE_C_INDICES:
        Δ_sum = Δ_sum + (γ_n[ck] * B_0 - ω_n(ck))
    Δ_total_MHz = Δ_N + 0.5 * Δ_sum
    return - float(Δ_total_MHz) * two_pi_1e6