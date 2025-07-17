# evolution.py
import torch
from typing import Callable, List

# Set dtype globally
dtype = torch.complex128


def get_time_grid(duration_ns: float, steps_per_ns: float) -> torch.Tensor:
    """Return a time grid from 0 to duration (in ns), with steps_per_ns resolution."""
    steps = int(duration_ns * steps_per_ns)
    return torch.linspace(0, duration_ns * 1e-9, steps)


def get_evolution_vector(
    get_u: Callable[[List[float], float, float], torch.Tensor],
    time_grid: torch.Tensor,
    drive: List[torch.Tensor],
    psi0: torch.Tensor
) -> List[torch.Tensor]:
    """Simulate state vector evolution under drive fields."""
    dt = time_grid[1] - time_grid[0]
    states = []
    U = torch.eye(len(psi0), dtype=dtype)

    for i, t in enumerate(time_grid):
        Omega_t = [d[i] for d in drive]
        U_step = get_u(Omega_t, dt.item(), t.item())
        U = U_step @ U
        psi = U @ psi0
        states.append(psi.clone())

    return states


def get_evolution_density(
    get_u: Callable[[List[float], float, float], torch.Tensor],
    time_grid: torch.Tensor,
    drive: List[torch.Tensor],
    rho0: torch.Tensor
) -> List[torch.Tensor]:
    """Simulate density matrix evolution under drive fields."""
    dt = time_grid[1] - time_grid[0]
    states = []
    U = torch.eye(rho0.shape[0], dtype=dtype)

    for i, t in enumerate(time_grid):
        Omega_t = [d[i] for d in drive]
        U_step = get_u(Omega_t, dt.item(), t.item())
        U = U_step @ U
        rho = U @ rho0 @ U.conj().T
        states.append(rho.clone())

    return states


def get_propagator(
    get_u: Callable[[List[float], float, float], torch.Tensor],
    time_grid: torch.Tensor,
    drive: List[torch.Tensor]
) -> torch.Tensor:
    """Compute the full propagator over the entire time grid."""
    dt = time_grid[1] - time_grid[0]
    U = torch.eye(12, dtype=dtype)

    for i, t in enumerate(time_grid):
        Omega_t = [d[i] for d in drive]
        U_step = get_u(Omega_t, dt.item(), t.item())
        U = U_step @ U

    return U