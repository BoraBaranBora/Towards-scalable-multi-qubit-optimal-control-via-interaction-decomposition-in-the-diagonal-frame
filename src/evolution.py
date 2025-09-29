# evolution.py
import torch
from typing import Callable, List

# Set dtype globally
dtype = torch.complex128


def get_time_grid(duration_ns: float, steps_per_ns: float) -> torch.Tensor:
    """Return a time grid from 0 to duration (in ns), with steps_per_ns resolution."""
    steps = int(duration_ns * steps_per_ns)
    return torch.linspace(0, duration_ns * 1e-9, steps)

def _dt_t(time_grid, k):
    dt = (time_grid[k+1] - time_grid[k]).item()
    t  = time_grid[k].item()
    return dt, t

def get_propagator(
    get_u: Callable[[List[float], float, float], torch.Tensor],
    time_grid: torch.Tensor,
    drive: List[torch.Tensor]
) -> torch.Tensor:
    """
    Drive: list/tuple of control arrays, each same length as time_grid.
    Dimension is inferred from the first U_step.
    """
    U = None
    for k in range(len(time_grid) - 1):
        dt, t = _dt_t(time_grid, k)
        # slice controls at step k
        Omega_t = [d[k] for d in drive]
        U_step = get_u(Omega_t, dt, t)   # must be square (n×n)

        if U is None:
            n = U_step.shape[0]
            U = torch.eye(n, dtype=U_step.dtype, device=U_step.device)

        U = U_step @ U
    return U

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