from dataclasses import dataclass

@dataclass
class PulseSettings:
    basis_type: str
    basis_size: int
    maximal_pulse: float
    maximal_amplitude: float
    maximal_frequency: float
    minimal_frequency: float
    maximal_phase: float
    channel_type: str  # Optional field with default

