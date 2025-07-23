import cma
import numpy as np


def initialize_cmaes(f, parameter_set, sigma_init=0.1):
    """
    Initialize CMA-ES optimizer.
    Returns:
        - es: CMAEvolutionStrategy object
        - solutions: list of sampled candidates
        - values: their FoM evaluations
    """
    es = cma.CMAEvolutionStrategy(parameter_set.tolist(), sigma_init, {'verb_log': 0, 'verbose': -9})
    solutions = es.ask()
    values = [f(np.array(x)) for x in solutions]
    return es, solutions, values


def cmaes_iteration_step(f, es, solutions, values):
    """
    Perform one CMA-ES step:
        - Tell optimizer current solutions and values
        - Ask for new solutions
        - Evaluate them
    Returns:
        - Updated (es, solutions, values)
    """
    es.tell(solutions, values)
    solutions = es.ask()
    values = [f(np.array(x)) for x in solutions]
    return es, solutions, values


def get_scaling_from_pulse_settings(pulse_settings_list):
    scale_factors = []

    for ps in pulse_settings_list:
        bs = ps.basis_size

        # Amplitudes: [0, ..., 0] + [max_amplitude, ..., max_amplitude]
        scale_factors += [ps.maximal_amplitude] * bs
        scale_factors += [max(abs(ps.maximal_frequency), abs(ps.minimal_frequency))] * bs
        scale_factors += [ps.maximal_phase] * bs

    return np.array(scale_factors)


def normalize_params(params, scale):
    return np.array(params) / scale

def unnormalize_params(norm_params, scale):
    return np.array(norm_params) * scale


def initialize_cmaes(f, parameter_set, pulse_settings_list, sigma_init=0.1):
    scale = get_scaling_from_pulse_settings(pulse_settings_list)
    x0_norm = normalize_params(parameter_set, scale)

    options = {
        'verb_log': 0,
        'verbose': -9,
        'CMA_stds': np.ones_like(scale)
    }

    es = cma.CMAEvolutionStrategy(x0_norm.tolist(), sigma_init, options)
    solutions_norm = es.ask()
    solutions = [unnormalize_params(x, scale) for x in solutions_norm]
    values = [f(x) for x in solutions]
    return es, solutions_norm, values, scale


def cmaes_iteration_step(f, es, solutions_norm, values, scale):
    es.tell(solutions_norm, values)
    new_solutions_norm = es.ask()
    new_solutions = [unnormalize_params(x, scale) for x in new_solutions_norm]
    new_values = [f(x) for x in new_solutions]
    return es, new_solutions_norm, new_values
