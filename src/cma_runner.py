import cma
import numpy as np


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
    """
    Initialize CMA-ES optimizer.
    Returns:
        - es: CMAEvolutionStrategy object
        - solutions: list of sampled candidates
        - values: their FoM evaluations
    """
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


def get_bounds_from_pulse_settings(pulse_settings_list):
    lower_bounds = []
    upper_bounds = []

    for ps in pulse_settings_list:
        bs = ps.basis_size

        # Amplitudes
        lower_bounds += [0.0] * bs
        upper_bounds += [ps.maximal_amplitude] * bs

        # Frequencies
        lower_bounds += [ps.minimal_frequency] * bs
        upper_bounds += [ps.maximal_frequency] * bs

        # Phases
        lower_bounds += [-ps.maximal_phase] * bs
        upper_bounds += [ps.maximal_phase] * bs

    return np.array(lower_bounds), np.array(upper_bounds)


def initialize_cmaes(f, parameter_set, pulse_settings_list, sigma_init=0.1):
    scale = get_scaling_from_pulse_settings(pulse_settings_list)
    x0_norm = normalize_params(parameter_set, scale)

    # Get bounds and normalize them
    lower_bounds, upper_bounds = get_bounds_from_pulse_settings(pulse_settings_list)
    lower_bounds_norm = normalize_params(lower_bounds, scale)
    upper_bounds_norm = normalize_params(upper_bounds, scale)

    options = {
        'verb_log': 0,
        'verbose': -9,
        'CMA_stds': np.ones_like(scale),
        'bounds': [lower_bounds_norm.tolist(), upper_bounds_norm.tolist()],
    }

    es = cma.CMAEvolutionStrategy(x0_norm.tolist(), sigma_init, options)
    solutions_norm = es.ask()
    solutions = [unnormalize_params(x, scale) for x in solutions_norm]
    values = [f(x) for x in solutions]
    return es, solutions_norm, values, scale




def cmaes_iteration_step(f, es, solutions_norm, values, scale):
    """
    Perform one CMA-ES step:
        - Tell optimizer current solutions and values
        - Ask for new solutions
        - Evaluate them
    Returns:
        - Updated (es, solutions, values)
    """
    es.tell(solutions_norm, values)
        # Clamp step size if it grows too large
    MAX_SIGMA = 0.2
    if es.sigma > MAX_SIGMA:
        es.sigma = MAX_SIGMA
    new_solutions_norm = es.ask()
    new_solutions = [unnormalize_params(x, scale) for x in new_solutions_norm]
    new_values = [f(x) for x in new_solutions]
    return es, new_solutions, new_values



def cmaes_iteration_step(f, es, solutions_norm, values, scale):
    try:
        es.tell(solutions_norm, values)
    except ValueError as e:
        print("CMA-ES tell() failed due to invalid solution:", e)
        print("Trying to clip values into bounds...")

        # Clip solutions_norm into bounds (defensive repair)
        bounds = es.opts.get('bounds')
        if bounds is not None:
            lb, ub = np.array(bounds[0]), np.array(bounds[1])
            solutions_norm = [np.clip(sol, lb, ub) for sol in solutions_norm]
            es.tell(solutions_norm, values)
        else:
            raise e  # No bounds? Then we can't fix this

    # Optionally clamp step size to avoid runaway
    #MAX_SIGMA = 0.5
    #if es.sigma > MAX_SIGMA:
    #    es.sigma = MAX_SIGMA

    new_solutions_norm = es.ask()
    new_solutions = [unnormalize_params(x, scale) for x in new_solutions_norm]
    new_values = [f(x) for x in new_solutions]

    return es, new_solutions_norm, new_values