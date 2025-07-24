import numpy as np


def get_simplex(parameter_set, spread=0.1):
    """Generate a simplex around the given parameter set."""
    d_n = len(parameter_set)
    unit_vectors = np.eye(d_n)
    vertices = [parameter_set + spread * uv for uv in unit_vectors]
    return [parameter_set] + vertices


def nelder_mead_iteration_step(f, simplex, values):
    α, β, γ = 1.0, 2.0, 0.5
    
    # Make sure everything is numpy
    simplex = [np.array(x) for x in simplex]

    # Order by function value (ascending)
    indices = np.argsort(values)
    simplex = [simplex[i] for i in indices]
    values = [values[i] for i in indices]

    sl = simplex[0]
    sh = simplex[-1]
    sm = np.mean(simplex[:-1], axis=0)

    sr = sm + α * (sm - sh)
    vr = f(sr)

    if vr < values[0]:  # better than best
        se = sm + β * (sr - sm)
        ve = f(se)
        if ve < vr:
            simplex[-1], values[-1] = se, ve
        else:
            simplex[-1], values[-1] = sr, vr
    elif vr >= values[-2]:  # worse than second worst
        if vr < values[-1]:
            simplex[-1], values[-1] = sr, vr
        sc = sm + γ * (sh - sm)
        vc = f(sc)
        if vc > values[-1]:
            for i in range(1, len(simplex)):
                simplex[i] = (simplex[0] + simplex[i]) / 2
                values[i] = f(simplex[i])
        else:
            simplex[-1], values[-1] = sc, vc
    else:
        simplex[-1], values[-1] = sr, vr

    return simplex, values


def nelder_mead(f, simplex, iterations, verbose=False):
    values = [f(x) for x in simplex]

    for i in range(iterations):
        simplex, values = nelder_mead_iteration_step(f, simplex, values)
        best_value = min(values)
        if verbose:
            print(f"Iteration {i+1}: best FoM = {best_value:.6e}")
    return simplex, values


def initialize_nelder_mead(f, parameter_set):
    simplex = get_simplex(parameter_set)
    values = [f(x) for x in simplex]
    return simplex, values


# Interface maps
initialization_map = {
    "Nelder Mead": initialize_nelder_mead
}

step_map = {
    "Nelder Mead": nelder_mead_iteration_step
}


def call_initialization(algo_type, f, parameter_set):
    if algo_type not in initialization_map:
        raise ValueError(f"Algorithm '{algo_type}' not available.")
    return initialization_map[algo_type](f, parameter_set)


def call_algo_step(algo_type, f, samples, values):
    if algo_type not in step_map:
        raise ValueError(f"Algorithm '{algo_type}' not available.")
    return step_map[algo_type](f, samples, values)
