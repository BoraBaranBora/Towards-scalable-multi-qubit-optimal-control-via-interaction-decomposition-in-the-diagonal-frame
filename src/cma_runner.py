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
