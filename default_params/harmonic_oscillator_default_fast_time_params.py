import numpy as np


def get_harmonic_oscillator_fast_time_params(Fast_time=False):
    eps = 0.001
    problem_params = dict()
    problem_params["mu"] = 0.0
    problem_params["kappa"] = 1.0
    problem_params["F0"] = 1.0
    problem_params["t0"] = 0.0

    if Fast_time:
        problem_params["u0"] = [2, 0]
        problem_params["dt"] = 0.1 / np.sqrt(eps)
        problem_params["F0"] = 1.0
        time = np.linspace(0, 6, 1000)
        time = time / np.sqrt(eps)
    else:
        problem_params["F0"] = 0.0
        problem_params["u0"] = [1, 0]
        problem_params["dt"] = 0.1
        time = np.linspace(0, 30, 1000)
    return problem_params, time