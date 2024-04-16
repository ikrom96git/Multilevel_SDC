import numpy as np


def get_harmonic_oscillator_default_params(Force=False):
    if Force:
        eps = 0.2
        kappa_hat = 0.2
        problem_params = dict()
        problem_params["mu"] = kappa_hat
        problem_params["kappa"] = 1 / eps
        problem_params["F0"] = 1 /eps
        problem_params["t0"] = 0.0
        problem_params["u0"] = [2, 0]
        problem_params["dt"] = 0.5
        time = np.linspace(0, 30, 1000)
    else:
        problem_params = dict()
        problem_params["mu"] = 0
        problem_params["kappa"] = 2
        problem_params["F0"] = None
        problem_params["t0"] = 0.0
        problem_params["u0"] = np.array([2, 0])
        problem_params["dt"] = 0.5
        time = np.linspace(0, 6, 1000)
    return problem_params, time
