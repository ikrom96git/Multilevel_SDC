import numpy as np


def get_harmonic_oscillator_reduced_model_params():
    mu = 0.001
    kappa = 0.8
    problem_params = dict()
    problem_params["mu"] = kappa
    problem_params["kappa"] = 1 / mu
    problem_params["F0"] = 1 / mu
    problem_params["t0"] = 0.0
    problem_params["u0"] = [2, 0]
    problem_params["dt"] = 0.1
    time = np.linspace(0, 1, 1000)

    return problem_params, time
