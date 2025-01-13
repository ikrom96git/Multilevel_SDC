import numpy as np

dt = 0.1


def get_harmonic_oscillator_params(eps=None):
    kappa_hat = 0.8
    problem_params = dict()
    problem_params["mu"] = kappa_hat
    problem_params["kappa"] = 1 / eps
    problem_params["F0"] = 1 / eps
    problem_params["t0"] = 0.0
    problem_params["u0"] = [2, 0]
    problem_params["dt"] = dt/eps
    time = np.linspace(0, 2*np.pi, 1000)
    return problem_params, time


def get_harmonic_oscillator_zeros_order_params(eps=None):

    problem_params = dict()
    problem_params["mu"] = 0.0
    problem_params["kappa"] = 1.0
    problem_params["F0"] = 1.0
    problem_params["t0"] = 0.0

    # if Fast_time:

    problem_params["u0"] = [2, 0]
    problem_params["dt"] = dt / np.sqrt(eps)
    problem_params["F0"] = 1.0
    time = np.linspace(0,  2*np.pi, 1000)
    time = time / np.sqrt(eps)
    # else:
    #     problem_params["F0"] = 0.0
    #     problem_params["u0"] = [1, 0]
    #     problem_params["dt"] = 0.5
    #     time = np.linspace(0, 30, 1000)
    return problem_params, time


def get_harmonic_oscillator_first_order_params(eps=None):

    problem_params = dict()
    problem_params["mu"] = 0.0
    problem_params["kappa"] = 1.0
    problem_params["F0"] = 1.0
    problem_params["t0"] = 0.0

    # if Fast_time:

    problem_params["u0"] = [0, 0]
    problem_params["dt"] = dt / np.sqrt(eps)
    problem_params["F0"] = 1.0
    time = np.linspace(0,  2*np.pi, 1000)
    time = time / np.sqrt(eps)
    # else:
    #     problem_params["F0"] = 0.0
    #     problem_params["u0"] = [1, 0]
    #     problem_params["dt"] = 0.5
    #     time = np.linspace(0, 30, 1000)
    return problem_params, time
