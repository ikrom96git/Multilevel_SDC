import numpy as np
eps_fast_time=0.1

def get_harmonic_oscillator_default_params(Force="Fast_time", eps=None):
    if Force == True:
        eps = 0.2
        kappa_hat = 0.2
        problem_params = dict()
        problem_params["mu"] = kappa_hat
        problem_params["kappa"] = 1 / eps
        problem_params["F0"] = 1 / eps
        problem_params["t0"] = 0.0
        problem_params["u0"] = [2, 0]
        problem_params["dt"] = 1.0
        time = np.linspace(0, 30, 1000)
    elif Force == "Fast_time":

        # eps = eps
        kappa_hat = 0.8
        problem_params = dict()
        problem_params["mu"] = kappa_hat
        problem_params["kappa"] = 1 / eps
        problem_params["F0"] = 1 / eps
        problem_params["t0"] = 0.0
        problem_params["u0"] = [2, 0]
        problem_params["dt"] = .1
        time = np.linspace(0, 30, 1000)
    else:
        eps = None
        problem_params = dict()
        problem_params["mu"] = 0
        problem_params["kappa"] = 2
        problem_params["F0"] = None
        problem_params["t0"] = 0.0
        problem_params["u0"] = np.array([2, 0])
        problem_params["dt"] = 0.5
        time = np.linspace(0, 6, 1000)
    return problem_params, time, eps
