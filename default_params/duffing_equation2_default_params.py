import numpy as np

dt = 0.4


def get_duffing_equation_params(eps=None):
    problem_params = dict()
    problem_params["omega"] = 1.0
    problem_params["eps"] = eps
    problem_params["u0"] = [0, 2]
    problem_params["dt"] = dt
    problem_params["t0"] = 0.0
    return problem_params


def get_duffing_zeros_order_params(eps=None):
    problem_params = dict()
    problem_params["omega"] = 1.0
    problem_params["eps"] = eps
    problem_params["u0"] = [0, 2]
    problem_params["dt"] = dt
    problem_params["t0"] = 0.0
    return problem_params


def get_duffing_first_order_params(eps=None):
    problem_params = dict()
    problem_params["omega"] = 1.0
    problem_params["eps"] = eps
    problem_params["u0"] = [0, 0]
    problem_params["dt"] = dt
    problem_params["t0"] = 0.0
    return problem_params
