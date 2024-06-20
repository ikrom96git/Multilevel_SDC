from default_params.harmonic_oscillator_default_params import (
    get_harmonic_oscillator_default_params,
)
from problem_class.HarmonicOscillator import HarmonicOscillator


def get_sdc_default_params(Force=False, eps=None):
    problem_params, *_ = get_harmonic_oscillator_default_params(Force=Force, eps=eps)
    collocation_params = dict()
    collocation_params["quad_type"] = "GAUSS"
    collocation_params["num_nodes"] = 5
    sweeper_params = dict()
    sweeper_params["Kiter"] = 10
    sweeper_params["initial_guess"] = "spread"
    problem_class = HarmonicOscillator
    return problem_params, collocation_params, sweeper_params, problem_class
