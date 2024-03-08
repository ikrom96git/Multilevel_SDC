from default_params.harmonic_oscillator_default_params import (
    get_harmonic_oscillator_default_params,
)
from problem_class.HarmonicOscillator import HarmonicOscillator


def get_mlsdc_default_params(Force=False):
    problem_params, *_ = get_harmonic_oscillator_default_params(Force=Force)
    collocation_params = dict()
    collocation_params["quad_type"] = "GAUSS"
    collocation_params["num_nodes"] = [5, 3]
    sweeper_params = dict()
    sweeper_params["Kiter"] = 10
    sweeper_params['coarse_solver']='no_coarse'
    sweeper_params["initial_guess"] = "collocation"
    problem_class = [HarmonicOscillator, HarmonicOscillator]
    return problem_params, collocation_params, sweeper_params, problem_class


if __name__ == "__main__":
    get_mlsdc_default_params()
