from default_params.harmonic_oscillator_default_params import (
    get_harmonic_oscillator_default_params,
)
from problem_class.HarmonicOscillator import HarmonicOscillator


def get_mlsdc_default_params(Force=False, eps=None):
    """
    Get default parameters for a Multilevel Spectral Deferred Correction (MLSDC) method.

    Args:
        Force (bool): Flag to force the computation of default parameters.

    Returns:
        tuple: A tuple containing problem_params (dict), collocation_params (dict), sweeper_params (dict), and problem_class (list).

    Examples:
        problem_params, collocation_params, sweeper_params, problem_class = get_mlsdc_default_params(Force=True)
    """

    problem_params, *_ = get_harmonic_oscillator_default_params(Force=Force, eps=eps)
    collocation_params = dict()
    collocation_params["quad_type"] = "GAUSS"
    collocation_params["num_nodes"] = [5, 5]
    sweeper_params = dict()
    sweeper_params["Kiter"] = 8
    sweeper_params["coarse_solver"] = "sdc"
    sweeper_params["initial_guess"] = "spread"
    problem_class = [HarmonicOscillator, HarmonicOscillator]
    return problem_params, collocation_params, sweeper_params, problem_class


if __name__ == "__main__":
    get_mlsdc_default_params()
