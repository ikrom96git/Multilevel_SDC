import numpy as np
from default_params.harmonic_oscillator_default_fast_time_params import (
    get_harmonic_oscillator_fast_time_params,
)

from default_params.mlsdc_defautl_params import get_mlsdc_default_params
from problem_class.HarmonicOscillator_fast_time_reduced_problem import (
    HarmonicOscillator_fast_time,
    HarmonicOscillator_fast_time_first_order,
)
from sweeper_class.mlsdc_class import Mlsdc_class
from copy import deepcopy
from plot_class.plot_residual import plot_residual
from sweeper_class.sdc_class import sdc_class
from default_params.sdc_default_params import get_sdc_default_params
from default_params.harmonic_oscillator_default_params import eps_fast_time
def test_mlsdc_diff_eps():
    eps_set=[.1, 0.01, 0.001, 0.0001]
    residual=[]
    for ee in range(len(eps_set)):
        eps_fast_time=eps_set[ee]
        problem_params, collocation_params, sweeper_params, problem_class = (
            get_mlsdc_default_params("Fast_time", eps=eps_fast_time)
        )
        problem_fast_time_params, *_ = get_harmonic_oscillator_fast_time_params(
            Fast_time=True, eps=eps_fast_time
        )
        problem_class_zeros = [
            problem_class[0],
            HarmonicOscillator_fast_time
        ]
        problem_class_first = [
            problem_class[0],
            HarmonicOscillator_fast_time,
            HarmonicOscillator_fast_time_first_order,
        ]
        problem_params_reduced = [problem_params, problem_fast_time_params]
        model_mlsdc = Mlsdc_class(
            problem_params, collocation_params, sweeper_params, problem_class, eps=eps_fast_time
        )
        model_reduced_mlsdc = Mlsdc_class(
            problem_params_reduced,
            collocation_params,
            sweeper_params,
            problem_class_zeros,eps=eps_fast_time
        )
        model_averaged_mlsdc = Mlsdc_class(
            problem_params_reduced,
            collocation_params,
            sweeper_params,
            problem_class_first,eps=eps_fast_time
        )

        model_averaged_first_order_mlsdc=Mlsdc_class(problem_params_reduced,
            collocation_params,
            sweeper_params,
            problem_class_first)
        model_mlsdc.get_mlsdc_iter_solution()
        model_averaged_mlsdc.get_mlsdc_iter_arg_min_first_order()
        model_reduced_mlsdc.get_mlsdc_iter_solution()
        model_averaged_first_order_mlsdc.get_mlsdc_iter_asyp_expan()
        Residual_mlsdc = model_mlsdc.sdc_fine_level.get_residual
        Residual_reduced = model_reduced_mlsdc.sdc_fine_level.get_residual
        Residual_averaged_zeros = model_averaged_mlsdc.sdc_fine_level.get_residual
        Residual_averaged_first=model_averaged_first_order_mlsdc.sdc_fine_level.get_residual
        residual.append(np.array(Residual_mlsdc)[:,0])
    Kiter = np.arange(1, sweeper_params["Kiter"] + 1, 1)

    Title = "ArgMin"
    # label_set = ["MLSDC ", "$1^{th}$ order arg min model", "$1^{th}$ order averaged model"]
    residual_set = [
        np.array(Residual_mlsdc)[:, 0],
        # np.array(Residual_reduced)[:, 0],
        np.array(Residual_averaged_zeros)[:, 0],
        np.array(Residual_averaged_first)[:,0]
    ]
    plot_residual(Kiter, residual, Title, eps_set)

if __name__=='__main__':
    test_mlsdc_diff_eps()