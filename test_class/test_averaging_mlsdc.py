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
def test_mlsdc_zeros_averaging():
    problem_params, collocation_params, sweeper_params, problem_class = (
        get_mlsdc_default_params("Fast_time")
    )
    problem_fast_time_params, *_ = get_harmonic_oscillator_fast_time_params(
        Fast_time=True
    )
    problem_class_reduced = [
        problem_class[0],
        HarmonicOscillator_fast_time,
    ]
    problem_params_reduced = [problem_params, problem_fast_time_params]
    # problem_class_reduced[1]=HarmonicOscillator_fast_time
    model_mlsdc = Mlsdc_class(
        problem_params, collocation_params, sweeper_params, problem_class
    )
    model_reduced_mlsdc = Mlsdc_class(
        problem_params_reduced,
        collocation_params,
        sweeper_params,
        problem_class_reduced,
    )
    model_averaged_mlsdc = Mlsdc_class(
        problem_params_reduced,
        collocation_params,
        sweeper_params,
        problem_class_reduced,
    )
    problem_params_sdc, collocation_params_sdc,sweeper_params_sdc,problem_class_sdc, = get_sdc_default_params(Force='Fast_time')
    model_sdc = sdc_class(
            problem_params_sdc,
            collocation_params_sdc,
            sweeper_params_sdc,
            problem_class_sdc)
    X_sdc, V_sdc = model_sdc.sdc_iter()
    Residual_sdc=model_sdc.get_residual
    model_mlsdc.get_mlsdc_iter_solution()
    model_averaged_mlsdc.get_mlsdc_iter_averaged()
    model_reduced_mlsdc.get_mlsdc_iter_solution()
    Residual_mlsdc = model_mlsdc.sdc_fine_level.get_residual
    Residual_reduced = model_reduced_mlsdc.sdc_fine_level.get_residual
    Residual_averaged = model_averaged_mlsdc.sdc_fine_level.get_residual
    Kiter = np.arange(1, sweeper_params["Kiter"] + 1, 1)

    Title = "Residual MLSDC VS Reduced model"
    label_set = ["MLSDC ", "SDC", "$0^{th}$ order averaged model"]
    residual_set = [
        np.array(Residual_mlsdc)[:, 0],
        np.array(Residual_sdc)[:, 0],
        np.array(Residual_averaged)[:, 0],
    ]
    plot_residual(Kiter, residual_set, Title, label_set)

def test_mlsdc_first_averaging():
    problem_params, collocation_params, sweeper_params, problem_class = (
        get_mlsdc_default_params("Fast_time")
    )
    problem_fast_time_params, *_ = get_harmonic_oscillator_fast_time_params(
        Fast_time=True
    )
    problem_class_zeros = [
        problem_class[0],
        HarmonicOscillator_fast_time,
    ]
    problem_class_first = [
        problem_class[0],
        HarmonicOscillator_fast_time,
        HarmonicOscillator_fast_time_first_order,
    ]
    problem_params_reduced = [problem_params, problem_fast_time_params]
    model_mlsdc = Mlsdc_class(
        problem_params, collocation_params, sweeper_params, problem_class
    )
    model_reduced_mlsdc = Mlsdc_class(
        problem_params_reduced,
        collocation_params,
        sweeper_params,
        problem_class_zeros,
    )
    model_averaged_mlsdc = Mlsdc_class(
        problem_params_reduced,
        collocation_params,
        sweeper_params,
        problem_class_zeros,
    )

    model_averaged_first_order_mlsdc=Mlsdc_class(problem_params_reduced,
        collocation_params,
        sweeper_params,
        problem_class_first)
    model_mlsdc.get_mlsdc_iter_solution()
    model_averaged_mlsdc.get_mlsdc_iter_averaged()
    model_reduced_mlsdc.get_mlsdc_iter_solution()
    model_averaged_first_order_mlsdc.get_mlsdc_iter_averaged()
    Residual_mlsdc = model_mlsdc.sdc_fine_level.get_residual
    Residual_reduced = model_reduced_mlsdc.sdc_fine_level.get_residual
    Residual_averaged_zeros = model_averaged_mlsdc.sdc_fine_level.get_residual
    Residual_averaged_first=model_averaged_first_order_mlsdc.sdc_fine_level.get_residual
    Kiter = np.arange(1, sweeper_params["Kiter"] + 1, 1)

    Title = "Residual MLSDC VS Reduced model"
    label_set = ["MLSDC ", "$0^{th}$ order model", "$0^{th}$ order averaged model", "$1^{th}$ order averaged model"]
    residual_set = [
        np.array(Residual_mlsdc)[:, 0],
        np.array(Residual_reduced)[:, 0],
        np.array(Residual_averaged_zeros)[:, 0],
        np.array(Residual_averaged_first)[:,0]
    ]
    plot_residual(Kiter, residual_set, Title, label_set)
if __name__ == "__main__":
    # test_mlsdc_zeros_averaging()
    test_mlsdc_first_averaging()
