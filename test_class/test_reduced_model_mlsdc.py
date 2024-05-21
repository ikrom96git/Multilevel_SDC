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


def test_mlsdc_slow_time_problem():
    problem_slow_time_params, collocation_params, sweeper_params, problem_class = (
        get_mlsdc_default_params(Force=True)
    )
    problem_reduced_params, *_ = get_harmonic_oscillator_fast_time_params(
        Fast_time=False
    )
    problem_class_reduced = deepcopy(problem_class)
    problem_class_reduced[1] = HarmonicOscillator_fast_time
    sweeper_params["initial_guess"] = "collocation"
    problem_params = [problem_slow_time_params, problem_reduced_params]
    model_reduced_mlsdc = Mlsdc_class(
        problem_params, collocation_params, sweeper_params, problem_class_reduced
    )
    model_mlsdc = Mlsdc_class(
        problem_slow_time_params, collocation_params, sweeper_params, problem_class
    )
    X_reduced, V_reduced = model_reduced_mlsdc.get_mlsdc_iter_solution()
    X_mlsdc, V_mlsdc = model_mlsdc.get_mlsdc_iter_solution()
    Residual_mlsdc = model_mlsdc.sdc_fine_level.get_residual
    Residual_reduced = model_reduced_mlsdc.sdc_fine_level.get_residual
    Kiter = np.arange(1, 11, 1)
    Title = "Residual MLSDC vs Reduced model"
    label_set = ["MLSDC", "$0^{th}$ order model on coarse level"]
    residual_set = [np.array(Residual_mlsdc)[:, 0], np.array(Residual_reduced)[:, 0]]
    plot_residual(Kiter, residual_set, Title, label_set)


def test_mlsdc_fast_time_problems():
    problem_params, collocation_params, sweeper_params, problem_class = (
        get_mlsdc_default_params("Fast_time")
    )
    problem_fast_time_params, *_ = get_harmonic_oscillator_fast_time_params(
        Fast_time=True
    )
    problem_class_reduced = [
        problem_class[0],
        HarmonicOscillator_fast_time,
        HarmonicOscillator_fast_time_first_order,
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
    model_mlsdc.get_mlsdc_iter_solution()
    model_reduced_mlsdc.get_mlsdc_iter_solution()
    Residual_mlsdc = model_mlsdc.sdc_fine_level.get_residual
    Residual_reduced = model_reduced_mlsdc.sdc_fine_level.get_residual
    Kiter = np.arange(1, sweeper_params["Kiter"] + 1, 1)

    Title = "Residual MLSDC VS Reduced model"
    label_set = ["MLSDC ", "$1^{th}$ order model"]
    residual_set = [np.array(Residual_mlsdc)[:, 0], np.array(Residual_reduced)[:, 0]]
    plot_residual(Kiter, residual_set, Title, label_set)


def test_mlsdc_first_order_model():
    problem_params, collocation_params, sweeper_params, problem_class = (
        get_mlsdc_default_params("Fast_time")
    )
    problem_fast_time_params, *_ = get_harmonic_oscillator_fast_time_params(
        Fast_time=True
    )
    problem_class_zeros_order = [problem_class[0], HarmonicOscillator_fast_time]
    problem_class_reduced = [
        problem_class[0],
        HarmonicOscillator_fast_time,
        HarmonicOscillator_fast_time_first_order,
    ]
    problem_params_reduced = [problem_params, problem_fast_time_params]
    # problem_class_reduced[1]=HarmonicOscillator_fast_time
    model_mlsdc = Mlsdc_class(
        problem_params, collocation_params, sweeper_params, problem_class
    )
    model_zeros_order = Mlsdc_class(
        problem_params_reduced,
        collocation_params,
        sweeper_params,
        problem_class_zeros_order,
    )
    model_reduced_mlsdc = Mlsdc_class(
        problem_params_reduced,
        collocation_params,
        sweeper_params,
        problem_class_reduced,
    )

    model_mlsdc.get_mlsdc_iter_solution()
    model_zeros_order.get_mlsdc_iter_solution()
    model_reduced_mlsdc.get_mlsdc_iter_solution()
    Residual_mlsdc = model_mlsdc.sdc_fine_level.get_residual
    Residual_zeros_order = model_zeros_order.sdc_fine_level.get_residual
    Residual_reduced = model_reduced_mlsdc.sdc_fine_level.get_residual
    Kiter = np.arange(1, sweeper_params["Kiter"] + 1, 1)

    Title = "Residual MLSDC VS Reduced model (Position)"
    label_set = ["MLSDC ", "$0^{th}$ order model", "$1^{st}$ order model"]
    residual_set = [
        np.array(Residual_mlsdc)[:, 0],
        np.array(Residual_zeros_order)[:, 0],
        np.array(Residual_reduced)[:, 0],
    ]
    plot_residual(Kiter, residual_set, Title, label_set)

def test_mlsdc_arg_min_sweep():
    problem_params, collocation_params, sweeper_params, problem_class = (
        get_mlsdc_default_params("Fast_time")
    )
    problem_fast_time_params, *_ = get_harmonic_oscillator_fast_time_params(
        Fast_time=True
    )
    
    problem_params_reduced = [problem_params, problem_fast_time_params]
    problem_class_zeros_order = [problem_class[0], HarmonicOscillator_fast_time]
    model_mlsdc = Mlsdc_class(
        problem_params, collocation_params, sweeper_params, problem_class
    )
    model_reduced_mlsdc = Mlsdc_class(
        problem_params_reduced,
        collocation_params,
        sweeper_params,
        problem_class_zeros_order
    )
    model_mlsdc.get_mlsdc_iter_solution()
    model_reduced_mlsdc.get_mlsdc_iter_arg_min()
    Residual_mlsdc = model_mlsdc.sdc_fine_level.get_residual
    Residual_reduced = model_reduced_mlsdc.sdc_fine_level.get_residual
    model_reduced_mlsdc.sdc_fine_level.get_residual=[]
    model_reduced_mlsdc.get_mlsdc_iter_averaged()
    Residual_averaged = model_reduced_mlsdc.sdc_fine_level.get_residual
    Kiter = np.arange(1, sweeper_params["Kiter"] + 1, 1)

    Title = "Residual MLSDC VS Reduced model"
    label_set = ["MLSDC ", "$0^{th}$ order arg min model", "$0^{th}$ order averaged model"]
    residual_set = [np.array(Residual_mlsdc)[:, 0], np.array(Residual_reduced)[:, 0], np.array(Residual_averaged)[:, 0]]
    plot_residual(Kiter, residual_set, Title, label_set)

def test_mlsdc_arg_min_first_order():
    problem_params, collocation_params, sweeper_params, problem_class = get_mlsdc_default_params("Fast_time")
    problem_fast_time_params, *_ = get_harmonic_oscillator_fast_time_params(Fast_time=True)
    problem_class_zeros_order = [problem_class[0], HarmonicOscillator_fast_time]
    problem_class_first_order = [problem_class[0], HarmonicOscillator_fast_time, HarmonicOscillator_fast_time_first_order]
    problem_params_reduced = [problem_params, problem_fast_time_params]
    model_mlsdc = Mlsdc_class(problem_params, collocation_params, sweeper_params, problem_class)
    model_zeros_order = Mlsdc_class(problem_params_reduced, collocation_params, sweeper_params, problem_class_zeros_order)
    model_first_order = Mlsdc_class(problem_params_reduced, collocation_params, sweeper_params, problem_class_first_order)
    model_mlsdc.get_mlsdc_iter_solution()
    model_zeros_order.get_mlsdc_iter_arg_min()
    model_first_order.get_mlsdc_iter_arg_min_first_order()
    Residual_mlsdc = model_mlsdc.sdc_fine_level.get_residual
    Residual_zeros_order = model_zeros_order.sdc_fine_level.get_residual
    Residual_first_order = model_first_order.sdc_fine_level.get_residual
    Kiter = np.arange(1, sweeper_params["Kiter"] + 1, 1)
    Title = "Residual MLSDC VS Reduced model"
    label_set = ["MLSDC ", "$0^{th}$ order arg min model", "$1^{st}$ order arg min model"]
    residual_set = [np.array(Residual_mlsdc)[:, 0], np.array(Residual_zeros_order)[:, 0], np.array(Residual_first_order)[:, 0]]
    plot_residual(Kiter, residual_set, Title, label_set)

def mlsdc_simple_test():
    problem_params, collocation_params, sweeper_params, problem_class = get_mlsdc_default_params("Fast_time")   
    problem_fast_time_params, *_ = get_harmonic_oscillator_fast_time_params(Fast_time=True)
    problem_class_zeros_order = [problem_class[0], HarmonicOscillator_fast_time]    
    problem_class_first_order = [problem_class[0], HarmonicOscillator_fast_time, HarmonicOscillator_fast_time_first_order]
    problem_params_reduced = [problem_params, problem_fast_time_params]
    model_mlsdc = Mlsdc_class(problem_params, collocation_params, sweeper_params, problem_class)
    model_mlsdc_simple=Mlsdc_class(problem_params, collocation_params, sweeper_params, problem_class_first_order)
    model_mlsdc.get_mlsdc_iter_solution()
    model_mlsdc_simple.get_mlsdc_simple_test()
    Residual_mlsdc = model_mlsdc.sdc_fine_level.get_residual
    Residual_simple = model_mlsdc_simple.sdc_fine_level.get_residual
    Kiter = np.arange(1, sweeper_params["Kiter"] + 1, 1)
    Title = "Residual MLSDC VS Reduced model"
    label_set = ["MLSDC ", "$1^{st}$ order simple model"]
    residual_set = [np.array(Residual_mlsdc)[:, 0], np.array(Residual_simple)[:, 0]]
    plot_residual(Kiter, residual_set, Title, label_set)
if __name__ == "__main__":
    # test_mlsdc_slow_time_problem()
    # test_mlsdc_fast_time_problems()
    # test_mlsdc_first_order_model()
    # test_mlsdc_arg_min_sweep()
    # test_mlsdc_arg_min_first_order()
    mlsdc_simple_test()
