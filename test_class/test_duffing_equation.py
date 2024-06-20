import numpy as np

from default_params.duffing_equation2_default_params import (
    get_duffing_equation_params,
    get_duffing_zeros_order_params,
    get_duffing_first_order_params,
)
from default_params.mlsdc_defautl_params import get_mlsdc_default_params
from problem_class.Duffing_Equation_2 import (
    DuffingEquation,
    DuffingEquation_zeros_order_problem,
    DuffingEquation_first_order_problem,
)
from sweeper_class.mlsdc_class import Mlsdc_class
from plot_class.plot_residual import plot_residual
from plot_class.plot_solutionvstime import plot_solution
from transfer_class.standart_restriction_class import StandartRestriction
from transfer_class.asymptotic_restriction_class import AsymptoticRestriction
from transfer_class.optimazation_restriction_class import OptimationRestriction, OptimazationResidual
from transfer_class.restriction import Restriction



EPSILON = 1
problem_params, collocation_params, sweeper_params, *_ = get_mlsdc_default_params()
    
def duffing_mlsdc():
    problem_duffing_params = get_duffing_equation_params(EPSILON)
    problem_class_mlsdc = [DuffingEquation, DuffingEquation]
    model_mlsdc = Mlsdc_class(
        problem_duffing_params,
        collocation_params,
        sweeper_params,
        problem_class_mlsdc,
        Restriction,
        eps=EPSILON,
    )
    model_mlsdc.get_mlsdc_iter_solution()
    return model_mlsdc

def duffing_asymptotic_restriction():
    problem_duffing_params = get_duffing_equation_params(EPSILON)
    problem_duffing_zeros_params = get_duffing_zeros_order_params(EPSILON)
    problem_duffing_first_params = get_duffing_first_order_params(EPSILON)
    problem_class_reduced = [
        DuffingEquation,
        DuffingEquation_zeros_order_problem,
        DuffingEquation_first_order_problem,
    ]
    problem_params_reduced = [
        problem_duffing_params,
        problem_duffing_zeros_params,
        problem_duffing_first_params,
    ]
    model_asymp_mlsdc = Mlsdc_class(
        problem_params_reduced,
        collocation_params,
        sweeper_params,
        problem_class_reduced,
        AsymptoticRestriction,
        eps=EPSILON,
    )
    model_asymp_mlsdc.get_mlsdc_iter_solution()
    return model_asymp_mlsdc

def duffing_standart_restriction():
    problem_duffing_params = get_duffing_equation_params(EPSILON)
    problem_duffing_zeros_params = get_duffing_zeros_order_params(EPSILON)
    problem_duffing_first_params = get_duffing_first_order_params(EPSILON)
    problem_class_reduced = [
        DuffingEquation,
        DuffingEquation_zeros_order_problem,
        DuffingEquation_first_order_problem,
    ]
    problem_params_reduced = [
        problem_duffing_params,
        problem_duffing_zeros_params,
        problem_duffing_first_params,
    ]
    model_standart_mlsdc = Mlsdc_class(
        problem_params_reduced,
        collocation_params,
        sweeper_params,
        problem_class_reduced,
        StandartRestriction,
        eps=EPSILON
    )
    model_standart_mlsdc.get_mlsdc_iter_solution()
    return model_standart_mlsdc

def duffing_minimize_restriction():
    problem_duffing_params = get_duffing_equation_params(EPSILON)
    problem_duffing_zeros_params = get_duffing_zeros_order_params(EPSILON)
    problem_duffing_first_params = get_duffing_first_order_params(EPSILON)
    problem_class_reduced = [
        DuffingEquation,
        DuffingEquation_zeros_order_problem,
        DuffingEquation_first_order_problem,
    ]
    problem_params_reduced = [
        problem_duffing_params,
        problem_duffing_zeros_params,
        problem_duffing_first_params,
    ]
    model_minimize_mlsdc = Mlsdc_class(
        problem_params_reduced,
        collocation_params,
        sweeper_params,
        problem_class_reduced,
        OptimazationResidual,
        eps=EPSILON
    )
    model_minimize_mlsdc.get_mlsdc_iter_solution()
    return model_minimize_mlsdc
def test_duffing_residual():
    model_mlsdc=duffing_mlsdc()
    model_asymp_mlsdc=duffing_asymptotic_restriction()
    model_standart_mlsdc=duffing_standart_restriction()
    model_minimize_mlsdc=duffing_minimize_restriction()
    Residual_mlsdc=model_mlsdc.sdc_fine_model.get_residual
    Residual_asymp_mlsdc=model_asymp_mlsdc.sdc_fine_model.get_residual
    Residual_standart_mlsdc=model_standart_mlsdc.sdc_fine_model.get_residual
    Residual_minimize_mlsdc=model_minimize_mlsdc.sdc_fine_model.get_residual
    Kiter = np.arange(1, sweeper_params["Kiter"] + 1, 1)
    Title = rf"$\varepsilon={EPSILON}$"
    label_set = [
        "MLSDC ",
        "Asymptotic restriction (M3LSDC)",
        "Arg minimize (M3LSDC)",
        "Standart Restriction (M3LSDC)",
    ]
    residual_set = [
        np.array(Residual_mlsdc)[:, 0],
        np.array(Residual_asymp_mlsdc)[:, 0],
        np.array(Residual_minimize_mlsdc)[:, 0],
        np.array(Residual_standart_mlsdc)[:, 0],
    ]
    plot_residual(Kiter, residual_set, Title, label_set)


def test_duffing_equation_solution():
    EPSILON = 0.1
    problem_params, collocation_params, sweeper_params, *_ = get_mlsdc_default_params()
    problem_duffing_params = get_duffing_equation_params(EPSILON)
    problem_duffing_zeros_params = get_duffing_zeros_order_params(EPSILON)
    problem_duffing_first_params = get_duffing_first_order_params(EPSILON)
    problem_class_reduced = [
        DuffingEquation,
        DuffingEquation_zeros_order_problem,
        DuffingEquation_first_order_problem,
    ]
    problem_class = [DuffingEquation, DuffingEquation]
    problem_params_reduced = [problem_duffing_params, problem_duffing_first_params]
    model_mlsdc = Mlsdc_class(
        problem_duffing_params,
        collocation_params,
        sweeper_params,
        problem_class,
        eps=EPSILON,
    )
    model_reduced_mlsdc = Mlsdc_class(
        problem_params_reduced,
        collocation_params,
        sweeper_params,
        problem_class_reduced,
        eps=EPSILON,
    )
    mlsdc_pos, mlsdc_vel = model_mlsdc.get_mlsdc_iter_solution()
    mlsdc_reduced_pos, mlsdc_reduced_vel = (
        model_reduced_mlsdc.get_mlsdc_iter_arg_min_first_order()
    )

    duffing_zeros_order = DuffingEquation_zeros_order_problem(
        problem_duffing_zeros_params
    )
    duffing_first_order = DuffingEquation_first_order_problem(
        problem_duffing_first_params
    )
    time = np.linspace(0, 2 * np.pi, 1000)
    duffing_zeros_order_solution = duffing_zeros_order.get_ntime_exact_solution(time)
    duffing_first_order_solution = duffing_first_order.get_ntime_exact_solution(time)
    duffing_pos = duffing_first_order.asyp_expansion(
        duffing_zeros_order_solution[0, :],
        duffing_first_order_solution[0, :],
        eps=EPSILON,
    )
    duffing_solution = [duffing_zeros_order_solution[0, :], duffing_pos]

    Time = np.append(0.0, model_mlsdc.sdc_fine_level.coll.nodes)
    Title = "Residual MLSDC VS Reduced model with new restiriction operator"
    label_set = ["MLSDC ", "Reduced model"]
    solution_set = [mlsdc_pos, mlsdc_reduced_pos]
    # plot_solution(Time, solution_set, Title, label_set)
    plot_solution(time, duffing_solution, Title, label_set)


if __name__ == "__main__":
    test_duffing_residual()
    # test_duffing_equation_solution()
