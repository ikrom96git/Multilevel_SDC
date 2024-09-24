import numpy as np
from sweeper_class.sdc_class import sdc_class
from problem_class.Duffing_Equation_2 import DuffingEquation
from plot_class.plot_solutionvstime import plot_solution
from plot_class.plot_residual import plot_residual
from default_params.sdc_default_params import get_sdc_default_params
from default_params.duffing_equation2_default_params import get_duffing_equation_params


def test_sdc_duffing_eqaution():
    problem_params = get_duffing_equation_params()
    *_, collocation_params, sweeper_params, problem_class = get_sdc_default_params()
    model = sdc_class(
        problem_params, collocation_params, sweeper_params, DuffingEquation
    )
    X, V = model.sdc_iter(5)
    time = 0.4 * np.append(0, model.coll.nodes)
    duffing_equation = DuffingEquation(problem_params)
    solution = duffing_equation.get_ntime_exact_solution(time)
    Title = "Solution of SDC"
    label_set = ["SDC solution", "Exact"]
    solution_set = [X, solution]
    plot_solution(time, solution_set, Title, label_set)


def test_sdc_residual(Force=False):
    problem_params = get_duffing_equation_params()
    *_, collocation_params, sweeper_params, problem_class = get_sdc_default_params(
        Force=Force
    )
    K_iter = 10
    model_sdc_sweeper = sdc_class(
        problem_params, collocation_params, sweeper_params, DuffingEquation
    )
    pos_solution, vel_solution = model_sdc_sweeper.sdc_iter(K_iter)
    residual_sdc = model_sdc_sweeper.get_residual
    Kiter = np.arange(1, K_iter + 1, 1)
    Title = "Residual"
    label_set = ["Residual_position", "Residual_velocity"]
    residual_set = [np.array(residual_sdc)[:, 0], np.array(residual_sdc)[:, 1]]
    plot_residual(Kiter, residual_set, Title, label_set)


def test_collocation_problem():
    problem_params = get_duffing_equation_params()
    *_, collocation_params, sweeper_params, problem_class = get_sdc_default_params(
        Force=True
    )
    model_collocation = sdc_class(
        problem_params, collocation_params, sweeper_params, DuffingEquation
    )
    X, V = model_collocation.get_collocation_fsolve()
    time = 0.4 * np.append(0, model_collocation.coll.nodes)
    duffing_equation = DuffingEquation(problem_params)
    # solution = harmonic_oscillator.get_solution_ntimeWithoutForce(time)
    solution = duffing_equation.get_ntime_exact_solution(time)
    Title = "Collocation"
    label_set = ["Collocation", "Exact"]
    Solution_set = [X, solution]
    plot_solution(time, Solution_set, Title, label_set)


def test_sdc_with_collocation_residual(Force=True):
    problem_params = get_duffing_equation_params()
    *_, collocation_params, sweeper_params, problem_class = get_sdc_default_params(
        Force=Force
    )
    sweeper_params["initial_guess"] = "10SDC"

    model_sdc = sdc_class(
        problem_params, collocation_params, sweeper_params, DuffingEquation
    )
    residual_sdc = model_sdc.get_max_norm_residual()
    Kiter = np.arange(1, sweeper_params["Kiter"] + 1, 1)
    Title = "Residual"
    label_set = ["Position residual", "Velocity Residual"]
    residual_set = [np.array(residual_sdc)[:, 0], np.array(residual_sdc)[:, 1]]
    plot_residual(Kiter, residual_set, Title, label_set)


if __name__ == "__main__":
    # test_sdc_duffing_eqaution()
    # test_sdc_residual(Force=False)
    test_collocation_problem()  # Test is failing
    # test_sdc_with_collocation_residual()
