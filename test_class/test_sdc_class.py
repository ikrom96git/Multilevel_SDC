import numpy as np
from sweeper_class.sdc_class import sdc_class
from problem_class.HarmonicOscillator import HarmonicOscillator
from plot_class.plot_solutionvstime import plot_solution
from plot_class.plot_residual import plot_residual
from default_params.sdc_default_params import get_sdc_default_params


def test_sdc_without_force():
    problem_params, collocation_params, sweeper_params, problem_class = (
        get_sdc_default_params()
    )
    model = sdc_class(problem_params, collocation_params, sweeper_params, problem_class)
    X, V = model.sdc_iter(5)
    time = 0.1 * np.append(0, model.coll.nodes)
    harmonic_oscillator = HarmonicOscillator(problem_params)
    solution = harmonic_oscillator.get_solution_ntimeWithoutForce(time)
    Title = "Solution of SDC"
    label_set = ["SDC", "Exact"]
    solution_set = [X, solution[0, :]]
    plot_solution(time, solution_set, Title, label_set)


def test_sdc_with_force():

    problem_params, collocation_params, sweeper_params, problem_class = (
        get_sdc_default_params(Force=True)
    )
    model = sdc_class(problem_params, collocation_params, sweeper_params, problem_class)
    X, V = model.sdc_iter(10)
    time = 0.1 * np.append(0, model.coll.nodes)
    harmonic_oscillator = HarmonicOscillator(problem_params)
    solution = harmonic_oscillator.get_solution_ntimeWithForce(time)
    Title = "Solution of SDC with Force"
    label_set = ["SDC", "Exact"]
    Solution_set = [X, solution[0, :]]
    plot_solution(time, Solution_set, Title, label_set)


def test_sdc_residual(Force=False):
    problem_params, collocation_params, sweeper_params, problem_class = (
        get_sdc_default_params(Force=Force)
    )
    K_iter = 10
    model_sdc_sweeper = sdc_class(
        problem_params, collocation_params, sweeper_params, problem_class
    )
    pos_solution, vel_solution = model_sdc_sweeper.sdc_iter(K_iter)
    residual_sdc = model_sdc_sweeper.get_residual
    Kiter = np.arange(1, K_iter + 1, 1)
    Title = "Residual"
    label_set = ["Residual_position", "Residual_velocity"]
    residual_set = [np.array(residual_sdc)[:, 0], np.array(residual_sdc)[:, 1]]
    plot_residual(Kiter, residual_set, Title, label_set)


def test_collocation_problem():

    problem_params, collocation_params, sweeper_params, problem_class = (
        get_sdc_default_params(Force=True)
    )
    model_collocation = sdc_class(
        problem_params, collocation_params, sweeper_params, problem_class
    )
    X, V = model_collocation.get_collocation_fsolve()
    time = 0.1 * np.append(0, model_collocation.coll.nodes)
    harmonic_oscillator = HarmonicOscillator(problem_params)
    # solution = harmonic_oscillator.get_solution_ntimeWithoutForce(time)
    solution = harmonic_oscillator.get_solution_ntimeWithForce(time)
    Title = "Collocation"
    label_set = ["Collocation", "Exact"]
    Solution_set = [X, solution[0, :]]
    plot_solution(time, Solution_set, Title, label_set)


def test_sdc_with_collocation_residual(Force=True):
    problem_params, collocation_params, sweeper_params, problem_class = (
        get_sdc_default_params(Force=Force)
    )
    sweeper_params["initial_guess"] = "spread"

    model_sdc = sdc_class(
        problem_params, collocation_params, sweeper_params, problem_class
    )
    residual_sdc = model_sdc.get_max_norm_residual()
    Kiter = np.arange(1, sweeper_params["Kiter"] + 1, 1)
    Title = "Residual"
    label_set = ["Position residual", "Velocity Residual"]
    residual_set = [np.array(residual_sdc)[:, 0], np.array(residual_sdc)[:, 1]]
    plot_residual(Kiter, residual_set, Title, label_set)


if __name__ == "__main__":
    # test_sdc_with_force()
    # test_sdc_residual(Force=True)
    # test_collocation_problem()
    test_sdc_with_collocation_residual()
