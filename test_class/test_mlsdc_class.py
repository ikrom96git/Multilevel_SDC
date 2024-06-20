import numpy as np
from default_params.mlsdc_defautl_params import get_mlsdc_default_params
from default_params.sdc_default_params import get_sdc_default_params
from default_params.harmonic_oscillator_default_fast_time_params import (
    get_harmonic_oscillator_fast_time_params,
)
from sweeper_class.mlsdc_class import Mlsdc_class
from sweeper_class.sdc_class import sdc_class
from problem_class.HarmonicOscillator import HarmonicOscillator
from problem_class.HarmonicOscillator_fast_time_reduced_problem import (
    HarmonicOscillator_fast_time,
    HarmonicOscillator_fast_time_first_order,
)
from plot_class.plot_solutionvstime import plot_solution
from plot_class.plot_residual import plot_residual
from transfer_class.restriction import Restriction


def test_solution(Force=True):
    EPSILON = 0.001
    problem_params, collocation_params, sweeper_params, problem_class = (
        get_mlsdc_default_params(Force="Fast_time", eps=EPSILON)
    )
    mlsdc_model = Mlsdc_class(
        problem_params, collocation_params, sweeper_params, problem_class, Restriction
    )
    X, V = mlsdc_model.get_mlsdc_iter_solution(10)
    time = 0.1 * np.append(
        mlsdc_model.sdc_fine_model.prob.t0, mlsdc_model.sdc_fine_model.coll.nodes
    )
    prob_fast_time_params, fast_time = get_harmonic_oscillator_fast_time_params(
        Fast_time=True, eps=EPSILON
    )
    time_fast = time / np.sqrt(EPSILON)
    model_zeros_order = HarmonicOscillator_fast_time(prob_fast_time_params)
    model_first_order = HarmonicOscillator_fast_time_first_order(prob_fast_time_params)
    solution_fast = model_zeros_order.get_ntime_exact_solution(fast_time)
    solution_first = model_first_order.get_ntime_exact_solution(fast_time)
    position = model_first_order.asyp_expansion(
        solution_fast[0, :], solution_first[0, :], eps=EPSILON
    )
    harmonic_oscillator = HarmonicOscillator(problem_params)
    Exact_solution = harmonic_oscillator.get_solution_ntimeWithForce(
        fast_time * np.sqrt(EPSILON)
    )

    Title = rf"Solution $\varepsilon=${EPSILON}"
    label_set = ["MLSDC", "Exact solution", "reduced order model"]
    solution_set = [X, Exact_solution[0, :], position]

    plot_solution(fast_time * np.sqrt(EPSILON), solution_set, Title, label_set)


def test_residual(Force=False):
    EPSILON = 0.001
    problem_params, collocation_params, sweeper_params, problem_class = (
        get_mlsdc_default_params(Force=Force, eps=EPSILON)
    )
    sweeper_params["initial_guess"] = "spread"
    mlsdc_model = Mlsdc_class(
        problem_params, collocation_params, sweeper_params, problem_class, Restriction
    )
    (
        problem_params_sdc,
        collocation_params_sdc,
        sweeper_params_sdc,
        problem_class_sdc,
    ) = get_sdc_default_params(Force=Force, eps=EPSILON)
    model_sdc = sdc_class(
        problem_params_sdc,
        collocation_params_sdc,
        sweeper_params_sdc,
        problem_class_sdc,
    )
    X_sdc, V_sdc = model_sdc.sdc_iter()
    Residual_sdc = model_sdc.get_residual
    X, V = mlsdc_model.get_mlsdc_iter_solution()
    Residual_mlsdc = mlsdc_model.sdc_fine_model.get_residual
    Kiter = np.arange(1, 10 + 1, 1)
    Title = "Residual velocity"
    label_set = ["MLSDC", "SDC"]
    residual_set = [np.array(Residual_mlsdc)[:, 1], np.array(Residual_sdc)[:, 1]]
    plot_residual(Kiter, residual_set, Title, label_set)


def test_mlsdc_vs_sdc_solution(Force=False):
    (
        problem_params_mlsdc,
        collocation_params_mlsdc,
        sweeper_params_mlsdc,
        problem_class_mlsdc,
    ) = get_mlsdc_default_params(Force=Force)
    (
        problem_params_sdc,
        collocation_params_sdc,
        sweeper_params_sdc,
        problem_class_sdc,
    ) = get_sdc_default_params(Force=Force)
    collocation_params_mlsdc["num_nodes"] = [5, 5]
    sweeper_params_mlsdc["coarse_solver"] = "spread"
    for kk in range(1, 5):
        sweeper_params_sdc["Kiter"] = 2 * kk
        sweeper_params_mlsdc["Kiter"] = kk
        model_mlsdc = Mlsdc_class(
            problem_params_mlsdc,
            collocation_params_mlsdc,
            sweeper_params_mlsdc,
            problem_class_mlsdc,
        )
        model_sdc = sdc_class(
            problem_params_sdc,
            collocation_params_sdc,
            sweeper_params_sdc,
            problem_class_sdc,
        )
        X_mlsdc, V_mlsdc = model_mlsdc.get_mlsdc_iter_solution()
        X_sdc, V_sdc = model_sdc.sdc_iter()

        if (X_mlsdc == X_sdc).all():
            print(f"Solutoin of position is the same for the iteration {kk}")
        else:
            print(f"Solution of position is not the same for the iteration {kk}")

        if (V_mlsdc == V_sdc).all():
            print(f"Solution of velocity is the same for the iteration {kk}")
        else:
            print(f"Solution of velocity is not the same for the iteration {kk}")


if __name__ == "__main__":
    # test_solution()
    test_residual(Force="Fast_time")
    # test_mlsdc_vs_sdc_solution()
