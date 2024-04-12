import numpy as np
from default_params.mlsdc_defautl_params import get_mlsdc_default_params
from default_params.sdc_default_params import get_sdc_default_params
from sweeper_class.mlsdc_class import Mlsdc_class
from sweeper_class.sdc_class import sdc_class
from problem_class.HarmonicOscillator import HarmonicOscillator
from plot_class.plot_solutionvstime import plot_solution
from plot_class.plot_residual import plot_residual


def test_solution(Force=True):
    problem_params, collocation_params, sweeper_params, problem_class = (
        get_mlsdc_default_params(Force=Force)
    )
    mlsdc_model = Mlsdc_class(
        problem_params, collocation_params, sweeper_params, problem_class
    )
    X, V = mlsdc_model.get_mlsdc_iter_solution(10)
    time = 0.1 * np.append(
        mlsdc_model.sdc_fine_level.prob.t0, mlsdc_model.sdc_fine_level.coll.nodes
    )
    harmonic_oscillator = HarmonicOscillator(problem_params)
    Exact_solution = harmonic_oscillator.get_solution_ntimeWithForce(time)
    Title = "MLSDC solution"
    label_set = ["Solution MLSDC", "Exact_solution"]
    solution_set = [X, Exact_solution[0, :]]

    plot_solution(time, solution_set, Title, label_set)


def test_residual(Force=False):
    problem_params, collocation_params, sweeper_params, problem_class = (
        get_mlsdc_default_params(Force=Force)
    )
    sweeper_params["initial_guess"] = "collocation"
    mlsdc_model = Mlsdc_class(
        problem_params, collocation_params, sweeper_params, problem_class
    )
    X, V = mlsdc_model.get_mlsdc_iter_solution()
    Residual_mlsdc = mlsdc_model.sdc_fine_level.get_residual
    Kiter = np.arange(1, 10 + 1, 1)
    Title = "Residual"
    label_set = ["Position residual", "Velocity residual"]
    residual_set = [np.array(Residual_mlsdc)[:, 0], np.array(Residual_mlsdc)[:, 1]]
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
        sweeper_params_sdc["Kiter"] = 2*kk
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
    # test_residual()
    test_mlsdc_vs_sdc_solution()
