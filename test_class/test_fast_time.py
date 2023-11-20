import numpy as np
from problem_class.asymptotic_problem import Fast_time
from problem_class.reduced_HO import Reduced_HO
import matplotlib.pyplot as plt


def problem_params():
    eps = 0.001
    prob_params = dict()
    prob_params["eps"] = eps
    prob_params["kappa"] = 0.8
    prob_params["c"] = 1 / eps
    prob_params["f0"] = 1 / eps
    prob_params["u0"] = np.array([2, 0])
    return prob_params


def fast_time():
    prob_params = problem_params()
    t = np.linspace(0, 0.1, 1000)
    model = Fast_time(prob_params)
    solution = model.fast_time(t)
    return solution, t


def test_problem():
    prob_params = problem_params()
    t = np.linspace(0, 0.1, 1000)
    model = Reduced_HO(prob_params)
    solution = model.get_sol(t)
    return solution, t


def test_plot():
    solution, t = test_problem()
    fast_solution, t = fast_time()
    plt.plot(t, fast_solution, label="fast solution")
    plt.plot(t, solution[0, :], label="exact solution")
    plt.show()


if __name__ == "__main__":
    test_plot()
