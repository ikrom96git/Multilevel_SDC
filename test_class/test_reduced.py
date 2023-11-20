import numpy as np
import matplotlib.pyplot as plt
from problem_class.reduced_HO import Reduced_HO


def params():
    prob_params = dict()
    prob_params["kappa"] = 0.2
    prob_params["c"] = 1 / 0.2
    prob_params["f0"] = 1 / 0.2
    prob_params["u0"] = np.array([2, 0])
    return prob_params


def test_problem():
    prob_params = params()
    t = np.linspace(0, 30, 1000)
    model = Reduced_HO(prob_params)
    solution = model.get_sol(t)
    return solution, t


def test_plot():
    solution, t = test_problem()
    plt.plot(t, solution[0, :])
    plt.show()


if __name__ == "__main__":
    test_plot()
