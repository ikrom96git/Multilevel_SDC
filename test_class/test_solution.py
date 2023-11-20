def prob_param():
    import numpy as np

    eps = 0.2
    prob_params = dict()
    prob_params["eps"] = eps
    prob_params["kappa"] = 0.2
    prob_params["c"] = 1 / eps
    prob_params["f0"] = 1 / eps
    prob_params["u0"] = np.array([2, 0])

    return prob_params


def test_solution():
    from problem_class.asymptotic_problem import OscillatorProblem
    import numpy as np

    prob_params = prob_param()
    t = np.linspace(0, 30, 1000)
    model = OscillatorProblem(prob_params)
    y0 = model.slow_time_solution(t, order=0)
    y1 = model.slow_time_solution(t, order=1)
    y2 = model.slow_time_solution(t, order=2)
    return y0, y1, y2, t


def exact_solution():
    from problem_class.reduced_HO import Reduced_HO
    import numpy as np

    prob_params = prob_param()
    t = np.linspace(0, 30, 1000)
    model = Reduced_HO(prob_params)
    y = model.get_sol(t)
    return y, t


def plot_solution():
    y0, y1, y2, t = test_solution()
    import matplotlib.pyplot as plt

    y, t = exact_solution()
    plt.plot(t, y[0, :], color="black", label="exact")
    plt.plot(t, y0, linestyle="--", label="order 0")
    plt.plot(t, y1, linestyle="-.", label="order 1")
    plt.plot(t, y2, linestyle=":", label="ordjr 2")
    plt.xlabel("t")
    plt.ylabel("y(t, eps)")
    plt.title("Solution of the slow time system")
    plt.tight_layout()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_solution()
