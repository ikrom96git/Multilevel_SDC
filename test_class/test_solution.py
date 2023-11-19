def prob_param():
    import numpy as np

    eps = 0.2
    kappa = 0.2
    mu = eps
    c = 1 / mu
    k = kappa / mu
    f0 = c
    x0 = 2 / c
    prob_params = dict()
    prob_params["k"] = k
    prob_params["c"] = c
    prob_params["dt"] = 0.1
    prob_params["oscillator_type"] = "free"
    prob_params["initial_guess"] = "spread"
    prob_params["u0"] = np.array([2 / c, 0])
    prob_params["Kiter"] = 5
    prob_params["f0"] = 1.0
    prob_params["omega"] = 1.0
    prob_params["t0"] = 0.0
    return prob_params


def test_solution():
    from problem_class.harmonicoscillator import OscillatorProblem
    import numpy as np

    prob_params = prob_param()
    t = np.linspace(0, 30, 1000)
    model = OscillatorProblem(prob_params)
    y0 = model.slow_time_solution(t, kappa_hat=0.2, eps=0.2, order=0)
    y1 = model.slow_time_solution(t, kappa_hat=0.2, eps=0.2, order=1)
    y2 = model.slow_time_solution(t, kappa_hat=0.2, eps=0.2, order=2)
    return y0, y1, y2, t


def exact_solution():
    from problem_class.harmonicoscillator import HarmonicOscillator
    import numpy as np

    prob_params = prob_param()
    t = np.linspace(0, 30, 1000)
    model = HarmonicOscillator(prob_params)
    y = model.compute_solution(2, 0, 0, t)
    return y, t


def plot_solution():
    y0, y1, y2, t = test_solution()
    import matplotlib.pyplot as plt

    y, t = exact_solution()
    plt.plot(t, y[0, :], label="exact")
    plt.plot(t, y0, label="order 0")
    plt.plot(t, y1, label="order 1")
    plt.plot(t, y2, label="order 2")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_solution()
