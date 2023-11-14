import numpy as np
import matplotlib.pyplot as plt


def plot_residual(x_axis, y_axis, labels, title, save_name=None):
    plt.figure()
    plt.semilogy(x_axis, y_axis, "r", label=labels)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Inf Norm of Residual")
    plt.legend()
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name)
    else:
        plt.show()


def compute_residual(residual):
    return np.max(np.abs(residual))


def sort_data(data):
    residual = {"pos": [], "vel": []}
    for i in range(data.prob_params.Kiter):
        if i == 0:
            continue
        else:
            X, V = np.split(data.residual[i], 2)
            residual["pos"].append(compute_residual(X))
            residual["vel"].append(compute_residual(V))
    return residual


def run_plot(model, value="pos", save_name=None):
    data = sort_data(model)
    x_axis = np.arange(1, model.prob_params.Kiter)
    plot_residual(x_axis, data[value], model.name, model.name)
