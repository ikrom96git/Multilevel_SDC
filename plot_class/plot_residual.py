import numpy as np
import matplotlib.pyplot as plt


def plot_residual(x_axis, y_axis, labels, title, save_name=None):
    for x, y, l in zip(x_axis, y_axis, labels):
        plt.semilogy(x, y, label=l)
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


def sort_data(data, Kiter):
    residual = {"pos": [], "vel": []}
    for i in range(Kiter):
        if False:
            continue
        else:
            X, V = np.split(data[i], 2)
            residual["pos"].append(compute_residual(X))
            residual["vel"].append(compute_residual(V))
    return residual


def run_plot(residual, Kiter, label, title="Plot", value="pos", save_name=None):
    res = []
    x_axis = []
    if len(residual) != len(Kiter):
        Kiter = len(residual) * Kiter
    for kk, rr in enumerate(residual):
        res.append(sort_data(rr, Kiter[kk])[value])
        x_axis.append(np.arange(Kiter[kk]))
    plot_residual(x_axis, res, label, title, save_name=save_name)
