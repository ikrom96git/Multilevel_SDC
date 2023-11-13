import numpy as np
import matplotlib.pyplot as plt


def plot_solution(x_axis, y_axis, labels, title, save_name=None):
    plt.figure()
    plt.plot(x_axis, y_axis, label=labels)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_name is not None:
        plt.savefig("data/" + save_name + ".png")
    else:
        plt.show()


def load_data(filename):
    data = np.loadtxt(filename, delimiter=",")
    return data


def sort_data(data):
    data = data[np.argsort(data[:, 0])]
    data = data[np.argsort(data[:, 1])]
    return data
