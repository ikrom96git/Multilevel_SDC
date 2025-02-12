import matplotlib.pyplot as plt
import numpy as np
from plot_class import plot_params
def plot_solution(time, solution, title, label_set):

    linestyle = ["-", "solid", "dashed", ":"]
    colors = ["black", "blue", "red", "brown"]
    marker = ["o", "s", "<", "*"]
    for ii in range(len(solution)):
        plt.plot(
            time,
            solution[ii],
            label=label_set[ii],
            color=colors[ii],
            linestyle=linestyle[ii],
        )
    # plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("$x(t, \\varepsilon)$")
    plt.legend()
    plt.title(title)
    plt.xlim(left=0)
    plt.xlim(right=2*np.pi)
    plt.ylim(bottom=-2.8)
    plt.ylim(top=2.8)
    plt.tight_layout()
    plt.grid()

    plt.savefig("solution_Duffing.pdf")
    plt.show()
