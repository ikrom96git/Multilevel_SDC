import matplotlib.pyplot as plt
from plot_class import plot_params

def plot_residual(Kiter, residual_set, Title, label_set,filename):
    fs = 12
    plt.rcParams.update({"font.size": fs})
    marker = ["*", "s", "o", "8"]
    for rr in range(len(residual_set)):
        plt.semilogy(Kiter, residual_set[rr], label=label_set[rr], marker=marker[rr])
    plt.title(Title)
    plt.xlabel("Iteration")
    plt.ylim(top=1e-1)
    plt.ylabel("$\|R\|_{\infty}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
