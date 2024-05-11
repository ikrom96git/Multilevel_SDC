import matplotlib.pyplot as plt


def plot_residual(Kiter, residual_set, Title, label_set):
    fs=12
    plt.rcParams.update({'font.size':fs})
    for rr in range(len(residual_set)):
        plt.semilogy(Kiter, residual_set[rr], label=label_set[rr])
    plt.title(Title)
    plt.xlabel("Iterations")
    plt.ylabel("$\|R\|_{\infty}$")
    plt.legend()
    plt.tight_layout()
    plt.show()
