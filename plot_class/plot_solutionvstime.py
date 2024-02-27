import matplotlib.pyplot as plt

def plot_solution(time, solution, title, label_set):
    for ii in range(len(solution)):
        plt.plot(time, solution[ii], label=label_set[ii])
    plt.title(title)
    plt.xlabel('time')
    plt.ylabel('Solution')
    plt.legend()
    plt.tight_layout()
    plt.show()
