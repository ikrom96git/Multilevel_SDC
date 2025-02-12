import numpy as np
from plot_class.plot_solutionvstime import plot_solution
from problem_class.Uniform_electric_field import Uniform_electric_field
from plot_class import plot_params
import matplotlib.pyplot as plt
def uniform_electric_filed_params():
    problem_params = {
        "x_0": np.array([1, 1, 1]),
        "v_0": np.array([1, 1, 1]),
        "t": np.linspace(0, 2 * np.pi, 10000),
        "epsilon": 0.1,
        "s": 0,
    }
    return problem_params

def test_uniform_electric_field_solution():
    problem_params = uniform_electric_filed_params()
    problem = Uniform_electric_field(problem_params)
    x_0 = problem.params.x_0
    v_0 = problem.params.v_0
    t = problem.params.t
    epsilon = problem.params.epsilon
    s = problem.params.s
    exact_solution = problem.n_time_exact_solution(x_0, v_0, t, epsilon, s)
    asymptotic_solution = problem.n_time_asymptotic_solution(x_0, v_0, t, epsilon, s)
    solution = [exact_solution[:,5], asymptotic_solution[:,5]]
    label_set = ["Exact solution", "Asymptotic solution"]
    title = "Uniform electric field"
    plot_solution(t, solution, title, label_set)

def test_solution_line_plot():
    problem_params = uniform_electric_filed_params()
    problem = Uniform_electric_field(problem_params)
    x_0 = problem.params.x_0
    v_0 = problem.params.v_0
    t = problem.params.t
    epsilon = problem.params.epsilon
    s = problem.params.s
    exact_solution = problem.n_time_exact_solution(x_0, v_0, t, epsilon, s)
    asymptotic_solution = problem.n_time_asymptotic_solution(x_0, v_0, t, epsilon, s)
    # 3d plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(exact_solution[:,0], exact_solution[:,1], exact_solution[:,2])
    # ax.plot(asymptotic_solution[:,0], asymptotic_solution[:,1], asymptotic_solution[:,2], label='Asymptotic solution')
    ax.set_xlabel('$X(t)$')
    ax.set_ylabel('$Y(t)$')
    ax.set_zlabel('$Z(t)$')
    # ax.set_title(rf'$\varepsilon={epsilon},$' +r'$\ t_{\mathrm{end}}=2\pi$')
    ax.legend()
    plt.tight_layout()
    plt.savefig("uniform_electric_field.pdf")
    plt.show()
    

def relative_error_plot():
    EPSILON=[0.1, 0.01, 0.001, 0.0001]
    problem_params = uniform_electric_filed_params()
    problem = Uniform_electric_field(problem_params)
    x_0 = problem.params.x_0
    v_0 = problem.params.v_0
    t = problem.params.t
    relative_error=np.zeros((len(EPSILON), len(t)))
    
    for ee, eps in enumerate(EPSILON):
        
        epsilon = eps
        print(epsilon)
        s = problem.params.s
        exact_solution = problem.n_time_exact_solution(x_0, v_0, t, epsilon, s)
        asymptotic_solution = problem.n_time_asymptotic_solution(x_0, v_0, t, epsilon, s)
        # l1 relative error
        for i in range(len(t)):
            relative_error[ee, i] = np.linalg.norm(exact_solution[i] - asymptotic_solution[i], 1)/np.linalg.norm(exact_solution[i], 1)
        l1_relative_error = np.linalg.norm(exact_solution - asymptotic_solution, 1)/np.linalg.norm(exact_solution, 1)
    # relative_error = np.linalg.norm(exact_solution - asymptotic_solution)/np.linalg.norm(exact_solution)
    # plot eror with log scale vs time step
    [plt.plot(t[5:-5], relative_error[ee][5:-5], label=rf"$\varepsilon={epsilon}$") for ee, epsilon in enumerate(EPSILON)]
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Relative error")
    plt.xlim(left=0)
    plt.xlim(right=2*np.pi)
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("relative_error_uniform_electric_field.pdf")
    plt.show()



if __name__ == "__main__":
    # test_uniform_electric_field_solution()
    test_solution_line_plot()
    # relative_error_plot()