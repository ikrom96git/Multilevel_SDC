import numpy as np
from problem_class.Magnetic_field import Magnetic_field
import matplotlib.pyplot as plt
EPSILON=0.01
def prob_params():
    problem_params=dict()
    problem_params['epsilon']=EPSILON
    problem_params['u0']=np.array([1.,1.,1.,1.,EPSILON, 0.])
    return problem_params


def get_solution(time):
    prob=prob_params()
    model=Magnetic_field(prob)
    return model.solve_orginal(time)

def plot_solution(solution, time):
    ax=plt.figure().add_subplot(projection='3d')
    ax.plot(solution[0, :], solution[1,:], solution[2,:], label='curve')
    ax.legend()
    plt.tight_layout()
    plt.show()

def test():
    import numpy as np
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt

    # Define the function M(x)
    def M(x):
        x1, x2, x3 = x  # Unpacking x
        norm = np.sqrt(x1**2 + x2**2)
        if norm == 0:
            return np.array([0, 0, 0])  # Handle division by zero
        return (1 / norm) * np.array([-x2, x1, 0])

    # Define the ODE system
    def ode_system(t, y, eps):
        x = y[:3]  # Position vector
        v = y[3:]  # Velocity vector

        dxdt = v
        dvdt = (1 / eps) * np.cross(v, M(x)) + np.cross(v, [0, 0, 1])

        return np.concatenate((dxdt, dvdt))

    # Set initial conditions
    x0 = np.array([0, 1, 1])
    v0 = np.array([1, 1e-2, 0])  # Assuming eps = 0.01 for this example
    y0 = np.concatenate((x0, v0))

    # Time span
    t_span = (0, 10)
    t_eval = np.linspace(*t_span, 1000)

    # Solve the ODE system
    eps = 0.01  # Define epsilon
    sol = solve_ivp(ode_system, t_span, y0, args=(eps,), t_eval=t_eval, method='RK45')
    ax=plt.figure().add_subplot(projection='3d')
    ax.plot(sol.y[0, :], sol.y[1,:], sol.y[2,:], label='curve')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Plot the solution
    plt.figure(figsize=(10, 5))
    plt.plot(sol.t, sol.y[0], label=r'$x_1$')
    plt.plot(sol.t, sol.y[1], label=r'$x_2$')
    plt.plot(sol.t, sol.y[2], label=r'$x_3$')
    plt.xlabel('Time')
    plt.ylabel('Position Components')
    plt.legend()
    plt.grid()
    plt.show()


if __name__=="__main__":
    time=np.linspace(0, 5, 100000)
    # solution=get_solution(time)
    # plot_solution(solution, 1)
    test()
    # breakpoint()

