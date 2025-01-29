import numpy as np
from problem_class.Magnetic_field import Magnetic_field
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from copy import deepcopy
EPSILON=0.01
t_end=50
dt=(2*np.pi*EPSILON)/80
def Error(G, X):
    abs_norm=np.sum(np.abs(G-X))
    # breakpoint()
    return abs_norm/np.sum(np.abs(X))
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

def test_RK45():
    # Define the function M(x)
    
    # Set initial conditions
    x0 = np.array([1, 1, 1])
    v0 = np.array([1, EPSILON, 0])  # Assuming eps = 0.01 for this example
    y0 = np.concatenate((x0, v0))

    # Time span
    t_span = (0, t_end)
    t_eval = np.linspace(*t_span, 1000)
    t_eval=np.arange(*t_span,dt)
    # Solve the ODE system
    eps = EPSILON  # Define epsilon
    sol = solve_ivp(ode_system, t_span, y0, args=(eps,), t_eval=t_eval, method='RK45')
    return sol
    # ax=plt.figure().add_subplot(projection='3d')
    
    # ax.plot(sol.y[0, :], sol.y[1,:], sol.y[2,:], label='curve')
    # ax.legend()
    # plt.tight_layout()
    # plt.show()
    # # breakpoint()
    # # Plot the solution
    # plt.figure(figsize=(10, 5))
    # plt.plot(sol.t, sol.y[0], label=r'$x_1$')
    # plt.plot(sol.t, sol.y[1], label=r'$x_2$')
    # plt.plot(sol.t, sol.y[2], label=r'$x_3$')
    # plt.xlabel('Time')
    # plt.ylabel('Position Components')
    # plt.legend()
    # plt.grid()
    # plt.show()

def A_bar(y):
     norm=y[0]**2+y[1]**2
     A=np.zeros((3,3))
     A[0,0]=y[1]**2/norm
     A[0,1]=(-y[0]*y[1])/norm
     A[1,0]=deepcopy(A[0,1])
     A[1,1]=y[0]**2/norm
     A[2,2]=A[2,0]=A[2,1]=A[1,2]=A[0,2]=1e-4

     return A

def beta_bar(y, u):
     norm=y[0]**2+y[1]**2
     prod=u[0]*y[1]-u[1]*y[0]
     beta=np.zeros(3)
     beta[0]=(u[1]*prod)/norm
     beta[1]=-(u[0]*prod)/norm
     beta[2]=1e-6
     return beta

def function_ode(t, y):
    y0 = y[:3]  # Position vector
    u0 = y[3:]  # Velocity vector
    dy0dt=A_bar(y0)@u0
    du0dt=beta_bar(y0, u0)

    return np.concatenate((dy0dt, du0dt))

def C_matrix(theta, x):
    norm=x[0]**2+x[1]**2
    sin_norm=np.sin(theta)/norm
    C=np.zeros((3,3))
    C[0,0]=(x[0]**2*np.cos(theta)+x[1]**2)/norm
    C[0,1]=C[1,0]=(x[0]*x[1]*(np.cos(theta)-1))/norm
    C[0,2]=-(x[0]*np.sin(theta))/np.sqrt(norm)
    C[1,1]=(x[1]**2*np.cos(theta)+x[0]**2)/norm
    C[1,2]=-(x[1]*np.sin(theta))/np.sqrt(norm)
    C[2,0]=-deepcopy(C[0,2])
    C[2,1]=-deepcopy(C[1,2])
    C[2,2]=np.cos(theta)
    return C


def G_asym(y, time, eps, s=0):
    theta=(time-s)/eps

    G=deepcopy(y)
    
    
    for ii in range(len(time)):
        theta=(time[ii]-s)/eps
        C=C_matrix(theta, y[:3,ii])

        G[3:, ii]=C@y[3:,ii]
    return G


def solution_asyp():
     # Set initial conditions
    x0 = np.array([0, 1, 1])
    v0 = np.array([1, EPSILON, 0])  # Assuming eps = 0.01 for this example
    y0 = np.concatenate((x0, v0))

    # Time span
    t_span = (0, t_end)
    t_eval = np.linspace(*t_span, 1000000)
    t_eval=np.arange(*t_span,dt)
    # Solve the ODE system
    eps = EPSILON  # Define epsilon
    sol = solve_ivp(function_ode, t_span, y0, t_eval=t_eval, method='RK45')
    G=G_asym(sol.y, t_eval, eps)
    return G, sol

def plot_solution(sol):
    
    ax=plt.figure().add_subplot(projection='3d')
    ax.plot(sol.y[0], sol.y[1], sol.y[2], label='curve')
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

def get_error():
    G, Solution=solution_asyp()
    sol=test_RK45()
    plot_solution(sol)
    # plot_solution(Solution)
    error=[]
    for ii in range(len(sol.t)):
        error=np.append(error, Error(G[:, ii], sol.y[:, ii]))
    plt.plot(sol.t, error)
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    time=np.linspace(0, 5, 100000)
    # solution=get_solution(time)
    # plot_solution(solution, 1)
    # solution_asyp()
    # test_RK45()
    get_error()
    # breakpoint()

