import numpy as np
from problem_class.Magnetic_field import Magnetic_field
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from copy import deepcopy
from plot_class import plot_params
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
EPSILON=0.01
t_end=5
dt=(2*np.pi*EPSILON)/80

def A_matrix(y):
    y1, y2 = y[0], y[1]
    factor = 1 / (y1**2 + y2**2)
    return np.array([
        [y2**2, -y1 * y2, 0],
        [-y1 * y2, y1**2, 0],
        [0, 0, 0]
    ]) * factor

def beta_vector(y, u):
    y1, y2 = y[0], y[1]
    u1, u2 = u[0], u[1]
    factor = 1 / (y1**2 + y2**2)
    return np.array([
        u2 * (u1 * y2 - u2 * y1) * factor,
        u1 * (u2 * y1 - u1 * y2) * factor,
        0
    ])

def system(t, state):
    y = state[:3]  # First 3 elements are y(t)
    u = state[3:]  # Last 3 elements are u(t)
    dy_dt = A_matrix(y) @ u
    du_dt = beta_vector(y, u)
    return np.concatenate((dy_dt, du_dt))

def runge_kutta_4(f, y0, t0, t_end, dt):
    t = t0
    state = y0
    solution = [state]
    time_points = [t]

    while t < t_end:
        k1 = f(t, state)
        k2 = f(t + dt / 2, state + dt / 2 * k1)
        k3 = f(t + dt / 2, state + dt / 2 * k2)
        k4 = f(t + dt, state + dt * k3)
        state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt

        solution.append(state)
        time_points.append(t)

    return np.array(solution)



def ode_solver(xx0):
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
    x0 = np.array([xx0, 1, 1])
    v0 = np.array([1, 1e-2, 0])  # Assuming eps = 0.01 for this example
    y0 = np.concatenate((x0, v0))
    params={'t0':0, 't_end':5, 'dt':0.01}
    magnetic=Magnetic_field(params)
    C_matrix=magnetic.C_matrix
    # Time span
    t_span = (0, 2500)
    dt=0.615
    t_eval=np.arange(*t_span,dt)
    # Solve the ODE system
    EPSILON = [0.1, 0.05, 0.01, 0.005]  # Define epsilon
    G=np.zeros((6, len(t_eval)))
    error=np.zeros((len(EPSILON), len(t_eval)))
    for ee, eps in enumerate(EPSILON):
    # eps = 0.01  # Define epsilon
        theta=(t_eval-0)/eps
        sol = solve_ivp(ode_system, t_span, y0, args=(eps,), t_eval=t_eval, method='RK45')
        asymptotic_solution=runge_kutta_4(system, y0, t_span[0], t_span[1], dt)
        y, u=np.split(asymptotic_solution, 2, axis=1)
        for ii in range(len(t_eval)):
            theta=(t_eval[ii]-0)/eps
            C=C_matrix(theta, y[ii, :])
            # breakpoint()
            G[3:, ii]=C@u[ii,:]
            G[:3, ii]=y[ ii, :]

        
    # compute relative error with l1 norm
        
        for jj in range(len(sol.t)):
            error[ee, jj]=np.linalg.norm(G[:,jj]-sol.y[:,jj], ord=1)/np.linalg.norm(sol.y[:,jj], ord=1)
    
    # plot the error vs time
    colors=['black', 'green', 'blue', 'red']
    fig, ax = plt.subplots()
    [ax.semilogy(sol.t, error[ii],  color=colors[ii]) for ii, eps in enumerate(EPSILON)]
    
    ax.set_ylabel('Relative Error')
    ax.set_xlabel('Time')
    ax.grid()
    ax.set_xlim(t_span[0],t_span[1])

    ax_inset=inset_axes(ax, width='60%', height='50%', loc='center right')
    [ax_inset.semilogy(sol.t[:300], error[ii][:300], label=rf'$\varepsilon={eps}$', color=colors[ii]) for ii, eps in enumerate(EPSILON)]
    ax_inset.tick_params(labelsize=8)
    ax_inset.legend(EPSILON, fontsize=10, loc='lower right')
    ax_inset.grid()
    ax_inset.set_xlim(0, sol.t[300])
    # plt.ylim(0, 1.2)
    # ax.yscale('log')
    ax.set_ylim(1e-3, 2)
    plt.tight_layout()
    ax.legend(EPSILON, loc='lower left')
    plt.savefig('magnetic_field_error.pdf')
    plt.show()
    
    # ax=plt.figure().add_subplot(projection='3d')
    
    # ax.grid(False)
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    # ax.set_zlim(0.8, 1.1)
    # x = ax.get_xticks()
    # y = ax.get_yticks()
    # x, y = np.meshgrid(x, y)
    # z = np.zeros_like(x)
    # ax.plot_wireframe(x, y,z, color='black', alpha=0.15)
    # ax.plot(sol.y[0, :], sol.y[1,:], sol.y[2,:], color='blue')
    # ax.set_xlim(-1.5, 1.5)
    # ax.set_ylim(-1.5, 1.5)
    # z=ax.get_zticks()
    # ax.set_zlim(np.min(z)-0.0000001, np.max(z)+0.00001)
    # ax.set_xlabel('$X(t)$', linespacing=3.2)
    # ax.set_ylabel('$Y(t)$', linespacing=3.2)
    # ax.set_zlabel('$Z(t)$', linespacing=5.2)

    # # z axis in log scale
    # # ax.set_zscale('log')
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig('magnetic_field.pdf')
    # plt.show()
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
    return sol

def plot_solution_projection():
    sol0=ode_solver(0)
    sol1=ode_solver(1)
    
    plt.plot(sol0.y[0], sol0.y[1], label='initial condition in $(5.87)$', color='blue')
    plt.plot(sol1.y[0], sol1.y[1], label='initial condition in $(5.88)$', color='purple')
    plt.legend(loc='upper left')
    plt.grid()
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('$X(t)$')
    plt.ylabel('$Y(t)$')
    plt.tight_layout()
    plt.savefig('magnetic_field_projection.pdf')
    plt.show()



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
    x0 = np.array([0, 1, 1])
    v0 = np.array([1, EPSILON, 0])  # Assuming eps = 0.01 for this example
    y0 = np.concatenate((x0, v0))

    # Time span
    t_span = (0, t_end)
    t_eval = np.linspace(*t_span, 10000)
    # t_eval=np.arange(*t_span,dt)
    # Solve the ODE system
    eps = EPSILON  # Define epsilon
    sol = solve_ivp(ode_system, t_span, y0, args=(eps,), t_eval=t_eval, method='RK45')
    return sol
    # ax=plt.figure().add_subplot(projection='3d')
    
    
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
    # plt.rcParams['axes3d.yaxis.panecolor']=''
    ax=plt.figure().add_subplot(projection='3d')
    # plt.rcParams['axes.xmargin']=False
   
    ax.grid(False)
    grid_lines = []
   
    
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    

    ax.plot(sol.y[0], sol.y[1], sol.y[2], label='curve')
    # make a mesh grid
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)    
    
    x = ax.get_xticks()
    y = ax.get_yticks()
    x, y = np.meshgrid(x, y)
    z = np.zeros_like(x)
    # plot the mesh grid
    # ax.plot_surface(x, y, z, alpha=0.5)
    #mesh like net with dashed lines
    ax.plot_wireframe(x, y, z, color='black', alpha=0.15)
    # set upper bound for x and y axis
    ax.legend()
    ax.set_xlabel('$X(t)$')
    ax.set_ylabel("$Y(t)$")
    ax.set_zlabel("$Z(t)$")

    
   
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
    # plt.show()

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

def plot_values():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Generate sample data (replace with actual numerical solution)
    t = np.linspace(0, 5, 1000)
    eps = 0.01

    X1 = np.cos(2 * np.pi * t)  # Example trajectory 1
    Y1 = np.sin(2 * np.pi * t)
    Z1 = 1 + eps * np.sin(20 * np.pi * t)  # Small oscillations in Z

    X2 = np.cos(2 * np.pi * t) + eps * t   # Example trajectory 2
    Y2 = np.sin(2 * np.pi * t)
    Z2 = 1 - eps * np.sin(20 * np.pi * t)

    fig = plt.figure(figsize=(12, 4))

    # Left: First 3D trajectory
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(X1, Y1, Z1, color='lightblue', alpha=0.7)
    ax1.set_xlabel("X(t)")
    ax1.set_ylabel("Y(t)")
    ax1.set_zlabel("Z(t)")

    # Middle: Second 3D trajectory
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot(X2, Y2, Z2, color='purple', alpha=0.7)
    ax2.set_xlabel("X(t)")
    ax2.set_ylabel("Y(t)")
    ax2.set_zlabel("Z(t)")

    # Right: 2D projection in XY-plane
    ax3 = fig.add_subplot(133)
    ax3.plot(X1, Y1, color='lightblue', label="Initial condition in (37)")
    ax3.plot(X2, Y2, color='purple', label="Initial condition in (36)")
    ax3.set_xlabel("X(t)")
    ax3.set_ylabel("Y(t)")
    ax3.legend()
    ax3.set_title(r"$\varepsilon=0.01, \, T=5$")

    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    time=np.linspace(0, 5, 100000)
    # solution=get_solution(time)
    # plot_solution(solution, 1)
    # solution_asyp()
    # sol=test_RK45()
    # plot_solution(sol)
    ode_solver(1)
    # get_error()
    # breakpoint()
    # plot_values()
    # plot_solution_projection()
# 
