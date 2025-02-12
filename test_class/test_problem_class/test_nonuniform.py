import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from plot_class import plot_params

EPS=0.00001
def nonuniform_electric_field():
    # Parameters
    eps = EPS
    c = 2

    # Right-hand side of the ODE system
    def ode_system(t, y):
        x, v = y[:3], y[3:]
        
        v_perp = np.array([0, v[2], -v[1]])
        E_x = c * np.array([-x[0], x[1] / 2, x[2] / 2])
        
        dxdt = v
        dvdt = (1 / eps) * v_perp + E_x
        
        return np.concatenate([dxdt, dvdt])

    # Initial conditions: [x1, x2, x3, v1, v2, v3]
    y0 = [1, 1, 1, 1, 1, 1]

    # Time span
    t_span = (0, 60)  # From t=0 to t=1
    t_eval = np.linspace(*t_span, 1000000)

    # Solve the system
    sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, method='RK45')

    # Plot results
    labels = ["$x_{1}$", "$x_{2}$", "$x_3$"]

    # [plt.plot(sol.t, sol.y[ii+3], label=rf'$v_{ii+1}$') for ii in range(3)]
    # plt.legend()
    # plt.xlim(left=0)
    # plt.xlim(right=sol.t[-1])
    # plt.ylim((-1.8,1.8))
    # plt.xlabel('Time')
    # plt.ylabel(r'$v_{\varepsilon}(t)$')
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig('velocity_scale.pdf')
    # plt.show()
    

    # [plt.plot(sol.t, sol.y[ii], label=rf'$x_{ii+1}$') for ii in range(3)]
    # plt.legend(loc='lower right')
    # plt.xlim(left=0)
    # plt.xlim(right=sol.t[-1])
    # plt.ylim((-1.8,1.8))
    # plt.xlabel('Time')
    # plt.ylabel(r'$x_{\varepsilon}(t)$')
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig('position_scale.pdf')
    # plt.show()
    return sol
    

def R_matrix(theta):
    R = np.zeros((3, 3))
    R[0, 0] = 1
    R[1, 1] = np.cos(theta)
    R[1, 2] = np.sin(theta)
    R[2, 1] = -np.sin(theta)
    R[2, 2] = np.cos(theta)
    return R

def RR_matrix(theta):
    RR = np.zeros((3, 3))
    RR[1, 1] = np.sin(theta)
    RR[1, 2] = 1 - np.cos(theta)
    RR[2, 1] = np.cos(theta) - 1
    RR[2, 2] = np.sin(theta)
    return RR

def E_field(x, c):
    return c * np.array([-x[0],  x[1]/2, x[2]/2])

def zeros_model_solution(x0, v0,c, t, eps):
    ct = np.sqrt(c) * (t)
    y0 = np.array([x0[0] * np.cos(ct) + (v0[0] / np.sqrt(c)) * np.sin(ct), x0[1], x0[2]])
    u0 = np.array([-x0[0] * np.sqrt(c) * np.sin(ct) + v0[0] * np.cos(ct), v0[1], v0[2]])
    c_half = 0.5 * c * (t )
    y1 = np.array([0, x0[2] * c_half, -x0[1] * c_half])
    u1 = np.array([0, -v0[2] * c_half, v0[1] * c_half])
    theta = t / eps
    A = np.concatenate((y0, R_matrix(theta) @ u0))
    B = np.concatenate((y1 + RR_matrix(theta) @ u0, R_matrix(theta) @ u1 + RR_matrix(theta) @ E_field(y0, c)))
    return A +eps*B

def compare_solutions():
    # Parameters
    eps = EPS
    c = 2

    # Initial conditions
    x0 = np.array([1, 1, 1])
    v0 = np.array([1, 1, 1])

    # Time span
    t_span = (0, 60)  # From t
    t_eval = np.linspace(*t_span, 1000000)
    sol=nonuniform_electric_field()
    asymp_solution=np.zeros((6, len(t_eval)))
    for ii, tt in enumerate(t_eval):
        asymp_solution[:, ii]=zeros_model_solution(x0, v0, c, tt, eps)
    # aymp_solution = zeros_model_solution(x0, v0, c, t_eval, eps)
    for ii in range(6):
        plt.plot(t_eval, sol.y[ii], label=f'odeint {ii}')
        plt.plot(t_eval, asymp_solution[ii], label=f'asymp_solution {ii}')
        plt.legend()
        plt.show()
    # plt.plot(t_eval, sol.y[1], label='odeint')
    # plt.plot(t_eval, asymp_solution[1], label='asymp_solution')
    # plt.legend()
    # plt.show()



def cos_function():
    eps=0.001

    time=np.arange(0,15, 0.1)
    

    scale=[2*np.pi*eps, 1.0, 2*np.pi/eps]
    

    func =[np.cos(time+theta) for theta in scale]


    [plt.plot(time, func[ii], label=f"{scale[ii]}") for ii in range(len(scale))]
    plt.legend()
    plt.show()

if __name__=='__main__':
    # nonuniform_electric_field()
    # cos_function()
    compare_solutions()

    