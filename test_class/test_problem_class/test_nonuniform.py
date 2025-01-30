import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from plot_class import plot_params


def nonuniform_electric_field():
    # Parameters
    eps = 0.1
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
    

    [plt.plot(sol.t, sol.y[ii], label=rf'$x_{ii+1}$') for ii in range(3)]
    plt.legend()
    plt.xlim(left=0)
    plt.xlim(right=sol.t[-1])
    plt.ylim((-1.8,1.8))
    plt.xlabel('Time')
    plt.ylabel(r'$x_{\varepsilon}(t)$')
    plt.grid()
    plt.tight_layout()
    plt.savefig('position_scale.pdf')
    plt.show()
    

    

def cos_function():
    eps=0.001

    time=np.arange(0,15, 0.1)
    

    scale=[2*np.pi*eps, 1.0, 2*np.pi/eps]
    

    func =[np.cos(time+theta) for theta in scale]


    [plt.plot(time, func[ii], label=f"{scale[ii]}") for ii in range(len(scale))]
    plt.legend()
    plt.show()

if __name__=='__main__':
    nonuniform_electric_field()
    # cos_function()

    