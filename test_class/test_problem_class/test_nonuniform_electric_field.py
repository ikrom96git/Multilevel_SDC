import numpy as np
import matplotlib.pyplot as plt
from problem_class.Nonuniform_electric_field import Nonuniform_electric_field
from scipy.integrate import odeint


def prob_params():
    prob_params=dict()
    prob_params['u0']=np.array([[1,1,1],[1,1,1]])
    prob_params['c']=2
    prob_params['epsilon']=0.0001
    return prob_params

def Error(G, X):
    abs_norm=np.sum(np.abs(G-X))
    # breakpoint()
    return abs_norm/np.sum(np.abs(X))

def get_solution(time):
    params=prob_params()
    problem_class=Nonuniform_electric_field(params)
    x_0=problem_class.prob.u0[0]
    v_0=problem_class.prob.u0[1]
    asymp_solution=problem_class.asymp_solution(x_0, v_0,time)
    exact_solution=problem_class.exact_solution(x_0, v_0, time)
    return exact_solution, asymp_solution

def get_Error(time):
    Error_data=[]
    for tt in time:
        X, G=get_solution(tt)
        Error_data=np.append(Error_data, Error(G, X))
    return Error_data

def plot_solution(time):
    G_sol=[]
    X_sol=[]
    eps=0.0001
    u0=np.ones(6)
    sol=odeint(right_hand_side, u0, time, args=(eps,))
    
    for tt in time:
        G, X=get_solution(tt)
        G_sol=np.append(G_sol, G[1])
        X_sol=np.append(X_sol, X[1])
    plt.plot(time, G_sol, label='asymptotic solution')
    plt.plot(time, X_sol, label='exact solution')
    plt.plot(time, sol[:,1], label='odeint')
    plt.legend()
    plt.tight_layout()
    plt.show()
        

def plot_Error(Data_error, time):
    plt.semilogy(time, Data_error)
    plt.show()

def right_hand_side(y, t, eps):
    c=2.0
    dydt=[y[3], y[4], y[5], -c*y[0], 1/eps*y[5]+0.5*y[1], -(1/eps)*y[4]+0.5*y[2]]
    return dydt

def get_odeint(time, eps):
    u0=np.ones(6)
    sol=odeint(right_hand_side, u0, time, args=(eps,))
    return sol


if __name__=="__main__":
    time=np.linspace(0, 15,100000)
    Error_data=get_Error(time)
    # plot_Error(Error_data, time)
    plot_solution(time)
    # get_odeint(time, eps=0.001)



