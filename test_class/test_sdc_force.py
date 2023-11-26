import numpy as np
from problem_class.asymptotic_problem import Fast_time
from sdc_class.sdc_method_fast_time import SDC_method_fast_time
import matplotlib.pyplot as plt
from sdc_class.sdc_method_force import SDC_method_force
from problem_class.reduced_HO import Reduced_HO
# initialize the problem parameters
def prob_params():
    problem_params = dict()
    eps = 0.001
    problem_params["kappa"] = 0.8
    problem_params["c"] = 1 / eps
    problem_params["eps"] = eps
    problem_params["dt"] = 0.1
    problem_params["u0"] = np.array([2.0, 0.0])
    problem_params["f0"] = 1 / eps
    problem_params["initial_guess"] = "spread"
    problem_params["Kiter"] = 10
    collocation_params = dict()
    collocation_params["num_nodes"] =5 
    collocation_params["quad_type"] = "GAUSS"
    return problem_params, collocation_params
def u_exact():
    problem_params, collocation_params = prob_params()
    u_ex=Reduced_HO(problem_params)
    model_sdc=SDC_method_fast_time(problem_params, collocation_params)
    t=np.append(0, problem_params['dt']*model_sdc.coll.nodes)
    reduced_sol=u_ex.get_sol(t)
    X, V=np.split(reduced_sol, 2) 
    return X, t
def sdc_method_force():
    problem_params, collocation_params = prob_params()
    sdc_method = SDC_method_force(problem_params, collocation_params)
    U=sdc_method.collocation_solution()
    X, V=np.split(U, 2)
    nodes=np.append(0, sdc_method.dt*sdc_method.coll.nodes)
    return X, nodes
    

def solution():
    problem_params, collocation_params = prob_params()
    model_sdc=SDC_method_fast_time(problem_params, collocation_params)
    t=np.append(0, problem_params['dt']*model_sdc.coll.nodes)
    T=np.linspace(0, problem_params['dt'], 1000)
    fast_time = Fast_time(problem_params)
    fast_sol=fast_time.fast_time(T)
    return fast_sol, T

def plot():
    X, nodes=sdc_method_force()
    fast_sol, T=solution()
    reduced_sol, t=u_exact()
    plt.plot(nodes, X, 'o', label='SDC')
    plt.plot(T, fast_sol, label='Exact')
    plt.plot(t, reduced_sol[0], label='Reduced')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot()
