import numpy as np
from problem_class.asymptotic_problem import Fast_time
from sdc_class.sdc_method_fast_time import SDC_method_fast_time
import matplotlib.pyplot as plt

# initialize the problem parameters
def prob_params():
    problem_params = dict()
    eps=0.001
    problem_params['kappan']=0.8
    problem_params['c']=1/eps
    problem_params['eps']=eps
    problem_params['dt']=0.01
    problem_params['u0']=np.array([2.0,0.0])
    problem_params['f0']=1/eps
    problem_params['initial_guess']='spread'
    problem_params['Kiter']=10
    collocation_params = dict()
    collocation_params['num_nodes'] =5
    collocation_params['quad_type'] = 'LOBATTO'
    return problem_params, collocation_params

# initialize the SDC parameters
def sdc_method():
    problem_params, collocation_params = prob_params()
    model=SDC_method_fast_time(problem_params,collocation_params)
    U=model.run_sdc()
    X, V=np.split(U,2)
    nodes=np.append(0,problem_params['dt']* model.coll.nodes)
    return X, nodes

def fast_time():
    problem_params, collocation_params = prob_params()
    model_sdc=SDC_method_fast_time(problem_params,collocation_params)
    t=np.linspace(0,problem_params['dt'],1000)

    nodes=np.append(0,problem_params['dt']* model_sdc.coll.nodes)
    model=Fast_time(problem_params)
    solution=model.fast_time(nodes)
    
    return solution, t

def test_plot():
    U, nodes = sdc_method()
    print(U)
    solution, t = fast_time()
    print(solution)
    plt.plot(nodes,solution, label='exact')
    plt.plot(nodes,U, label='SDC')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    test_plot()




