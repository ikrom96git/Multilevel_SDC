import numpy as np
from problem_class.HarmonicOscillator import HarmonicOscillator
from plot_class.plot_solutionvstime import plot_solution
def harmonic_oscillator_test_params():
    problem_params=dict()
    problem_params['mu']=3
    problem_params['kappa']=2
    problem_params['F0']=0.0
    problem_params['t0']=0.0
    problem_params['u0']=[1, 0]
    problem_params['F0']=1
    time=np.linspace(0, 6, 1000)
    return problem_params, time
def harmonic_oscillator_test_force_params():
    mu=0.2
    kappa=0.2
    problem_params=dict()
    problem_params['mu']=kappa
    problem_params['kappa']=1/mu
    problem_params['F0']=1/mu
    problem_params['t0']=0.0
    problem_params['u0']=[2, 0]
    time=np.linspace(0, 30, 1000)
    return problem_params,time

def test_solutionWithoutFriction():
    problem_params, time =harmonic_oscillator_test_params()
    problem_params['kappa']=1
    problem_params['u0']=[0, 2]
    problem_params['mu']=0.0
    time=np.linspace(0, 25, 1000)
    title='Without Friction (Figure 4)'
    model=HarmonicOscillator(problem_params=problem_params)
    solution=model.get_solution_ntimeWithoutForce(time)
    exact_solution=exact_solution_WithoutFriction(time, 2, 1)
    solution_set=[solution[0,:], exact_solution]
    label_set=['Reduced model', 'Exact solution']
    plot_solution(time, solution_set, title, label_set)

def exact_solution_WithoutFriction(time, C, omega_0):
    return C*np.sin(omega_0*time)

def test_solution_WithFriction():
    problem_params, time=harmonic_oscillator_test_params()
    model=HarmonicOscillator(problem_params=problem_params)
    label_set=['D<0']
    Title='Figure 5'
    solution=model.get_solution_ntimeWithoutForce(time)
    solution_set=[solution[0,:]]
    plot_solution(time, solution_set, Title, label_set)

def test_solution_WithForce():
    problem_params, time=harmonic_oscillator_test_force_params()
    model=HarmonicOscillator(problem_params=problem_params)
    label_set=['Solution with force']
    Title='Figure 10'
    solution=model.get_solution_ntimeWithForce(time)
    solution_set=[solution[0, :]]
    plot_solution(time, solution_set, Title, label_set)
if __name__=='__main__':
    # test_solutionWithoutFriction()
    # test_solution_WithFriction()
    test_solution_WithForce()
    # problem_params, time=harmonic_oscillator_test_params()
    # model=HarmonicOscillator(problem_params=problem_params)
    # label_set=['solution']
    # solution=model.get_solution_ntimeWithoutForce(time)
    # solution_set=[solution[0,:]]
    # Title='Solution'
    # plot_solution(time, solution_set, Title, label_set)
    