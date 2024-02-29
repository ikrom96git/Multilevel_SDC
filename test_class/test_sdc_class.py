import numpy as np
from sweeper_class.sdc_class import sdc_class
from problem_class.HarmonicOscillator import HarmonicOscillator
from plot_class.plot_solutionvstime import plot_solution
from test_class.test_harmonicoscillator import harmonic_oscillator_test_params, harmonic_oscillator_test_force_params

def sdc_params_without_force():
    problem_params, *_ =harmonic_oscillator_test_params()
    problem_params['dt']=0.1
    problem_params['F0']=None
    collocation_params=dict()
    collocation_params['quad_type']='GAUSS'
    collocation_params['num_nodes']=5
    sweeper_params=dict()
    sweeper_params['Kiter']=5
    return problem_params, collocation_params, sweeper_params


def test_sdc_without_force():
    problem_params, collocation_params, sweeper_params=sdc_params_without_force()
    model=sdc_class(problem_params, collocation_params, sweeper_params)
    X, V=model.sdc_iter(5)
    time=0.1*np.append(0, model.coll.nodes)
    harmonic_oscillator=HarmonicOscillator(problem_params)
    solution=harmonic_oscillator.get_solution_ntimeWithoutForce(time)
    Title='Solution of SDC'
    label_set=['SDC', 'Exact']
    solution_set=[X, solution[0,:]]
    plot_solution(time, solution_set, Title, label_set)


def test_sdc_with_force():
    problem_params, *_=harmonic_oscillator_test_force_params()
    problem_params['dt']=0.1
    *_, collocation_params, sweeper_params=sdc_params_without_force()
    model=sdc_class(problem_params, collocation_params, sweeper_params)
    X, V=model.sdc_iter(10)
    time=0.1*np.append(0, model.coll.nodes)
    harmonic_oscillator=HarmonicOscillator(problem_params)
    solution=harmonic_oscillator.get_solution_ntimeWithForce(time)
    Title='Solution of SDC with Force'
    label_set=['SDC', 'Exact']
    Solution_set=[X, solution[0,:]]
    plot_solution(time, Solution_set, Title, label_set)

def test_sdc_residual():
    pass

def test_sdc_with_collocation_residual():
    pass

def test_sdc_with_scipy_solve():
    pass






if __name__=='__main__':
    # test_sdc_without_force()
    test_sdc_with_force()