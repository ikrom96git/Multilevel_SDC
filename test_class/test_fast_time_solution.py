import numpy as np
from plot_class.plot_solutionvstime import plot_solution
from plot_class.plot_residual import plot_residual
from default_params.harmonic_oscillator_default_fast_time_params import (
    get_harmonic_oscillator_fast_time_params,
)
from default_params.harmonic_oscillator_reduced_order_model_params import (
    get_harmonic_oscillator_reduced_model_params,
)
from default_params.harmonic_oscillator_default_params import (
    get_harmonic_oscillator_default_params,
)
from sweeper_class.sdc_class import sdc_class
from default_params.sdc_default_params import get_sdc_default_params
from problem_class.HarmonicOscillator import HarmonicOscillator
from problem_class.HarmonicOscillator_fast_time_reduced_problem import (
    HarmonicOscillator_fast_time, HarmonicOscillator_fast_time_first_order
)


def test_compare_solution_fast_time_reduced():
    prob_reduced_model_params, time_reduced = (
        get_harmonic_oscillator_reduced_model_params()
    )
    prob_fast_time_params, time_fast = get_harmonic_oscillator_fast_time_params(
        Fast_time=True
    )
    model_reduced = HarmonicOscillator(prob_reduced_model_params)
    model_fast_time = HarmonicOscillator_fast_time(prob_fast_time_params)
    solution_reduced = model_reduced.get_solution_ntimeWithForce(time_reduced)
    solution_fast = model_fast_time.get_ntime_exact_solution(time_fast)
    Title = "Solution of Fast time"
    label_set = ["Exact solution", "Reduced model solution"]
    solution_set = [solution_reduced[0, :], solution_fast[0, :]]
   
    plot_solution(time_reduced, solution_set, Title, label_set)
    


def test_compare_solution_slow_time_reduced():
    prob_reduced_model_params, time_reduced = get_harmonic_oscillator_default_params(
        Force=True
    )
    prob_fast_time_params, time_fast = get_harmonic_oscillator_fast_time_params(
        Fast_time=False
    )
    model_reduced = HarmonicOscillator(prob_reduced_model_params)
    model_fast_time = HarmonicOscillator_fast_time(prob_fast_time_params)
    solution_reduced = model_reduced.get_solution_ntimeWithForce(time_reduced)
    solution_fast = model_fast_time.get_ntime_exact_solution(time_fast)
    Title = "Solution of Fast time"
    label_set = ["Exact solution", "Reduced model solution"]
    solution_set = [solution_reduced[0, :], solution_fast[0, :]]
    plot_solution(time_reduced, solution_set, Title, label_set)

def test_fast_time_SDC():
    eps = 0.001
    problem_fast_time_params,time_fast =get_harmonic_oscillator_fast_time_params(Fast_time=True)
    *_, collocation_params, sweeper_params, problem_class=get_sdc_default_params()
    problem_class_fast_time=HarmonicOscillator_fast_time
    model_sdc=sdc_class(problem_fast_time_params, collocation_params, sweeper_params, problem_class_fast_time)
    X, V=model_sdc.sdc_iter(10)
    
    time=0.1*np.append(0, model_sdc.coll.nodes)
    time_fast=time/np.sqrt(eps)
    model_fast_time = HarmonicOscillator_fast_time(problem_fast_time_params)
    solution_fast = model_fast_time.get_ntime_exact_solution(time_fast)
    prob_reduced_model_params, time_reduced = (
        get_harmonic_oscillator_reduced_model_params()
    )
    model_reduced = HarmonicOscillator(prob_reduced_model_params)
    solution_reduced = model_reduced.get_solution_ntimeWithForce(time)
    title='Solution of SDC'
    label_set=['SDC', 'fast time', 'Exact solution']
    solution_set=[X, solution_fast[0,:], solution_reduced[0,:]]
    plot_solution(time, solution_set, title, label_set)

def test_residual_fast_time():
    K=20
    problem__fast_time_params, *_=get_harmonic_oscillator_fast_time_params(Fast_time=True)
    *_, collocation_params, sweeper_params, problem_class=get_sdc_default_params()
    problem_class_fast_time=HarmonicOscillator_fast_time
    sweeper_params['initial_guess']='collocation'
    model_sdc=sdc_class(problem__fast_time_params, collocation_params, sweeper_params, problem_class_fast_time)
    pos_solution, vel_solution=model_sdc.sdc_iter(K)
    residual_sdc=model_sdc.get_residual
    Kiter=np.arange(1, K+1, 1)
    Title='Residual'
    label_set=['Position', 'Velocity']
    resdual_set=[np.array(residual_sdc)[:,0], np.array(residual_sdc)[:,1]]
    plot_residual(Kiter, resdual_set, Title, label_set)

def test_fast_time_first_order_model():
    
    prob_reduced_model_params, time_reduced = (
        get_harmonic_oscillator_reduced_model_params()
    )
    prob_fast_time_params, time_fast = get_harmonic_oscillator_fast_time_params(
        Fast_time=True
    )
    
    model_reduced = HarmonicOscillator(prob_reduced_model_params)
    model_fast_time = HarmonicOscillator_fast_time(prob_fast_time_params)
    model_first_order=HarmonicOscillator_fast_time_first_order(prob_fast_time_params)
    solution_reduced = model_reduced.get_solution_ntimeWithForce(time_reduced)
    solution_fast = model_fast_time.get_ntime_exact_solution(time_fast)
    solution_first=model_first_order.get_ntime_exact_solution(time_fast)
    position=model_first_order.asyp_expansion(solution_fast[0,:], solution_first[0,:], eps=0.001)
    Title = "Solution of Fast time"
    label_set = ["Exact solution", "Reduced model solution", "First order"]
    solution_set = [solution_reduced[0, :], solution_fast[0, :], position]
   
    plot_solution(time_reduced, solution_set, Title, label_set)
    

if __name__ == "__main__":
    # test_compare_solution_fast_time_reduced()
    # test_compare_solution_slow_time_reduced()
    # test_fast_time_SDC()
    # test_residual_fast_time()
    test_fast_time_first_order_model()
