import numpy as np
from plot_class.plot_solutionvstime import plot_solution
from default_params.harmonic_oscillator_default_fast_time_params import get_harmonic_oscillator_fast_time_params
from default_params.harmonic_oscillator_reduced_order_model_params import get_harmonic_oscillator_reduced_model_params
from default_params.harmonic_oscillator_default_params import get_harmonic_oscillator_default_params
from problem_class.HarmonicOscillator import HarmonicOscillator
from problem_class.HarmonicOscillator_fast_time_reduced_problem import HarmonicOscillator_fast_time

def test_compare_solution_fast_time_reduced():
    prob_reduced_model_params, time_reduced=get_harmonic_oscillator_reduced_model_params()
    prob_fast_time_params, time_fast=get_harmonic_oscillator_fast_time_params(Fast_time=True)
    model_reduced=HarmonicOscillator(prob_reduced_model_params)
    model_fast_time=HarmonicOscillator_fast_time(prob_fast_time_params)
    solution_reduced=model_reduced.get_solution_ntimeWithForce(time_reduced)
    solution_fast=model_fast_time.get_ntime_exact_solution(time_fast)
    Title='Solution of Fast time'
    label_set=['Exact solution', 'Reduced model solution']
    solution_set=[solution_reduced[0,:], solution_fast[0, :]]
    plot_solution(time_reduced, solution_set, Title, label_set)

def test_compare_solution_slow_time_reduced():
    prob_reduced_model_params, time_reduced=get_harmonic_oscillator_default_params(Force=True)
    prob_fast_time_params, time_fast=get_harmonic_oscillator_fast_time_params(Fast_time=False)
    model_reduced=HarmonicOscillator(prob_reduced_model_params)
    model_fast_time=HarmonicOscillator_fast_time(prob_fast_time_params)
    solution_reduced=model_reduced.get_solution_ntimeWithForce(time_reduced)
    solution_fast=model_fast_time.get_ntime_exact_solution(time_fast)
    Title='Solution of Fast time'
    label_set=['Exact solution', 'Reduced model solution']
    solution_set=[solution_reduced[0,:], solution_fast[0, :]]
    plot_solution(time_reduced, solution_set, Title, label_set)

if __name__=='__main__':
    test_compare_solution_fast_time_reduced()
    # test_compare_solution_slow_time_reduced()
