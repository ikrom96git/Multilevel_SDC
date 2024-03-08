import numpy as np
from default_params.harmonic_oscillator_default_fast_time_params import get_harmonic_oscillator_fast_time_params
from default_params.harmonic_oscillator_reduced_order_model_params import get_harmonic_oscillator_reduced_model_params
from default_params.mlsdc_defautl_params import get_mlsdc_default_params
from problem_class.HarmonicOscillator_fast_time_reduced_problem import HarmonicOscillator_fast_time
from sweeper_class.mlsdc_class import Mlsdc_class
from copy import deepcopy
from plot_class.plot_residual import plot_residual
def test_mlsdc_slow_time_problem():
    problem_slow_time_params, collocation_params, sweeper_params, problem_class=get_mlsdc_default_params(Force=True)
    problem_reduced_params, *_=get_harmonic_oscillator_fast_time_params(Fast_time=False)
    problem_class_reduced=deepcopy(problem_class)
    problem_class_reduced[1]=HarmonicOscillator_fast_time
    problem_params=[problem_slow_time_params, problem_reduced_params]
    model_reduced_mlsdc=Mlsdc_class(problem_params, collocation_params, sweeper_params, problem_class_reduced)
    model_mlsdc=Mlsdc_class(problem_slow_time_params, collocation_params, sweeper_params, problem_class)
    X_reduced, V_reduced=model_reduced_mlsdc.get_mlsdc_iter_solution()
    X_mlsdc, V_mlsdc=model_mlsdc.get_mlsdc_iter_solution()
    Residual_mlsdc=model_mlsdc.sdc_fine_level.get_residual
    Residual_reduced=model_reduced_mlsdc.sdc_fine_level.get_residual
    Kiter=np.arange(1, 11, 1)
    Title='Residual MLSDC'
    label_set=['MLSDC', 'Reduced model on coarse level']
    residual_set=[np.array(Residual_mlsdc)[:, 0], np.array(Residual_reduced)[:, 0]]
    plot_residual(Kiter, residual_set, Title, label_set)

if __name__=='__main__':
    test_mlsdc_slow_time_problem()


    




    
