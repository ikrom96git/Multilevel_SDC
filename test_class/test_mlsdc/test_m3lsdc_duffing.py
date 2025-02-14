import numpy as np

from default_params.duffing_equation_default_params import (
    get_duffing_equation_params,
    get_duffing_zeros_order_params,
    get_duffing_first_order_params,
)
from default_params.mlsdc_defautl_params import get_mlsdc_default_params
from problem_class.DuffingEquation import (
    DuffingEquation,
    DuffingEquation_zeros_order_problem,
    DuffingEquation_first_order_problem,
)
from sweeper_class.sdc_class import sdc_class
from sweeper_class.mlsdc_class import Mlsdc_class
from plot_class.plot_residual import plot_residual
from plot_class.plot_solutionvstime import plot_solution
from transfer_class.standart_restriction_class import StandartRestriction
from transfer_class.asymptotic_restriction_class import AsymptoticRestriction
from transfer_class.optimazation_restriction_class import (
    OptimationRestriction,
    OptimazationResidual,
)
from transfer_class.restriction import Restriction
from sweeper_class.sdc_class import sdc_class
from scipy.integrate import solve_ivp
from copy import deepcopy


EPSILON=0.01

problem_params, collocation_params, sweeper_params, *_=get_mlsdc_default_params()


def duffing_mlsdc():
    problem_duffing_params = get_duffing_equation_params(EPSILON)
    problem_class_mlsdc = [DuffingEquation, DuffingEquation]
    model_mlsdc = Mlsdc_class(
        problem_duffing_params,
        collocation_params,
        sweeper_params,
        problem_class_mlsdc,
        Restriction,
        eps=EPSILON,
    )
    mlsdc_pos, mlsdc_vel=model_mlsdc.get_mlsdc_iter_solution()
    residual_mlsdc=model_mlsdc.sdc_fine_model.get_residual
    nodes=model_mlsdc.sdc_fine_model.coll.nodes
    
    return residual_mlsdc, mlsdc_pos, nodes

def duffing_m3lsdc_standart():
    problem_duffing_params=get_duffing_equation_params(EPSILON)
    problem_duffing_zeros_params=get_duffing_zeros_order_params(EPSILON)
    problem_duffing_first_params = get_duffing_first_order_params(EPSILON)
    problem_class_reduced = [
        DuffingEquation,
        DuffingEquation_zeros_order_problem,
        # DuffingEquation_first_order_problem,
    ]
    problem_params_reduced = [
        problem_duffing_params,
        problem_duffing_zeros_params,
        # problem_duffing_first_params,
    ]
    model_standart_mlsdc = Mlsdc_class(
        problem_params_reduced,
        collocation_params,
        sweeper_params,
        problem_class_reduced,
        StandartRestriction,
        eps=EPSILON,
    )
    m3lsdc_pos, m3lsdc_vel=model_standart_mlsdc.get_mlsdc_iter_solution()
    residual_m3lsdc=model_standart_mlsdc.sdc_fine_model.get_residual
    return residual_m3lsdc, m3lsdc_pos

def duffing_sdc():
    problem_param=get_duffing_equation_params(eps=EPSILON)
    collocation_sdc=deepcopy(collocation_params)
    collocation_sdc['num_nodes']=5
    model=sdc_class(problem_param, collocation_sdc, sweeper_params, DuffingEquation)
    model.sdc_iter()
    # breakpoint()
    return model.get_residual


def duffing_m3lsdc_Asyptotic(order=1):
    problem_duffing_params=get_duffing_equation_params(EPSILON)
    problem_duffing_zeros_params=get_duffing_zeros_order_params(EPSILON)
    problem_duffing_first_params = get_duffing_first_order_params(EPSILON)
    if order==1:
        problem_class_reduced = [
            DuffingEquation,
            DuffingEquation_zeros_order_problem,
            DuffingEquation_first_order_problem,
        ]
        problem_params_reduced = [
            problem_duffing_params,
            problem_duffing_zeros_params,
            problem_duffing_first_params,
        ]
    else:
        problem_class_reduced = [
            DuffingEquation,
            DuffingEquation_zeros_order_problem,
            # DuffingEquation_first_order_problem,
        ]
        problem_params_reduced = [
            problem_duffing_params,
            problem_duffing_zeros_params,
            # problem_duffing_first_params,
        ]
    model_standart_mlsdc = Mlsdc_class(
        problem_params_reduced,
        collocation_params,
        sweeper_params,
        problem_class_reduced,
        AsymptoticRestriction,
        eps=EPSILON,
    )
    m3lsdc_pos, m3lsdc_vel=model_standart_mlsdc.get_mlsdc_iter_solution()
    residual_m3lsdc=model_standart_mlsdc.sdc_fine_model.get_residual
    return residual_m3lsdc, m3lsdc_pos
def test_residual():
    residual_mlsdc, *_=duffing_mlsdc()
    # residual_m3lsdc_standart, *_=duffing_m3lsdc_standart()
    residual_m3lsdc_asyp, *_=duffing_m3lsdc_Asyptotic()
    # breakpoint()
    residual_m3lsdc_asyp_zero, *_=duffing_m3lsdc_Asyptotic(order=0)
    residual_sdc=duffing_sdc()
    # breakpoint()
    nodes=collocation_params['num_nodes']
    Kiter = np.arange(1, sweeper_params["Kiter"] + 1, 1)
    Title = rf"$\varepsilon={EPSILON}$, $M_f={nodes[0]}, \ M_c={nodes[1]}$"
    label_set = [
        'SDC',
        "MLSDC ",
        # "M3LSDC standart",
        r"M3LSDC Avg. $\mathcal{O}(\varepsilon^{0})$",
        r"M3LSDC Avg. $\mathcal{O}(\varepsilon^{1})$"
    ]
    residual_set = [
        np.array(residual_sdc)[:,0],
        np.array(residual_mlsdc)[:, 0],
        # np.array(residual_m3lsdc_standart)[:, 0],
        np.array(residual_m3lsdc_asyp_zero)[:,0],
        np.array(residual_m3lsdc_asyp)[:, 0]
       
    ]
    filename=f'residual_duffing{EPSILON}.pdf'
    plot_residual(Kiter, residual_set, Title, label_set, filename)

def test_solution():
    *_, mlsdc_pos, nodes =duffing_mlsdc()
    *_, m3lsdc_pos=duffing_m3lsdc_standart()
    problem_params=get_duffing_equation_params(EPSILON)
    prob=DuffingEquation(problem_params)
    nodes=4*np.append(0, nodes)
    solution=prob.get_ntime_exact_solution(nodes)
    Title=f"$\epsilon={EPSILON}$ Solution"
    label_set=['Asyptotic solution','mlsdc', 'm3lsdc_standart']
    solution_set=[solution, mlsdc_pos, m3lsdc_pos]
    plot_solution(nodes, solution_set, Title, label_set) 

if __name__=='__main__':
    test_residual()
    # test_solution()
