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


EPSILON=0.1

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
        DuffingEquation_first_order_problem,
    ]
    problem_params_reduced = [
        problem_duffing_params,
        problem_duffing_zeros_params,
        problem_duffing_first_params,
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
    residual_m3lsdc_asyp_zero, *_=duffing_m3lsdc_Asyptotic(order=0)
    Kiter = np.arange(1, sweeper_params["Kiter"] + 1, 1)
    Title = rf"$\varepsilon={EPSILON}$, fine-level residual"
    label_set = [
        "MLSDC ",
        # "M3LSDC standart",
        "M3LSDC asyptotic $\mathcal{O}^{0}$",
        "M3LSDC asyptotic $\mathcal{O}^{1}$"
    ]
    residual_set = [
        np.array(residual_mlsdc)[:, 0],
        # np.array(residual_m3lsdc_standart)[:, 0],
        np.array(residual_m3lsdc_asyp_zero)[:,0],
        np.array(residual_m3lsdc_asyp)[:, 0]
       
    ]
    plot_residual(Kiter, residual_set, Title, label_set)

def test_solution():
    *_, mlsdc_pos, nodes =duffing_mlsdc()
    *_, m3lsdc_pos=duffing_m3lsdc_standart()
    Title="Solution"
    label_set=['mlsdc', 'm3lsdc_standart']
    solution_set=[mlsdc_pos, m3lsdc_pos]
    nodes=np.append(0, nodes)
    plot_solution(nodes, solution_set, Title, label_set) 

if __name__=='__main__':
    test_residual()
    test_solution()
