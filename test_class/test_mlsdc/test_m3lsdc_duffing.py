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
    nodes=model_mlsdc.sdc_fine_model.coll.nodes*model_mlsdc.sdc_fine_model.prob.dt
    
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
    pos, vel=model.sdc_iter()
    # breakpoint()
    return model.get_residual, pos


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
        r"M3LSDC averaging $\mathcal{O}(\varepsilon^{0})$",
        r"M3LSDC averaging $\mathcal{O}(\varepsilon^{1})$"
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

# Define the system of ODEs
def duffing_deriv(t, y, epsilon):
    x, v = y  # v = dx/dt
    dxdt = v
    dvdt = -x - epsilon * x**3  # Duffing equation
    return [dxdt, dvdt]
def duffing_solution(t_eval):
    # Parameters
    epsilon = EPSILON  # Nonlinearity parameter
    x0 = 2  # Initial displacement
    v0 = 0  # Initial velocity
    y0 = [x0, v0]  # Initial condition

    # Time span
    # t_span = (0, 20)  # From t=0 to t=20
    # t_eval = np.linspace(0, 20, 1000)  # High-resolution output times
    t_span=(np.min(t_eval), np.max(t_eval))
    # Solve the ODE with high precision using RK45 (or DOP853 for even higher order)
    sol = solve_ivp(
        duffing_deriv, t_span, y0, args=(epsilon,), method="DOP853",
        t_eval=t_eval, rtol=1e-16, atol=1e-16  # High precision tolerances
    )

    # Extract solutions
    t = sol.t
    x = sol.y[0]
    v = sol.y[1]
    return x


def test_convergence():
    import matplotlib.pyplot as plt
    # Get computed solutions
    _, mlsdc_pos, nodes = duffing_mlsdc()
    residual_m3lsdc_asyp, m3lsdc_asym1_pos, *_ = duffing_m3lsdc_Asyptotic()
    residual_m3lsdc_asyp_zero, m3lsdc_asym0_pos, *_ = duffing_m3lsdc_Asyptotic(order=0)
    residual_sdc, sdc_pos = duffing_sdc()
    exact_solution_pos = duffing_solution(nodes)

    # Compute absolute errors
    error_mlsdc = np.abs(mlsdc_pos[1:] - exact_solution_pos)
    error_m3lsdc_asym1 = np.abs(m3lsdc_asym1_pos[1:] - exact_solution_pos)
    error_m3lsdc_asym0 = np.abs(m3lsdc_asym0_pos[1:] - exact_solution_pos)
    error_sdc = np.abs(sdc_pos[1:] - exact_solution_pos)

    # Plot absolute errors
    plt.figure(figsize=(10, 6))
    plt.plot(nodes, error_mlsdc, label="MLSDC Error", linestyle="solid")
    plt.plot(nodes, error_m3lsdc_asym1, label="M3LSDC (Asymptotic, order=1) Error", linestyle="dashed")
    plt.plot(nodes, error_m3lsdc_asym0, label="M3LSDC (Asymptotic, order=0) Error", linestyle="dotted")
    plt.plot(nodes, error_sdc, label="SDC Error", linestyle="dashdot")

    plt.yscale("log")  # Log scale for better visualization
    plt.xlabel("Time")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error Comparison of Duffing Equation Solvers")
    plt.legend()
    plt.grid()
    plt.show()


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
    # test_residual()
    # test_solution()
    test_convergence()