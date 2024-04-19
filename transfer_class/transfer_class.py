import numpy as np
from sweeper_class.sdc_class import sdc_class
from copy import deepcopy
from core.Lagrange import LagrangeApproximation


class transfer_class(object):
    def __init__(
        self, problem_params, collocation_params, sweeper_params, problem_class
    ):
        self.get_sorted_params(
            problem_params, collocation_params, sweeper_params, problem_class
        )
        self.Pcoll = self.get_transfer_matrix_Q(
            self.sdc_fine_level.coll.nodes, self.sdc_coarse_level.coll.nodes
        )

        self.Rcoll = self.get_transfer_matrix_Q(
            self.sdc_coarse_level.coll.nodes, self.sdc_fine_level.coll.nodes
        )

    def get_sorted_params(
        self, problem_params, collocation_params, sweeper_params, problem_class
    ):

        if len(problem_params) == 2:
            problem_params_fine = problem_params[0]
            problem_params_coarse = problem_params[1]
        else:
            problem_params_fine = problem_params
            problem_params_coarse = problem_params
        problem_class_fine = problem_class[0]
        collocation_params_fine = deepcopy(collocation_params)
        collocation_params_fine["num_nodes"] = collocation_params["num_nodes"][0]
        collocation_params_coarse = deepcopy(collocation_params)
        collocation_params_coarse["num_nodes"] = collocation_params["num_nodes"][1]
        if len(problem_class) == 1:
            problem_class_coarse = problem_class[0]
        elif len(problem_class) == 2:
            problem_class_coarse = problem_class[1]
        else:
            problem_class_coarse = problem_class[1]
            problem_class_coarse_first = problem_class[2]
            self.sdc_coarse_first_order = sdc_class(
                problem_params_coarse,
                collocation_params_coarse,
                sweeper_params,
                problem_class_coarse_first,
            )

        self.sdc_fine_level = sdc_class(
            problem_params_fine,
            collocation_params_fine,
            sweeper_params,
            problem_class_fine,
        )
        self.sdc_coarse_level = sdc_class(
            problem_params_coarse,
            collocation_params_coarse,
            sweeper_params,
            problem_class_coarse,
        )

    @staticmethod
    def get_transfer_matrix_Q(f_nodes, c_nodes):
        approx = LagrangeApproximation(c_nodes)
        return approx.getInterpolationMatrix(f_nodes)

    def restrict(self, U):
        return self.Rcoll @ U

    def interpolate(self, U):
        return np.append(U[0], self.Pcoll @ U[1:])

    def fas_correction(self, X_fine, V_fine, fine_level=None, coarse_level=None):
        if fine_level is None:
            fine_level = self.sdc_fine_level
        if coarse_level is None:
            coarse_level = self.sdc_coarse_level
        X_coarse = self.restrict(X_fine[1:])
        V_coarse = self.restrict(V_fine[1:])
        dt_fine = fine_level.prob.dt
        dt_coarse = coarse_level.prob.dt
        F_fine = fine_level.build_f(
            X_fine[1:], V_fine[1:], dt_fine * fine_level.coll.nodes
        )
        F_coarse = coarse_level.build_f(
            X_coarse, V_coarse, dt_coarse * coarse_level.coll.nodes
        )
        RF_fine_vel = self.restrict(fine_level.coll.Q[1:, 1:] @ F_fine)
        RF_coarse_vel = coarse_level.coll.Q[1:, 1:] @ F_coarse
        RF_fine_pos = self.restrict(fine_level.coll.QQ[1:, 1:] @ F_fine)
        RF_coarse_pos = coarse_level.coll.QQ[1:, 1:] @ F_coarse
        tau_pos = ((dt_fine**2) * RF_fine_pos) - ((dt_coarse) ** 2 * RF_coarse_pos)
        tau_vel = dt_fine * RF_fine_vel - dt_coarse * RF_coarse_vel
        X_coarse = np.append(X_fine[0], X_coarse)
        V_coarse = np.append(V_fine[0], V_coarse)
        tau_pos = np.append(0.0, tau_pos)
        tau_vel = np.append(0.0, tau_vel)
        return tau_pos, tau_vel, X_coarse, V_coarse
