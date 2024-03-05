import numpy as np
from sweeper_class.sdc_class import sdc_class
from copy import deepcopy
from core.Lagrange import LagrangeApproximation


class transfer_class(object):
    def __init__(
        self, problem_params, collocation_params, sweeper_params, problem_class
    ):
        if len(problem_class) == 1:
            problem_class_fine = problem_class[0]
            problem_class_coarse = problem_class[0]
        else:
            problem_class_fine = problem_class[0]
            problem_class_coarse = problem_class[1]
        collocation_params_fine = deepcopy(collocation_params)
        collocation_params_fine["num_nodes"] = collocation_params["num_nodes"][0]
        collocation_params_coarse = deepcopy(collocation_params)
        collocation_params_coarse["num_nodes"] = collocation_params["num_nodes"][1]
        self.sdc_fine_level = sdc_class(
            problem_params, collocation_params_fine, sweeper_params, problem_class_fine
        )
        self.sdc_coarse_level = sdc_class(
            problem_params,
            collocation_params_coarse,
            sweeper_params,
            problem_class_coarse,
        )
        self.Pcoll = self.get_transfer_matrix_Q(
            self.sdc_fine_level.coll.nodes, self.sdc_coarse_level.coll.nodes
        )

        self.Rcoll = self.get_transfer_matrix_Q(
            self.sdc_coarse_level.coll.nodes, self.sdc_fine_level.coll.nodes
        )

    @staticmethod
    def get_transfer_matrix_Q(f_nodes, c_nodes):
        approx = LagrangeApproximation(c_nodes)
        return approx.getInterpolationMatrix(f_nodes)

    def restrict(self, U):
        U_restricted = self.Rcoll @ U
        return U_restricted

    def interpolate(self, U):
        U_interpolated = self.Pcoll @ U[1:]
        U_interpolated = np.append(U[0], U_interpolated)
        return U_interpolated

    def fas_correction(self, X_fine, V_fine):
        X = X_fine[1:]
        V = V_fine[1:]
        X_coarse = self.restrict(X)
        V_coarse = self.restrict(V)
        F_fine = self.sdc_fine_level.build_f(X, V, self.sdc_fine_level.coll.nodes)
        F_coarse = self.sdc_coarse_level.build_f(
            X_coarse, V_coarse, self.sdc_coarse_level.coll.nodes
        )
        RF_fine_vel = self.restrict(self.sdc_fine_level.coll.Q[1:, 1:] @ F_fine)
        RF_coarse_vel = self.sdc_coarse_level.coll.Q[1:, 1:] @ F_coarse
        RF_fine_pos = self.restrict(self.sdc_fine_level.coll.QQ[1:, 1:] @ F_fine)
        RF_coarse_pos = self.sdc_coarse_level.coll.QQ[1:, 1:] @ F_coarse
        tau_pos = (self.sdc_fine_level.prob.dt**2) * (RF_fine_pos - RF_coarse_pos)
        tau_vel = (self.sdc_fine_level.prob.dt) * (RF_fine_vel - RF_coarse_vel)
        X_coarse = np.append(X_fine[0], X_coarse)
        V_coarse = np.append(V_fine[0], V_coarse)
        tau_pos = np.append(0.0, tau_pos)
        tau_vel = np.append(0.0, tau_vel)
        return tau_pos, tau_vel, X_coarse, V_coarse
