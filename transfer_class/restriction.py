import numpy as np
from transfer_class.transfer_class import transfer_class


class Restriction(transfer_class):

    def __init__(self, restrict_nodes):
        self.restrict = restrict_nodes

    def restriction_operator(
        self,
        X_fine,
        V_fine,
        fine_model=None,
        coarse_zero_model=None,
        coarse_first_model=None,
        eps=None,
    ):
        X_coarse = np.append(X_fine[0], self.restrict(X_fine[1:]))
        V_coarse = np.append(V_fine[0], self.restrict(V_fine[1:]))

        return X_coarse, V_coarse

    def fas_correction_operator(
        self,
        X_fine,
        V_fine,
        fine_prob=None,
        coarse_zeros_model=None,
        coarse_first_model=None,
    ):
        X_coarse = self.restrict(X_fine[1:])
        V_coarse = self.restrict(V_fine[1:])
        dt_fine = fine_prob.prob.dt
        dt_coarse = coarse_zeros_model.prob.dt
        F_fine = fine_prob.build_f(
            X_fine[1:], V_fine[1:], dt_fine * fine_prob.coll.nodes
        )
        F_coarse = coarse_zeros_model.build_f(
            X_coarse, V_coarse, dt_coarse * coarse_zeros_model.coll.nodes
        )
        RF_fine_vel = self.restrict(fine_prob.coll.Q[1:, 1:] @ F_fine)
        RF_coarse_vel = coarse_zeros_model.coll.Q[1:, 1:] @ F_coarse
        RF_fine_pos = self.restrict(fine_prob.coll.QQ[1:, 1:] @ F_fine)
        RF_coarse_pos = coarse_zeros_model.coll.QQ[1:, 1:] @ F_coarse
        tau_pos = ((dt_fine**2) * RF_fine_pos) - ((dt_coarse) ** 2 * RF_coarse_pos)
        tau_vel = dt_fine * RF_fine_vel - dt_coarse * RF_coarse_vel
        tau_pos = np.append(0.0, tau_pos)
        tau_vel = np.append(0.0, tau_vel)
        return tau_pos, tau_vel
