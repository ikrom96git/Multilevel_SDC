import numpy as np
from transfer_class.transfer_class import transfer_class


class AveragingRestriction(transfer_class):

    def __init__(self, restrict_nodes):
        self.restriction_node = restrict_nodes

    def averaging_over_time(self, U, level=None):
        return level.coll.weights @ U

    def averaging_over_time_pos(self, U, level=None):
        return level.coll.weights @ U

    def averaging_first_order(self, U, U_averaged):
        return (1 / np.sqrt(self.eps)) * (U - U_averaged)

    def restriction_operator(
        self,
        X_fine,
        V_fine,
        fine_model=None,
        coarse_zero_model=None,
        coarse_first_model=None,
        eps=None,
    ):
        X_averaged = self.averaging_over_time(X_fine[1:], level=fine_model)
        V_averaged = self.averaging_over_time(V_fine[1:], level=fine_model)
        X_zero = X_averaged * np.ones(coarse_zero_model.coll.num_nodes)
        V_zero = V_averaged * np.ones(coarse_zero_model.coll.num_nodes)
        if eps is None:
            return X_zero, V_zero
        else:
            X_first = (X_fine - X_zero) / eps
            V_first = (V_fine - V_zero) / eps
            return X_zero, V_zero, X_first, V_first

    def fas_correction_zeros(
        self, X, V, fine_prob=None, coarse_zeros_model=None, eps=None
    ):
        X_zero, V_zero = self.restriction_operator(
            X, V, fine_model=fine_prob, coarse_zero_model=coarse_zeros_model
        )
        Rfine_pos, Rfine_vel = fine_prob.collocation_operator(X, V)

        Rcoarse_zeros_pos, Rcoarse_zeros_vel = coarse_zeros_model.collocation_operator(
            X_zero, V_zero
        )

        Rfine_zeros_pos, Rfine_zeros_vel = self.restriction_operator(
            Rfine_pos,
            Rfine_vel,
            fine_model=fine_prob,
            coarse_zero_model=coarse_zeros_model,
        )
        tau_pos_zeros = Rcoarse_zeros_pos - Rfine_zeros_pos
        tau_vel_zeros = Rcoarse_zeros_vel - Rfine_zeros_vel
        return tau_pos_zeros, tau_vel_zeros

    def fas_correction_first(
        self,
        X,
        V,
        fine_prob=None,
        coarse_zeros_model=None,
        coarse_first_model=None,
        eps=None,
    ):
        X_zero, V_zero, X_first, V_first = self.restriction_operator(
            X, V, fine_model=fine_prob, coarse_zero_model=coarse_zeros_model, eps=eps
        )
        Rfine_pos, Rfine_vel = fine_prob.collocation_operator(X, V)

        Rcoarse_zeros_pos, Rcoarse_zeros_vel = coarse_zeros_model.collocation_operator(
            X_zero, V_zero
        )

        V0first_order = 0.0 * V_first
        Rcoarse_first_pos, Rcoarse_first_vel = coarse_first_model.collocation_operator(
            X_first, V_first, V0=V0first_order
        )
        Rfine_zeros_pos, Rfine_zeros_vel, Rfine_first_pos, Rfine_first_vel = (
            self.restriction_operator(
                Rfine_pos,
                Rfine_vel,
                fine_model=fine_prob,
                coarse_zero_model=coarse_zeros_model,
                eps=eps,
            )
        )
        tau_pos_zeros = Rcoarse_zeros_pos - Rfine_zeros_pos
        tau_vel_zeros = Rcoarse_zeros_vel - Rfine_zeros_vel
        tau_pos_first = Rcoarse_first_pos - Rfine_first_pos
        tau_vel_first = Rcoarse_first_vel - Rfine_first_vel
        return tau_pos_zeros, tau_vel_zeros, tau_pos_first, tau_vel_first

    def fas_correction_operator(
        self,
        X,
        V,
        fine_prob=None,
        coarse_zeros_model=None,
        coarse_first_model=None,
        eps=None,
    ):
        if eps is None:
            return self.fas_correction_zeros(
                X, V, fine_prob=fine_prob, coarse_zeros_model=coarse_zeros_model
            )
        else:
            return self.fas_correction_first(
                X,
                V,
                fine_prob=fine_prob,
                coarse_zeros_model=coarse_zeros_model,
                coarse_first_model=coarse_first_model,
                eps=eps,
            )
