import numpy as np
from transfer_class.transfer_class import transfer_class


class StandartRestriction(transfer_class):
    def __init__(self, restrict_nodes):
        self.restriction_node = restrict_nodes

    def restriction_operator(
        self,
        X_fine,
        V_fine,
        fine_model=None,
        coarse_zero_model=None,
        coarse_first_model=None,
        eps=None,
    ):
        # X_zero = np.append(X_fine[0], self.restrict(X_fine[1:]))
        # V_zero = np.append(V_fine[0], self.restrict(V_fine[1:]))
        X_zero, V_zero=self.restriction_operator_nodes(X_fine, V_fine)
        print("Just copying")
        # breakpoint()
        if eps is None:
            return X_zero, V_zero
        else:
            X_first = np.zeros(len(X_zero))
            V_first = np.zeros(len(V_zero))
            # breakpoint()
            return X_zero, V_zero, X_first, V_first
    def restriction_operator_nodes(
        self,
        X_fine,
        V_fine,
    ):
        X_coarse = np.append(X_fine[0], self.restriction_node(X_fine[1:]))
        V_coarse = np.append(V_fine[0], self.restriction_node(V_fine[1:]))
        return X_coarse, V_coarse

    def fas_correction_zeros(self, X, V, fine_prob=None, coarse_zeros_model=None):
        X_zero, V_zero = self.restriction_operator(X, V)
        Rfine_pos, Rfine_vel = fine_prob.collocation_operator(X, V)

        Rcoarse_zeros_pos, Rcoarse_zeros_vel = coarse_zeros_model.collocation_operator(
            X_zero, V_zero
        )

        Rfine_zeros_pos, Rfine_zeros_vel = self.restriction_operator(
            Rfine_pos, Rfine_vel
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
        X_zero, V_zero, X_first, V_first = self.restriction_operator(X, V, eps=eps)
        Rfine_pos, Rfine_vel = fine_prob.collocation_operator(X, V)

        Rcoarse_zeros_pos, Rcoarse_zeros_vel = coarse_zeros_model.collocation_operator(
            X_zero, V_zero
        )

        V0first_order = 0.0 * V_first
        Rcoarse_first_pos, Rcoarse_first_vel = coarse_first_model.collocation_operator(
            X_first, V_first, V0=V0first_order
        )
        Rfine_zeros_pos, Rfine_zeros_vel, Rfine_first_pos, Rfine_first_vel = (
            self.restriction_operator(Rfine_pos, Rfine_vel, eps=eps)
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
