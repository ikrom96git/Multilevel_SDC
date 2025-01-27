import numpy as np
from transfer_class.transfer_class import transfer_class


class AsymptoticRestriction(transfer_class):

    def __init__(self, restrict_nodes):
        self.restriction_node = restrict_nodes

    def restriction_operator_nodes(
        self,
        X_fine,
        V_fine,
    ):
        X_coarse = np.append(X_fine[0], self.restriction_node(X_fine[1:]))
        V_coarse = np.append(V_fine[0], self.restriction_node(V_fine[1:]))

        return X_coarse, V_coarse
    def restriction_operator0(
        self,
        X,
        V,
        fine_model=None,
        coarse_zero_model=None,
        coarse_first_model=None,
        eps=None,
    ):
        T = fine_model.prob.dt * np.append(0, fine_model.coll.nodes)
        dt_coarse = fine_model.prob.dt
        X_zeros = np.ones(len(X)) * X[0] + dt_coarse * fine_model.coll.Q @ V
        duffin2 = True
        if not duffin2:
            V_zeros = np.ones(len(V)) * V[0] - dt_coarse * fine_model.coll.Q @ (
                fine_model.prob.omega**2 * X
            )
            print("Something")

        else:
            V_zeros = (
                np.ones(len(V)) * V[0]
                - dt_coarse * fine_model.coll.Q @ (fine_model.prob.omega**2 * X)
                + dt_coarse * fine_model.coll.Q @ (eps * (V**2) * X)
            )
            
        # breakpoint()
        return X_zeros, V_zeros

    def restriction_operator(
        self,
        X_fine,
        V_fine,
        fine_model=None,
        coarse_zero_model=None,
        coarse_first_model=None,
        eps=None,
    ):
        X_zero, V_zero = fine_model.compute_integral(X_fine, V_fine)
        X_zero_nodes, V_zero_nodes=self.restriction_operator_nodes(X_zero, V_zero)
        # breakpoint()
        if eps is None:
            return X_zero_nodes, V_zero_nodes
        else:
            X_first = (X_fine - X_zero) / eps
            V_first = (V_fine - V_zero) / eps
            X_first_nodes, V_first_nodes=self.restriction_operator_nodes(X_first, V_first)
            
            return X_zero_nodes, V_zero_nodes, X_first_nodes, V_first_nodes

    def fas_correction_zeros(
        self, X, V, fine_prob=None, coarse_zeros_model=None, eps=None
    ):
        X_zero, V_zero = self.restriction_operator(X, V, fine_model=fine_prob)
        Rfine_pos, Rfine_vel = fine_prob.collocation_operator(X, V)

        Rcoarse_zeros_pos, Rcoarse_zeros_vel = coarse_zeros_model.collocation_operator(
            X_zero, V_zero
        )

        Rfine_zeros_pos, Rfine_zeros_vel = self.restriction_operator(
            Rfine_pos, Rfine_vel, fine_model=fine_prob
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

    def interpolation_operator(self, X_zeros, X_first, eps=None, model=None):
        if model is None:
            return X_zeros, X_first
        else:
            return model.problem_class.asyp_expansion(X_zeros, X_first, eps=eps)
