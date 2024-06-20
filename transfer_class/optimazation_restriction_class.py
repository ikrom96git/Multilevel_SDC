import numpy as np
from transfer_class.transfer_class import transfer_class
from scipy.optimize import minimize

from transfer_class.standart_restriction_class import StandartRestriction


class OptimationRestriction(transfer_class):
    def __init__(self, restrict_nodes):
        self.restriction = restrict_nodes
        self.initial_condition = StandartRestriction(restrict_nodes)

    def objective_function_first_order(self, y, y_star, num_nodes):
        return (
            np.linalg.norm(y[:num_nodes] - y_star[:num_nodes]) ** 2
            + self.eps * np.linalg.norm((y[num_nodes:] - y_star[num_nodes:])) ** 2
        )

    def objective_function_zeros_order(self, y, y_star, num_nodes):
        return np.linalg.norm(y[:num_nodes] - y_star[:num_nodes]) ** 2

    def arg_minimize(self, U, y_star, num_nodes, eps=None):
        cons_first = {
            "type": "eq",
            "fun": lambda y: y[:num_nodes] + np.sqrt(self.eps) * y[num_nodes:] - U,
        }
        cons_zero = {
            "type": "eq",
            "fun": lambda y: y[:num_nodes] - U,
        }

        y0 = y_star
        if eps is not None:
            min_values = minimize(
                self.objective_function_first_order,
                y0,
                args=(y_star, num_nodes),
                constraints=cons_first,
                tol=1e-13,
            )
        else:
            min_values = minimize(
                self.objective_function_zero_order,
                y0,
                args=(y_star, num_nodes),
                constraints=cons_zero,
                tol=1e-13,
            )
        print("Description: ", min_values.message)
        print("Termination status: ", min_values.status)
        print("Number of evaluations: ", min_values.nfev)
        print("Values of objective function: ", min_values.fun)
        print("optimizer exited successfully: ", min_values.success)
        return min_values.x

    def restriction_first_order(self, X_fine, V_fine, eps=None):
        X_zero_average, V_zero_average, X_first_average, V_first_average = (
            self.initial_condition.restriction_operator(X_fine, V_fine, eps=eps)
        )
        X_star = np.block([X_zero_average, X_first_average])
        V_star = np.block([V_zero_average, V_first_average])
        X_zero, X_first = np.split(self.arg_minimize(X_fine, X_star, 6, eps), 2)
        V_zero, V_first = np.split(self.arg_minimize(V_fine, V_star, 6, eps), 2)
        return X_zero, V_zero, X_first, V_first

    def restriction_zero_order(self, X_fine, V_fine, eps=None):
        X_zero_average, V_zero_average = self.initial_condition.restriction_operator(
            X_fine, V_fine
        )
        X_star = X_zero_average
        V_star = V_zero_average
        X_zero = np.split(self.arg_minimize(X_fine, X_star, 6), 2)
        V_zero = np.split(self.arg_minimize(V_fine, V_star, 6), 2)
        return X_zero, V_zero

    def restriction_operator(self, X_fine, V_fine, fine_model=None,  coarse_zero_model=None, coarse_first_model=None, eps=None):
        if eps is None:
            return self.restriction_zero_order(X_fine, V_fine)
        else:
            return self.restriction_first_order(X_fine, V_fine, eps)

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
            self.restriction_operator(Rfine_pos, Rfine_vel,eps= eps)
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


class OptimazationResidual(transfer_class):
    def __init__(self, restrict_nodes):
        self.restriction = restrict_nodes
        self.initial_condition = StandartRestriction(restrict_nodes)

    def objective_restriction_first_order(
        self, y, coarse_zero_model=None, coarse_first_model=None, eps=None
    ):
        X_zero, V_zero, X_first, V_first = np.split(y, 4)
        residual_zero_order = coarse_zero_model.get_coll_residual(
            X_zero, V_zero, tau_pos=[None], tau_vel=[None]
        )
        residual_first_order = coarse_first_model.get_coll_residual(
            X_first, V_first, tau_pos=[None], tau_vel=[None]
        )
        return (
            np.linalg.norm(residual_zero_order, 2) ** 2
            + eps * np.linalg.norm(residual_first_order, 2) ** 2
        )

    def objective_restriction_zeros_order(self, y, coarse_zero_model=None):
        X_zero, V_zero = np.split(y, 2)
        residual_zero_order = coarse_zero_model.get_coll_residual(
            X_zero, V_zero, tau_pos=[None], tau_vel=[None]
        )
        return np.linalg.norm(residual_zero_order, 2) ** 2

    def arg_minimize_restriction(
        self,
        U,
        y_star,
        num_nodes,
        coarse_zero_model=None,
        coarse_first_model=None,
        eps=None,
    ):
        
        y0 = y_star
        if eps is not None:
            cons_first = {
            "type": "eq",
            "fun": lambda y: y[:2*num_nodes] + eps * y[2*num_nodes:] - U,
            }
            
            min_values = minimize(
                self.objective_restriction_first_order,
                y0,
                args=(coarse_zero_model, coarse_first_model, eps),
                constraints=cons_first,
                tol=1e-13,
            )
        else:
            cons_zero = {
            "type": "eq",
            "fun": lambda y: y[:num_nodes] - U,
            }
            breakpoint()
            min_values = minimize(
                self.objective_restriction_zeros_order,
                y0,
                args=(y_star, coarse_zero_model),
                constraints=cons_zero,
                tol=1e-13,
            )
        print("Description: ", min_values.message)
        print("Termination status: ", min_values.status)
        print("Number of evaluations: ", min_values.nfev)
        print("Values of objective function: ", min_values.fun)
        print("optimizer exited successfully: ", min_values.success)
        return min_values.x

    def restriction_first_order(
        self, X_fine, V_fine, coarse_zero_model=None, coarse_first_model=None, eps=None
    ):
        X_zero_average, V_zero_average, X_first_average, V_first_average = (
            self.initial_condition.restriction_operator(X_fine, V_fine, eps=eps)
        )
        U=np.block([X_fine, V_fine])
        U_star = np.block([X_zero_average, V_zero_average, X_first_average, V_first_average])
        X_zero, V_zero, X_first, V_first = np.split(
            self.arg_minimize_restriction(
                U, U_star, 6, coarse_zero_model, coarse_first_model, eps
            ),
            4,
        )
        return X_zero, V_zero, X_first, V_first

    def restriction_zero_order(self, X_fine, V_fine, coarse_zero_model=None):
        X_zero_average, V_zero_average = self.initial_condition.restriction_operator(
            X_fine, V_fine
        )
        X_star = X_zero_average
        V_star = V_zero_average
        X_zero = np.split(
            self.arg_minimize_restriction(X_fine, X_star, 6, coarse_zero_model), 2
        )
        V_zero = np.split(
            self.arg_minimize_restriction(V_fine, V_star, 6, coarse_zero_model), 2
        )
        return X_zero, V_zero

    def restriction_operator(
        self, X_fine, V_fine,fine_model=None,  coarse_zero_model=None, coarse_first_model=None, eps=None
    ):
        if eps is None:
            return self.restriction_zero_order(X_fine, V_fine, coarse_zero_model)
        else:
            return self.restriction_first_order(
                X_fine, V_fine, coarse_zero_model, coarse_first_model, eps
            )

    def fas_correction_zeros(self, X, V, fine_prob=None, coarse_zeros_model=None):
        X_zero, V_zero = self.restriction_operator(X, V, coarse_zero_model=coarse_zeros_model)
        Rfine_pos, Rfine_vel = fine_prob.collocation_operator(X, V)

        Rcoarse_zeros_pos, Rcoarse_zeros_vel = coarse_zeros_model.collocation_operator(
            X_zero, V_zero
        )

        Rfine_zeros_pos, Rfine_zeros_vel = self.restriction_operator(
            Rfine_pos, Rfine_vel, coarse_zero_model=coarse_zeros_model
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
            X, V, coarse_zero_model=coarse_zeros_model, coarse_first_model=coarse_first_model, eps=eps
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
                Rfine_pos, Rfine_vel, coarse_zero_model=coarse_zeros_model, coarse_first_model=coarse_first_model, eps=eps
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
