import numpy as np
from sweeper_class.sdc_class import sdc_class
from copy import deepcopy
from core.Lagrange import LagrangeApproximation
from scipy.optimize import minimize

class transfer_class(object):
    def __init__(
        self, problem_params, collocation_params, sweeper_params, problem_class, eps
    ):
        if eps is None:
            self.eps = 0.001
        else:
            self.eps = eps
        self.get_sorted_params(
            problem_params, collocation_params, sweeper_params, problem_class
        )
        self.Pcoll = self.get_transfer_matrix_Q(
            self.sdc_fine_level.coll.nodes, self.sdc_coarse_level.coll.nodes
        )

        self.Rcoll = self.get_transfer_matrix_Q(
            self.sdc_coarse_level.coll.nodes, self.sdc_fine_level.coll.nodes
        )
        self.Raverage = self.get_transfer_interp_matrix_Q(
            self.sdc_coarse_level.coll.nodes, self.sdc_fine_level.coll.nodes
        )
        self.Paverage = self.get_transfer_interp_matrix_Q(
            self.sdc_fine_level.coll.nodes, self.sdc_coarse_level.coll.nodes
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

    @staticmethod
    def get_transfer_interp_matrix_Q(f_nodes, c_nodes, eps=0.1):
        approx = LagrangeApproximation(c_nodes)
        return approx.getIntegrationMatrix(
            [(t / eps - 0.5, t / eps + 0.5) for t in f_nodes]
        )

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

    def compression(self, U):
        return self.Raverage @ U

    def reconstruction(self, U):
        return np.append(U[0], self.Paverage @ U[1:])
    

    def averaging(self, X_fine, V_fine, fine_level=None, coarse_level=None):
        if fine_level is None:
            fine_level = self.sdc_fine_level
        if coarse_level is None:
            coarse_level = self.sdc_coarse_level
        X_coarse = self.compression(X_fine[1:])
        V_coarse = self.compression(V_fine[1:])
        dt_fine = fine_level.prob.dt
        dt_coarse = coarse_level.prob.dt
        F_fine = fine_level.build_f(
            X_fine[1:], V_fine[1:], dt_fine * fine_level.coll.nodes
        )
        F_coarse = coarse_level.build_f(
            X_coarse, V_coarse, dt_coarse * coarse_level.coll.nodes
        )
        RF_fine_vel = self.compression(fine_level.coll.Q[1:, 1:] @ F_fine)
        RF_coarse_vel = coarse_level.coll.Q[1:, 1:] @ F_coarse
        RF_fine_pos = self.compression(fine_level.coll.QQ[1:, 1:] @ F_fine)
        RF_coarse_pos = coarse_level.coll.QQ[1:, 1:] @ F_coarse
        tau_pos = ((dt_fine**2) * RF_fine_pos) - ((dt_coarse) ** 2 * RF_coarse_pos)
        tau_vel = dt_fine * RF_fine_vel - dt_coarse * RF_coarse_vel
        X_coarse = np.append(X_fine[0], X_coarse)
        V_coarse = np.append(V_fine[0], V_coarse)
        tau_pos = np.append(0.0, tau_pos)
        tau_vel = np.append(0.0, tau_vel)
        return tau_pos, tau_vel, X_coarse, V_coarse

    def averaging_over_time_vel(self, U, level=None):
        return (level.coll.weights @ U)
    
    def averaging_over_time_pos(self, U, level=None):
        return level.coll.weights@U

    def averaging_first_order(self, U, U_averaged):
        return  (1/np.sqrt(self.eps))*(U - U_averaged)
    
    def FAS_averaging_first_order(self, X_fine, V_fine, fine_level=None, coarse_level=None, coarse_level_first=None):
        X_averag = self.averaging_over_time_pos(X_fine[1:], level=fine_level)
        V_averag = self.averaging_over_time_vel(V_fine[1:], level=fine_level)
        X_zeros=X_averag*np.ones(fine_level.coll.num_nodes)
        V_zeros=V_averag*np.ones(fine_level.coll.num_nodes)
        X_first=self.averaging_first_order(X_fine[1:], X_zeros)
        V_first=self.averaging_first_order(V_fine[1:], V_zeros)
        dt_fine=fine_level.prob.dt
        dt_zeros=coarse_level.prob.dt
        dt_first=coarse_level_first.prob.dt
        F_fine = fine_level.build_f(
            X_fine[1:], V_fine[1:], dt_fine * fine_level.coll.nodes
        )
        F_zeros = coarse_level.build_f(
            X_zeros, V_zeros, dt_zeros * coarse_level.coll.nodes
        )
        F_first=coarse_level_first.build_f(X_first, V_first, dt_first*coarse_level_first.coll.num_nodes)
        QF_fine_pos=fine_level.coll.QQ[1:,1:]@F_fine
        QF_fine_vel=fine_level.coll.Q[1:,1:]@F_fine
        RF_averaging_zeros_pos=self.averaging_over_time_pos(QF_fine_pos, level=fine_level)
        RF_averaging_zeros_vel=self.averaging_over_time_vel(QF_fine_vel, level=fine_level)
        RF_averaging_zeros_pos=RF_averaging_zeros_pos*np.ones(coarse_level.coll.num_nodes)
        RF_averaging_zeros_vel=RF_averaging_zeros_vel*np.ones(coarse_level.coll.num_nodes)
        RF_averaging_first_pos=self.averaging_first_order(QF_fine_pos, RF_averaging_zeros_pos)
        RF_averaging_first_vel=self.averaging_first_order(QF_fine_vel, RF_averaging_zeros_vel)

        RC_averaging_zeros_pos=(dt_zeros**2)*coarse_level.coll.QQ[1:,1:]@F_zeros
        RC_averaging_zeros_vel=dt_zeros*coarse_level.coll.Q[1:,1:]@F_zeros
        RC_averaging_first_pos=(dt_first**2)*coarse_level_first.coll.QQ[1:,1:]@F_first
        RC_averaging_first_vel=dt_first*coarse_level_first.coll.Q[1:,1:]@F_first
        
        tau_pos_zeros=np.append(0,dt_fine**2* RF_averaging_zeros_pos-RC_averaging_zeros_pos)
        tau_vel_zeros=np.append(0, dt_fine*RF_averaging_zeros_vel-RC_averaging_zeros_vel)
        tau_pos_first=np.append(0, dt_fine**2*RF_averaging_first_pos-RC_averaging_first_pos)
        tau_vel_first=np.append(0, dt_fine* RF_averaging_first_vel-RC_averaging_first_vel)
        X_zeros=np.append(X_fine[0], X_zeros)
        V_zeros=np.append(V_fine[0], V_zeros)
        X_first=np.append(0.0, X_first)
        V_first=np.append(0.0, V_first)

        return tau_pos_zeros, tau_vel_zeros, tau_pos_first, tau_vel_first, X_zeros, V_zeros, X_first, V_first

    
    def FAS_averaging(self, X_fine, V_fine, fine_level=None, coarse_level=None):
        X_averag = self.averaging_over_time_pos(X_fine[1:], level=fine_level)
        V_averag = self.averaging_over_time_vel(V_fine[1:], level=fine_level)
        X_coarse = X_averag * np.ones(coarse_level.coll.num_nodes)
        V_coarse = V_averag * np.ones(coarse_level.coll.num_nodes)
        dt_fine = fine_level.prob.dt
        dt_coarse = coarse_level.prob.dt
        F_fine = fine_level.build_f(
            X_fine[1:], V_fine[1:], dt_fine * fine_level.coll.nodes
        )
        F_coarse = coarse_level.build_f(
            X_coarse, V_coarse, dt_coarse * coarse_level.coll.nodes
        )
        RF_fine_vel_average = self.averaging_over_time_vel(
            fine_level.coll.Q[1:, 1:] @ F_fine, level=fine_level
        )
        RF_fine_vel = RF_fine_vel_average * np.ones(coarse_level.coll.num_nodes)
        RF_coarse_vel = coarse_level.coll.Q[1:, 1:] @ F_coarse
        RF_fine_pos_average = self.averaging_over_time_pos(
            fine_level.coll.QQ[1:, 1:] @ F_fine, level=fine_level
        )
        RF_fine_pos = RF_fine_pos_average * np.ones(coarse_level.coll.num_nodes)
        RF_coarse_pos = coarse_level.coll.QQ[1:, 1:] @ F_coarse
        tau_pos = ((dt_fine**2) * RF_fine_pos) - ((dt_coarse) ** 2 * RF_coarse_pos)
        tau_vel = dt_fine * RF_fine_vel - dt_coarse * RF_coarse_vel
        X_coarse = np.append(X_fine[0], X_coarse)
        V_coarse = np.append(V_fine[0], V_coarse)
        tau_pos = np.append(0.0, tau_pos)
        tau_vel = np.append(0.0, tau_vel)
        return tau_pos, tau_vel, X_coarse, V_coarse

    def arg_min_function(self, y, y_star, num_nodes):
        if len(y_star)==num_nodes:
            func=np.linalg.norm(y-y_star)**2
        else:
            func=np.linalg.norm(y[0:num_nodes]-y_star[0:num_nodes])**2+np.linalg.norm(np.sqrt(self.eps)*(y[num_nodes:]-y_star[num_nodes:]))**2
        return func

    def arg_min(self, U, y_star, num_nodes):
        if len(y_star)==num_nodes:
            cons=({'type':'eq', 'fun': lambda y: y-U})
            y0=np.ones(len(U))*U[0]
        else:
            cons=({'type':'eq', 'fun': lambda y: y[0:num_nodes]+np.sqrt(self.eps)*y[num_nodes:]-U})
            y0=np.block([np.ones(len(U))*U[0], np.zeros(len(U))])
        res=minimize(self.arg_min_function, y0, args=(y_star, num_nodes), constraints=cons)
        print(res.message)
        return res.x

    def FAS_with_arg_min(self, X_fine, V_fine, fine_level=None, coarse_level=None):
        if fine_level is None:
            fine_level = self.sdc_fine_level
        if coarse_level is None:
            coarse_level = self.sdc_coarse_level
        X_star=np.ones(fine_level.coll.num_nodes)*X_fine[0]
        V_star=np.ones(fine_level.coll.num_nodes)*V_fine[0]
        X_coarse=self.arg_min(X_fine[1:], X_star, fine_level.coll.num_nodes)
        V_coarse=self.arg_min(V_fine[1:], V_star, fine_level.coll.num_nodes)
        dt_fine = fine_level.prob.dt
        dt_coarse = coarse_level.prob.dt
        F_fine = fine_level.build_f(
            X_fine[1:], V_fine[1:], dt_fine * fine_level.coll.nodes
        )
        F_coarse = coarse_level.build_f(
            X_coarse, V_coarse, dt_coarse * coarse_level.coll.nodes
        )
        RF_coarse_vel = coarse_level.coll.Q[1:, 1:] @ F_coarse
        RF_fine_vel = self.arg_min(fine_level.coll.Q[1:, 1:] @ F_fine, RF_coarse_vel, coarse_level.coll.num_nodes)
        RF_coarse_pos = coarse_level.coll.QQ[1:, 1:] @ F_coarse
        RF_fine_pos = self.arg_min(fine_level.coll.QQ[1:, 1:] @ F_fine, RF_coarse_pos, coarse_level.coll.num_nodes)
        
        tau_vel=dt_fine*RF_fine_vel-dt_coarse*RF_coarse_vel
        tau_pos=(dt_fine**2)*RF_fine_pos-(dt_coarse**2)*RF_coarse_pos
        X_coarse=np.append(X_fine[0], X_coarse)
        V_coarse=np.append(V_fine[0], V_coarse)
        tau_pos=np.append(0.0, tau_pos)
        tau_vel=np.append(0.0, tau_vel)
        return tau_pos, tau_vel, X_coarse, V_coarse
    
    def FAS_with_arg_min_first_order(self, X_fine, V_fine, fine_level=None, coarse_level=None, coarse_level_first=None):
        if fine_level is None:
            fine_level = self.sdc_fine_level
        if coarse_level is None:
            coarse_level = self.sdc_coarse_level
        if coarse_level_first is None:
            coarse_level_first = self.sdc_coarse_first_order
        X_star=np.block([np.ones(fine_level.coll.num_nodes)*X_fine[0], np.zeros(fine_level.coll.num_nodes)])
        V_star=np.block([np.ones(fine_level.coll.num_nodes)*V_fine[0], np.zeros(fine_level.coll.num_nodes)])
        X_coarse=self.arg_min(X_fine[1:], X_star, fine_level.coll.num_nodes)
        V_coarse=self.arg_min(V_fine[1:], V_star, fine_level.coll.num_nodes)
        X_zeros, X_first=np.split(X_coarse, 2)
        V_zeros, V_first=np.split(V_coarse, 2)
        dt_fine = fine_level.prob.dt
        dt_zeros = coarse_level.prob.dt
        dt_first=coarse_level_first.prob.dt
        F_fine = fine_level.build_f(
            X_fine[1:], V_fine[1:], dt_fine * fine_level.coll.nodes
        )
        F_zeros = coarse_level.build_f(
            X_zeros, V_zeros, dt_zeros * coarse_level.coll.nodes
        )
        F_first=coarse_level_first.build_f(X_first, V_first, dt_first*coarse_level_first.coll.num_nodes)
        QF_fine_pos=fine_level.coll.QQ[1:,1:]@F_fine
        QF_fine_vel=fine_level.coll.Q[1:,1:]@F_fine
        RF_zeros_vel = coarse_level.coll.Q[1:, 1:] @ F_zeros
        RF_zeros_pos = coarse_level.coll.QQ[1:, 1:] @ F_zeros
        RF_first_vel = coarse_level_first.coll.Q[1:, 1:] @ F_first
        RF_first_pos = coarse_level_first.coll.QQ[1:, 1:] @ F_first
        RF_star_pos=np.block([RF_zeros_pos, RF_first_pos])
        RF_star_vel=np.block([RF_zeros_vel, RF_first_vel])
        RF_fine_pos=self.arg_min(QF_fine_pos, RF_star_pos, coarse_level.coll.num_nodes)
        RF_fine_vel=self.arg_min(QF_fine_vel, RF_star_vel, coarse_level.coll.num_nodes)
        RF_zeros_pos_first, RF_first_pos_first=np.split(RF_fine_pos, 2)
        RF_zeros_vel_first, RF_first_vel_first=np.split(RF_fine_vel, 2)
        tau_zeros_pos=(dt_fine**2)*RF_zeros_pos_first-(dt_zeros**2)*RF_zeros_pos
        tau_zeros_vel=dt_fine*RF_zeros_vel_first-dt_zeros*RF_zeros_vel
        tau_first_pos=(dt_fine**2)*RF_first_pos_first-(dt_first**2)*RF_first_pos
        tau_first_vel=dt_fine*RF_first_vel_first-dt_first*RF_first_vel
        X_zeros=np.append(X_fine[0], X_zeros)
        V_zeros=np.append(V_fine[0], V_zeros)
        X_first=np.append(0.0, X_first)
        V_first=np.append(0.0, V_first)
        tau_zeros_pos=np.append(0.0, tau_zeros_pos)
        tau_zeros_vel=np.append(0.0, tau_zeros_vel)
        tau_first_pos=np.append(0.0, tau_first_pos)
        tau_first_vel=np.append(0.0, tau_first_vel)
        return tau_zeros_pos, tau_zeros_vel, tau_first_pos, tau_first_vel, X_zeros, V_zeros, X_first, V_first








