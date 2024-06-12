import numpy as np
from sweeper_class.sdc_class import sdc_class
from copy import deepcopy
from core.Lagrange import LagrangeApproximation
from scipy.optimize import minimize
from default_params.harmonic_oscillator_default_params import eps_fast_time
class transfer_class(object):
    def __init__(
        self, problem_params, collocation_params, sweeper_params, problem_class, eps
    ):
        if eps is None:
            self.eps = eps_fast_time
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
        V0_fist=np.zeros(6)
        self.X0_zero=self.sdc_fine_level.prob.u0[0]*np.ones(self.sdc_coarse_level.coll.num_nodes+1)
        self.V0_zero=self.sdc_fine_level.prob.u0[1]*np.ones(self.sdc_coarse_level.coll.num_nodes+1)
        
        


    def get_sorted_params(
        self, problem_params, collocation_params, sweeper_params, problem_class
    ):

        if len(problem_params) == 2:
            problem_params_fine = problem_params[0]
            problem_params_coarse = problem_params[1]
            problem_params_first=problem_params[1]
        elif len(problem_params)==3:
            problem_params_fine = problem_params[0]
            problem_params_coarse = problem_params[1]
            problem_params_first=problem_params[2]
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
                problem_params_first,
                collocation_params_coarse,
                sweeper_params,
                problem_class_coarse_first,
            )
            self.X0_first=np.zeros(self.sdc_coarse_first_order.coll.num_nodes+1)
            self.V0_first=np.zeros(self.sdc_coarse_first_order.coll.num_nodes+1)
            


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

    def fas_asyp_model(self, X_fine, V_fine, fine_level=None, coarse_zeros_level=None, coarse_first_order=None):
        RX_zeros_order, RV_zeros_order, RX_first_order, RV_first_order=self.restriction_operator(X_fine, V_fine)
        
        Rfine_pos, Rfine_vel=fine_level.collocation_operator(X_fine, V_fine)

        Rcoarse_zeros_pos, Rcoarse_zeros_vel=coarse_zeros_level.collocation_operator(RX_zeros_order, RV_zeros_order)
        
        V0first_order=0.0*RV_first_order
        Rcoarse_first_pos, Rcoarse_first_vel=coarse_first_order.collocation_operator(RX_first_order, RV_first_order, V0=V0first_order)
        Rfine_zeros_pos, Rfine_zeros_vel, Rfine_first_pos, Rfine_first_vel=self.restriction_operator(Rfine_pos, Rfine_vel)
        tau_pos_zeros=Rcoarse_zeros_pos-Rfine_zeros_pos
        tau_vel_zeros=Rcoarse_zeros_vel-Rfine_zeros_vel
        tau_pos_first=Rcoarse_first_pos-Rfine_first_pos
        tau_vel_first=Rcoarse_first_vel-Rfine_first_vel
        return tau_pos_zeros, tau_vel_zeros, tau_pos_first, tau_vel_first

    
    def restriction_operator(self, X_fine, V_fine):
        RX_zeros_averaged=self.averaging_over_time_pos(X_fine[1:], level=self.sdc_fine_level)
        RV_zeros_averaged=self.averaging_over_time_vel(V_fine[1:], level=self.sdc_fine_level)
        RX_zeros_order=np.ones(len(X_fine[1:]))*RX_zeros_averaged
        RV_zeros_order=np.ones(len(V_fine[1:]))*RV_zeros_averaged
        RX_first_order=self.averaging_first_order(X_fine[1:], RX_zeros_order)
        RV_first_order=self.averaging_first_order(V_fine[1:], RV_zeros_order)
        RX_zeros_order=np.append(X_fine[0], RX_zeros_order)
        RV_zeros_order=np.append(V_fine[0], RV_zeros_order)
        RX_first_order=np.append(0.0, RX_first_order)
        RV_first_order=np.append(0.0, RV_first_order)
        return RX_zeros_order, RV_zeros_order, RX_first_order, RV_first_order


    def averaging_over_time_vel(self, U, level=None):
        return (level.coll.weights @ U)
    
    def averaging_over_time_pos(self, U, level=None):
        return level.coll.weights@U

    def averaging_first_order(self, U, U_averaged):
        return  (1/np.sqrt(self.eps))*(U - U_averaged)
    

    
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
        func=np.linalg.norm(y[0:num_nodes]-y_star[0:num_nodes])**2+np.linalg.norm((y[num_nodes:]-y_star[num_nodes:]))**2
        return func

    def arg_min(self, U, y_star, num_nodes):
        cons=({'type':'eq', 'fun': lambda y: y[0:num_nodes]+np.sqrt(self.eps)*y[num_nodes:]-U})
        y0=y_star
        res=minimize(self.arg_min_function, y0, args=(y_star, num_nodes), constraints=cons, tol=1e-13)
        print(res.message)
        print(res.status)
        print(res.nfev)
        print(res.fun)
        print(res.success)
        breakpoint()
        return res.x
    
    def arg_min_restriction_operator(self, X_fine, V_fine, operator=False):
        # if operator:
        #     Ac_zeros_pos, Ac_zeros_vel=self.sdc_coarse_level.collocation_operator(self.X0_zero, self.V0_zero)
        #     print(self.X0_first)
        #     Ac_first_pos, Ac_first_vel=self.sdc_coarse_first_order.collocation_operator(self.X0_first, self.V0_first, self.V0_first*0.0)
        #     X_star=np.block([Ac_zeros_pos, Ac_first_pos])
        #     V_star=np.block([Ac_zeros_vel, Ac_first_vel])
        # else:
        #     X_star=np.block([self.X0_zero, self.X0_first])
        #     V_star=np.block([self.V0_zero, self.V0_first])
        X_zero_average, V_zero_average, X_first_average, V_first_average=self.restriction_operator(X_fine, V_fine)
        X_star=np.block([X_zero_average, X_first_average])
        V_star=np.block([V_zero_average, V_first_average])
        X_zero_min, V_zero_min=np.split(self.arg_min(X_fine, X_star, 6), 2)
        X_first_min, V_first_min=np.split(self.arg_min(V_fine, V_star, 6), 2)
        return X_zero_min, V_zero_min, X_first_min, V_first_min
    
    def fas_asyp_arg_min_model(self, X_fine, V_fine, fine_level=None, coarse_zeros_level=None, coarse_first_order=None):
        RX_zeros_order, RV_zeros_order, RX_first_order, RV_first_order=self.arg_min_restriction_operator(X_fine, V_fine)
        
        Rfine_pos, Rfine_vel=fine_level.collocation_operator(X_fine, V_fine)

        Rcoarse_zeros_pos, Rcoarse_zeros_vel=coarse_zeros_level.collocation_operator(RX_zeros_order, RV_zeros_order)
        
        V0first_order=0.0*RV_first_order
        Rcoarse_first_pos, Rcoarse_first_vel=coarse_first_order.collocation_operator(RX_first_order, RV_first_order, V0=V0first_order)
        Rfine_zeros_pos, Rfine_zeros_vel, Rfine_first_pos, Rfine_first_vel=self.arg_min_restriction_operator(Rfine_pos, Rfine_vel, operator=True)
        tau_pos_zeros=Rcoarse_zeros_pos-Rfine_zeros_pos
        tau_vel_zeros=Rcoarse_zeros_vel-Rfine_zeros_vel
        tau_pos_first=Rcoarse_first_pos-Rfine_first_pos
        tau_vel_first=Rcoarse_first_vel-Rfine_first_vel
        return tau_pos_zeros, tau_vel_zeros, tau_pos_first, tau_vel_first

    def last_idea_for_restriction(self, X, V, fine_level=None, coarse_zeros_order=None, coarse_first_order=None):
        X_zero=np.ones(len(X))*X[0]+coarse_zeros_order.prob.dt*coarse_zeros_order.coll.Q@V
        X_first=(X-X_zero)/np.sqrt(self.eps)
        
        T=coarse_zeros_order.prob.dt*np.append(0, coarse_zeros_order.coll.nodes)
        V_zero=np.ones(len(V))*V[0]+np.sin(T)-coarse_zeros_order.prob.dt*coarse_zeros_order.coll.Q@X
        V_first=(V-V_zero)/np.sqrt(self.eps)
        return X_zero, V_zero, X_first, V_first
    
    def last_idea_more_restriction(self, X, V, fine_level=None, coarse_zeros_order=None, coarse_first_order=None):
        T=coarse_zeros_order.prob.dt*np.append(0, coarse_zeros_order.coll.nodes)
        V_sin=np.ones(len(V))*V[0]+np.sin(T)
        X_zero=np.ones(len(X))*X[0]+coarse_zeros_order.prob.dt*coarse_zeros_order.coll.Q@V_sin-(coarse_zeros_order.prob.dt**2)*coarse_zeros_order.coll.QQ@X
        X_first=(X-X_zero)/np.sqrt(self.eps)
        # X_first=-0.8*(coarse_zeros_order.prob.dt**2)*coarse_first_order.coll.QQ@V
        
        V_zero=np.ones(len(V))*V[0]+np.sin(T)-coarse_zeros_order.prob.dt*coarse_zeros_order.coll.Q@X
        # V_first=-0.8*coarse_first_order.prob.dt*coarse_first_order.coll.Q@V
        V_first=(V-V_zero)/np.sqrt(self.eps)
        # V_first[0]=0
        # X_first[0]=0
        return X_zero, V_zero, X_first, V_first
    
    def last_idea_for_fas(self, X, V, fine_prob=None, coarse_zeros_prob=None, coarse_first_prob=None):
        X_zero, V_zero, X_first, V_first=self.restriction_duffing_equation(X, V, fine_level=fine_prob, coarse_zeros_order=coarse_zeros_prob, coarse_first_order=coarse_first_prob)
        Rfine_pos, Rfine_vel=fine_prob.collocation_operator(X, V)

        Rcoarse_zeros_pos, Rcoarse_zeros_vel=coarse_zeros_prob.collocation_operator(X_zero, V_zero)
        
        V0first_order=0.0*V_first
        Rcoarse_first_pos, Rcoarse_first_vel=coarse_first_prob.collocation_operator(X_first, V_first, V0=V0first_order)
        Rfine_zeros_pos, Rfine_zeros_vel, Rfine_first_pos, Rfine_first_vel=self.restriction_duffing_equation(Rfine_pos, Rfine_vel, fine_level=fine_prob, coarse_zeros_order=coarse_zeros_prob, coarse_first_order=coarse_first_prob)
        tau_pos_zeros=Rcoarse_zeros_pos-Rfine_zeros_pos
        tau_vel_zeros=Rcoarse_zeros_vel-Rfine_zeros_vel
        tau_pos_first=Rcoarse_first_pos-Rfine_first_pos
        tau_vel_first=Rcoarse_first_vel-Rfine_first_vel
        # breakpoint()
        return tau_pos_zeros, tau_vel_zeros, tau_pos_first, tau_vel_first

    def restriction_duffing_equation(self, X, V, fine_level=None, coarse_zeros_order=None, coarse_first_order=None):
        T=coarse_zeros_order.prob.dt*np.append(0, coarse_zeros_order.coll.nodes)
        dt_coarse=coarse_zeros_order.prob.dt
        X_zeros=np.ones(len(X))*X[0]+dt_coarse*coarse_zeros_order.coll.Q@V
        duffin2=True
        if not duffin2:
            V_zeros=np.ones(len(V))*V[0]-dt_coarse*coarse_zeros_order.coll.Q@(self.sdc_fine_level.prob.omega**2*X)
        else:
            V_zeros=np.ones(len(V))*V[0]-dt_coarse*coarse_zeros_order.coll.Q@(self.sdc_fine_level.prob.omega**2*X)+dt_coarse*coarse_zeros_order.coll.Q@(self.eps*(V**2)*X)
        X_first=(X-X_zeros)/self.eps
        V_first=(V-V_zeros)/self.eps
        return X_zeros, V_zeros, X_first, V_first
    



        


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

    









