import numpy as np
from sweeper_class.sdc_class import sdc_class
from copy import deepcopy
from core.Lagrange import LagrangeApproximation
from scipy.optimize import minimize
from transfer_class.transfer_class import transfer_class
class restriction_operator(transfer_class):
    
    def __init__(self, restrict_nodes):
        self.restrict=restrict_nodes

    def restriction_operator(self, X_fine, V_fine):
        X_coarse=np.append(X_fine[0], self.restrict(X_fine[1:]))
        V_coarse=np.append(V_fine[0], self.restrict(V_fine[1:]))
        return X_coarse, V_coarse

    def fas_correction_operator(self, X_fine, V_fine, fine_model=None, coarse_model=None):
        X_coarse = self.restrict(X_fine[1:])
        V_coarse = self.restrict(V_fine[1:])
        dt_fine = fine_model.prob.dt
        dt_coarse = coarse_model.prob.dt
        F_fine = fine_model.build_f(
            X_fine[1:], V_fine[1:], dt_fine * fine_model.coll.nodes
        )
        F_coarse = coarse_model.build_f(
            X_coarse, V_coarse, dt_coarse * coarse_model.coll.nodes
        )
        RF_fine_vel = self.restrict(fine_model.coll.Q[1:, 1:] @ F_fine)
        RF_coarse_vel = coarse_model.coll.Q[1:, 1:] @ F_coarse
        RF_fine_pos = self.restrict(fine_model.coll.QQ[1:, 1:] @ F_fine)
        RF_coarse_pos = coarse_model.coll.QQ[1:, 1:] @ F_coarse
        tau_pos = ((dt_fine**2) * RF_fine_pos) - ((dt_coarse) ** 2 * RF_coarse_pos)
        tau_vel = dt_fine * RF_fine_vel - dt_coarse * RF_coarse_vel
        X_coarse = np.append(X_fine[0], X_coarse)
        V_coarse = np.append(V_fine[0], V_coarse)
        tau_pos = np.append(0.0, tau_pos)
        tau_vel = np.append(0.0, tau_vel)
        return tau_pos, tau_vel, X_coarse, V_coarse

    def simple_restriction_operator(self, X_fine, V_fine):
        RX_zeros_order=np.append(X_fine[0], self.restrict(X_fine[1:]))
        RV_zeros_order=np.append(V_fine[0], self.restrict(V_fine[1:]))
        RX_first_order=(X_fine-RX_zeros_order)/self.eps
        RV_first_order=(V_fine-RV_zeros_order)/self.eps
        return RX_zeros_order, RV_zeros_order, RX_first_order, RV_first_order

    def fas_asyp_model(self, X_fine, V_fine, fine_model=None, coarse_zeros_model=None, coarse_first_model=None):
        RX_zeros_order, RV_zeros_order, RX_first_order, RV_first_order=self.simple_restriction_operator(X_fine, V_fine)
        
        Rfine_pos, Rfine_vel=fine_model.collocation_operator(X_fine, V_fine)

        Rcoarse_zeros_pos, Rcoarse_zeros_vel=coarse_zeros_model.collocation_operator(RX_zeros_order, RV_zeros_order)
        
        V0first_order=0.0*RV_first_order
        Rcoarse_first_pos, Rcoarse_first_vel=coarse_first_model.collocation_operator(RX_first_order, RV_first_order, V0=V0first_order)
        Rfine_zeros_pos, Rfine_zeros_vel, Rfine_first_pos, Rfine_first_vel=self.simple_restriction_operator(Rfine_pos, Rfine_vel)
        tau_pos_zeros=Rcoarse_zeros_pos-Rfine_zeros_pos
        tau_vel_zeros=Rcoarse_zeros_vel-Rfine_zeros_vel
        tau_pos_first=Rcoarse_first_pos-Rfine_first_pos
        tau_vel_first=Rcoarse_first_vel-Rfine_first_vel
        return tau_pos_zeros, tau_vel_zeros, tau_pos_first, tau_vel_first

    
    

    def averaging_over_time_vel(self, U, level=None):
        return (level.coll.weights @ U)
    
    def averaging_over_time_pos(self, U, level=None):
        return level.coll.weights@U

    def averaging_first_order(self, U, U_averaged):
        return  (1/np.sqrt(self.eps))*(U - U_averaged)
    

    
    def FAS_zeros_order_averaged(self, X_fine, V_fine, fine_model=None, coarse_model=None):
        X_averag = self.averaging_over_time_pos(X_fine[1:], level=fine_model)
        V_averag = self.averaging_over_time_vel(V_fine[1:], level=fine_model)
        X_coarse = X_averag * np.ones(coarse_model.coll.num_nodes)
        V_coarse = V_averag * np.ones(coarse_model.coll.num_nodes)
        dt_fine = fine_model.prob.dt
        dt_coarse = coarse_model.prob.dt
        F_fine = fine_model.build_f(
            X_fine[1:], V_fine[1:], dt_fine * fine_model.coll.nodes
        )
        F_coarse = coarse_model.build_f(
            X_coarse, V_coarse, dt_coarse * coarse_model.coll.nodes
        )
        RF_fine_vel_average = self.averaging_over_time_vel(
            fine_model.coll.Q[1:, 1:] @ F_fine, level=fine_model
        )
        RF_fine_vel = RF_fine_vel_average * np.ones(coarse_model.coll.num_nodes)
        RF_coarse_vel = coarse_model.coll.Q[1:, 1:] @ F_coarse
        RF_fine_pos_average = self.averaging_over_time_pos(
            fine_model.coll.QQ[1:, 1:] @ F_fine, level=fine_model
        )
        RF_fine_pos = RF_fine_pos_average * np.ones(coarse_model.coll.num_nodes)
        RF_coarse_pos = coarse_model.coll.QQ[1:, 1:] @ F_coarse
        tau_pos = ((dt_fine**2) * RF_fine_pos) - ((dt_coarse) ** 2 * RF_coarse_pos)
        tau_vel = dt_fine * RF_fine_vel - dt_coarse * RF_coarse_vel
        X_coarse = np.append(X_fine[0], X_coarse)
        V_coarse = np.append(V_fine[0], V_coarse)
        tau_pos = np.append(0.0, tau_pos)
        tau_vel = np.append(0.0, tau_vel)
        return tau_pos, tau_vel, X_coarse, V_coarse

    def arg_min_function(self, y, y_star, num_nodes):
        func=np.linalg.norm(y[0:num_nodes]-y_star[0:num_nodes])**2+self.eps*np.linalg.norm((y[num_nodes:]-y_star[num_nodes:]))**2
        return func
    
    def arg_min_restriction(self, y, num_nodes):
        X_zero, V_zero, X_first, V_first=np.split(y, 4)
        Residual_zeros=self.sdc_coarse_model.get_coll_residual(X_zero, V_zero, tau_pos=[None], tau_vel=[None])
        Residual_first=self.sdc_coarse_first_order.get_coll_residual(X_first, V_first, tau_pos=[None], tau_vel=[None])
        func=np.linalg.norm(Residual_zeros, 2)**2+self.eps*np.linalg.norm(Residual_first, 2)**2
        return func

    def arg_min(self, U, y_star, num_nodes):
        cons=({'type':'eq', 'fun': lambda y: y[0:num_nodes]+np.sqrt(self.eps)*y[num_nodes:]-U})
        y0=y_star
        res=minimize(self.arg_min_function, y0, args=(y_star, num_nodes),  tol=1e-13)
        print(res.message)
        print(res.status)
        print(res.nfev)
        print(res.fun)
        print(res.success)
        # breakpoint()
        return res.x
    def arg_min_res(self, U, y_star, num_nodes):
        
        cons=({'type':'eq', 'fun': lambda y: y[0:2*num_nodes]+self.eps*y[2*num_nodes:]-U})
        y0=y_star
        res=minimize(self.arg_min_restriction, y0, args=(num_nodes), constraints=cons,  tol=1e-13)
        print(res.message)
        print(res.status)
        print(res.nfev)
        print(res.fun)
        print(res.success)
        # breakpoint()
        return res.x
    
    def arg_min_restriction_operator(self, X_fine, V_fine, operator=False):
        X_zero_average, V_zero_average, X_first_average, V_first_average=self.restriction_operator(X_fine, V_fine)
        X_star=np.block([X_zero_average, X_first_average])
        V_star=np.block([V_zero_average, V_first_average])
        X_zero_min, V_zero_min=np.split(self.arg_min(X_fine, X_star, 6), 2)
        X_first_min, V_first_min=np.split(self.arg_min(V_fine, V_star, 6), 2)
        return X_zero_min, V_zero_min, X_first_min, V_first_min
    
    def arg_min_res_operator(self, X_fine, V_fine):
        X_zero_average, V_zero_average, X_first_average, V_first_average=self.restriction_operator(X_fine, V_fine)
        U=np.block([X_fine, V_fine])
        Y=np.block([X_zero_average, V_zero_average, X_first_average, V_first_average])
        X_zeros_min, V_zeros_min, X_first_min, V_first_min=np.split(self.arg_min_res(U, Y, 6), 4)
        return X_zeros_min, V_zeros_min, X_first_min, V_first_min 
    
    def fas_asyp_arg_min_model(self, X_fine, V_fine, fine_model=None, coarse_zeros_model=None, coarse_first_model=None):
        RX_zeros_order, RV_zeros_order, RX_first_order, RV_first_order=self.arg_min_res_operator(X_fine, V_fine)
        
        Rfine_pos, Rfine_vel=fine_model.collocation_operator(X_fine, V_fine)

        Rcoarse_zeros_pos, Rcoarse_zeros_vel=coarse_zeros_model.collocation_operator(RX_zeros_order, RV_zeros_order)
        
        V0first_order=0.0*RV_first_order
        Rcoarse_first_pos, Rcoarse_first_vel=coarse_first_model.collocation_operator(RX_first_order, RV_first_order, V0=V0first_order)
        Rfine_zeros_pos, Rfine_zeros_vel, Rfine_first_pos, Rfine_first_vel=self.arg_min_res_operator(Rfine_pos, Rfine_vel)
        tau_pos_zeros=Rcoarse_zeros_pos-Rfine_zeros_pos
        tau_vel_zeros=Rcoarse_zeros_vel-Rfine_zeros_vel
        tau_pos_first=Rcoarse_first_pos-Rfine_first_pos
        tau_vel_first=Rcoarse_first_vel-Rfine_first_vel
        return tau_pos_zeros, tau_vel_zeros, tau_pos_first, tau_vel_first

    def last_idea_for_restriction(self, X, V, fine_model=None, coarse_zeros_model=None, coarse_first_model=None):
        X_zero=np.ones(len(X))*X[0]+coarse_zeros_model.prob.dt*coarse_zeros_model.coll.Q@V
        X_first=(X-X_zero)/np.sqrt(self.eps)
        
        T=coarse_zeros_model.prob.dt*np.append(0, coarse_zeros_model.coll.nodes)
        V_zero=np.ones(len(V))*V[0]+np.sin(T)-coarse_zeros_model.prob.dt*coarse_zeros_model.coll.Q@X
        V_first=(V-V_zero)/np.sqrt(self.eps)
        return X_zero, V_zero, X_first, V_first
    
    def last_idea_more_restriction(self, X, V, fine_model=None, coarse_zeros_model=None, coarse_first_model=None):
        T=coarse_zeros_model.prob.dt*np.append(0, coarse_zeros_model.coll.nodes)
        V_sin=np.ones(len(V))*V[0]+np.sin(T)
        X_zero=np.ones(len(X))*X[0]+coarse_zeros_model.prob.dt*coarse_zeros_model.coll.Q@V_sin-(coarse_zeros_model.prob.dt**2)*coarse_zeros_model.coll.QQ@X
        X_first=(X-X_zero)/np.sqrt(self.eps)
        # X_first=-0.8*(coarse_zeros_model.prob.dt**2)*coarse_first_order.coll.QQ@V
        
        V_zero=np.ones(len(V))*V[0]+np.sin(T)-coarse_zeros_model.prob.dt*coarse_zeros_model.coll.Q@X
        # V_first=-0.8*coarse_first_order.prob.dt*coarse_first_order.coll.Q@V
        V_first=(V-V_zero)/np.sqrt(self.eps)
        # V_first[0]=0
        # X_first[0]=0
        return X_zero, V_zero, X_first, V_first
    
    def last_idea_for_fas(self, X, V, fine_prob=None, coarse_zeros_model=None, coarse_first_model=None):
        X_zero, V_zero, X_first, V_first=self.restriction_duffing_equation(X, V, fine_model=fine_prob, coarse_zeros_model=coarse_zeros_model, coarse_first_order=coarse_first_model)
        Rfine_pos, Rfine_vel=fine_prob.collocation_operator(X, V)

        Rcoarse_zeros_pos, Rcoarse_zeros_vel=coarse_zeros_model.collocation_operator(X_zero, V_zero)
        
        V0first_order=0.0*V_first
        Rcoarse_first_pos, Rcoarse_first_vel=coarse_first_model.collocation_operator(X_first, V_first, V0=V0first_order)
        Rfine_zeros_pos, Rfine_zeros_vel, Rfine_first_pos, Rfine_first_vel=self.restriction_duffing_equation(Rfine_pos, Rfine_vel, fine_model=fine_prob, coarse_zeros_model=coarse_zeros_model, coarse_first_order=coarse_first_model)
        tau_pos_zeros=Rcoarse_zeros_pos-Rfine_zeros_pos
        tau_vel_zeros=Rcoarse_zeros_vel-Rfine_zeros_vel
        tau_pos_first=Rcoarse_first_pos-Rfine_first_pos
        tau_vel_first=Rcoarse_first_vel-Rfine_first_vel
        # breakpoint()
        return tau_pos_zeros, tau_vel_zeros, tau_pos_first, tau_vel_first

    

    









