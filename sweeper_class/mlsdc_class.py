import numpy as np
from transfer_class.transfer_class import transfer_class
from copy import deepcopy


class Mlsdc_class(transfer_class):
    def __init__(
        self, problem_params, collocation_params, sweeper_params, problem_class
    ):
        super().__init__(
            problem_params, collocation_params, sweeper_params, problem_class
        )

    def mlsdc_sweep(self, X_old, V_old):
        tau_pos, tau_vel, X_coarse_old, V_coarse_old = self.fas_correction(X_old, V_old)
        X_coarse, V_coarse=self.sdc_coarse_level.sdc_sweep(X_coarse_old, V_coarse_old, tau_pos=tau_pos, tau_vel=tau_vel)
        # X_coarse, V_coarse = self.get_without_coarse(
        #         X_coarse_old, V_coarse_old, tau_pos, tau_vel
        #     )
        # interpolation
        
        X_inter = X_old + self.interpolate(X_coarse - X_coarse_old)
        V_inter = V_old + self.interpolate(V_coarse - V_coarse_old)
        print(X_inter)

        X_fine, V_fine = self.sdc_fine_level.sdc_sweep(X_inter, V_inter)
        print(X_fine)
        breakpoint()
        return X_fine, V_fine

    # def get_coarse_solver(self, X_coarse_old, V_coarse_old, tau_pos, tau_vel):
    #     coarse_solver = self.sdc_coarse_level.sweeper.coarse_solver
    #     if coarse_solver == "sdc":
            
            
    #     elif coarse_solver == "no_coarse":

    #         
    #     else:
    #         raise ValueError(
    #             'coarse solver is not defined. Set coarse solver to "sdc" or "no_coarse"'
    #         )
    #     return X_coarse, V_coarse

    def get_mlsdc_iter_solution(self, K_iter=None, initial_guess=None):
        if K_iter is None:
            K_iter = self.sdc_fine_level.sweeper.Kiter
        if initial_guess == None:
            X = deepcopy(self.sdc_fine_level.X0)
            V = deepcopy(self.sdc_fine_level.V0)
        else:
            X, V = self.sdc_fine_level.get_initial_guess(initial_guess=initial_guess)

        for ii in range(K_iter):
            X_new, V_new = self.mlsdc_sweep(X, V)
            X=deepcopy(X_new)
            V=deepcopy(V_new)
            # 
        return X_new, V_new

    def get_without_coarse(self, X, V, tau_pos, tau_vel):
        return X - tau_pos, V - tau_vel
