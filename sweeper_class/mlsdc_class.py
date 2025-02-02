import numpy as np
from copy import deepcopy
from core.sort_params import SortParams


class Mlsdc_class(SortParams):
    def __init__(
        self,
        problem_params,
        collocation_params,
        sweeper_params,
        problem_class,
        restriction_class,
        eps=None,
    ):
        super().__init__(
            problem_params,
            collocation_params,
            sweeper_params,
            problem_class,
            restriction_class,
            eps,
        )
        self.first_order_model = len(problem_class)
        print(problem_class)

    def m3lsdc_sweep(self, X_old, V_old):
        # breakpoint()
        print('M3LSDC is used')
        X_zero, V_zero, X_first, V_first = self.transfer_operator.restriction_operator(
            X_old,
            V_old,
            fine_model=self.sdc_fine_model,
            coarse_zero_model=self.sdc_coarse_model,
            coarse_first_model=self.sdc_coarse_first_model,
            eps=self.eps,
        )
        tau_zero_pos, tau_zero_vel, tau_first_pos, tau_first_vel = (
            self.transfer_operator.fas_correction_operator(
                X_old,
                V_old,
                fine_prob=self.sdc_fine_model,
                coarse_zeros_model=self.sdc_coarse_model,
                coarse_first_model=self.sdc_coarse_first_model,
                eps=self.eps,
            )
        )

        X_coarse_zero, V_coarse_zero = self.sdc_coarse_model.sdc_sweep(
            X_zero, V_zero, tau_zero_pos, tau_zero_vel
        )
        X_coarse_first, V_coarse_first = self.sdc_coarse_first_model.sdc_sweep(
            X_first, V_first, tau_first_pos, tau_first_vel
        )

        X_zero_diff = X_coarse_zero - X_zero
        V_zero_diff = V_coarse_zero - V_zero
        X_first_diff = X_coarse_first - X_first
        V_first_diff = V_coarse_first - V_first

        X_expan = self.sdc_coarse_first_model.problem_class.asyp_expansion(
            X_zero_diff, X_first_diff, self.eps
        )
        V_expan = self.sdc_coarse_first_model.problem_class.asyp_expansion(
            V_zero_diff, V_first_diff, self.eps
        )

        X_inter = X_old + self.interpolation_node(X_expan)
        V_inter = V_old + self.interpolation_node(V_expan)
        X_fine, V_fine = self.sdc_fine_model.sdc_sweep(X_inter, V_inter)
        # print("position (Asyp): ", X_old - X_fine)
        # breakpoint()
        return X_fine, V_fine

    def sweep(self, X_old, V_old):
        if self.first_order_model == 3:
            return self.m3lsdc_sweep(X_old, V_old)
        else:
            return self.mlsdc_sweep(X_old, V_old)

    def mlsdc_sweep(self, X_old, V_old):
        print('MLSDC sweeper is used')
        X_coarse_old, V_coarse_old = self.transfer_operator.restriction_operator(
            X_old,
            V_old,
            fine_model=self.sdc_fine_model,
            coarse_zero_model=self.sdc_coarse_model,
            coarse_first_model=None,
            eps=None,
        )
        # breakpoint()
        tau_pos, tau_vel = self.transfer_operator.fas_correction_operator(
            X_old,
            V_old,
            fine_prob=self.sdc_fine_model,
            coarse_zeros_model=self.sdc_coarse_model,
            
        )
        X_coarse, V_coarse = self.get_coarse_solver(
            X_coarse_old, V_coarse_old, tau_pos, tau_vel
        )
        # interpolation
        X_inter = X_old + self.interpolation_node(X_coarse - X_coarse_old)
        V_inter = V_old + self.interpolation_node(V_coarse - V_coarse_old)
        X_fine, V_fine = self.sdc_fine_model.sdc_sweep(X_inter, V_inter)
        # print("position (MLSDC): ", X_old - X_fine)
        # breakpoint()
        return X_fine, V_fine

    def get_coarse_solver(
        self, X_coarse_old, V_coarse_old, tau_pos, tau_vel, coarse_solver=None
    ):
        if coarse_solver is None:
            coarse_solver = self.sdc_coarse_model.sweeper.coarse_solver

        if coarse_solver == "sdc":
            X_coarse, V_coarse = self.sdc_coarse_model.sdc_sweep(
                X_coarse_old, V_coarse_old, tau_pos=tau_pos, tau_vel=tau_vel
            )
        elif coarse_solver == "no_coarse":
            print(coarse_solver)
            X_coarse, V_coarse = self.get_without_coarse(
                X_coarse_old, V_coarse_old, tau_pos, tau_vel
            )

        else:
            raise ValueError(
                'coarse solver is not defined. Set coarse solver to "sdc" or "no_coarse"'
            )
        return X_coarse, V_coarse

    def get_mlsdc_iter_solution(self, K_iter=None, initial_guess=None):
        if K_iter is None:
            K_iter = self.sdc_fine_model.sweeper.Kiter
        if initial_guess is None:
            X = deepcopy(self.sdc_fine_model.X0)
            V = deepcopy(self.sdc_fine_model.V0)
        else:
            X, V = self.sdc_fine_model.get_initial_guess(initial_guess=initial_guess)

        for _ in range(K_iter):
            X_new, V_new = self.sweep(X, V)
            X = deepcopy(X_new)
            V = deepcopy(V_new)
        return X_new, V_new
    def get_ml3sdc_iter_exact(self, K_iter=None, initial_guess=None):
        if K_iter is None:
            K_iter = self.sdc_fine_model.sweeper.Kiter
        if initial_guess is None:
            X = deepcopy(self.sdc_fine_model.X0)
            V = deepcopy(self.sdc_fine_model.V0)
        else:
            X, V = self.sdc_fine_model.get_initial_guess(initial_guess=initial_guess)

        for _ in range(K_iter):
            X_new, V_new = self.ml3sdc_sweep_exact(X, V)
            X = deepcopy(X_new)
            V = deepcopy(V_new)
        return X_new, V_new
    def get_without_coarse(self, X, V, tau_pos, tau_vel):
        tau_pos_node_to_node = np.append(0, tau_pos[1:] - tau_pos[:-1])

        tau_vel_node_to_node = np.append(0, tau_vel[1:] - tau_vel[:-1])
        for m in range(self.sdc_coarse_model.coll.num_nodes):
            X[m + 1] = X[m] + tau_pos_node_to_node[m + 1]
            V[m + 1] = V[m] + tau_vel_node_to_node[m + 1]
        return X, V

    def ml3sdc_sweep_exact(self, X_old, V_old):
        print('MLSDC sweeper is used')
        X_coarse_old, V_coarse_old = self.transfer_operator.restriction_operator(
            X_old,
            V_old,
            fine_model=self.sdc_fine_model,
            coarse_zero_model=self.sdc_coarse_model,
            coarse_first_model=None,
            eps=None,
        )
        # breakpoint()
        tau_pos, tau_vel = self.transfer_operator.fas_correction_operator(
            X_old,
            V_old,
            fine_prob=self.sdc_fine_model,
            coarse_zeros_model=self.sdc_coarse_model,
            
        )
        
        T=np.append(0,self.sdc_coarse_model.prob.dt*self.sdc_coarse_model.coll.nodes)
        
        X_exact, V_exact=np.split(self.sdc_coarse_model.problem_class.get_ntime_exact_solution(T),2)
        X_first, V_first=np.split(self.sdc_coarse_first_model.problem_class.get_ntime_exact_solution(T), 2)
        X_coarse=self.sdc_coarse_first_model.problem_class.asyp_expansion(X_exact[0], X_first[0], self.eps)
        V_coarse=self.sdc_coarse_first_model.problem_class.asyp_expansion(V_exact[0], V_first[0], self.eps)
        
        X_coarse, V_coarse=self.sdc_coarse_model.get_collocation_fsolve(tau_pos+X_exact[0], tau_vel+V_exact[0])
        # interpolation
        X_inter = X_old + self.interpolation_node(X_coarse - X_coarse_old)
        V_inter = V_old + self.interpolation_node(V_coarse - V_coarse_old)
        X_fine, V_fine = self.sdc_fine_model.sdc_sweep(X_inter, V_inter)
        breakpoint()
        # print("position (MLSDC): ", X_old - X_fine)
        # breakpoint()
        return X_fine, V_fine

    def get_mlsdc_ntime_sweep(self, time):
        pass
