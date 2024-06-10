import numpy as np
from core.Pars import _Pars
from transfer_class.CollocationMatrix import CollocationMatrix
from copy import deepcopy
from scipy.optimize import fsolve


class sdc_class(object):
    def __init__(
        self, problem_params, collocation_params, sweeper_params, problem_class
    ):
        self.prob = _Pars(problem_params)
        self.sweeper = _Pars(sweeper_params)
        self.coll = CollocationMatrix(collocation_params)
        self.problem_class = problem_class(problem_params)
        self.build_f = self.problem_class.build_f
        self.get_residual = []
        self.get_rhs = self.problem_class.get_rhs
        self.X0, self.V0 = self.get_initial_guess()
    def collocation_operator(self, X, V, V0=None):
        if V0 is None:
            V0=np.ones(len(V))*V[0]
        T = self.prob.dt * np.append(self.prob.t0, self.coll.nodes)
        X_pos=X-self.prob.dt*self.coll.Q@V0-self.prob.dt**2*self.coll.QQ@self.build_f(X, V, T)
        V_pos=V-self.prob.dt*self.coll.Q@self.build_f(X, V, T)
        return X_pos, V_pos
    
    def get_update_step(self, X, V):
        T = self.prob.dt * self.coll.nodes
        X_update=X[0]+self.prob.dt*V[0]+self.prob.dt**2*(self.coll.q@self.coll.Q[1:,1:])@self.build_f(X, V, T)
        V_update=V[0]+self.prob.dt*self.coll.q@self.build_f(X, V, T)
        return X_update, V_update
        



    def sdc_sweep(self, X_old, V_old, tau_pos=[None], tau_vel=[None]):

        M = self.coll.num_nodes
        T = self.prob.dt * np.append(self.prob.t0, self.coll.nodes)
        X = deepcopy(X_old)
        V = deepcopy(V_old)
        F_old = self.build_f(X_old, V_old, T)
        SQF = self.prob.dt**2 * (self.coll.SQ @ F_old - self.coll.Sx @ F_old)
        SF = self.prob.dt * self.coll.S @ F_old
        if None not in tau_pos:

            tau_pos_nn = np.append(0, tau_pos[1:] - tau_pos[:-1])
            tau_vel_nn = np.append(0, tau_vel[1:] - tau_vel[:-1])

            SQF += tau_pos_nn
            SF += tau_vel_nn

        for m in range(M):
            F_new = self.build_f(X, V, T)
            SXF = self.prob.dt**2 * self.coll.Sx @ self.build_f(X, V, T)
            X[m + 1] = (
                X[m]
                + self.prob.dt * self.coll.delta_m[m] * V[0]
                + SXF[m + 1]
                + SQF[m + 1]
            )
            rhs = (
                V[m]
                - 0.5 * self.prob.dt * self.coll.delta_m[m] * (F_old[m + 1] + F_old[m])
                + 0.5 * self.prob.dt * self.coll.delta_m[m] * F_new[m]
                + SF[m + 1]
            )

            def func(v):
                return (
                    rhs
                    + 0.5
                    * self.prob.dt
                    * self.coll.delta_m[m]
                    * self.build_f(X[m + 1], v, T[m + 1])
                    - v
                )

            V[m + 1] = fsolve(func, V[m])
        self.get_residual.append(self.compute_residual(X, V, tau_pos, tau_vel))

        return X, V

    def sdc_iter(self, K=None, initial_guess=None):
        """
        Performs K iterations of the SDC sweep process and returns the final X and V values.

        Args:
            K (int, optional): Number of iterations. Defaults to None.
            initial_guess (any, optional): Initial values for X and V. Defaults to None.

        Returns:
            tuple: Final X and V values after K iterations of the SDC sweep process.
        """

        if K is None:
            K = self.sweeper.Kiter
        if initial_guess is None:

            X = deepcopy(self.X0)
            V = deepcopy(self.V0)
        else:
            X, V = self.get_initial_guess(initial_guess=initial_guess)

        for _ in range(K):

            X, V = self.sdc_sweep(X, V)

        return X, V

    def get_max_norm_residual(self):
        self.sdc_iter()
        return self.get_residual

    def compute_residual(self, X, V, tau_pos, tau_vel):
        X0 = self.prob.u0[0] * np.ones(self.coll.num_nodes + 1)
        V0 = self.prob.u0[1] * np.ones(self.coll.num_nodes + 1)
        T = self.prob.dt * np.append(self.prob.t0, self.coll.nodes)
        vel_residual = V0 + self.prob.dt * self.coll.Q @ self.build_f(X, V, T) - V
        pos_residual = (
            X0
            + self.prob.dt * self.coll.Q @ V0
            + self.prob.dt**2 * self.coll.QQ @ self.build_f(X, V, T)
            - X
        )
        if None not in tau_pos:
            pos_residual += tau_pos
            vel_residual += tau_vel
        vel_max_norm = self.max_norm_residual(vel_residual)
        pos_max_norm = self.max_norm_residual(pos_residual)
        return [pos_max_norm, vel_max_norm]

    def max_norm_residual(self, residual):
        return np.max(np.abs(residual))

    def get_collocation_fsolve(self):
        X0 = self.prob.u0[0] * np.ones(self.coll.num_nodes)
        V0 = self.prob.u0[1] * np.ones(self.coll.num_nodes)
        U0 = np.concatenate([X0, V0])
        U = fsolve(self.get_collocation_problem, U0)
        X, V = np.split(U, 2)
        X = np.append(self.prob.u0[0], X)
        V = np.append(self.prob.u0[1], V)
        return X, V

    def get_collocation_problem(self, U):
        T = self.prob.dt * self.coll.nodes
        ret = []

        for ii in range(self.coll.num_nodes):
            Y = np.array([U[ii], U[self.coll.num_nodes + ii]])
            pos_equation = np.zeros([1])
            vel_equation = np.zeros([1])
            for jj in range(self.coll.num_nodes):
                X = np.array([U[jj], U[self.coll.num_nodes + jj]])
                pos_equation += (
                    -self.prob.dt * self.coll.Q[ii + 1, jj + 1] * self.prob.u0[1]
                    - self.prob.dt**2
                    * self.coll.QQ[ii + 1, jj + 1]
                    * self.get_rhs(X, t=T[jj])[1]
                )
                vel_equation += (
                    -self.prob.dt
                    * self.coll.Q[ii + 1, jj + 1]
                    * self.get_rhs(X, t=T[jj])[1]
                )

            equation = np.concatenate([pos_equation, vel_equation])
            ret = np.append(ret, Y - self.prob.u0 + equation)

        return ret

    def get_initial_guess(self, initial_guess=None):
        if initial_guess is None:
            initial_guess = self.sweeper.initial_guess
        if initial_guess == "spread":
            X0 = self.prob.u0[0] * np.ones(self.coll.num_nodes + 1)
            V0 = self.prob.u0[1] * np.ones(self.coll.num_nodes + 1)
        elif initial_guess == "collocation":
            X0, V0 = self.get_collocation_fsolve()

        elif initial_guess == "zeros":
            X0 = np.zeros(self.coll.num_nodes + 1)
            V0 = np.zeros(self.coll.num_nodes + 1)
            X0[0] = self.prob.u0[0]
            V0[0] = self.prob.u0[1]
        elif initial_guess == "exact":
            time = self.prob.dt * np.append(self.prob.t0, self.coll.nodes)
            if self.prob.F0 is None:
                exact_solution = self.problem_class.get_solution_ntimeWithoutForce(time)
            else:
                exact_solution = self.problem_class.get_solution_ntimeWithForce(time)
            X0 = exact_solution[0, :]
            V0 = exact_solution[1, :]

        elif initial_guess == "10SDC":
            X0, V0 = self.sdc_iter(K=10, initial_guess="spread")
            self.get_residual = []
        else:
            raise (
                ValueError(
                    'Initial guess is not given. Set initial guess: "spread", "collocation" or "zeros"'
                )
            )
        return X0, V0


if __name__ == "__main__":
    pass
