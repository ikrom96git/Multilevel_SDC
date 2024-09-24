import numpy as np
from core.Pars import _Pars
from transfer_class.CollocationMatrix import CollocationMatrix
from scipy.optimize import fsolve
from problem_class.Duffing_Equation_2 import DuffingEquation
from sweeper_class.sdc_class import sdc_class


dt = 0.1

eps = 0.1


class collocation_problem(object):
    def __init__(self, problem_params, collocation_params):

        self.prob = _Pars(problem_params)
        self.coll = CollocationMatrix(collocation_params)

    def get_collocation_fine_model(self, U):
        equation_system = []

        for ii in range(self.coll.num_nodes):
            U0 = np.array([U[ii], U[self.coll.num_nodes + ii]])
            pos_equation = np.zeros([1])
            vel_equation = np.zeros([1])
            for jj in range(self.coll.num_nodes):
                u = np.array([U[jj], U[self.coll.num_nodes + jj]])
                pos_equation += (
                    -self.prob.dt * self.coll.Q[ii + 1, jj + 1] * self.prob.u0[1]
                    - self.prob.dt**2
                    * self.coll.QQ[ii + 1, jj + 1]
                    * self.get_rhs_fine(u)[1]
                )
                vel_equation += (
                    -self.prob.dt
                    * self.coll.Q[ii + 1, jj + 1]
                    * self.get_rhs_fine(u)[1]
                )
            equation = np.concatenate([pos_equation, vel_equation])
            equation_system = np.append(equation_system, U0 - self.prob.u0 + equation)
        return equation_system

    def get_collocation_coarse_model(self, U, X0, V0):
        equation_system = []

        for ii in range(self.coll.num_nodes):
            U0 = np.array([U[ii], U[self.coll.num_nodes + ii]])
            sol = np.array([X0[ii], V0[ii]])
            pos_equation = np.zeros([1])
            vel_equation = np.zeros([1])
            for jj in range(self.coll.num_nodes):
                u = np.array([U[jj], U[self.coll.num_nodes + jj]])
                pos_equation += (
                    -self.prob.dt * self.coll.Q[ii + 1, jj + 1] * self.prob.u0[1]
                    - self.prob.dt**2
                    * self.coll.QQ[ii + 1, jj + 1]
                    * self.get_rhs_coarse(u)[1]
                )
                vel_equation += (
                    -self.prob.dt
                    * self.coll.Q[ii + 1, jj + 1]
                    * self.get_rhs_coarse(u)[1]
                )
            equation = np.concatenate([pos_equation, vel_equation])
            equation_system = np.append(equation_system, U0 - sol + equation)
        return equation_system

    def get_rhs_fine(self, u):
        u = np.array(u)
        x_dot = u[1]
        v_dot = -self.prob.omega**2 * u[0] + self.prob.eps * (u[1] ** 2) * u[0]
        return np.asarray([x_dot, v_dot])

    def get_rhs_coarse(self, u):
        u = np.array(u)
        x_dot = u[1]
        v_dot = -self.prob.omega**2 * u[0]
        return np.asarray([x_dot, v_dot])

    def build_f_fine(self, X, V):
        return -self.prob.omega**2 * X + self.prob.eps * (V**2) * X

    def build_f_coarse(self, X, V):
        return -self.prob.omega**2 * X

    def compute_integral(self, X, V, model=None):
        if model == "fine":
            build_f = self.build_f_fine(X, V)
        elif model == "coarse":
            build_f = self.build_f_coarse(X, V)
        else:
            raise ValueError("not defined")

        vel_int = self.prob.dt * self.coll.Q @ build_f
        pos_int = self.prob.dt**2 * self.coll.QQ @ build_f
        return pos_int, vel_int

    def compute_integral_coarse(self, X, V):
        build_f = self.build_f_coarse(X, V)
        V0 = V[0] * np.ones(len(V))
        vel_int = V - self.prob.dt * self.coll.Q @ build_f
        pos_int = X - (
            self.prob.dt * self.coll.Q @ V0 + self.prob.dt**2 * self.coll.QQ @ build_f
        )
        return pos_int, vel_int

    def compute_residual(self, X, V, X0, V0, model=None):
        pos_int, vel_int = self.compute_integral(X, V, X0, V0, model)
        return pos_int - X, vel_int - V

    def get_collocation_fsolve_fine(self, X0, V0):
        U0 = np.concatenate([X0, V0])
        U = fsolve(self.get_collocation_fine_model, U0)
        X, V = np.split(U, 2)
        X = np.append(self.prob.u0[0], X)
        V = np.append(self.prob.u0[1], V)
        return X, V

    def get_collocation_fsolve_coarse(self, X, V, X0, V0):

        U0 = np.concatenate([X[1:], V[1:]])
        U = fsolve(self.get_collocation_coarse_model, U0, args=(X0[1:], V0[1:]))
        X, V = np.split(U, 2)
        X = np.append(self.prob.u0[0], X)
        V = np.append(self.prob.u0[1], V)
        return X, V


def fine_params():
    problem_params = dict()
    problem_params["omega"] = 1.0
    problem_params["b"] = 1.0
    problem_params["eps"] = eps
    problem_params["u0"] = [2, 0]
    problem_params["dt"] = dt
    problem_params["t0"] = 0.0
    return problem_params


def coarse_params():
    problem_params = dict()
    problem_params["omega"] = 1.0
    problem_params["b"] = 1.0
    problem_params["eps"] = eps
    problem_params["u0"] = [2, 0]
    problem_params["dt"] = dt
    problem_params["t0"] = 0.0
    return problem_params


def m3lsdc():
    collocation_params = dict()
    collocation_params["quad_type"] = "GAUSS"
    collocation_params["num_nodes"] = 5
    sweeper_params = dict()
    sweeper_params["Kiter"] = 10
    sweeper_params["coarse_solver"] = "sdc"
    sweeper_params["initial_guess"] = "10SDC"
    f_params = fine_params()
    c_params = coarse_params()
    problem_class = DuffingEquation

    fine_model = collocation_problem(f_params, collocation_params)
    coarse_model = collocation_problem(c_params, collocation_params)
    sdc_model = sdc_class(f_params, collocation_params, sweeper_params, problem_class)
    X_sdc, V_sdc = sdc_model.sdc_iter()
    X0 = fine_model.prob.u0[0] * np.ones(fine_model.coll.num_nodes + 1)
    V0 = fine_model.prob.u0[1] * np.ones(fine_model.coll.num_nodes + 1)
    # ML3SDC
    Xf, Vf = fine_model.compute_integral(X_sdc, V_sdc, model="fine")
    A_cx, A_cv = coarse_model.compute_integral_coarse(Xf, Vf)
    # RX, RV=fine_model.compute_residual(X_sdc, V_sdc, X0, V0, model='fine')
    RX = 0.0 * Xf
    RV = 0.0 * Vf
    A_crx, A_crv = fine_model.compute_integral(RX, RV, model="fine")
    X0c = A_cx + A_crx
    V0c = A_cv + A_crv
    Xc, Vc = coarse_model.get_collocation_fsolve_coarse(Xf, Vf, X0c, V0c)
    breakpoint()
    Rcx, Rcv = coarse_model.compute_residual(Xc, Vc, X0c, V0c, model="coarse")

    print(Rcx)


if __name__ == "__main__":
    m3lsdc()
