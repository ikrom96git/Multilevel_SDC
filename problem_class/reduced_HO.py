import numpy as np

from problem_class.Pars import _Pars


class Reduced_HO(object):
    def __init__(self, prob_params):
        self.prob_params = _Pars(prob_params)

    def build_f(self, x, v, t):
        f = (
            -self.prob_params.kappa * x
            - self.prob_params.c * v
            + self.prob_params.f0 * np.cos(t)
        )
        return f

    def get_force_const(self):
        Omega = (self.prob_params.c - 1.0) / (self.prob_params.kappa)
        Ap = (self.prob_params.f0 / self.prob_params.kappa) * (1.0 + Omega**2) ** (-1)
        Bp = Omega * Ap
        return Ap, Bp

    def get_force(self, t):
        Ap, Bp = self.get_force_const()
        fx = Ap * np.sin(t) + Bp * np.cos(t)
        dfx = Ap * np.cos(t) - Bp * np.sin(t)
        return fx, dfx

    def get_const(self, t):
        exp = np.exp(-0.5 * self.prob_params.kappa * t)
        omega = np.sqrt(self.prob_params.c - self.prob_params.kappa**2 / 4)
        x = np.cos(omega * t) * exp
        v = np.sin(omega * t) * exp
        derx = -0.5 * self.prob_params.kappa * x + omega * v
        derv = -0.5 * self.prob_params.kappa * v - omega * x
        return np.array([x, v]), np.array([derx, derv])

    def find_const(self, t):
        x, derx = self.get_const(t)
        fx, dfx = self.get_force(t)
        x0 = self.prob_params.u0
        C = np.linalg.solve(
            np.array([[x[0], x[1]], [derx[0], derx[1]]]),
            np.array([x0[0] - fx, x0[1] - dfx]),
        )
        return C

    def get_sol(self, t):
        C = self.find_const(t[0])
        x, derx = self.get_const(t)

        fx, dfx = self.get_force(t)
        X = C[0] * x[0] + C[1] * x[1] + fx
        V = C[0] * derx[0] + C[1] * derx[1] + dfx
        return np.array([X, V])
