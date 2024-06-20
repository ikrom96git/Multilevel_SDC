import numpy as np
from core.Pars import _Pars


class HarmonicOscillator_fast_time(object):
    def __init__(self, problem_params):
        self.params = _Pars(problem_params)

    def build_f(self, X, V, T):
        return -self.params.mu * V - self.params.kappa * X + self.params.F0

    def get_rhs(self, u, t):
        u = np.asarray(u)
        y_dot = u[1]
        v_dot = -self.params.mu * u[1] - self.params.kappa * u[0] + self.params.F0
        return np.asarray([y_dot, v_dot])

    def get_exact_solution(self, t):
        a_0 = 0
        b_0 = 1

        dy = a_0 * np.sin(t) + b_0 * np.cos(t) + self.params.F0
        dv = a_0 * np.cos(t) - b_0 * np.sin(t)
        return np.array([dy, dv])

    def get_ntime_exact_solution(self, time):
        solution = np.zeros([2, len(time)])
        for tt in range(len(time)):
            solution[:, tt] = self.get_exact_solution(time[tt])
        return solution


class HarmonicOscillator_fast_time_first_order(object):
    def __init__(self, problem_params):
        self.params = _Pars(problem_params)
        self.kappa_hat = 0.8

    def build_f(self, X, V, T):
        a0 = 0
        b0 = 1
        rhs = a0 * np.cos(T) - b0 * np.sin(T)
        return -self.params.mu * V - self.params.kappa * X - self.kappa_hat * rhs

    def get_rhs(self, u, t):
        a0 = 0
        b0 = 1
        rhs = a0 * np.cos(t) - b0 * np.sin(t)
        u = np.asarray(u)
        y_dot = u[0]
        v_dot = -self.params.mu * u[1] - self.params.kappa * u[0] - self.kappa_hat * rhs
        return np.asarray([y_dot, v_dot])

    def asyp_expansion(self, zeros_order_sol, first_order_sol, eps):
        return zeros_order_sol + np.sqrt(eps) * first_order_sol

    def get_dy0(self, t):
        a_0 = 0
        b_0 = 1

        dy = 0.5 * self.kappa_hat * (a_0 * np.sin(t) + b_0 * np.cos(t))
        dv = dy + 0.5 * self.kappa_hat * t * (a_0 * np.cos(t) - b_0 * np.sin(t))
        return t * dy, dv

    def get_exact_solution(self, t):
        dy0, dy1 = self.get_dy0(t)
        a1, b1 = self.get_dy0(self.params.t0)
        b1 = b1
        dy = a1 * np.sin(t) + b1 * np.cos(t) - dy0
        dv = a1 * np.cos(t) - b1 * np.sin(t) - dy1
        return np.array([dy, dv])

    def get_ntime_exact_solution(self, time):
        solution = np.zeros([2, len(time)])
        for tt in range(len(time)):
            solution[:, tt] = self.get_exact_solution(time[tt])
        return solution
