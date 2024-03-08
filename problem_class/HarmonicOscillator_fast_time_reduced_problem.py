import numpy as np
from core.Pars import _Pars


class HarmonicOscillator_fast_time(object):
    def __init__(self, problem_params):
        self.params = _Pars(problem_params)

    def build_f(self, X, V, T):
        return -self.params.mu * V - self.params.kappa * X + self.params.F0

    def get_rhs(self, u, t):
        u = np.asarray(u)
        y_dot = u[0]
        v_dot = -self.params.mu * u[1] - self.params.kappa * u[0] + self.params.F0
        return np.asarray([y_dot, v_dot])

    def get_exact_solution(self, t):
        a_0 = 0
        b_0 = 1
        dy=np.cos(t)
        
        dy = a_0 * np.sin(t) + b_0 * np.cos(t) + self.params.F0
        dv = a_0 * np.cos(t) - b_0 * np.sin(t)
        return np.array([dy, dv])

    def get_ntime_exact_solution(self, time):
        solution = np.zeros([2, len(time)])
        for tt in range(len(time)):
            solution[:, tt] = self.get_exact_solution(time[tt])
        return solution
