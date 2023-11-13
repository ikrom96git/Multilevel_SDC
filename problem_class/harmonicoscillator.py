import numpy as np
import matplotlib.pyplot as plt
from problem_class.Pars import _Pars


class HarmonicOscillator:
    def __init__(self, prob_params):
        self.prob_params = _Pars(prob_params)

    def build_f(self, x, v, t):
        if self.prob_params.oscillator_type == "free":
            f = -self.prob_params.k * v - self.prob_params.c * x
        elif self.prob_params.oscillator_type == "forced":
            f = (
                -self.prob_params.k * v
                - self.prob_params.c * x
                + self.prob_params.f0 * np.cos(self.prob_params.w * t)
            )
        return f

    def solution_form(self, t):
        if self.prob_params.k == 0 and self.prob_params.c > 0:
            omega_0 = np.sqrt(self.prob_params.c)
            x = np.cos(omega_0 * t)
            v = np.sin(omega_0 * t)
            derx = -omega_0 * np.sin(omega_0 * t)
            derv = omega_0 * np.cos(omega_0 * t)
            return np.array([x, v]), np.array([derx, derv])
        elif self.prob_params.k > 0 and self.prob_params.c > 0:
            D = self.prob_params.k**2 / 4 - self.prob_params.c
            exp = np.exp(-self.prob_params.k * t / 2)
            if D < 0:
                omega = np.sqrt(-D)
                x = exp * np.cos(omega * t)
                v = exp * np.sin(omega * t)
                derx = exp * (
                    -omega * np.sin(omega * t)
                    - self.prob_params.k / 2 * np.cos(omega * t)
                )
                derv = exp * (
                    omega * np.cos(omega * t)
                    - self.prob_params.k / 2 * np.sin(omega * t)
                )
                return np.array([x, v]), np.array([derx, derv])
            elif D == 0:
                x = exp
                v = t * exp
                derx = exp * (-self.prob_params.k / 2)
                derv = exp * (-self.prob_params.k / 2 + t)
                return np.array([x, v]), np.array([derx, derv])
            else:
                omega = np.sqrt(D)
                x = exp * np.exp(omega * t)
                v = exp * np.exp(-omega * t)
                derx = exp * (
                    -self.prob_params.k / 2 * np.exp(omega * t)
                    + omega * np.exp(omega * t)
                )
                derv = exp * (
                    -self.prob_params.k / 2 * np.exp(-omega * t)
                    - omega * np.exp(-omega * t)
                )
                return np.array([x, v]), np.array([derx, derv])

    def compute_const(self, x0, v0, t0):
        x, v = self.solution_form(t0)
        u0 = np.array([x0, v0])
        A = np.array([[x[0], v[0]], [x[1], v[1]]])
        return np.linalg.solve(A, u0)

    def compute_solution(self, x0, v0, t0, t):
        c = self.compute_const(x0, v0, t0)
        x, v = self.solution_form(t)
        return c[0] * x + c[1] * v
