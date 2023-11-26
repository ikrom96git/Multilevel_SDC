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
                + self.prob_params.f0 * np.cos(self.prob_params.omega * t)
            )
        return f
    def force(self, t):
        return 0.0*t
    def forced_const(self):
        if self.prob_params.k != 0:
            const = (self.prob_params.c - self.prob_params.omega**2) / (
                self.prob_params.k * self.prob_params.omega
            )
            Ap = (self.prob_params.f0 / self.prob_params.omega) * (1 / (1 + const**2))
            Bp = const * Ap
        elif self.prob_params.k == 0 and (c - omega**2) != 0:
            Ap = 0
            Bp = self.prob_params.f0 / (
                self.prob_params.c - self.prob_params.omega**2
            )
        elif self.prob_params.k == 0 and (c - omega**2) == 0:
            Ap = self.prob_params.f0 / (2 * self.prob_params.omega)
            Bp = 0
        else:
            raise ValueError("The system is not a harmonic oscillator")
        return Ap, Bp

    def compute_force(self, t):
        Ap, Bp = self.forced_const()
        if self.prob_params.k != 0:
            fx = Ap * np.sin(self.prob_params.omega * t) + Bp * np.cos(
                self.prob_params.omega * t
            )
            dfx = Ap * self.prob_params.omega * np.cos(
                self.prob_params.omega * t
            ) - Bp * self.prob_params.omega * np.sin(self.prob_params.omega * t)
        elif (
            self.prob_params.k == 0
            and (self.prob_params.c - self.prob_params.omega**2) != 0
        ):
            fx = Ap * np.sin(self.prob_params.omega * t) + Bp * np.cos(
                self.prob_params.omega * t
            )
            dfx = Ap * self.prob_params.omega * np.cos(
                self.prob_params.omega * t
            ) - Bp * self.prob_params.omega * np.sin(self.prob_params.omega * t)
        elif (
            self.prob_params.k == 0
            and (self.prob_params.c - self.prob_params.omega**2) == 0
        ):
            fx = Ap * t * np.sin(self.prob_params.omega * t)
            dfx = Ap * np.sin(
                self.prob_params.omega * t
            ) + Ap * t * self.prob_params.omega * np.cos(self.prob_params.omega * t)
        else:
            raise ValueError("The system is not a harmonic oscillator")
        return fx, dfx

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
                print(D)
                x = exp
                v = t * exp
                derx = exp * (-self.prob_params.k / 2)
                derv = exp * (-self.prob_params.k / 2 * t + 1)
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
        fx, dfx = self.compute_force(t0)
        x, v = self.solution_form(t0)
        u0 = np.array([x0 - fx, v0 - dfx])
        A = np.array([[x[0], x[1]], [v[0], v[1]]])
        return np.linalg.solve(A, u0)

    def compute_solution(self, x0, v0, t0, t):
        fx, dfx = self.compute_force(t)
        c = self.compute_const(x0, v0, t0)
        x, v = self.solution_form(t)
        Ap, Bp = self.forced_const()
        x_sol = c @ x + fx
        v_sol = c @ v + dfx
        return np.block([[x_sol], [v_sol]])


class OscillatorProblem_old(HarmonicOscillator):
    def __init__(self, prob_params):
        super().__init__(prob_params)
        self.mu = self.prob_params.omega**2 / self.prob_params.c
        self.kappa = (self.prob_params.k * self.prob_params.omega) / self.prob_params.c
        self.y0 = (self.prob_params.u0[0] * self.prob_params.c) / self.prob_params.f0
        self.dy0 = (self.prob_params.u0[1] * self.prob_params.c) / (
            self.prob_params.omega * self.prob_params.f0
        )
        self.tau0 = self.prob_params.t0 * self.prob_params.omega

    def time_rescale(self, t):
        return t * self.prob_params.omega

    def slow_time(self, t, kappa_hat, order: int = 0):
        tau = self.time_rescale(t)
        if order == 0:
            return np.cos(tau)
        elif order == 1:
            return np.cos(tau) + kappa_hat * np.sin(tau)
        elif order == 2:
            return (1 - kappa_hat**2) * np.cos(tau) + 2 * kappa_hat * np.sin(tau)
        else:
            raise ValueError("The order must be 0, 1 or 2")

    def slow_time_solution(self, t, kappa_hat=None, eps=None, order: int = 0):
        if eps is None:
            eps = self.mu
        if kappa_hat is None:
            kappa_hat = self.kappa
        tau = self.time_rescale(t)
        y0 = self.slow_time(tau, kappa_hat, order=0)
        y1 = y0 + eps * self.slow_time(tau, kappa_hat, order=1)
        y2 = y1 + eps * self.slow_time(tau, kappa_hat, order=2)
        if order == 0:
            return y0
        elif order == 1:
            return y1
        elif order == 2:
            return y2
        else:
            raise ValueError("The order must be 0, 1 or 2")

    def none_eps_solution(self, t):
        tau = self.time_rescale(t)
        return np.cos(tau)
