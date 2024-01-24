
import numpy as np
from core.Pars import _Pars


class HarmonicOscillator:

    def __init__(self, problem_params):
        """


        Parameters
        ----------
        problem_params : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.params = _Pars(problem_params)
        self.determinant = self.params.mu**2*0.25-self.params.kappa

    def get_cos(self, omega, t):
        return np.cos(omega*t)

    def get_sin(self, omega, t):
        return np.sin(omega, t)

    def get_dcos(self, omega, t):
        return -omega*np.sin(omega*t)

    def get_dsin(self, omega, t):
        return omega*np.cos(omega*t)

    def get_exp(self, omega,  t):
        return np.exp(omega*t)

    def get_dexp(self, omega, t):
        return omega*self.get_exp(omega, t)

    def const_withoutFriction(self, t):
        omega = np.sqrt(self.params.kappa)
        A = np.zeros((2, 2))
        A[0, 0], A[0, 1] = self.get_cos(omega, t), self.get_sin(omega, t)
        A[1, 0], A[1, 1] = self.get_dcos(omega, t), self.get_dsin(omega, t)
        b = np.linalg.sove(A, self.params.u0)
        return A, b

    def const_withFriction(self, t):
        A = np.zeros((2, 2))
        omega_exp = -0.5*self.params.mu
        exp = self.get_exp(omega_exp, t)
        dexp = self.get_dexp(omega_exp, t)
        if self.determinant < 0:
            omega = np.sqrt(-self.determinant)
            A[0, 0], A[0, 1] = exp * \
                self.get_cos(omega, t), exp*self.get_sin(omega, t)
            A[1, 0] = dexp*self.get_cos(omega, t)+exp*self.get_dcos(omega, t)
            A[1, 1] = dexp*self.get_sin(omega, t)+exp*self.get_dsin(omega, t)

        if self.determinant > 0:
            omega = np.sqrt(self.determinant)
            A[0, 0], A[0, 1] = exp * \
                self.get_exp(omega, t), exp*self.get_exp(-omega, t)
            A[1, 0] = dexp*self.get_exp(omega, t)+exp*self.get_dexp(omega, t)
            A[1, 1] = dexp*self.get_exp(-omega, t)+exp*self.get_dexp(-omega, t)

        if self.determinant == 0:
            A[0, 0], A[0, 1] = exp, t*exp
            A[1, 0], A[1, 1] = dexp, t*dexp+exp
        b = np.linalg.solve(A, self.params.u0)
        return A, b

    def get_solution(self, t):
        if self.mu == 0:
            *_, b = self.const_withoutFriction(self.params.t0)
            A, *_ = self.const_withoutFriction(t)
        if self.mu > 0:
            *_, b = self.const_withFriction(self.params.t0)
            A, *_ = self.const_withFriction(t)
        return A@b

    def get_solution_interval(self, time: np.narray) -> np.narray:
        solution_store = np.zeros([2, len(time)])
        for tt in range(time):
            solution_store[:, tt] = self.get_solution(tt)
        return solution_store
