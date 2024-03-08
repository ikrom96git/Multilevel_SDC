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
        self.determinant = self.params.mu**2 * 0.25 - self.params.kappa

    def build_f(self, X, V, T):
        withoutforce = -self.params.mu * V - self.params.kappa * X
        if self.params.F0 is None:
            func = withoutforce
        else:
            func = withoutforce + self.params.F0 * np.cos(T)
        return func

    def get_rhs(self, u, t):
        u = np.asarray(u)
        x_dot = u[0]
        if self.params.F0 is None:
            v_dot = -self.params.mu * u[1] - self.params.kappa * u[0]
        else:
            v_dot = (
                -self.params.mu * u[1]
                - self.params.kappa * u[0]
                + self.params.F0 * np.cos(t)
            )

        return np.asarray([x_dot, v_dot])

    def get_cos(self, omega, t):
        return np.cos(omega * t)

    def get_sin(self, omega, t):
        return np.sin(omega * t)

    def get_dcos(self, omega, t):
        return -omega * np.sin(omega * t)

    def get_dsin(self, omega, t):
        return omega * np.cos(omega * t)

    def get_exp(self, omega, t):
        return np.exp(omega * t)

    def get_dexp(self, omega, t):
        return omega * self.get_exp(omega, t)

    def const_withoutFriction(self, t, F0=None, omega_force=None):
        omega = np.sqrt(self.params.kappa)
        A = np.zeros((2, 2))
        A[0, 0], A[0, 1] = self.get_cos(omega, t), self.get_sin(omega, t)
        A[1, 0], A[1, 1] = self.get_dcos(omega, t), self.get_dsin(omega, t)
        if omega_force is None:
            omega_force = 1.0

        if F0 is None:
            u0 = self.params.u0
        else:
            forceterm = self.get_forceTerm(t, F0=F0, omega_force=omega_force)
            u0 = self.params.u0 - forceterm

        b = np.linalg.solve(A, u0)
        return A, b

    def const_withFriction(self, t, F0=None, omega_force=None):
        A = np.zeros((2, 2))
        omega_exp = -0.5 * self.params.mu
        exp = self.get_exp(omega_exp, t)
        dexp = self.get_dexp(omega_exp, t)
        if self.determinant < 0:
            omega = np.sqrt(-self.determinant)
            A[0, 0], A[0, 1] = exp * self.get_cos(omega, t), exp * self.get_sin(
                omega, t
            )
            A[1, 0] = dexp * self.get_cos(omega, t) + exp * self.get_dcos(omega, t)
            A[1, 1] = dexp * self.get_sin(omega, t) + exp * self.get_dsin(omega, t)

        elif self.determinant > 0:
            omega = np.sqrt(self.determinant)
            A[0, 0], A[0, 1] = exp * self.get_exp(omega, t), exp * self.get_exp(
                -omega, t
            )
            A[1, 0] = dexp * self.get_exp(omega, t) + exp * self.get_dexp(omega, t)
            A[1, 1] = dexp * self.get_exp(-omega, t) + exp * self.get_dexp(-omega, t)

        else:
            A[0, 0], A[0, 1] = exp, t * exp
            A[1, 0], A[1, 1] = dexp, t * dexp + exp

        if omega_force is None:
            omega_force = 1.0

        if F0 is None:
            u0 = self.params.u0
        else:

            forceterm = self.get_forceTerm(t, F0=F0, omega_force=omega_force)

            u0 = self.params.u0 - forceterm
        b = np.linalg.solve(A, u0)

        return A, b

    def get_solutionWithoutForce(self, t, F0=None, omega_force=None):
        if self.params.mu == 0:
            *_, b = self.const_withoutFriction(self.params.t0, F0, omega_force)
            A, *_ = self.const_withoutFriction(t)
        elif self.params.mu > 0:

            *_, b = self.const_withFriction(self.params.t0, F0, omega_force)
            A, *_ = self.const_withFriction(t)
        else:
            raise ValueError("Solution without Force is not working")
        return A @ b

    def get_solution_ntimeWithoutForce(self, time):
        solution_store = np.zeros([2, len(time)])
        for tt in range(len(time)):
            solution_store[:, tt] = self.get_solutionWithoutForce(time[tt])
        return solution_store

    def get_constForce(self, t, F0, omega_force):

        c_omega = self.params.kappa - omega_force**2
        if self.params.mu != 0:
            const_omega = c_omega / (self.params.mu * omega_force)
            Ap = (F0 / (self.params.mu * omega_force)) * (1 / (1 + const_omega**2))
            Bp = const_omega * Ap
        elif self.params.mu == 0 and c_omega != 0:
            Ap = 0.0
            Bp = F0 / c_omega
        elif self.params.mu == 0 and c_omega == 0:
            Ap = 0.5 * F0 * t / omega_force
            Bp = 0.0
        else:
            raise (ValueError("Check get_constForce"))

        return Ap, Bp, c_omega

    def get_forceTerm(self, t, F0, omega_force):

        Ap, Bp, c_omega = self.get_constForce(t, F0, omega_force)
        position_force = Ap * self.get_sin(omega_force, t) + Bp * self.get_cos(
            omega_force, t
        )
        if c_omega == 0:
            velocity_force = Ap * self.get_dsin(omega_force, t) + (
                0.5 * F0 / (omega_force)
            ) * self.get_sin(omega_force, t)
        else:
            velocity_force = Ap * self.get_dsin(omega_force, t) + Bp * self.get_dcos(
                omega_force, t
            )

        return np.array([position_force, velocity_force])

    def get_solutionWithForce(self, t, F0, omega_force=None):
        if omega_force is None:
            omega_force = 1.0
        solutionWithoutForce = self.get_solutionWithoutForce(t, F0, omega_force)

        forceTerm = self.get_forceTerm(t, F0, omega_force)
        solutionWithForce = solutionWithoutForce + forceTerm
        return solutionWithForce

    def get_solution_ntimeWithForce(self, time):
        F0 = self.params.F0
        solutionStore = np.zeros((2, len(time)))
        for tt in range(len(time)):
            solutionStore[:, tt] = self.get_solutionWithForce(time[tt], F0)
        return solutionStore
