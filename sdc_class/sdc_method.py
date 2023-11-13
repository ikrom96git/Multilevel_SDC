import numpy as np
from problem_class.harmonicoscillator import HarmonicOscillator
from transfer_class.matrix_sdc import MatrixSDC
from problem_class.Pars import _Pars


# Define SDC class
class SDC_method(HarmonicOscillator):
    def __init__(self, problem_params, collocation_params):
        self.collocation_params = _Pars(collocation_params)
        super().__init__(problem_params)
        self.dt = self.prob_params.dt
        self.coll = MatrixSDC(
            self.collocation_params.num_nodes, self.collocation_params.quad_type
        )

        self.residual=dict()
        self.name = "SDC"

    def get_residual(self, U0, U):
        # Get the residual of the problem

        F = np.block([[self.get_f()], [self.get_f()]])
        O = np.zeros([self.coll.num_nodes + 1, self.coll.num_nodes + 1])
        QQ = self.dt**2 * (self.coll.QQ)
        Q = self.dt * (self.coll.Qmat)
        A = np.block([[QQ, O], [O, Q]])
        R = U - A @ F @ U + U0
        Rabs = np.abs(R)
        return Rabs

    def get_initial_guess(self, type="spread"):
        x = self.prob_params.u0[0]
        v = self.prob_params.u0[1]

        if type == "spread":
            # Get the initial guess for the problem
            X = x * np.ones([self.coll.num_nodes + 1])
            V = v * np.ones([self.coll.num_nodes + 1])
        elif type == "zeros":
            X = np.zeros([self.coll.num_nodes + 1])
            V = np.zeros([self.coll.num_nodes + 1])
            X[0] = x
            V[0] = v
        elif type == "manual":
            X, V = self.get_initial_condition()
        else:
            raise ValueError("Unknown initial guess type")
        return np.block([X, V])

    def get_initial_condition(self, X, V):
        # Get the initial condition for the problem
        return X, V

    def get_f(self):
        # Get the function f for the problem
        I = np.eye(self.coll.num_nodes + 1)
        O = np.zeros([self.coll.num_nodes + 1, self.coll.num_nodes + 1])
        Fx = self.build_f(I, O, 0)
        Fv = self.build_f(O, I, 0)
        F = np.block([Fx, Fv])
        return F

    def sdc_method(self, U0, U):
        # Run the SDC method
        F = np.block([[self.get_f()], [self.get_f()]])
        O = np.zeros([self.coll.num_nodes + 1, self.coll.num_nodes + 1])
        I = np.eye(self.coll.num_nodes + 1)
        QQ = self.dt**2 * (self.coll.QQ - self.coll.Qx)
        Q = self.dt * (self.coll.Qmat - self.coll.QT)
        bQ = np.block([[QQ, O], [O, Q]])

        b0 = np.block([[I, self.dt * self.coll.Qmat], [O, I]])
        b = bQ @ F @ U + b0 @ U0
        AQ = np.block([[self.dt**2 * self.coll.Qx, O], [O, self.dt * self.coll.QT]])
        A = np.eye(2 * (self.coll.num_nodes + 1)) - AQ @ F
        U = np.linalg.solve(A, b)
        return U

    def run_sdc(self):
        # Run the SDC method
        U0 = self.get_initial_guess()
        U = self.get_initial_guess()
        for i in range(self.prob_params.Kiter):
            U = self.sdc_method(U0, U)
            self.residual[i] = self.get_residual(U0, U)
        return U

    def collocation_solution(self):
        # Get the collocation solution
        U0 = self.get_initial_guess()

        F = np.block([[self.get_f()], [self.get_f()]])
        O = np.zeros([self.coll.num_nodes + 1, self.coll.num_nodes + 1])
        QQ = self.dt**2 * (self.coll.QQ)
        Q = self.dt * (self.coll.Qmat)
        A = np.eye(2 * (self.coll.num_nodes + 1)) - np.block([[QQ, O], [O, Q]]) @ F
        U = np.linalg.solve(A, U0)
        return U
