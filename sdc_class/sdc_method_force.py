import numpy as np
from transfer_class.matrix_sdc import MatrixSDC
from problem_class.Pars import _Pars
from scipy.optimize import fsolve
from problem_class.reduced_HO import Reduced_HO
from copy import deepcopy
from problem_class.asymptotic_problem import Fast_time

class SDC_method_force(Reduced_HO):
    def __init__(self, problem_params, collocation_params):
        self.collocation_params = _Pars(collocation_params)
        super().__init__(problem_params)
        self.dt = self.prob_params.dt
        self.coll = MatrixSDC(
            self.collocation_params.num_nodes, self.collocation_params.quad_type
        )
        self.residual = []
        self.name = "SDC_method_force"

    def get_residual(self, U0, U):
        X, V =np.split(U, 2)
        X0, V0 =np.split(U0, 2)
        T=np.append(0, self.dt*self.coll.nodes)
        F=self.build_f(X, V, 0)+self.force(T)
        Rx=X0+self.dt*self.coll.Qmat@V0+self.dt**2*self.coll.QQ@F-X
        Rv=V0+self.dt*self.coll.Qmat@F-V
        R=np.block([Rx, Rv])
        return np.abs(R)

        

    def get_initial_guess(self, type=None):
        if type is None:
            type = self.prob_params.initial_guess
        else:
            type = type
        x = self.prob_params.u0[0]
        v = self.prob_params.u0[1]
        if type == "spread":
            X = np.ones(self.coll.num_nodes+1) * x
            V = np.ones(self.coll.num_nodes+1) * v
            U = np.block([X, V])
        elif type == "zeros":
            X = np.zeros(self.coll.num_nodes+1)
            V = np.zeros(self.coll.num_nodes+1)
            X[0] = x
            V[0] = v
            U = np.block([X, V])
        
        elif type == "collocation":
            U = self.collocation_solution()
        elif type == "manual":
            U = self.get_initial_guess()
        else:
            raise ValueError("Initial guess type not recognized")
        return U

    def get_f(self):
        I = np.eye(self.coll.num_nodes + 1)
        O = np.zeros((self.coll.num_nodes + 1, self.coll.num_nodes + 1))
        Fx = self.build_f(I, O, 0)
        Fv = self.build_f(O, I, 0)
        F = np.block([Fx, Fv])
        return F

    def sdc_method(self, U0=None, U=None, tau=[None, None]):
        if U0 is None:
            U0 = self.get_initial_guess(type="spread")
        if U is None:
            U = self.get_initial_guess()

        T=np.append(0, self.dt*self.coll.nodes)
        F=np.block([[self.get_f()], [self.get_f()]])
        I = np.eye(self.coll.num_nodes + 1)
        O = np.zeros((self.coll.num_nodes + 1, self.coll.num_nodes + 1))
        QQ=self.dt**2*(self.coll.QQ)
        Q=self.dt*(self.coll.Qmat)
        QT=self.dt*self.coll.QT
        Qx=self.dt**2*(self.coll.Qx)
        AQ=np.block([[Qx, O],[O, QT]])
        B=np.block([[QQ, O], [O, Q]])
        b0=np.block([[I, Q], [O, I]])
        force=np.block([self.force(T), self.force(T)])
        b = (B-AQ) @ F @ U + b0 @ U0 + B @ force
        func=lambda Z: b+AQ@F@Z-Z
        U=fsolve(func, U)
        self.residual.append(self.get_residual(U0, U))
        return U
    def run_sdc(self):
        U0=deepcopy(self.get_initial_guess())
        U=deepcopy(self.get_initial_guess())
        for i in range(self.prob_params.Kiter):
            U=self.sdc_method(U0=U0, U=U)
        return U
    def collocation_solution(self):
        U0=self.get_initial_guess(type="spread")
        F=np.block([[self.get_f()], [self.get_f()]])
        I = np.eye(self.coll.num_nodes + 1)
        O = np.zeros((self.coll.num_nodes + 1, self.coll.num_nodes + 1))
        T=np.append(0, self.dt*self.coll.nodes)
        QQ=self.dt**2*(self.coll.QQ)
        Q=self.dt*(self.coll.Qmat)
        B=np.block([[QQ, O], [O, Q]])
        b0=np.block([[I, Q], [O, I]])
        force=np.block([self.force(T), self.force(T)])
        A=np.eye(2*(self.coll.num_nodes+1))-B@F
        U=np.linalg.solve(A, b0@U0+B@force)
        breakpoint()
        return U
        







