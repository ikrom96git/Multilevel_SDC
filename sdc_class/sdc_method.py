import numpy as np
from problem_class.harmonicoscillator import HarmonicOscillator
from transfer_class.matrix_sdc import MatrixSDC
from problem_class.Pars import _Pars
from scipy.optimize import fsolve
from problem_class.reduced_HO import Reduced_HO
from copy import deepcopy

# Define SDC class
class SDC_method(Reduced_HO):
    def __init__(self, problem_params, collocation_params):
        self.collocation_params = _Pars(collocation_params)
        super().__init__(problem_params)
        self.dt = self.prob_params.dt
        self.coll = MatrixSDC(
            self.collocation_params.num_nodes, self.collocation_params.quad_type
        )
        self.residual = []

        self.name = "SDC"

    def get_residual(self, U0, U):
        # Get the residual of the problem

        F = np.block([[self.get_f()], [self.get_f()]])
        O = np.zeros([self.coll.num_nodes + 1, self.coll.num_nodes + 1])

        I = np.eye(self.coll.num_nodes + 1)
        b0 = np.block([[I, self.dt * self.coll.Qmat], [O, I]])
        QQ = self.dt**2 * (self.coll.QQ)
        Q = self.dt * (self.coll.Qmat)
        A = np.block([[QQ, O], [O, Q]])
        R = U - (A @ F @ U + b0 @ U0)
        Rabs = np.abs(R)
        return Rabs

    def get_initial_guess(self, type=None):
        if type is None:
            type = self.prob_params.initial_guess
        else:
            type = type
        x = self.prob_params.u0[0]
        v = self.prob_params.u0[1]

        if type == "spread":
            # Get the initial guess for the problem
            X = x * np.ones([self.coll.num_nodes + 1])
            V = v * np.ones([self.coll.num_nodes + 1])
            U = np.block([X, V])
        elif type == "zeros":
            X = np.zeros([self.coll.num_nodes + 1])
            V = np.zeros([self.coll.num_nodes + 1])
            X[0] = x
            V[0] = v
            U = np.block([X, V])
        elif type == "collocation":
            U = self.collocation_solution()
        elif type == "manual":
            U = self.get_initial_condition(X, V)
        else:
            raise ValueError("Unknown initial guess type")
        return U

    def get_initial_condition(self, U):
        # Get the initial condition for the problem
        return U

    def get_f(self):
        # Get the function f for the problem
        I = np.eye(self.coll.num_nodes + 1)
        O = np.zeros([self.coll.num_nodes + 1, self.coll.num_nodes + 1])
        nodes = np.append(0, self.coll.nodes)
        D = np.diag(nodes)
        Fx = self.build_f(I, O, D)
        Fv = self.build_f(O, I, D)
        F = np.block([Fx, Fv])
        return F

    def eval_f(self, U):
        F = np.block([[self.get_f()], [self.get_f()]])
        return F @ U
    
    def sdc_node_node(self, U=None, tau=[None, None]):
        if U is None:
            U = self.get_initial_guess()

        X, V = np.split(U, 2)
        Xnew=deepcopy(X)
        Vnew=deepcopy(V)
        intpos=[]
        intvel=[]
        for m in range(self.coll.num_nodes):
            Sq=0
            S=0
            for j in range(self.coll.num_nodes+1):
                Sq+=self.dt**2*(self.coll.SQ[m+1,j]-self.coll.Sx[m+1, j])*self.build_f(X[j], V[j], self.dt*self.coll.nodes[j-1])
                S+=self.dt*(self.coll.S[m+1,j]-self.coll.ST[m+1, j])*self.build_f(X[j], V[j], self.dt*self.coll.nodes[j-1])
            intpos.append(Sq)
            intvel.append(S)
    
        for m in range(self.coll.num_nodes):
            tmppos=intpos[m]     
            tmpvel=intvel[m]+Vnew[m]
            for j in range(m+1):
                tmppos+=self.dt**2*(self.coll.Sx[m+1,j]*self.build_f(Xnew[j], Vnew[j], self.dt*self.coll.nodes[j]))
            Xnew[m+1]=tmppos+Xnew[m]+self.dt*self.coll.delta_m[m]*V[0]
            func=lambda v: 0.5*self.dt*self.coll.delta_m[m]*self.build_f(Xnew[m+1], v, self.dt*self.coll.nodes[m])+0.5*self.dt*self.coll.delta_m[m]*self.build_f(Xnew[m], Vnew[m], self.dt*self.coll.nodes[m])
            Vnew[m+1]=fsolve(func, Vnew[m])
        Unew=np.block([Xnew, Vnew])
        return Unew




    def sdc_method(self, U0=None, U=None, tau=[None, None]):
        # Run the SDC method
        if U0 is None:
            U0 = self.get_initial_guess(type="spread")
        if U is None:
            U = self.get_initial_guess()

        X, V = np.split(U, 2)
        X0, V0 = np.split(U0, 2)
        T=np.append(0, self.coll.nodes)
        for m in range(self.coll.num_nodes):
            for j in range(self.coll.num_nodes + 1):
                
                X0[j] = X0[j] + self.dt * V0[j]
                V0[j] = V0[j] - self.dt * X0[j]
        F = np.block([[self.get_f()], [self.get_f()]])
        O = np.zeros([self.coll.num_nodes + 1, self.coll.num_nodes + 1])
        I = np.eye(self.coll.num_nodes + 1)
        QQ = self.dt**2 * (self.coll.QQ - self.coll.Qx)
        Q = self.dt * (self.coll.Qmat - self.coll.QT)
        bQ = np.block([[QQ, O], [O, Q]])
        b0 = np.block([[I, self.dt * self.coll.Qmat], [O, I]])
        b = bQ @ F @ U + b0 @ U0
        
        if None not in tau:
            b += tau

        AQ = np.block([[self.dt**2 * self.coll.Qx, O], [O, self.dt * self.coll.QT]])
        func = lambda Z: b + AQ @ F @ Z - Z
        U = fsolve(func, U)
        A = np.eye(2 * (self.coll.num_nodes + 1)) - AQ @ F
        UL = np.linalg.solve(A, b)
        self.residual.append(self.get_residual(U0, U))
        return U

    def run_sdc(self):
        # Run the SDC method
        U0 = self.get_initial_guess(type="spread")
        U = self.get_initial_guess()
        for i in range(self.prob_params.Kiter):
            U = self.sdc_method(U0, U)
            Usol=self.sdc_node_node(U)
            breakpoint()
        return U

    def get_Qmat(self):
        O = np.zeros([self.coll.num_nodes + 1, self.coll.num_nodes + 1])
        QQ = self.dt**2 * (self.coll.QQ)
        Q = self.dt * (self.coll.Qmat)
        A = np.block([[QQ, O], [O, Q]])
        return A

    def collocation_solution(self):
        # Get the collocation solution
        U0 = self.get_initial_guess(type="spread")
        F = np.block([[self.get_f()], [self.get_f()]])
        O = np.zeros([self.coll.num_nodes + 1, self.coll.num_nodes + 1])
        I = np.eye(self.coll.num_nodes + 1)
        b0 = np.block([[I, self.dt * self.coll.Qmat], [O, I]])
        QQ = self.dt**2 * (self.coll.QQ)
        Q = self.dt * (self.coll.Qmat)
        A = np.eye(2 * (self.coll.num_nodes + 1)) - np.block([[QQ, O], [O, Q]]) @ F
        U = np.linalg.solve(A, b0 @ U0)
        return U
