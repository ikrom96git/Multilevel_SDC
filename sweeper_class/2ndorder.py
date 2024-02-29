import sys
sys.path.append("../../old")
from functions import fcs as fcs
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp



# van der Pol oscillator
def vanderPol(x, mu=1):
    x = np.asarray(x)
    y_dot = x[1]
    x_dot = mu*(1-x[0]**2)*x[1] - x[0]
    return np.asarray([y_dot, x_dot])


# general RK solver... is very slow because will always use fsolve
class ReferenceSolver:
    """docstrin for ReferenceSolver"""
    def __init__(self, f, y0, dt=0.02, a=0, b=40, M=5):
        n, w = fcs.nodes_weights(M, 0, 1)
        Q = fcs.Qmatrix(n, a)
        self.weights = w
        self.Q = Q
        self.M = M
        self.f = f
        self.y = np.asarray(y0)
        self.dt = dt
        self.dim = len(y0)
        self.stages = np.zeros([self.M, self.dim])
        self.shape = self.stages.shape
        self.stages = self.stages.ravel()
        self.ind = np.asarray([k for k in range(self.dim)])
        self.steps = int(b/dt)
        self.storage = []
        self.storage.append(y0)
        self.t = np.arange(a, b+dt/10, dt)
        print(f'Running from t={a} to t={b} with dt={dt} and a total amount of {self.steps} steps')
        # self.stages.reshape(self.shape)


    def rk(self, x):
        # sets up stages based in self.Q is then used in fully implicit solve by fsolve
        from itertools import chain
        ret = []
        for i in range(self.M):
            temp = np.zeros(self.dim)
            for j in range(self.M):
                temp += -self.dt*self.Q[i,j]*self.f(x[self.ind + self.dim*j])
            ret = list(chain(ret, list(x[self.ind + self.dim*i] - self.y + temp)))
        return ret

    def nystrom(self, x):
        # sets up stages based in self.Q is then used in fully implicit solve by fsolve
        from itertools import chain
        ret = []
        QQ = self.Q@self.Q
        for i in range(self.M):
            temp1 = np.zeros(int(self.dim/2))
            temp2 = np.zeros(int(self.dim/2))
            for j in range(self.M):
                temp1 += -self.dt*self.Q[i,j]*self.y[1]  -self.dt*QQ[i,j]*self.f(x[self.ind + self.dim*j])[0]
                temp2 += -self.dt*self.Q[i,j]*self.f(x[self.ind + self.dim*j])[1]
            temp = np.concatenate([temp1, temp2])
            ret = list(chain(ret, list(x[self.ind + self.dim*i] - self.y + temp)))
        return ret

    def solve_stages(self):
        from scipy.optimize import fsolve
        self.stages = fsolve(self.nystrom, self.stages)

    def quadrature(self):
        stages = self.stages.reshape(self.shape)
        temp = np.sum(np.asarray([self.weights[i]*self.f(stages[i]) for i in range(self.M)]), axis=0)
        self.y = np.asarray(self.y + self.dt*temp)
        stages[:] = self.y
        self.stages = stages.ravel()

    def run(self):
        from itertools import chain
        for i in range(self.steps):
            self.solve_stages()
            self.quadrature()
            self.storage.append(self.y)
        self.storage = np.asarray(self.storage)


f = vanderPol
init = np.asarray([2, 0])
ref = ReferenceSolver(f, init)
s = ref.run()
