import numpy as np
from core.Pars import _Pars
from transfer_class.CollocationMatrix import CollocationMatrix
from problem_class.HarmonicOscillator import HarmonicOscillator
from copy import deepcopy
from scipy.optimize import fsolve


class sdc_class(object):
    def __init__(self, problem_params, collocation_params, sweeper_params):
        self.prob = _Pars(problem_params)
        self.sweeper = _Pars(sweeper_params)
        self.coll = CollocationMatrix(collocation_params)
        problem_class = HarmonicOscillator(problem_params)
        self.build_f=problem_class.build_f

    def sdc_sweep(self, X, V):
        M = self.coll.num_nodes
        T = self.prob.dt*np.append(self.prob.t0, self.coll.nodes)
        X_old = deepcopy(X)
        V_old = deepcopy(V)
        SQF = self.prob.dt**2*self.coll.SQ@self.build_f(X_old, V_old, T)
        SF = self.prob.dt*self.coll.S@self.build_f(X_old, V_old, T)
        F_old = self.build_f(X_old, V_old, T)
        F_new = self.build_f(X, V, T)
        for m in range(M):
            SXF = self.prob.dt**2*self.coll.Sx@(self.build_f(X, V, T) -
                                self.build_f(X_old, V_old, T))
            X[m+1] = X[m]+self.prob.dt*self.coll.delta_m[m]*V_old[0] + SXF[m+1] + SQF[m+1]
            rhs = V[m]-self.prob.dt*self.coll.delta_m[m]*0.5 * \
                (F_old[m+1]+F_old[m])+0.5 * \
                self.prob.dt*self.coll.delta_m[m]*F_new[m]+SF[m+1]

            def func(v): return rhs+0.5 * \
                self.prob.dt*self.coll.delta_m[m]*self.build_f(X[m+1], v, T[m+1])-v
            V[m+1] = fsolve(func, V[0])
        return X, V

    def sdc_iter(self, K):
        X = self.prob.u0[0]*np.ones(self.coll.num_nodes+1)
        V = self.prob.u0[1]*np.ones(self.coll.num_nodes+1)
        for ii in range(K):
            X, V = self.sdc_sweep(X, V)
        return X, V

if __name__=='__main__':
    pass
