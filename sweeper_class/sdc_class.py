import numpy as np
from core.Pars import _Pars
from transfer_class.CollocationMatrix import CollocationMatrix
from problem_class.HarmonicOscillator import HarmonicOscillator
from copy import deepcopy


class sdc_class(object):
    def __init__(self, problem_params, collocation_params, sweeper_params):
        self.prob = _Pars(problem_params)
        self.sweeper = _Pars(sweeper_params)
        self.coll = CollocationMatrix(collocation_params)
        self.build_f = HarmonicOscillator(problem_params)

    def sdc_sweep(self, U0, U_old=None):
        if U_old is None:
            U_old = deepcopy(U0)
        X0, V0 = U0[0], U0[1]
        X_old, V_old = U_old[0], U_old[1]
        X = deepcopy(X0)
        V = deepcopy(V0)
        for kk in range(self.sweeper.Kiter):
            QX, QV = self.sdc_sweep_integrate(X0, V0, X_old, V_old)
            for mm in range(self.coll.num_nodes+1):
                X[mm+1] = QX[mm+1]+self.prob.dt**2 * \
                    (self.coll.Qx @ self.build_f.get_rhs(X, V))[mm+1]

                def V_func(v): return QV[mm+1]+self.prob.dt*

    def sdc_sweep_integrate(self, X0, V0, X_old, V_old):
        QX = X0+self.prob.dt*self.coll.Q@V0+self.prob.dt**2 * \
            (self.coll.QQ-self.coll.Qx)@self.build_f.get_rhs(X_old, V_old)
        QV = V0+self.prob.dt * \
            (self.coll.Q-self.coll.QT)@self.build_f.get_rhs(X_old, V_old)
        return QX, QV
