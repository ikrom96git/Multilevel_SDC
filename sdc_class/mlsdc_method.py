import numpy as np
from transfer_class.transfer_operators import LevelMatrixSDC
from sdc_class.sdc_method import SDC_method
from copy import deepcopy
from sdc_class.sdc_method_fast_time import SDC_method_fast_time


class Mlsdc(LevelMatrixSDC):
    # constructor for the ml-sdc method
    def __init__(self, problem_params, collocation_params):
        LevelMatrixSDC.__init__(self, collocation_params)
        coll_f = deepcopy(collocation_params)
        coll_f["num_nodes"] = collocation_params["num_nodes"][0]
        coll_c = deepcopy(collocation_params)
        coll_c["num_nodes"] = collocation_params["num_nodes"][1]
        self.sdc_f = SDC_method(problem_params, coll_f)
        self.sdc_c = SDC_method(problem_params, coll_c)
        self.sdc_f_fast = SDC_method_fast_time(problem_params, coll_f)
        self.sdc_c_fast = SDC_method_fast_time(problem_params, coll_c)

    # ml-sdc method for the problem
    def mlsdc_method(self, Uf=None):
        if Uf is None:
            Uf = self.sdc_f.get_initial_guess()
        Uc, tau = self.tau_correction(Uf)
        Uc_new = self.sdc_c.sdc_method(U0=None, U=Uc, tau=tau)
        Uf_new, tau = self.interpolate(Uc, Uc_new)
        U = Uf_new + Uf
        U_new = self.sdc_f.sdc_method(U0=None, U=U, tau=tau)
        return U_new

    def mlsdc_method_fast(self, Uf=None):
        if Uf is None:
            Uf = self.sdc_f.get_initial_guess()
        Uc, tau = self.tau_correction(Uf)
        Uc_new = self.sdc_c_fast.sdc_method(U0=None, U=Uc, tau=tau)
        Uf_new, tau = self.interpolate(Uc, Uc_new)
        U = Uf_new + Uf
        U_new = self.sdc_f.sdc_method(U0=None, U=U, tau=tau)
        return U_new

    # tau correction from fine to coarse
    def tau_correction(self, U_f):
        F_f = self.sdc_f.eval_f(U_f)
        Q_f = self.sdc_f.get_Qmat()
        restrict_Q = self.trans.restrict(Q_f @ F_f)

        U_c = self.trans.restrict(U_f)
        F_c = self.sdc_c.eval_f(U_c)
        Q_c = self.sdc_c.get_Qmat()

        tau = restrict_Q - Q_c @ F_c
        return U_c, tau

    # interpolation from coarse to fine
    def interpolate(self, U_c, U_new):
        diffU = U_new - U_c
        tau = np.empty(shape=self.sdc_f.coll.num_nodes, dtype=object)
        return self.trans.prolongate(diffU), tau

    # Ml-sdc method iteration
    def run_mlsdc(self, Kiter=None, U=None):
        if Kiter is None:
            Kiter = self.sdc_f.prob_params.Kiter
        if U is None:
            U = self.sdc_f.get_initial_guess()
        for ii in range(Kiter):
            U = self.mlsdc_method(U)

        return U

    def run_mlsdc_fast(self, Kiter=None, U=None):
        if Kiter is None:
            Kiter = self.sdc_f.prob_params.Kiter
        if U is None:
            U = self.sdc_f.get_initial_guess()
        for ii in range(Kiter):
            U = self.mlsdc_method_fast(U)

        return U
