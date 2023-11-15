import numpy as np
from core.Lagrange import LagrangeApproximation
from transfer_class.matrix_sdc import MatrixSDC
from problem_class.Pars import _Pars


class Transfer(object):
    def __init__(self, fine_nodes, coarse_nodes):
        self.Pcoll = self.get_transfer_matrix(fine_nodes, coarse_nodes)
        self.Rcoll = self.get_transfer_matrix(coarse_nodes, fine_nodes)

    def get_transfer_matrix(self, fine_nodes, coarse_nodes):
        approx = LagrangeApproximation(coarse_nodes)
        return approx.getInterpolationMatrix(fine_nodes)

    def restrict(self, U):
        X, V = np.split(U, 2)

        X_new = self.Rcoll @ X[1:]
        V_new = self.Rcoll @ V[1:]
        X_new = np.append(X[0], X_new)
        V_new = np.append(V[0], V_new)
        return np.concatenate((X_new, V_new))

    def prolongate(self, U):
        X, V = np.split(U, 2)
        X_new = self.Pcoll @ X[1:]
        V_new = self.Pcoll @ V[1:]
        X_new = np.append(X[0], X_new)
        V_new = np.append(V[0], V_new)

        return np.concatenate((X_new, V_new))


class LevelMatrixSDC(object):
    def __init__(self, collocation_params):
        self.coll = _Pars(collocation_params)
        self.fine = MatrixSDC(
            num_nodes=self.coll.num_nodes[0], quad_type=self.coll.quad_type
        )
        self.coarse = MatrixSDC(
            num_nodes=self.coll.num_nodes[1], quad_type=self.coll.quad_type
        )
        self.trans = Transfer(self.fine.nodes, self.coarse.nodes)
