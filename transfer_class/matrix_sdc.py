import numpy as np
from core.Collocation import CollBase


class MatrixSDC(CollBase):
    def __init__(self, num_nodes, quad_type):
        super().__init__(num_nodes=num_nodes, quad_type=quad_type)
        self._num_nodes = num_nodes
        self._quad_type = quad_type
        self.QQ, self.Qx, self.QT = self._compute_collocation_matrix()

    def _compute_collocation_matrix(self):
        # Compute collocation matrix
        Q = self.Qmat
        QE = np.zeros(self.Qmat.shape)
        for m in range(self.num_nodes + 1):
            QE[m, 0:m] = self.delta_m[0:m]
        QI = np.zeros(self.Qmat.shape)
        for m in range(self.num_nodes + 1):
            QI[m, 1 : m + 1] = self.delta_m[0:m]

        # Trapzoidal rule
        QT = 0.5 * (QI + QE)

        # Qx as in the paper
        Qx = np.dot(QE, QT) + 0.5 * QE * QE

        # QQ matrix
        QQ = np.dot(Q, Q)
        return QQ, Qx, QT
