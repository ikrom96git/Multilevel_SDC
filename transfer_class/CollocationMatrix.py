import numpy as np
from core.Collocation import CollBase
from core.Pars import _Pars


class CollocationMatrix(CollBase):
    def __init__(self, collocation_params):
        self.collocation_params = _Pars(collocation_params)

        super().__init__(
            num_nodes=self.collocation_params.num_nodes,
            quad_type=self.collocation_params.quad_type,
        )
        [
            self.S,
            self.ST,
            self.SQ,
            self.Sx,
            self.QQ,
            self.QI,
            self.QT,
            self.Qx,
            self.Q,
        ] = self.__get_Qmatrix()

    def __get_Qmatrix(self):
        QI = self.get_Qdelta_implicit("IE")
        QE = self.get_Qdelta_implicit("EE")

        QT = 1 / 2 * (QI + QE)
        Qx = QE @ QT + 1 / 2 * QE * QE
        Sx = np.zeros(self.Qmat.shape)
        ST = np.zeros(self.Qmat.shape)
        S = np.zeros(self.Qmat.shape)
        Sx[0, :] = Qx[0, :]
        ST[0, :] = QT[0, :]
        S[0, :] = self.Qmat[0, :]
        for m in range(self.num_nodes):
            Sx[m + 1, :] = Qx[m + 1, :] - Qx[m, :]
            ST[m + 1, :] = QT[m + 1, :] - QT[m, :]
            S[m + 1, :] = self.Qmat[m + 1, :] - self.Qmat[m, :]
        SQ = S @ self.Qmat
        QQ = self.Qmat @ self.Qmat
        return [S, ST, SQ, Sx, QQ, QI, QT, Qx, self.Qmat]

    def get_Qdelta_implicit(self, qd_type):
        QDmat = np.zeros(self.Qmat.shape)
        if qd_type == "IE":
            for m in range(self.num_nodes + 1):
                QDmat[m, 1 : m + 1] = self.delta_m[0:m]
        elif qd_type == "EE":
            for m in range(self.num_nodes + 1):
                QDmat[m, 0:m] = self.delta_m[0:m]
        return QDmat


if __name__ == "__main__":
    collocation_params = dict()
    collocation_params["num_nodes"] = 5
    collocation_params["quad_type"] = "GAUSS"
    aa = CollocationMatrix(collocation_params)
