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

    def restirct(self):
        pass

    def prolongate(self):
        pass


class LevelMatrixSDC(object):
    def __init__(self, collocation_params):
        self.coll = _Pars(collocation_params)
        self.fine = MatrixSDC(
            num_nodes=self.coll.num_nodes[0], quad_type=self.coll.quda_type
        )
        self.coarse = MatrixSDC(
            num_nodes=self.coll.num_nodes[1], quad_type=self.coll.quda_type
        )
        self.trans = Transfer(self.fine.nodes, self.coarse.nodes)
