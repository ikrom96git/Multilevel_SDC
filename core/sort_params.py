from sweeper_class.sdc_class import sdc_class
from core.Lagrange import LagrangeApproximation
from copy import deepcopy
import numpy as np


class SortParams(object):
    def __init__(
        self,
        problem_params,
        collocation_params,
        sweeper_params,
        problem_class,
        restriction_class,
        eps,
    ):
        self.eps = eps
        self.get_sorted_params(
            problem_params, collocation_params, sweeper_params, problem_class
        )
        self.Pcoll = self.get_transfer_matrix_Q(
            self.sdc_fine_model.coll.nodes, self.sdc_coarse_model.coll.nodes
        )

        self.Rcoll = self.get_transfer_matrix_Q(
            self.sdc_coarse_model.coll.nodes, self.sdc_fine_model.coll.nodes
        )
        self.transfer_operator = restriction_class(self.restriction_node)

    def get_sorted_params(
        self, problem_params, collocation_params, sweeper_params, problem_class
    ):

        if len(problem_params) == 2:
            problem_params_fine = problem_params[0]
            problem_params_coarse = problem_params[1]
            problem_params_first = problem_params[1]
        elif len(problem_params) == 3:
            problem_params_fine = problem_params[0]
            problem_params_coarse = problem_params[1]
            problem_params_first = problem_params[2]
        else:
            problem_params_fine = problem_params
            problem_params_coarse = problem_params
        problem_class_fine = problem_class[0]
        collocation_params_fine = deepcopy(collocation_params)
        collocation_params_fine["num_nodes"] = collocation_params["num_nodes"][0]
        collocation_params_coarse = deepcopy(collocation_params)
        collocation_params_coarse["num_nodes"] = collocation_params["num_nodes"][1]
        if len(problem_class) == 1:
            problem_class_coarse = problem_class[0]
        elif len(problem_class) == 2:
            problem_class_coarse = problem_class[1]
        else:
            problem_class_coarse = problem_class[1]
            problem_class_coarse_first = problem_class[2]
            self.sdc_coarse_first_model = sdc_class(
                problem_params_first,
                collocation_params_coarse,
                sweeper_params,
                problem_class_coarse_first,
            )

        self.sdc_fine_model = sdc_class(
            problem_params_fine,
            collocation_params_fine,
            sweeper_params,
            problem_class_fine,
        )
        self.sdc_coarse_model = sdc_class(
            problem_params_coarse,
            collocation_params_coarse,
            sweeper_params,
            problem_class_coarse,
        )

    @staticmethod
    def get_transfer_matrix_Q(f_nodes, c_nodes):
        approx = LagrangeApproximation(c_nodes)
        return approx.getInterpolationMatrix(f_nodes)

    def restriction_node(self, U):
        return self.Rcoll @ U

    def interpolation_node(self, U):
        return np.append(U[0], self.Pcoll @ U[1:])
