import numpy as np
from sdc_class.sdc_method import SDC_method
from plot_class.plot_residual import run_plot
from sdc_class.mlsdc_method import Mlsdc
from sdc_class.sdc_method_fast_time import SDC_method_fast_time


def asymptotic_params():
    prob_params = dict()
    eps = 0.2
    prob_params["kappa"] = 0.2
    prob_params["eps"] = eps
    prob_params["c"] = 1 / eps
    prob_params["f0"] = 1 / eps
    prob_params["u0"] = np.array([2, 0])
    prob_params["dt"] = 0.1
    prob_params["initial_guess"] = "spread"
    prob_params["Kiter"] = 10
    collocation_params = dict()
    collocation_params["num_nodes"] = 4
    collocation_params["quad_type"] = "LOBATTO"
    collocation_params_mlsdc = dict()
    collocation_params_mlsdc["quad_type"] = "GAUSS"
    collocation_params_mlsdc["num_nodes"] = [5, 5]
    return prob_params, collocation_params, collocation_params_mlsdc


def prob_params():
    prob_params = dict()
    prob_params["k"] = 3.0
    prob_params["c"] = 1.0
    prob_params["dt"] = 0.1
    prob_params["oscillator_type"] = "free"
    prob_params["initial_guess"] = "spread"
    prob_params["u0"] = np.array([2, 1])
    prob_params["eps"] = 1.0
    prob_params["Kiter"] = 5
    prob_params["f0"] = 1.0
    prob_params["omega"] = 1.0
    collocation_params = dict()
    collocation_params["num_nodes"] = 6
    collocation_params["quad_type"] = "GAUSS"
    collocation_params_mlsdc = dict()
    collocation_params_mlsdc["quad_type"] = "GAUSS"
    collocation_params_mlsdc["num_nodes"] = [5, 5]
    return prob_params, collocation_params, collocation_params_mlsdc


if __name__ == "__main__":
    prob_params, collocation_params, collocation_params_mlsdc = asymptotic_params()
    model = SDC_method(prob_params, collocation_params)
    model_mlsdc = Mlsdc(prob_params, collocation_params_mlsdc)
    model.run_sdc()

    model_mlsdc.run_mlsdc()
    model_mlsdc.run_mlsdc_fast()
    run_plot(
        [model.residual, model_mlsdc.sdc_f.residual, model_mlsdc.sdc_f.residual],
        [model.prob_params.Kiter],
        [model.name, "mlsdc", "mlsdc_fast"],
    )
    # run_plot(model_mlsdc.sdc_f.residual, model_mlsdc.sdc_f.prob_params.Kiter, model_mlsdc.sdc_f.name)
