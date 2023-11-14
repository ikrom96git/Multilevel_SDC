import numpy as np
from problem_class.harmonicoscillator import HarmonicOscillator
from sdc_class.sdc_method import SDC_method
from plot_class.plot_residual import run_plot

if __name__ == "__main__":
    prob_params = dict()
    prob_params["k"] = 3.0
    prob_params["c"] = 1.0
    prob_params["dt"] = 0.1
    prob_params["oscillator_type"] = "free"
    prob_params["initial_guess"] = "collocation"
    prob_params["u0"] = np.array([2, 1])
    prob_params["Kiter"] = 5
    collocation_params = dict()
    collocation_params["num_nodes"] = 3
    collocation_params["quad_type"] = "GAUSS"

    model = SDC_method(prob_params, collocation_params)
    model.run_sdc()

    run_plot(model)
