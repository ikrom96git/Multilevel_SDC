import numpy as np
import matplotlib.pyplot as plt

from default_params.harmonic_oscillator_default_fast_time_params import (
    get_harmonic_oscillator_params,
    get_harmonic_oscillator_zeros_order_params,
    get_harmonic_oscillator_first_order_params,
)
from problem_class.HarmonicOscillator import HarmonicOscillator
from problem_class.HarmonicOscillator_fast_time_reduced_problem import (
    HarmonicOscillator_fast_time,
    HarmonicOscillator_fast_time_first_order,
)
from default_params.mlsdc_defautl_params import get_mlsdc_default_params
from sweeper_class.sdc_class import sdc_class
from sweeper_class.mlsdc_class import Mlsdc_class
from plot_class.plot_residual import plot_residual
from transfer_class.restriction import Restriction
from transfer_class.asymptotic_restriction_class import AsymptoticRestriction
from transfer_class.standart_restriction_class import StandartRestriction

EPSILON = 0.1
problem_params, collocation_params, sweeper_params, *_ = get_mlsdc_default_params()


def sdc_iteration():
    params, *_ = get_harmonic_oscillator_params(EPSILON)
    collocation_params["num_nodes"] = 5
    model = sdc_class(params, collocation_params, sweeper_params, HarmonicOscillator)
    model.sdc_iter()

    return (model.get_residual,)


def mlsdc_iteration():
    params, *_ = get_harmonic_oscillator_params(EPSILON)

    model = Mlsdc_class(
        params,
        collocation_params,
        sweeper_params,
        [HarmonicOscillator, HarmonicOscillator],
        Restriction,
        eps=EPSILON,
    )
    model.get_mlsdc_iter_solution()
    return model.sdc_fine_model.get_residual


def m3lsdc_zeros():
    params, *_ = get_harmonic_oscillator_params(EPSILON)
    params_zeros, *_ = get_harmonic_oscillator_zeros_order_params(EPSILON)
    params_frist, *_ = get_harmonic_oscillator_first_order_params(EPSILON)
    model = Mlsdc_class(
        [params, params_zeros],
        collocation_params,
        sweeper_params,
        [HarmonicOscillator, HarmonicOscillator_fast_time, HarmonicOscillator_fast_time_first_order,],
        Restriction,
        eps=EPSILON,
    )
    model.get_ml3sdc_iter_exact()
    return model.sdc_fine_model.get_residual


def m3lsdc_first():
    params, *_ = get_harmonic_oscillator_params(EPSILON)
    params_zeros, *_ = get_harmonic_oscillator_zeros_order_params(EPSILON)
    params_frist, *_ = get_harmonic_oscillator_first_order_params(EPSILON)
    Params = [params, params_zeros, params_frist]
    Problems = [
        HarmonicOscillator,
        HarmonicOscillator_fast_time,
        HarmonicOscillator_fast_time_first_order,
    ]
    model = Mlsdc_class(
        Params,
        collocation_params,
        sweeper_params,
        Problems,
        StandartRestriction,
        eps=EPSILON,
    )
    model.get_mlsdc_iter_solution()
    return model.sdc_fine_model.get_residual


if __name__ == "__main__":
    mlsdc_residual = mlsdc_iteration()
    m3lsdc_zeros_model = m3lsdc_zeros()
    # m3lsdc_first_model=m3lsdc_first()
    sdc_residual = sdc_iteration()

    iter = np.arange(0, 8, 1)
    title = "Residual"
    label = ["SDC", "MLSDC", "M3LSDC zeros", "M3LSDC first"]
    Residual = [
        np.array(sdc_residual[0])[:, 0],
        np.array(mlsdc_residual)[:, 0],
        np.array(m3lsdc_zeros_model)[:, 0],
        # np.array(m3lsdc_first_model)[:, 0],
    ]
    # breakpoint()
    plot_residual(iter, Residual, title, label, "somethind.png")
