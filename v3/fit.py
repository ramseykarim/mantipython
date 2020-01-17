import numpy as np
from scipy.optimize import minimize
import sys

from .src import mpy_utils as mpu
from .src.greybody import Greybody
from .src.dust import Dust

__author__ = "Ramsey Karim"


# Standard guesses for T, N, beta
# Nh, Nc are log10
standard_x0 = {'T': 10, 'N': 22, 'beta': 2}
standard_bounds = {'T': (0, None), 'N': (18, 25), 'beta': (1, 2.5)}

def gen_goodness_of_fit_f(src_fn, dof):
    def goodness_of_fit_f(x, obs, err, instr):
        src = src_fn(x)
        return sum((d.detect(src) - o)**2 / (e*e) for d, o, e in zip(instr, obs, err)) / dof
    return goodness_of_fit_f


def fit_source(observations, errors, detectors, goodness_of_fit_f,
    x0=None, bounds=None, **min_kwargs):
    result = minimize(goodness_of_fit_f,
        x0=x0,
        args=(observations, errors, detectors),
        bounds=bounds,
        options={'maxiter': 50},
        **min_kwargs,
    )
    return result.x


def fit_full_image(observation_maps, error_maps, detectors, src_fn,
    initial_guess, bounds, chisq=False):
    # maps should be sequences of numpy.ndarrays
    n_pixels = observation_maps[0].size
    img_shape = observation_maps[0].shape
    n_params = len(initial_guess)
    n_data = len(observation_maps)
    dof = n_data - n_params
    obs_seq = zip(*(o.flat for o in observation_maps))
    err_seq = zip(*(e.flat for e in error_maps))
    result_seq = np.full((n_params, n_pixels), np.nan)
    goodness_of_fit_f = gen_goodness_of_fit_f(src_fn, dof)
    if chisq:
        chisq_seq = np.full((n_pixels,), np.nan)
    for i, obs, err in zip(range(n_pixels), obs_seq, err_seq):
        if not np.any(np.isnan(obs)):
            result_seq[:, i] = fit_source(obs, err, detectors, goodness_of_fit_f,
                x0=initial_guess, bounds=bounds)
            if chisq:
                chisq_seq[i] = goodness_of_fit_f(result_seq[:, i], obs, err, detectors)
    if chisq:
        return result_seq.reshape((n_params, *img_shape)), chisq_seq.reshape(img_shape)
    else:
        return result_seq.reshape((n_params, *img_shape))
