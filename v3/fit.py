import numpy as np
from scipy.optimize import minimize
import sys

import .mpy_utils as mpu
from .greybody import Greybody
from .dust import Dust

__author__ = "Ramsey Karim"


# Standard guesses for T, N, beta
# Nh, Nc are log10
standard_x0 = [10, 22, 2]

def goodness_of_fit_f(x, obs, err, instr, dof):
    # x is [T, N, beta] (N in log10)
    T, N, beta = x
    src = Greybody(T, N, Dust(beta=beta))
    return sum((d.detect(src) - o)**2 / (e*e) for d, o, e in zip(instr, obs, err)) / dof


def fit_source(observations, errors, detectors, dof=1., **min_kwargs):
    result = minimize(goodness_of_fit_f,
        x0=standard_x0,
        args=(observations, errors, detectors, dof),
        bounds=((0, None), (18, 25), (1, 2.5)),
        options={'maxiter': 50},
        **min_kwargs,
    )
    return result.x


def fit_full_image(observation_maps, error_maps, detectors,
    bounds, dof=1., chisq=False):
    # maps should be sequences of numpy.ndarrays
    n_pixels = observation_maps[0].size
    img_shape = observation_maps[0].shape
    n_params = len(initial_guess)
    obs_seq = zip(*(o.flat for o in observation_maps))
    err_seq = zip(*(e.flat for e in error_maps))
    result_seq = np.full((n_params, n_pixels), np.nan)
    if chisq:
        chisq_seq = np.full((n_pixels,), np.nan)
    for i, obs, err in zip(range(n_pixels), obs_seq, err_seq):
        result_seq[:, i] = fit_source(obs, err, detectors, dof=dof)
        if chisq:
            chisq_seq[i] = goodness_of_fit_f(result_seq[:, i], obs, err, detectors, dof)
    if chisq:
        return result_seq.reshape((n_params, *img_shape)), chisq_seq.reshape(img_shape)
    else:
        return result_seq.reshape((n_params, *img_shape))
