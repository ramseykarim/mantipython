import numpy as np
from scipy.optimize import minimize
import sys

__author__ = "Ramsey Karim"


# Standard guesses for T, N, tau, beta
# N, tau are log10
standard_x0 = {'T': 15, 'N': 22, 'tau': -4, 'beta': 2}
standard_bounds = {'T': (0, None), 'N': (18, 25), 'tau': (-7, 0), 'beta': (0, 3)}

def gen_goodness_of_fit_f(src_fn, dof):
    def goodness_of_fit_f(x, obs, err, instr):
        src = src_fn(x)
        return sum((d.detect(src) - o)**2 / (e*e) for d, o, e in zip(instr, obs, err)) / dof
    return goodness_of_fit_f

def gen_model_maps_f(src_fn):
    def model_maps_f(x, obs, err, instr):
        src = src_fn(x)
        return np.array([d.detect(src) for d in instr])
    return model_maps_f

def gen_diff_maps_f(src_fn):
    def diff_maps_f(x, obs, err, instr):
        # return 1d array with len(obs) elements
        src = src_fn(x)
        return np.array([(d.detect(src) - o) for d, o in zip(instr, obs)])
    return diff_maps_f


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
    initial_guess, bounds, mask=None, chisq=False, log_func=None):
    # maps should be sequences of numpy.ndarrays
    # chisq=True will also yield model and diff maps
    # mask (bool), if present, should have same shape as maps. 0 if skip.
    # log_func, if present, should take a single string argument to log
    n_pixels = observation_maps[0].size
    img_shape = observation_maps[0].shape
    n_params = len(initial_guess)
    n_data = len(observation_maps)
    valid_mask = ~np.any([np.isnan(obs) for obs in observation_maps], axis=0)
    if mask is None:
        mask = valid_mask
    else:
        mask = mask & valid_mask
    dof = n_data - n_params
    obs_seq = zip(*(o.flat for o in observation_maps))
    err_seq = zip(*(e.flat for e in error_maps))
    result_seq = np.full((n_params, n_pixels), np.nan)
    goodness_of_fit_f = gen_goodness_of_fit_f(src_fn, dof)
    if chisq:
        model_maps_f = gen_model_maps_f(src_fn)
        diff_maps_f = gen_diff_maps_f(src_fn)
        # n_data, not n_params!
        model_seq = np.full((n_data, n_pixels), np.nan)
        diff_seq = np.full((n_data, n_pixels), np.nan)
    # Keep track of progress
    num_valid_pixels = np.sum(mask.astype(int))
    num_finished_pixels = 0
    keeping_track_of_progress = log_func is not None
    completed_logs = set()
    if chisq:
        chisq_seq = np.full((n_pixels,), np.nan)
    for i, obs, err, valid_pixel in zip(range(n_pixels), obs_seq, err_seq, mask.flat):
        if valid_pixel:
            result_seq[:, i] = fit_source(obs, err, detectors, goodness_of_fit_f,
                x0=initial_guess, bounds=bounds)
            if chisq:
                chisq_seq[i] = goodness_of_fit_f(result_seq[:, i], obs, err, detectors)
                model_seq[:, i] = model_maps_f(result_seq[:, i], obs, err, detectors)
                diff_seq[:, i] = diff_maps_f(result_seq[:, i], obs, err, detectors)
            if keeping_track_of_progress:
                num_finished_pixels += 1
                nearest_10_pct = round(int(100.*num_finished_pixels / num_valid_pixels), -1)
                if nearest_10_pct not in completed_logs:
                    log_func("{:3d} percent done".format(nearest_10_pct))
                    completed_logs.add(nearest_10_pct)
    if chisq:
        return result_seq.reshape((n_params, *img_shape)), chisq_seq.reshape(img_shape), model_seq.reshape((n_data, *img_shape)), diff_seq.reshape((n_data, *img_shape))
    else:
        return result_seq.reshape((n_params, *img_shape))
