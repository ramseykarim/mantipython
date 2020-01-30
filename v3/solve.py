import numpy as np
from scipy.optimize import minimize
import sys

"""
Utilities to solve a single map. Assumes the entire map is in the arrays passed to
fit_full_image. Returns a dictionary containing numpy arrays.
Created: probably January 25 2020 or something
"""
__author__ = "Ramsey Karim"


# Standard guesses for T, N, tau, beta
# N, tau are log10
standard_x0 = {'T': 15, 'N': 22, 'tau': -4, 'beta': 2}
standard_bounds = {'T': (0, None), 'N': (18, 25), 'tau': (-7, 0), 'beta': (0, 3)}


def fit_pixel_standard(observations, errors, detectors, src_fn,
    x0=None, bounds=None, **min_kwargs):
    """
    Fit one single pixel. No derivatives.
    observations, errors, and detectors are all the same length and should
    be one-dimensional
    src_fn is a function taking x and returning a greybody
    x0 and bounds are inputs of the same name to scipy.optimize.minimize
    min_kwargs can be any valid keyword arguments to scipy.optimize.minimize
    """
    # Set up goodness of fit function for fitting procedure
    def goodness_of_fit_f(x, obs, err, instr):
        src = src_fn(x)
        return sum((d.detect(src) - o)**2 / (e*e) for d, o, e in zip(instr, obs, err))

    result = minimize(goodness_of_fit_f,
        x0=x0,
        args=(observations, errors, detectors),
        bounds=bounds,
        options={'maxiter': 50},
        **min_kwargs,
    )
    return result

def fit_pixel_jac(observations, errors, detectors, src_fn,
    x0=None, bounds=None, **min_kwargs):
    """
    Fit one single pixel. Uses Jacobian to help minimization.
    observations, errors, and detectors are all the same length and should
    be one-dimensional
    src_fn is a function taking x and returning a greybody
    x0 and bounds are inputs of the same name to scipy.optimize.minimize
    min_kwargs can be any valid keyword arguments to scipy.optimize.minimize
    """
    # Set up goodness of fit function for fitting procedure
    def goodness_of_fit_f(x, obs, err, instr):
        src = src_fn(x)
        return sum((d.detect(src) - o)**2 / (e*e) for d, o, e in zip(instr, obs, err))

    def dgof_f(x, obs, err, instr):
        src = src_fn(x, p=len(x))
        # src(nu) is same as src.radiate(nu)
        return np.sum([2*(d.detect(src.radiate) - o)*d.detect(src.dradiate)/(e*e) for d, o, e in zip(instr, obs, err)], axis=0)

    result = minimize(goodness_of_fit_f,
        x0=x0,
        args=(observations, errors, detectors),
        jac=dgof_f,
        bounds=bounds,
        options={'maxiter': 50},
        **min_kwargs,
    )
    return result


def fit_array(observation_maps, error_maps, detectors, src_fn,
    initial_guess, bounds, mask=None, log_func=None,
    fit_pixel_func=fit_pixel_standard):
    """
    Fit an array of pixels, with data and uncertainties passed in as
    sequences of numpy.ndarrays.
    Dimensionality does not matter, as everything gets flattened anyway.
    The original dimensionality is returned.
    Returns dictionary of numpy arrays including solution, model fluxes,
    differences, and chi squared.
    Keeps track of progress using the logging function passed to log_func.

    maps should be sequences of numpy.ndarrays
    mask (bool), if present, should have same shape as maps. 0 if skip.
    log_func, if present, should take a single string argument to log
    """
    # Get some basic information about the shape and size of the inputs
    n_pixels = observation_maps[0].size
    img_shape = observation_maps[0].shape
    n_params = len(initial_guess)
    n_data = len(observation_maps)
    dof = n_data - n_params

    # Set up the mask so we don't spend time on NaN pixels and can skip
    # manually masked pixels via mask argument
    valid_mask = ~np.any([np.isnan(obs) for obs in observation_maps], axis=0)
    if mask is None:
        mask = valid_mask
    else:
        mask = mask & valid_mask
    mask_seq = mask.flat

    # create low-memory zip iterator out of <n_data> iterators over the arrays
    obs_seq = zip(*(o.flat for o in observation_maps))
    err_seq = zip(*(e.flat for e in error_maps))
    # create solution sequence with parameters as fastest index
    solution_seq = np.full((n_pixels, n_params), np.nan)
    # do the same with other useful maps
    jac_seq = np.full((n_pixels, n_params), np.nan)
    model_seq = np.full((n_pixels, n_data), np.nan) # n_data, not n_params!
    diff_seq = np.full((n_pixels, n_data), np.nan)
    chisq_seq = np.full(n_pixels, np.nan)
    nit_seq = np.full(n_pixels, np.nan)
    success_seq = np.full(n_pixels, np.nan)

    # Keep track of progress
    num_valid_pixels = np.sum(mask.astype(int))
    num_finished_pixels = 0
    keeping_track_of_progress = log_func is not None
    completed_logs = set()

    # Loop through pixels and solve
    for i, obs, err, valid_pixel in zip(range(n_pixels), obs_seq, err_seq, mask_seq):
        if not valid_pixel:
            continue
        soln_object = fit_pixel_func(obs, err, detectors, src_fn,
            x0=initial_guess, bounds=bounds)
        solution_seq[i, :] = soln_object.x
        jac_seq[i, :] = soln_object.jac
        solution_src = src_fn(solution_seq[i, :])
        model_seq[i, :] = np.array([d.detect(solution_src) for d in detectors])
        # chisq_seq[i] = sum((m - o)**2 / (e*e) for m, o, e in zip(model_seq[i, :], obs, err)) / dof
        diff_seq[i, :] = np.array([m - o for m, o in zip(model_seq[i, :], obs)])
        chisq_seq[i] = soln_object.fun / dof
        nit_seq[i] = soln_object.nit
        success_seq[i] = float(soln_object.success)
        if keeping_track_of_progress:
            # Print every 10%, and only print it once
            num_finished_pixels += 1
            nearest_10_pct = round(int(100.*num_finished_pixels / num_valid_pixels), -1)
            if nearest_10_pct not in completed_logs:
                log_func("{:3d} percent done".format(nearest_10_pct))
                completed_logs.add(nearest_10_pct)

    # Reshape all the new maps, delete to avoid approximately doubling
    # memory usage during this last step.
    solution_maps = solution_seq.T.reshape((n_params, *img_shape))
    del solution_seq
    jac_maps = jac_seq.T.reshape((n_params, *img_shape))
    del jac_seq
    model_maps = model_seq.T.reshape((n_data, *img_shape))
    del model_seq
    diff_maps = diff_seq.T.reshape((n_data, *img_shape))
    del diff_seq
    chisq_map = chisq_seq.reshape(img_shape)
    del chisq_seq
    nit_map = nit_seq.reshape(img_shape)
    del nit_seq
    success_map = success_seq.reshape(img_shape)
    del success_seq

    # Build the return value of this function
    result_dict = {
        'solution': solution_maps,
        'model_flux': model_maps,
        'diff_flux': diff_maps,
        'chisq': chisq_map,
        'jacobian': jac_maps,
        'n_iter': nit_map,
        'success': success_map,
    }
    return result_dict
