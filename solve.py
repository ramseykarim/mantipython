import numpy as np
from scipy.optimize import minimize
import sys

"""
Utilities to solve a single map. Assumes the entire map is in the arrays passed to
fit_full_image. Returns a dictionary containing numpy arrays.
Created: probably January 25 2020 or something
Updated with uncertainties on April 10, 2020
"""
__author__ = "Ramsey Karim"


# Standard guesses for T, N, tau, beta
# N, tau are log10
standard_x0 = {'T': 15, 'N': 22, 'tau': -2, 'beta': 2,
    'T_bg': 15, 'N_bg': 22, 'tau_bg': -2,}
standard_bounds = {'T': (5, None), 'N': (18, 25), 'tau': (-7, 0), 'beta': (0, 3),
    'T_bg': (5, None), 'N_bg': (18, 25), 'tau_bg': (-7, 0),}

# Useful for creating the dictionary to return
result_frames = {
    'solution': lambda p, d: p,
    'model_flux': lambda p, d: d,
    'diff_flux': lambda p, d: d,
    'chisq': lambda p, d: 1,
    'error': lambda p, d: p,
    'n_iter': lambda p, d: 1,
    'success': lambda p, d: 1,
}


def make_goodness_of_fit_f(src_fn):
    """
    Define a simple goodness_of_fit_f
    """
    def goodness_of_fit_f(x, obs, err, instr):
        src = src_fn(x)
        return sum((d.detect(src) - o)**2 / (e*e) for d, o, e in zip(instr, obs, err))
    return goodness_of_fit_f


def make_goodness_of_fit_f_jacobian_vector(src_fn):
    def dgof_f(x, obs, err, instr):
        src = src_fn(x, p=len(x))
        # src(nu) is same as src.radiate(nu)
        return np.sum([2*(d.detect(src.radiate) - o)*d.detect(src.dradiate)/(e*e) for d, o, e in zip(instr, obs, err)], axis=0)
    return dgof_f


def residual_sumofsquares(obs, model):
    return sum((o - m)**2 for o, m in zip(obs, model))


def jacobian_matrix(err, instr, src):
    return np.array([d.detect(src.dradiate) for d in instr])


def weighting_matrix(err, instr, src):
    return np.diag(1./np.array(err)**2.)


def parameter_error_vector(*args):
    # From http://people.duke.edu/~hpgavin/ce281/lm.pdf section 4.2
    j = jacobian_matrix(*args)
    w = weighting_matrix(*args)
    return np.sqrt(np.diag(np.linalg.inv(j.T @ w @ j))) # Cool operator!


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
    goodness_of_fit_f = make_goodness_of_fit_f(src_fn)

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
    goodness_of_fit_f = make_goodness_of_fit_f(src_fn)

    # Set up derivatives of goodness of fit function
    dgof_f = make_goodness_of_fit_f_jacobian_vector(src_fn)

    result = minimize(goodness_of_fit_f,
        x0=x0,
        args=(observations, errors, detectors),
        jac=dgof_f,
        bounds=bounds,
        options={'maxiter': 50},
        **min_kwargs,
    )
    return result


def sample_uncertainty(observations, errors, detectors, src_fn,
    x0=None, bounds=None, jac=False, **min_kwargs):
    """
    Sample every combination of observations +1sigma, +0, -1sigma
    to get a very rough handle on the parameter uncertainty.
    Uses the standard deviation of the results
    Useful when dof=0
    """
    fit_pixel_func = fit_pixel_jac if jac else fit_pixel_standard
    obs_arrays = []
    for o, e in zip(observations, errors):
        obs_arrays.append([o-e, o, o+e])
    obs_cubes = np.meshgrid(*obs_arrays, indexing='ij')
    obs_flatcubes = (x.ravel() for x in obs_cubes)
    results = []
    for sampled_obs in zip(*obs_flatcubes):
        sampled_result = fit_pixel_func(sampled_obs, errors, detectors, src_fn,
            x0=x0, bounds=None, **min_kwargs)
        results.append(sampled_result.x)
    results = np.array(results)
    return np.std(results, axis=0)


def fit_array(observation_maps, error_maps, detectors, src_fn,
    initial_guess, bounds, mask=None, log_func=None,
    fit_pixel_func=fit_pixel_standard, grid_sample=False):
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
    error_seq = np.full((n_pixels, n_params), np.nan)
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

    # Decide if we are grid sampling for the parameter uncertainties
    if dof == 0:
        # Cannot explicitly calculate uncertainty; need to sample for it
        grid_sample = True
        if keeping_track_of_progress:
            log_func("Grid sample turned on because no degrees of freedom.")
    else:
        # Calculating uncertainty via Jacobian matrix (4/10/20)
        # I have compared these two methods and they produce similar results
        # for the test case
        pass


    # Loop through pixels and solve
    for i, obs, err, valid_pixel in zip(range(n_pixels), obs_seq, err_seq, mask_seq):
        if not valid_pixel:
            continue
        # Get the result
        soln_object = fit_pixel_func(obs, err, detectors, src_fn,
            x0=initial_guess, bounds=bounds)
        # Save the parameter combination
        solution_seq[i, :] = soln_object.x
        # Make the solution source object
        solution_src = src_fn(solution_seq[i, :], p=n_params)
        # Get the predicted fluxes
        model_seq[i, :] = np.array([d.detect(solution_src) for d in detectors])

        # Uncertainty
        if grid_sample:
            error_seq[i, :] = sample_uncertainty(obs, err, detectors, src_fn,
                x0=initial_guess, bounds=bounds, jac=False)
        else:
            error_seq[i, :] = parameter_error_vector(err, detectors, solution_src)

        # Get the model minus observation differences
        diff_seq[i, :] = np.array([m - o for m, o in zip(model_seq[i, :], obs)])
        # Get the chi squared
        if dof != 0:
            chisq_seq[i] = soln_object.fun / dof
        else:
            chisq_seq[i] = np.inf
        # Get the number of iterations
        nit_seq[i] = soln_object.nit
        # Get the success flag
        success_seq[i] = float(soln_object.success)
        # Manage log
        if keeping_track_of_progress:
            # Print every 10%, and only print it once
            num_finished_pixels += 1
            nearest_10_pct = round(int(100.*num_finished_pixels / num_valid_pixels), -1)
            if nearest_10_pct not in completed_logs:
                log_func("{:3d} percent done".format(nearest_10_pct))
                completed_logs.add(nearest_10_pct)

    # Build the return value of this function
    result_dict = {
        'solution': solution_seq,
        'model_flux': model_seq,
        'diff_flux': diff_seq,
        'chisq': chisq_seq,
        'error': error_seq,
        'n_iter': nit_seq,
        'success': success_seq,
    }
    # Delete to avoid approximately doubling memory usage during this last step.
    del solution_seq, model_seq, diff_seq, chisq_seq, err_seq, nit_seq, success_seq
    # Reshape all the new maps
    for k in result_dict:
        i_shape = result_frames[k](n_params, n_data)
        result_dict[k] = result_dict[k].T.reshape(i_shape, *img_shape)
    return result_dict


def check_and_refit(result_dict, observation_maps,
    error_maps, detectors, src_fn,
    initial_guess, bounds, log_func=None,
    fit_pixel_func=fit_pixel_standard, grid_sample=False):
    """
    Takes in a completed results dictionary (the entire thing is best,
    to minimize edge effects) and compares every pixel to its surrounding
    pixels.
    Pixels with differences greater than both the surrounding errors and
    std. of surrounding pixels will be refit with tighter bounds.
    This procedure should minimize noise spikes that are totally unrealistic.
    This will only work if unrealistic solutions are few and far between, and
    are always surrounded by good fits.
    Does not return anything; modifies result_dict in place
    """
    # Pick the first parameter (arbitrary)
    # If the fit is screwed up, all the parameters should be off, so it doesn't
    # matter which one we check.
    p = 0
    solution = result_dict['solution']
    perror = result_dict['error']
    # Set up a small mask around 1 pixel
    local_mask = np.ones((3, 3))
    local_mask[1,1] = 0
    local_mask = local_mask.astype(bool)
    # Gather problem pixels
    problem_pixels = []
    for i in range(1, solution.shape[1]-1):
        for j in range(1, solution.shape[2]-1):
            local_cube = solution[p, i-1:i+2, j-1:j+2]
            local_perror_cube = perror[p, i-1:i+2, j-1:j+2]
            local_perror = np.mean(local_perror_cube[local_mask])
            local_solution = solution[p, i, j]
            # Get mean and standard deviation of surrounding pixels
            local_mean = np.mean(local_cube[local_mask])
            local_std = np.std(local_cube[local_mask])
            if np.isnan(local_mean):
                # Skip it if there are any NaNs, not trustworthy anyway
                continue
            # Check the difference between the pixel and its surroudings
            # against the surrounding mean error and standard
            # deviation of values
            local_diff = np.abs(local_solution - local_mean)
            # Also mark this pixel if success flag came back False
            if ((local_diff > local_std) and (local_diff > local_perror)) or (result_dict['success'][0, i, j] == 0):
                # Save location, mean surrounding value, and local parameter error
                problem_pixels.append((i, j, local_mean, local_perror))
    # Iterate through the problem pixels and re-fit
    for i, j, local_mean, local_perror in problem_pixels:
        observations = [o[i, j] for o in observation_maps]
        errors = [e[i, j] for e in error_maps]
        # Perturb the observations using the errors
        for idx in range(len(observations)):
            observations[idx] += errors[idx] * 0.01 * (-1)**idx
        new_init_guesses = [x for x in initial_guess]
        new_init_guesses[p] = local_mean
        new_bounds = [x for x in bounds]
        new_bounds[p] = (local_mean - local_perror, local_mean + local_perror)
        new_result = fit_pixel_func(observations, errors, detectors, src_fn,
            x0=new_init_guesses, bounds=new_bounds)
        solution[:, i, j] = new_result.x
        # Other stuff
        dof = len(observations) - solution.shape[0]
        solution_src = src_fn(solution[:, i, j], p=solution.shape[0])
        # Get the model fluxes
        result_dict['model_flux'][:, i, j] = np.array([d.detect(solution_src) for d in detectors])
        # Get the model minus observation differences
        result_dict['diff_flux'][:, i, j] = np.array([m - o for m, o in zip(model_seq[i, :], obs)])
        # Calculate chi squared
        if dof != 0:
            chisq = new_result.fun / dof
        else:
            chisq = np.inf
        result_dict['chisq'][0, i, j] = chisq
        # Number of iterations INCLUDES PREVIOUS COUNT
        result_dict['n_iter'][0, i, j] += new_result.nit
        # Success flag: 2 if successful this time and also "success" last time
        # 3 if unsuccessful last time and successful this time
        # -1 if unsuccessful both times
        # -2 if successful last time and unsuccessful this time
        # This leaves 0: unsuccessful only once; and 1: successful only once
        prev_success_flag = result_dict['success'][0, i, j]
        new_success_flag = float(new_result.success)
        # Presently, only possibilities for either success flag are 0 or 1
        if prev_success_flag:
            if new_success_flag:
                final_success_flag = 2.0
            else:
                final_success_flag = -2.0
        else:
            if new_success_flag:
                final_success_flag = 3.0
            else:
                final_success_flag = -1.0
        # This success flag system basically means: the greater the number,
        # the better the re-fit. The smaller the number, the worse the re-fit
        result_dict['success'][0, i, j] = final_success_flag
        # Uncertainty
        if grid_sample or (dof == 0) or (final_success_flag < 0):
            # Also use grid sample if success flag looks bad
            result_dict['error'][:, i, j] = sample_uncertainty(observations, errors, detectors, src_fn,
                x0=initial_guess, bounds=bounds, jac=False)
        else:
            result_dict['error'][:, i, j] = parameter_error_vector(errors, detectors, solution_src)
        # 'dog it and log it
        log_func(f"refit pixel at {i}, {j} with success {final_success_flag:.0f}")
