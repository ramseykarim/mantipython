import numpy as np
import mpy_utils as mpu
from Greybody import Greybody
from scipy.optimize import minimize
import sys



# Standard guesses for Tc, Nh, Nc
# Nh, Nc are log10
standard_x0 = [10, 22, 23]

def goodness_of_fit_f_3p(x, dusts, obs, err, instr, Th, dof):
    # x is [Tc, Nh, Nc] (Ns in log10)
    src = Greybody([Th, x[0]], [10**N for N in x[1:]], dusts)
    # Computes reduced chi^2 given a source model and observations/errors
    return sum((d.detect(src) - o)**2 / (e*e) for d, o, e in zip(instr, obs, err))/dof


def fit_source_3p(observations, errors, detectors, dusts,
    Th=15., dof=1., **min_kwargs):
    result = minimize(goodness_of_fit_f_3p,
        x0=standard_x0,
        args=(dusts, observations, errors, detectors, Th, dof),
        bounds=((0, None), (18, 25), (18, 25)),
        options={'maxiter': 50},
        **min_kwargs,
    )
    # print("RESULT:", result.message)
    return result.x


def goodness_of_fit_f_2p(x, dust, obs, err, instr, dof):
    # x is [Tc, Nc] (N in log10)
    src = Greybody(x[0], 10**x[1], dust)
    # Computes reduced chi^2 given a source model and observations/errors
    return sum((d.detect(src) - o)**2 / (e*e) for d, o, e in zip(instr, obs, err))/dof

def fit_source_2p(observations, errors, detectors, dust,
    dof=2., **min_kwargs):
    result = minimize(goodness_of_fit_f_2p,
        x0=[standard_x0[0], standard_x0[2]],
        args=(dust, observations, errors, detectors, dof),
        bounds=((0, None), (18, 25)),
        options={'maxiter': 50}
        **min_kwargs,
    )
    # print("RESULT:", result.message)
    return result.x


def goodness_of_fit_f_1p(x, dust, obs, err, instr, Th, dof):
    # x is [Tc, Nc] (N in log10)
    src = Greybody(Th, 10**x[0], dust)
    # Computes reduced chi^2 given a source model and observations/errors
    return sum((d.detect(src) - o)**2 / (e*e) for d, o, e in zip(instr, obs, err))/dof


def fit_source_1p(observations, errors, detectors, dust,
    Th=15., dof=3., **min_kwargs):
    result = minimize(goodness_of_fit_f_1p,
        x0=[20,],
        args=(dust, observations, errors, detectors, Th, dof),
        bounds=((18, 25),),
        options={'maxiter': 50}
        **min_kwargs,
    )
    return result.x


def gen_goodness_of_fit_f(obs, err, instr, src_fn, dof):
    # src_fn should be preset with kwargs such as Th, Nh, whatever, AND DUST
    # src_fn should expect some X vector of params and return a Greybody
    def goodness_of_fit_f(x):
        src = src_fn(x)
        return sum((d.detect(src) - o)**2. / (e*e) for d, o, e in zip(instr, obs, err))/dof
    return goodness_of_fit_f


def fit_source(observations, errors, detectors, src_fn, initial_guess, bounds, dof=1.,
    goodness_of_fit_f=None, **min_kwargs):
    # see src_fn requirements in gen_goodness_of_fit_f
    if goodness_of_fit_f is None:
        goodness_of_fit_f = gen_goodness_of_fit_f(observations, errors, detectors, src_fn, dof)
    result = minimize(goodness_of_fit_f,
        x0=initial_guess, bounds=bounds, options={'maxiter': 50},
        **min_kwargs)
    return result.x


def bootstrap_errors(observations, errors, detectors, dusts,
    niter=30, fit_f=fit_source_2p, verbose=False,
    perturbation_f=np.random.normal, samples_only=False,
    **kwargs):
    # perturbation_f should take obs, err and return something like obs
    observations = np.array(observations)
    errors = np.array(errors)
    results = mpu.deque()
    for i in range(niter):
        obs_perturbed = perturbation_f(observations, errors)
        if verbose:
            sys.stdout.write(f"{i+1}/{niter}..\r")
            sys.stdout.flush()
        current_result = fit_f(obs_perturbed, errors, detectors, dusts, **kwargs)
        current_result[1:] = [10**x for x in current_result[1:]]
        results.append(current_result)
    if verbose:
        print()
    fitted_param_sets = list(results)
    param_uncertainties = [np.std(x) for x in zip(*results)]
    if samples_only:
        return fitted_param_sets
    else:
        return fitted_param_sets, param_uncertainties


def fit_full_image(observation_maps, error_maps, detectors,
    src_fn, initial_guess, bounds, dof=1., chisq=False):
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
        result_seq[:, i] = fit_source(obs, err, detectors, src_fn, initial_guess, bounds, dof=dof)
        if chisq:
            chisq_seq[i] = gen_goodness_of_fit_f(obs, err, detectors, src_fn, dof)(result_seq[:, i])
    if chisq:
        return result_seq.reshape((n_params, *img_shape)), chisq_seq.reshape(img_shape)
    else:
        return result_seq.reshape((n_params, *img_shape))
