import datetime
import multiprocessing

import .io_utils

"""
The "core" of mantipython
Coordinate everything and fit an entire map.
Takes paths to Herschel maps and fits SEDs in parallel.
Created: January 29, 2020
"""
__author__ = "Ramsey Karim"


def fit_entire_map(data_filenames, bands_to_fit, parameters_to_fit,
    initial_param_vals=None, param_bounds=None, dust='tau', dust_kwargs=None,
    data_directory="", cutout=None, log_name_func=None, n_procs=2):
    """
    Fit entire Herschel map in several bands.
    :param data_filenames: dictionary mapping integer band wavelength in
        micron to 2-tuples of strings pointing towards that Herschel image and
        its corresponding error map. Not all entries of the dictionary need
        to be part of the fit; see bands_to_fit
    :param bands_to_fit: a sequence of integer band wavelengths in micron that
        should be used for this fit. Must be a subset of keys for data_filenames
    :param parameters_to_fit: sequence of strings describing the parameters by
        which to fit the SEDs. Must be recognized by solve.py
    :param initial_param_vals: dictionary mapping recognized parameter names
        to their initial guesses. Values default to those in solve.py.
        This is also how to fix a parameter that is not being varied.
        Note: if using an opacity table, set kw 'dustmodel' to the table input
        described in dust.py documentation.
    :param param_bounds: dictionary, in same format as initial_param_vals,
        giving boundaries for fitting parameters. No effect for parameters not
        in parameters_to_fit.
    :param dust: tau or kappa, describing type of dust to use.
    :param dust_kwargs: extra kwargs to be passed to the Dust object.
        For tweaking other power law parameters (kappa0, nu0). Only valid for
        dust='kappa'
    :param data_directory: a string to be prepended to all strings contained
        in data_filenames
    :param cutout: descriptor for fitting a subset of the input maps.
        Can be a Cutout2D that is valid for use on these maps,
        or could be a tuple ((i_center, j_center), (i_width, j_width)).
    :param log_name_func:
    :param n_procs:
    """
    # Sanitize list
    bands_to_fit = list(bands_to_fit)
    # Get basic info for input data
    original_shape, hdr = io_utils.check_size_and_header(data_directory + data_filenames[bands_to_fit[0]][0])
    # Parse the cutout argument
    if cutout is None:
        ct_slices = (slice(None), slice(None)) # Entire map
        ct_wcs = io_utils.WCS(hdr)
    elif isinstance(cutout, Cutout2D):
        ct_slices = cutout.slices_original
        ct_wcs = cutout.wcs
    else:
        center, new_shape = cutout
        cutout = io_utils.Cutout2D(np.empty(original_shape), (center[1], center[0]),
            new_shape, wcs=io_utils.WCS(hdr), copy=True)
        ct_slices = cutout.slices_original
        ct_wcs = cutout.wcs

    # Set up some fit conditions
    if initial_param_vals is None:
        initial_param_vals = {}
    for param in solve.standard_x0:
        # Fill in initial values for all parameters not already specified.
        # Unused parameters have no effect.
        if param not in initial_param_vals:
            initial_param_vals[param] = solve.standard_x0[param]
    if param_bounds is None:
        param_bounds = {}
    for param in solve.standard_bounds:
        # Repeat with fit boundaries
        if param not in param_bounds:
            param_bounds[param] = solve.standard_bounds[param]


    # That's enough for now.......
    # need a better name
    # should probably implement before picking names... chickens before they hatch

    pass


def run_single_process(data_filenames, bands_to_fit, directory, log_name,
    f_to_map=None):
    """
    Either the serialized case of running the fit, or the work of one process
    in the parallel case.
    """
    data_lookup = io_utils.Data(data_filenames, bands_to_fit, prefix=directory)

    # oh god i'm so confused at this point

    def logger(message):
        with open(log_name, 'a') as f:
            f.write(message+"\n")

    t0 = datetime.datetime.now()
    logger(f"Beginning fit on {data_lookup[bands_to_fit[0]][0].shape}")
    t1 = datetime.datetime.now()
