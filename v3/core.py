import multiprocessing

from . import io_utils
from . import physics
from . import solve

"""
The "core" of mantipython
Coordinate everything and fit an entire map.
Takes paths to Herschel maps and fits SEDs in parallel.
Created: January 29, 2020
"""
__author__ = "Ramsey Karim"


def fit_entire_map(data_filenames, bands_to_fit, parameters_to_fit,
    initial_param_vals=None, param_bounds=None, dust='tau', dust_kwargs=None,
    data_directory="", cutout=None, log_name_func=None, n_procs=2,
    destination_filename=None):
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
    :param log_name_func: take a string, return a string filename
    :param n_procs:
    :param destination_filename: exactly what it sounds like.
        string path to write to.
    """
    # Sanitize list
    bands_to_fit = list(bands_to_fit)
    # Get basic info for input data
    original_shape, hdr = io_utils.check_size_and_header(data_directory + data_filenames[bands_to_fit[0]][0])
    # Parse the cutout argument
    if cutout is None:
        ct_slices = (slice(None), slice(None)) # Entire map
        ct_wcs = io_utils.WCS(hdr)
        ct_shape = original_shape
    elif isinstance(cutout, io_utils.Cutout2D):
        ct_slices = cutout.slices_original
        ct_wcs = cutout.wcs
        ct_shape = cutout.shape
    else:
        center, ct_shape = cutout
        cutout = io_utils.Cutout2D(solve.np.empty(original_shape), (center[1], center[0]),
            ct_shape, wcs=io_utils.WCS(hdr), copy=True)
        ct_slices = cutout.slices_original
        ct_wcs = cutout.wcs
    del cutout # Reduce memory to be copied during multiprocessing
    ct_size = ct_shape[0]*ct_shape[1]

    # Set up some fit conditions
    if initial_param_vals is None:
        initial_param_vals = {}
    for param in solve.standard_x0:
        # Fill in initial values for all parameters not already specified.
        # Unused parameters have no effect.
        if param not in initial_param_vals:
            initial_param_vals[param] = solve.standard_x0[param]
    # Repeat the above steps for parameter boundary conditions
    if param_bounds is None:
        param_bounds = {}
    for param in solve.standard_bounds:
        # Repeat with fit boundaries
        if param not in param_bounds:
            param_bounds[param] = solve.standard_bounds[param]

    # Set up source function, function of "x", len(x) = len(parameters_to_fit)
    src_fn = generate_source_function(parameters_to_fit, initial_param_vals,
        dust, dust_kwargs)

    if n_procs > 1:
        # Parallel case, use multiprocessing
        procs = []
        result_queue = multiprocessing.Queue()
        for proc_idx in range(n_procs):
            p = multiprocessing.Process(target=run_single_process,
                args=(
                    proc_idx, n_procs, data_filenames, bands_to_fit, data_directory,
                    ct_slices, log_name_func(f"_{proc_idx}"), src_fn,
                    [initial_param_vals[pn] for pn in parameters_to_fit],
                    [param_bounds[pn] for pn in parameters_to_fit],
                ), kwargs=dict(result_queue=result_queue))
            p.start()
            procs.append(p)
        # Now they're running in parallel
        for p in procs:
            p.join()
        # Now it's serial again
        # Make a result dictionary to hold the synthesized results
        result_dict = {}
        for k in solve.result_keys:
            array_shape = (solve.result_keys[k](len(parameters_to_fit),
                len(bands_to_fit)), ct_size)
            result_dict[k] = solve.np.full(array_shape, solve.np.nan)
        # Unload the queue into the result dictionary
        while not result_queue.empty():
            proc_idx, result_dict_subset = result_queue.get()
            subproc_slice = array_slice(proc_idx, n_procs, ct_size)
            for k in result_dict_subset:
                result_dict[k][:, subproc_slice] = result_dict_subset[k]
        # Reshape the flattened entries from the result dictionary
        for k in result_dict:
            result_dict[k] = result_dict[k].reshape(result_dict[k].shape[0], *ct_shape)
    else:
        # Serial case, no need for multiprocessing
        result_dict = run_single_process(0, 1,
            data_filenames, bands_to_fit, data_directory,
            ct_slices, log_name_func(""), src_fn,
            [initial_param_vals[pn] for pn in parameters_to_fit],
            [param_bounds[pn] for pn in parameters_to_fit],
        )[1]
    # Now both result dictionaries are similar
    if destination_filename is None:
        destination_filename = data_directory + "mantipython_solution.fits"
    io_utils.write_result(destination_filename,
        result_dict, parameters_to_fit, initial_param_vals,
        bands_to_fit, ct_wcs)
    print("Done, written to "+destination_filename)
    # This function needs a better name


def run_single_process(proc_idx, n_procs, data_filenames, bands_to_fit,
    directory, cutout_slices, log_name, src_fn, init_vals, bounds,
    result_queue=None):
    """
    Either the serialized case of running the fit, or the work of one process
    in the parallel case.
    """
    if result_queue is None and n_procs > 1:
        # Parallel is parallel
        raise RuntimeError("Parallel needs to be parallel. Give a Queue.")

    data_lookup = io_utils.Data(data_filenames, bands_to_fit, prefix=directory)
    # oh god i'm so confused at this point
    # don't worry buddy we got this

    # Function to map onto data_lookup.loaded_data to apply all the slicing
    def f_to_map(obserr, k):
        obserr_modified = []
        for x in obserr:
            # Simple cutout function first
            x = x[cutout_slices]
            if n_procs > 1:
                # Flatten so that dividing evenly is easier
                # Slice again based on process ID to divide the work
                x = x.flatten()[array_slice(proc_idx, n_procs, x.size)]
            obserr_modified.append(x)
        return tuple(obserr_modified)

    data_lookup.map(f_to_map) # Slice the data somehow

    def logger(message):
        with open(log_name, 'a') as f:
            f.write(message+"\n")

    t0 = io_utils.datetime.datetime.now()
    logger(f"Beginning fit on {data_lookup[bands_to_fit[0]][0].shape}")
    result_dict = solve.fit_array(*zip(*(data_lookup[k] for k in bands_to_fit)),
        physics.get_instrument(bands_to_fit), src_fn,
        init_vals, bounds, log_func=logger,
    )
    t1 = io_utils.datetime.datetime.now()
    logger(f"Finished at {t1}\nTook {(t1-t0).total_seconds()/60.} minutes")
    result = (proc_idx, result_dict)
    if result_queue is None:
        return result
    else:
        result_queue.put(result)



def array_slice(i, total_i, array_size):
    """
    Utility for dividing arrays roughly evenly to be worked on by
    several processes.
    :param i: index of this process (0-indexed)
    :param total_i: total number of processes involved
    :array_size: total number of pixels in the 1D array
    """
    arr_slice_size = int(array_size//total_i) + 1
    return slice(arr_slice_size*i, arr_slice_size*(i+1))


def generate_source_function(parameters_to_fit, initial_param_vals,
    dust, dust_kwargs):
    # Standardize the dust functions into a lambda function with one argument
    if dust == 'tau':
        # Make sure dust parameter matches dust type
        if 'N' in parameters_to_fit:
            raise RuntimeError("Parameter N not compatible with dust type tau")
        dust_function = lambda b: physics.TauOpacity(beta=b)
        column_parameter_name = 'tau'
    elif dust == 'kappa':
        if 'tau' in parameters_to_fit:
            raise RuntimeError("Parameter tau not compatible with dust type kappa")
        dust_function = lambda b: physics.Dust(beta=b, **dust_kwargs)
        column_parameter_name = 'N'
    else:
        raise RuntimeError(f"Dust type {dust} not recognized.")
    # Make function for retrieving parameters from parameter array,
    # or from initial values if parameter is to be fixed.
    def retrieve_parameter(x, param_name):
        if param_name in parameters_to_fit:
            return x[parameters_to_fit.index(param_name)]
        else:
            return initial_param_vals[param_name]
    # Build the source function; this will be used in the fitting algorithm
    def src_fn(x):
        return physics.Greybody(
            retrieve_parameter(x, 'T'),
            retrieve_parameter(x, column_parameter_name),
            dust_function(retrieve_parameter(x, 'beta'))
        )
    return src_fn
