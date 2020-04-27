import multiprocessing
import numpy as np # # TODO: REMOVE THIS move the stuff into solve
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
    destination_filename=None, fitting_function=None, grid_sample=False):
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
    :param fitting_function: 'jac' for jacobian, anything else for standard
    """
    # BEGINNING SETUP
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

    # Fitting function selection
    if fitting_function == 'jac':
        fitting_function = solve.fit_pixel_jac
    else:
        fitting_function = solve.fit_pixel_standard
    # FINISHED SETUP

    if n_procs > 1:
        ########################################
        # Parallel case, use multiprocessing

        # (2/6/20) was using Queue, but pickling/piping(?) has data limit
        # Now using shared arrays
        # Set up some structure with which to gather data from processes
        result_dict = {}
        for k in solve.result_frames:
            array_size = ct_size * solve.result_frames[k](len(parameters_to_fit), len(bands_to_fit))
            shared_array = multiprocessing.Array('d', array_size)
            result_dict[k] = shared_array

        # <PARALLEL>
        # Run the processes
        procs = []
        for proc_idx in range(n_procs):
            p = multiprocessing.Process(target=run_single_process,
                args=(
                    proc_idx, n_procs, data_filenames, bands_to_fit, data_directory,
                    ct_slices, ct_size, log_name_func(f"_{proc_idx}"), src_fn,
                    [initial_param_vals[pn] for pn in parameters_to_fit],
                    [param_bounds[pn] for pn in parameters_to_fit],
                    fitting_function, grid_sample,
                ), kwargs=dict(shared_array_dict=result_dict))
            p.start()
            procs.append(p)
        # Now they're running in parallel
        for p in procs:
            p.join()
        # </PARALLEL>

        # Now it's serial again
        # We basically already have the completed results dictionary, we just
        # need to transform it back to numpy arrays
        for k in result_dict:
            i_shape = solve.result_frames[k](len(parameters_to_fit), len(bands_to_fit))
            tmp = solve.np.full((i_shape, *ct_shape), solve.np.nan)
            tmp[:] = solve.np.frombuffer(result_dict[k].get_obj()).reshape(i_shape, *ct_shape)
            result_dict[k] = tmp
            # Now the shared arrays are gone, it's just pure numpy arrays
        ########################################
    else:
        ########################################
        # Serial case, no need for multiprocessing
        result_dict = run_single_process(0, 1,
            data_filenames, bands_to_fit, data_directory,
            ct_slices, ct_size, log_name_func(""), src_fn,
            [initial_param_vals[pn] for pn in parameters_to_fit],
            [param_bounds[pn] for pn in parameters_to_fit],
            fitting_function, grid_sample,
        )[1]
        ########################################

    # Make the new data lookup object for use in both check_and_refit and write_result
    data_lookup = io_utils.Data(data_filenames, bands_to_fit, prefix=data_directory)
    data_lookup.map(lambda oe, k: tuple(x[ct_slices] for x in oe))
    # This would be the place to check for bad fits (spikes, etc)
    # Check for bad pixels and refit using the function in solve.py
    solve.check_and_refit(result_dict,
        *zip(*(data_lookup[k] for k in bands_to_fit)),
        physics.get_instrument(bands_to_fit), src_fn,
        [initial_param_vals[pn] for pn in parameters_to_fit],
        [param_bounds[pn] for pn in parameters_to_fit],
        log_func=io_utils.make_logger(log_name_func("_0")),
        fit_pixel_func=fitting_function, grid_sample=grid_sample)

    """
    #### Stuff from earlier
    # Get solution and error maps
    soln = result_dict['solution']
    err = result_dict['error']
    # All the parameters should be affected so just use the first one
    p = 0
    # Set up a small mask around 1 pixel
    local_mask = np.ones((3, 3))
    local_mask[1,1] = 0
    local_mask = local_mask.astype(bool)
    # Gather problem pixels
    problem_pixels = []
    for i in range(1, soln.shape[1]-1):
        for j in range(1, soln.shape[2]-1):
            local_cube = soln[p, i-1:i+2, j-1:j+2]
            local_err_cube = err[p, i-1:i+2, j-1:j+2]
            local_err = np.mean(local_err_cube[local_mask])
            local_val = soln[p, i, j]
            # Get mean and standard deviation of surrounding pixels
            local_mean = np.mean(local_cube[local_mask])
            local_std = np.std(local_cube[local_mask])
            if np.isnan(local_mean):
                # Skip it if there are any NaNs, not trustworthy anyway
                continue
            local_diff = np.abs(local_val - local_mean)
            if (local_diff > local_std) & (local_diff > local_err):
                # Save location, mean surrounding value, and local parameter error
                problem_pixels.append((i, j, local_mean, local_err))
    # Sort out the observations and their errors into arrays
    i_idxs, j_idxs, local_means, local_errs = zip(*problem_pixels)
    # Fit the problem pixels with tight bounds
    instr = physics.get_instrument(bands_to_fit)
    for i, j, local_mean, local_err in problem_pixels:
        observations, errors = zip(*((arr[i, j] for arr in data_lookup[k]) for k in bands_to_fit))
        new_init_vals = [initial_param_vals[pn] for pn in parameters_to_fit]
        new_init_vals[p] = local_mean
        new_bounds = [param_bounds[pn] for pn in parameters_to_fit]
        new_bounds[p] = (local_mean - local_err, local_mean + local_err)
        new_result = fitting_function(observations, errors, instr, src_fn, x0=new_init_vals, bounds=new_bounds)
        print(new_result.x)


        # observations, errors, detectors, src_fn,
        #     x0=None, bounds=None, **min_kwargs


    # solve.fit_array(*zip(*(data_lookup[k] for k in bands_to_fit)),
    #     physics.get_instrument(bands_to_fit), src_fn,
    #     init_vals, bounds, log_func=logger, fit_pixel_func=fitting_function,
    # )
    """
    return result_dict


    # Now both result dictionaries are similar
    if destination_filename is None:
        destination_filename = data_directory + "mantipython_solution.fits"
    data_lookup = io_utils.Data(data_filenames, bands_to_fit, prefix=data_directory)
    data_lookup.map(lambda oe, k: tuple(x[ct_slices] for x in oe))
    io_utils.write_result(destination_filename,
        result_dict, data_lookup, parameters_to_fit, initial_param_vals,
        bands_to_fit, ct_wcs)
    print("Done, written to "+destination_filename)
    # This function needs a better name


def run_single_process(proc_idx, n_procs, data_filenames, bands_to_fit,
    directory, cutout_slices, cutout_size, log_name, src_fn, init_vals, bounds,
    fitting_function, grid_sample, shared_array_dict=None):
    """
    Either the serialized case of running the fit, or the work of one process
    in the parallel case.
    """
    if shared_array_dict is None and n_procs > 1:
        # Parallel is parallel
        raise RuntimeError("Parallel needs to be parallel. Give a Queue.")

    data_lookup = io_utils.Data(data_filenames, bands_to_fit, prefix=directory)
    # oh god i'm so confused at this point
    # don't worry buddy we got this

    # Function to map onto data_lookup.loaded_data to apply all the slicing
    def f_to_map(obserr, k):
        # k is unused but theoretically gives me ability to do different
        # things to different wavelengths (blow up errors, etc)
        obserr_modified = []
        for x in obserr:
            # Simple cutout function first
            x = x[cutout_slices]
            if n_procs > 1:
                # Flatten so that dividing evenly is easier
                # Slice again based on process ID to divide the work
                x = x.flatten()[array_slice(proc_idx, n_procs, cutout_size)]
            obserr_modified.append(x)
        return tuple(obserr_modified)

    data_lookup.map(f_to_map) # Slice the data somehow

    logger = io_utils.make_logger(log_name)
    t0 = io_utils.datetime.datetime.now()
    logger(f"Beginning fit on {data_lookup[bands_to_fit[0]][0].shape}")
    result_dict = solve.fit_array(*zip(*(data_lookup[k] for k in bands_to_fit)),
        physics.get_instrument(bands_to_fit), src_fn,
        init_vals, bounds, log_func=logger, fit_pixel_func=fitting_function,
        grid_sample=grid_sample,
    )
    t1 = io_utils.datetime.datetime.now()
    logger(f"Finished at {t1}\nTook {(t1-t0).total_seconds()/60.} minutes")
    if shared_array_dict is None:
        return (proc_idx, result_dict)
    else:
        # We have results in a dictionary with shapes according to
        # solve.result_frames[k](len(parameters_to_fit), len(bands_to_fit)), ct_size
        # Need to reshape each shared array to this shape and then assign
        # the result_dict entry to the array_slice() of the shared array
        for k in result_dict:
            i_shape = result_dict[k].shape[0]
            process_slice = array_slice(proc_idx, n_procs, cutout_size)
            shared_arr_np = solve.np.frombuffer(shared_array_dict[k].get_obj()).reshape((i_shape, cutout_size))
            shared_arr_np[:, process_slice] = result_dict[k]


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
        if 'N' in parameters_to_fit or "N_bg" in parameters_to_fit:
            raise RuntimeError("Parameter N not compatible with dust type tau")
        dust_function = lambda b: physics.TauOpacity(beta=b)
        column_parameter_name = 'tau'
        if 'tau_bg' in parameters_to_fit:
            df_bg = lambda b: physics.TauOpacity(beta=b)
            cpn_bg = 'tau_bg'
    elif dust == 'kappa':
        if 'tau' in parameters_to_fit or 'tau_bg' in parameters_to_fit:
            raise RuntimeError("Parameter tau not compatible with dust type kappa")
        dust_function = lambda b: physics.Dust(beta=b, **dust_kwargs)
        column_parameter_name = 'N'
        if 'N_bg' in parameters_to_fit:
            df_bg = lambda b: physics.Dust(beta=b, **dust_kwargs)
            cpn_bg = 'N_bg'
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
    if any(('bg' in param_name) for param_name in parameters_to_fit):
        def src_fn(x, **kwargs):
            return physics.MultiGreybody(
                [retrieve_parameter(x, 'T_bg'), retrieve_parameter(x, 'T')],
                [retrieve_parameter(x, cpn_bg), retrieve_parameter(x, column_parameter_name)],
                [df_bg(retrieve_parameter(x, 'beta')), dust_function(retrieve_parameter(x, 'beta'))],
                **kwargs
            )
            pass
    else:
        def src_fn(x, **kwargs):
            return physics.Greybody(
                retrieve_parameter(x, 'T'),
                retrieve_parameter(x, column_parameter_name),
                dust_function(retrieve_parameter(x, 'beta')),
                **kwargs,
            )
    return src_fn
