from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy.wcs import WCS
"""
The "core" of mantipython
Coordinate everything and fit an entire map.
Takes paths to Herschel maps and fits SEDs in parallel.
Created: January 29, 2020
"""
__author__ = "Ramsey Karim"


def fit_entire_map(data_filenames, bands_to_fit, parameters_to_fit,
    initial_param_vals=None, param_bounds=None, dust='tau', dust_kwargs=None,
    data_directory="", cutout=None):
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
    :param dust: tau or kappa, describing type of dust to use.
    :param dust_kwargs: extra kwargs to be passed to the Dust object.
        For tweaking other power law parameters (kappa0, nu0). Only valid for
        dust='kappa'
    :param data_directory: a string to be prepended to all strings contained
        in data_filenames
    :param cutout: descriptor for fitting a subset of the input maps.
        Can be a Cutout2D that is valid for use on these maps,
        or could be a tuple ((i_center, j_center), (i_width, j_width)).
    """
    # Sanitize list
    bands_to_fit = list(bands_to_fit)
    # Get basic info for input data
    tmp, hdr = fits.getdata(data_directory + data_filenames[bands_to_fit[0]][0],
        header=True)
    original_shape = tmp.shape
    del tmp # Save memory
    # Parse the cutout argument
    if isinstance(cutout, Cutout2D):
        ct_slices = cutout.slices_original
        ct_wcs = cutout.wcs
    else:
        center, new_shape = cutout
        cutout = Cutout2D(np.empty(original_shape), (center[1], center[0]),
            new_shape, wcs=WCS(hdr), copy=True)
        ct_slices = cutout.slices_original
        ct_wcs = cutout.wcs
    # Set up some fit conditions
    initial_guesses = [solve.standard_x0[pn] for pn in parameters_to_fit]
    bounds = [list(solve.standard_bounds[pn]) for pn in parameters_to_fit]
    # That's enough for now.......

    # need a better name
    # should probably implement before picking names... chickens before they hatch

    pass
