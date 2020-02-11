from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
import datetime

from . import solve

"""
Utilities for file IO, such as reading and writing FITS files,
managing WCS, and logging progress.
"""
__author__ = "Ramsey Karim"

class Data:

    def __init__(self, path_dict, keys, prefix=None):
        """
        :param path_dict: int->(str(data_path), str(err_path)) dict
        :param keys: sequence indicating the subset of path_dict keys that
            matter.
        """
        self.prefix = prefix if prefix is not None else ""
        self.filenames = {k: path_dict[k] for k in keys}
        self.loaded_data = {}

    def __getitem__(self, key):
        if key not in self.filenames:
            raise RuntimeError(f"{key} not valid Data key")
        elif key not in self.loaded_data:
            self.loaded_data[key] = tuple(fits.getdata(self.prefix+filename)
                for filename in self.filenames[key])
        return self.loaded_data[key]

    def __setitem__(self, key, item):
        self.loaded_data[key] = item

    def __iter__(self):
        return iter(self.filenames)

    def load_all(self):
        for k in self:
            self[k]

    def map(self, f):
        """
        Maps the function f onto all data in self.loaded_data,
            reassinging the return values to self.loaded_data
        :param f: callable function f(d_e, k) -> d_e_modified
            d_e is a tuple(data, error) where data and error
                are numpy arrays of the same shape.
            d_e_modified, same as above
            k is the dict key (probably int) referring to a given band
        """
        for k in self:
            self[k] = f(self[k], k)


def check_size_and_header(filename):
    tmp, hdr = fits.getdata(filename, header=True)
    return tmp.shape, hdr


def make_logger(log_name):
    def logger(message):
        with open(log_name, 'a') as f:
            f.write(message+'\n')
    return logger

def write_result(filename, result_dictionary, data_obj, parameters_to_fit, initial_param_vals, bands_to_fit,
    wcs):
    """
    :param result_dictionary: this should be in the same form as the
        dictionary returned by solve.fit_array, with they keys in
        solve.result_keys
    """
    phdu = fits.PrimaryHDU()
    hdu_list = [phdu]
    for k in result_dictionary:
        n_panels = solve.result_frames[k](len(parameters_to_fit), len(bands_to_fit))
        if n_panels == 1:
            suffixes = ("",)
        elif n_panels == len(parameters_to_fit):
            suffixes = parameters_to_fit
        elif n_panels == len(bands_to_fit):
            suffixes = map(str, bands_to_fit)
        else:
            raise RuntimeError(f"Strange number of panels in {k.upper()}: {n_panels}")
        for i, suffix in zip(range(n_panels), suffixes):
            ihdu = fits.ImageHDU(data=result_dictionary[k][i], header=fits.Header())
            ihdu.header.update(wcs.to_header())
            ihdu.header['EXTNAME'] = k + suffix
            ihdu.header['BUNIT'] = "use your best judgement"
            hdu_list.append(ihdu)
    for k in data_obj:
        for img, stub, idx in zip(data_obj[k], ('', 'd'), range(2)):
            ihdu = fits.ImageHDU(data=img, header=fits.Header())
            ihdu.header['EXTNAME'] = stub+"BAND"+str(int(k))
            ihdu.header['BUNIT'] = "MJy/sr"
            ihdu.header['HISTORY'] = f"Original file: {data_obj.filenames[k][idx]}"
            ihdu.header.update(wcs.to_header())
            hdu_list.append(ihdu)
    phdu.header['DATE'] = (datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(), "File creation date")
    phdu.header['CREATOR'] = (f"mantipython, by {__author__}", "FITS file creator")
    phdu.header['COMMENT'] = f"Bands {','.join(map(str, bands_to_fit))} solved for {','.join(parameters_to_fit)}."
    phdu.header['HISTORY'] = f"Data from:"
    phdu.header['HISTORY'] = data_obj.prefix
    phdu.header.update(wcs.to_header())
    if 'beta' not in parameters_to_fit:
        phdu.header['COMMENT'] = f"Beta set to {initial_param_vals['beta']}"
    hdul = fits.HDUList(hdu_list)
    hdul.writeto(filename, overwrite=True)
