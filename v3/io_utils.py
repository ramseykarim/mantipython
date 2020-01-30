from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D

"""
Utilities for file IO, such as reading and writing FITS files,
managing WCS, and logging progress.
"""

class Data:

    def __init__(self, path_dict, keys, prefix=None):
        """
        :param path_dict: int->(str(data_path), str(err_path)) dict
        :param keys: sequence indicating the subset of path_dict keys that
            matter.
        """
        self.prefix = prefix if prefix is not None else ""
        self.filenames = {k: data_dict[k] for k in keys}
        self.loaded_data = {}

    def __getitem__(self, x):
        if x not in self.filenames:
            raise RuntimeError(f"{x} not valid Data key")
        elif x not in self.loaded_data:
            self.loaded_data[x] = tuple(fits.getdata(prefix+filename)
                for filename in self.filenames[x])
        return self.loaded_data[x]

    def __iter__(self):
        return iter(self.loaded_data)

    def load_all(self):
        for k in self:
            self[k]

    def map(self, f):
        for k in self:
            self[k] = f(self[k], k)



def check_size_and_header(filename):
    tmp, hdr = fits.getdata(filename, header=True)
    return tmp.shape, hdr
