
"""
Utilities for dividing a map and handling the WCS.
Created January 29 2020
"""
__author__ = "Ramsey Karim"



class Data:

    def __init__(self, path_dict, keys, prefix=""):
        """
        :param path_dict: int->(str(data_path), str(err_path)) dict
        :param keys: sequence indicating the subset of path_dict keys that
            matter.
        """
        self.prefix = prefix
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

    def map(self, f):
        for k in self:
            self[k] = f(self[k], k)
