import os

from ..core import fit_entire_map

"""
Test fit_entire_map in core.py.
The directory boilerplate is taken from solveTest.py on Feb 5 2020
This version includes multiprocessing, so this will be interesting!
"""
__author__ = "Ramsey Karim"

# Directory with actual RCW 49 data
# Desktop path
data_dir = "/sgraraid/filaments/data/TEST4/pacs70_cal_test/RCW49/processed/1342255009_reproc350/"
if not os.path.isdir(data_dir):
    # Laptop path (updated 4/9/20 after clean install)
    data_dir = "/home/ramsey/Documents/Research/Feedback/ancillary_data/herschel/processed/1342255009/"
data_fns = {
    70: "PACS70um-image-remapped-conv-plus000080.fits", # -plus000102
    160: "PACS160um-image-remapped-conv-plus000370.fits", # -plus000343
    250: "SPIRE250um-image-remapped-conv.fits",
    350: "SPIRE350um-image-remapped-conv.fits",
    # 500: "SPIRE500um-image-remapped-conv.fits",
}
err_fns = {
    70: "PACS70um-error-remapped-conv-plus6.0pct.fits",
    160: "PACS160um-error-remapped-conv-plus8.0pct.fits",
    250: "SPIRE250um-error-remapped-conv-plus5.5pct.fits",
    350: "SPIRE350um-error-remapped-conv-plus5.5pct.fits",
    # 500: "SPIRE500um-error-remapped-conv.fits",
}

# organize filenames
data_dictionary = {}
for k in data_fns:
    data_dictionary[k] = (data_fns[k], err_fns[k])
# select small cutout area
i0, j0 = 543, 337
width_i, width_j = 15, 15
# decide whether or not this is parallel
n_processes = 6
write_fn = "/home/ramsey/Downloads/test.fits"
d = fit_entire_map(data_dictionary, [70, 160,], ('T', 'tau'),
    data_directory=data_dir, log_name_func=lambda s: f"/home/ramsey/Downloads/log{s}.log",
    n_procs=n_processes, destination_filename=write_fn,
    cutout=((i0, j0), (width_i, width_j)), fitting_function='jac',
)

print("finished with fit")
import matplotlib.pyplot as plt
import numpy as np
# from .. import solve
# Still working...
soln = d['solution']

plt.subplot(121)
plt.imshow(soln[0], origin='lower')
plt.subplot(122)
plt.imshow(d['success'][0], origin='lower')
plt.show()
