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
    # Laptop path
    data_dir = "/home/ramsey/Documents/Research/Feedback/ancillary_data/herschel/"
data_fns = {
    70: "PACS70um-image-remapped-conv-plus000184.fits", # -plus000102
    160: "PACS160um-image-remapped-conv-plus000615.fits", # -plus000343
    250: "SPIRE250um-image-remapped-conv.fits",
    350: "SPIRE350um-image-remapped-conv.fits",
    # 500: "SPIRE500um-image-remapped-conv.fits",
}
err_fns = {
    70: "PACS70um-error-remapped-conv.fits",
    160: "PACS160um-error-remapped-conv.fits",
    250: "SPIRE250um-error-remapped-conv.fits",
    350: "SPIRE350um-error-remapped-conv.fits",
    # 500: "SPIRE500um-error-remapped-conv.fits",
}

# organize filenames
data_dictionary = {}
for k in data_fns:
    data_dictionary[k] = (data_fns[k], err_fns[k])
# select small cutout area
i0, j0 = 150, 150
width_i, width_j = 40, 40
# decide whether or not this is parallel
n_processes = 4
write_fn = "/home/ramsey/Downloads/test.fits"
fit_entire_map(data_dictionary, [70, 160, 250, 350], ('T', 'tau'),
    data_directory=data_dir, log_name_func=lambda s: f"/home/ramsey/Downloads/log{s}.log",
    n_procs=n_processes, destination_filename=write_fn,
    cutout=((i0, j0), (width_i, width_j)),
)
