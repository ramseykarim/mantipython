from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy.wcs import WCS
import datetime
import sys
import os

from .. import solve
from ..physics import Greybody, TauOpacity, get_instrument

"""
Test fit_full_image in solve_map.py
Created 1/28/20, after renaming fit.py to solve_map.py
Updated 4/9/20
"""
__author__ = "Ramsey Karim"


"""
Took most of the code below from FITSTest.py on 1/28/20
My plan is to use this file to just test the solver in solve_map.py,
especially the new jacobian implementation
I'll do the more I/O-FITS stuff in FITSTest, and the actual solver stuff will
be running smoothly.
"""
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
mask = None # fits.getdata(data_dir+"dim_region_mask.fits").astype(bool)
wavelens = [70, 160, 250,]
# assume we've set up arguments for a valid Cutout2D
# these would be (center_i, center_j), (full_width_i, full_width_j)

# i0, j0 = 726, 466
# width_i, width_j = 220, 280

i0, j0 = 450, 450
width_i, width_j = 4, 4
cutout_args = ((j0, i0), (width_i, width_j))

img, hdr = fits.getdata(data_dir+data_fns[160], header=True)
ct = Cutout2D(img, *cutout_args, wcs=WCS(hdr), copy=True)
ct_wcs = ct.wcs
ct_slice = ct.slices_original

# set up parameters and bands
# Choose the parameters to use
param_names = ('T', 'tau', 'beta')
# Loop througuh parameters to make init values and bounds
initial_guesses = [solve.standard_x0[pn] for pn in param_names]
bounds = [list(solve.standard_bounds[pn]) for pn in param_names]
# Adjust bounds if needed:
pass

# Loop through bands and apply cutout
imgs, errs = [], []
for wl in wavelens:
    imgs.append(fits.getdata(data_dir+data_fns[wl])[ct_slice])
    errs.append(fits.getdata(data_dir+err_fns[wl])[ct_slice])
if mask is not None:
    mask = Cutout2D(mask, *cutout_args, wcs=WCS(hdr), copy=True)

print('Fit region shape:', imgs[0].shape)
print("center", ct.input_position_cutout, ct.input_position_original)

def src_fn(x):
    # x = [T, tau]
    return Greybody(x[0], x[1], TauOpacity(x[2]))

LOG_NAME = "log.log"
def logger(text):
    with open(LOG_NAME, 'a') as f:
        f.write(text+"\n")

t0 = datetime.datetime.now()
logger(f"\nNew Session!\nCutout shape: {imgs[0].shape}\nstarted at {t0}")
result_dict = solve.fit_array(imgs, errs, get_instrument(wavelens), src_fn,
    initial_guesses,  bounds, mask=None, log_func=logger)
t1 = datetime.datetime.now()
logger(f"finished at {t1}\ntook {(t1-t0).total_seconds()/60.} minutes")
print('done')
for k in result_dict:
    print(k)
    print(result_dict[k])
    print()
