import datetime
import sys
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy.wcs import WCS

from .. import solve
from ..physics import Greybody, TauOpacity, get_instrument

"""
Test a FITS-saving wrapper for the the fit routine.

CURRENTLY NOT FUNCTIONAL. need to figure out a good work flow for these fits
best parallelization is via command line (processes), so need to
integrate bash shell script into this pipeline. then python script cleans up after it.
whole thing must thus be called from bash script

need to figure out how to run 4 things at the same time and then wait for all of them to be done before continuing
Update (1/28/20): can do something like ~$ sleep 2 & sleep 2 & sleep 2 & ; wait
"""
__author__ = "Ramsey Karim"

data_dir = "/n/sgraraid/filaments/data/TEST4/pacs70_cal_test/RCW49/processed/1342255009_reproc350/"
data_fns = {
    70: "PACS70um-image-remapped-conv-plus000102.fits", # -plus000102
    160: "PACS160um-image-remapped-conv-plus000343.fits", # -plus000343
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
mask = None # fits.getdata(data_dir+"dim_region_mask.fits").astype(bool)
wavelens = [70, 160, 250, 350]
# assume we've set up arguments for a valid Cutout2D
# these would be (center_i, center_j), (full_width_i, full_width_j)
i0, j0 = 726, 466
width_i, width_j = 220, 280
cutout_args = ((j0, i0), (width_i, width_j))

# set up parameters and bands
# Choose the parameters to use
param_names = ('T', 'tau')
# Loop througuh parameters to make init values and bounds
initial_guesses = [fit.standard_x0[pn] for pn in param_names]
bounds = [list(fit.standard_bounds[pn]) for pn in param_names]
# Adjust bounds if needed:
pass
import numpy as np
import matplotlib.pyplot as plt

# Loop through bands and apply cutout
imgs, errs = [], []
for wl in [70]:
    img, hdr = fits.getdata(data_dir+data_fns[wl], header=True)
    plt.subplot(121)
    plt.imshow(np.log10(img), origin='lower', vmin=1, vmax=4)
    ct = Cutout2D(img, *cutout_args, wcs=WCS(hdr), copy=True)
    imgs.append(ct)
    plt.subplot(122)
    plt.imshow(np.log10(ct.data), origin='lower', vmin=1, vmax=4)
    plt.show()
    err, hdr = fits.getdata(data_dir+err_fns[wl], header=True)
    errs.append(Cutout2D(err, *cutout_args, wcs=WCS(hdr), copy=True))
if mask is not None:
    mask = Cutout2D(mask, *cutout_args, wcs=WCS(hdr), copy=True)
################## WORK IN PROGRESS
print('Fit region shape:', imgs[0].shape)
print("center", imgs[0].input_position_cutout, imgs[0].input_position_original)

"""
Finish the FITS stuff tomorrow :) (rkarim, 1/21/20)
hah (rkarim, 1/28/20)
I'm not planning on finishing this file (rkarim, 2/5/20)
"""
