import datetime
import sys

from .. import fit
from ..src.greybody import Greybody
from ..src.dust import TauOpacity
from ..src.instrument import get_instrument

"""
Test a FITS-saving wrapper for the the fit routine.
"""

data_dir = "/sgraraid/filaments/data/TEST4/pacs70_cal_test/RCW49/processed/1342255009/"
data_fns = {
    70: "PACS70um-image-remapped-conv-plus000102.fits",
    160: "PACS160um-image-remapped-conv-plus000343.fits",
    250: "SPIRE250um-image-remapped-conv.fits",
    350: "SPIRE350um-image-remapped-conv.fits",
    500: "SPIRE500um-image-remapped-conv.fits",
}
err_fns = {
    70: "PACS70um-error-remapped-conv.fits",
    160: "PACS160um-error-remapped-conv.fits",
    250: "SPIRE250um-error-remapped-conv.fits",
    350: "SPIRE350um-error-remapped-conv.fits",
    500: "SPIRE500um-error-remapped-conv.fits",
}
mask = fits.getdata(data_dir+"dim_region_mask.fits").astype(bool)

# set up cutout
# Describe cutout
i0, j0 = 470, 200
width_i, width_j = 80, 80
# Make cutout
cutout = (sl(i0, i0+width_i), sl(j0, j0+width_j))
result_shape = (width_i, width_j)

# set up parameters and bands
# Choose the parameters to use
param_names = ('T', 'tau')
# Loop througuh parameters to make init values and bounds
initial_guesses = [fit.standard_x0[pn] for pn in param_names]
bounds = [list(fit.standard_bounds[pn]) for pn in param_names]
# Adjust bounds if needed:
pass
# Choose the bands to use
wavelens = [70, 160, 250, 350, 500]
# Loop through bands and apply cutout
imgs, errs = [], []
for wl in wavelens:
    img = fits.getdata(data_dir+data_fns[wl])
    imgs.append(img[cutout])
    err = fits.getdata(data_dir[err_fns[wl]])
    errs.append(err[cutout])
mask = mask[cutout]

print('Fit region shape:', imgs[0].shape)

"""
Finish the FITS stuff tomorrow :) (rkarim, 1/21/20)
"""
