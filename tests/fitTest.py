from astropy.io import fits
import pickle
import datetime
import sys

from .. import solve
from ..physics.greybody import Greybody
from ..physics.dust import TauOpacity
from ..physics.instrument import get_instrument

"""
Test the fit routine fit_full_image in (formerly) fit.py

Update (1/28/20): this has mostly been used as the "official fitting routine"
Since I actually want to test a rewrite of the fitting routine, now in
solve_map.py, I am probably going to retire this file.
"""
__author__ = "Ramsey Karim"

# Directory with actual RCW 49 data
# data_dir = "/home/ramsey/Documents/Research/Feedback/ancillary_data/herschel/"
data_dir = "/sgraraid/filaments/data/TEST4/pacs70_cal_test/RCW49/processed/1342255009_reproc350/"
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

# CUTOUTS
width_i, width_j = 220//2, 280//2
# center of full region
i0, j0 = 726, 466
# now listing bottom-left corners of each tile
i1, j1 = i0, j0-width_j # top left tile
i2, j2 = i0, j0 # top right tile
i3, j3 = i0-width_i, j0 # bottom right tile
i4, j4 = i0-width_i, j0-width_j # bottom left tile
def make_cutout(*bottom_left_corner):
    i, j = bottom_left_corner
    return (slice(i, i+width_i), slice(j, j+width_j))
cutout1 = make_cutout(i1, j1)
cutout2 = make_cutout(i2, j2)
cutout3 = make_cutout(i3, j3)
cutout4 = make_cutout(i4, j4)

CUTOUT = cutout4
NAME = "RCW49large_350grid_3p_TILE4.pkl"
LOG_NAME = "./log14.log"
# use this to run nicely
# { python -m v3.tests.fitTest 2>&1; } 1>>log2.log &

param_names = ('T', 'tau', 'beta')
initial_guesses = [solve.standard_x0[param_name] for param_name in param_names]
bounds = [list(solve.standard_bounds[param_name]) for param_name in param_names]
# bounds[0][1] = 35
# dust = TauOpacity(2.0)
def src_fn(x):
    # x = [T, tau]
    return Greybody(x[0], x[1], TauOpacity(x[2]))

def log_fn(text):
    with open(LOG_NAME, 'a') as f:
        f.write(str(text)+"\n")

imgs, errs = [], []
width = 8//2
for wl in wavelens:
    img = fits.getdata(data_dir+data_fns[wl])
    imgs.append(img[CUTOUT])
    err = fits.getdata(data_dir+err_fns[wl])
    # if wl == 70 or wl == 160:
    #     err[:] = 1e10
    errs.append(err[CUTOUT])
# mask = mask[CUTOUT]

mask = None # NO MASK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

print("Cutout shape:", imgs[0].shape)

t0 = datetime.datetime.now()
with open(LOG_NAME, 'a') as f:
    f.write(f"Cutout shape: {imgs[0].shape}\n")
    f.write(f"started at {t0}\n")
result = solve.fit_array(imgs, errs, get_instrument(wavelens), src_fn,
    initial_guesses, bounds, mask=mask,
    chisq=True, log_func=log_fn)
t1 = datetime.datetime.now()
with open(LOG_NAME, 'a') as f:
    f.write(f"finished at {t1}\n")
    f.write(f"took {(t1-t0).total_seconds()/60.} minutes\n")


print('done')
savename = "/home/rkarim/Research/Feedback/ancillary_data/herschel/"
savename = savename + NAME
with open(savename, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(result, f)
print('saved')
with open(LOG_NAME, 'a') as f:
    f.write(f"wrote {savename} at {datetime.datetime.now()}\n")
