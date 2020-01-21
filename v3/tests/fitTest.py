from astropy.io import fits
import pickle
import datetime
import sys

from .. import fit
from ..src.greybody import Greybody
from ..src.dust import TauOpacity
from ..src.instrument import get_instrument

"""
Test the fit routine
"""

# Directory with actual RCW 49 data
# data_dir = "/home/ramsey/Documents/Research/Feedback/ancillary_data/herschel/"
data_dir = "/n/sgraraid/filaments/data/TEST4/pacs70_cal_test/RCW49/processed/1342255009/"
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
wavelens = [70, 160, 250, 350, 500]

LOG_NAME = "./log2.log"
# use this to run nicely
# { python -m v3.tests.fitTest 2>&1; } 1>>log2.log &

param_names = ('T', 'tau')
initial_guesses = [fit.standard_x0[param_name] for param_name in param_names]
bounds = [list(fit.standard_bounds[param_name]) for param_name in param_names]
# bounds[2][1] = 3.0
bounds[0][1] = 35
dust = TauOpacity(2.0)
def src_fn(x):
    # x = [T, tau]
    return Greybody(x[0], x[1], dust)

def log_fn(text):
    with open(LOG_NAME, 'a') as f:
        f.write(str(text)+"\n")

imgs, errs = [], []
width = 8//2
for wl in wavelens:
    img = fits.getdata(data_dir+data_fns[wl])
    i, j = img.shape
    i2, j2 = i//2, j//2
    i2, j2 = 175, 535
    cutout = (slice(i2-width, i2+width), slice(j2-width, j2+width))
    imgs.append(img)
    err = fits.getdata(data_dir+err_fns[wl])
    if wl == 70 or wl == 160:
        err[:] = 1e10
    errs.append(err)
# mask = mask[cutout]
print("Cutout shape:", imgs[0].shape)

t0 = datetime.datetime.now()
with open(LOG_NAME, 'a') as f:
    f.write(f"Cutout shape: {imgs[0].shape}\n")
    f.write(f"started at {t0}\n")
result = fit.fit_full_image(imgs, errs, get_instrument(wavelens), src_fn,
    initial_guesses, bounds, mask=mask,
    chisq=True, log_func=log_fn)
t1 = datetime.datetime.now()
with open(LOG_NAME, 'a') as f:
    f.write(f"finished at {t1}\n")
    f.write(f"took {(t1-t0).total_seconds()/60.} minutes\n")


print('done')
savename = "/home/rkarim/Research/Feedback/ancillary_data/herschel/"
savename = savename + "RCW49large_2p_3b.pkl"
with open(savename, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(result, f)
print('saved')
with open(LOG_NAME, 'a') as f:
    f.write(f"wrote {savename} at {datetime.datetime.now()}\n")
