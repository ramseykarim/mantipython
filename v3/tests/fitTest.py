from astropy.io import fits
import pickle
import datetime

from .. import fit
from ..src.greybody import Greybody
from ..src.dust import Dust
from ..src.instrument import get_instrument

"""
Test the fit routine
"""

# Directory with actual RCW 49 data
data_dir = "/home/ramsey/Documents/Research/Feedback/ancillary_data/herschel/"
data_fns = {
    70: "PACS70um-image-remapped-conv-plus000184.fits",
    160: "PACS160um-image-remapped-conv-plus000615.fits",
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
wavelens = [70, 160, 250, 350, 500]

# dust180 = Dust(beta=1.80)
def src_fn(x):
    # x = [T, N]
    return Greybody(x[0], x[1], Dust(beta=x[2]))


imgs, errs = [], []
width = 8//2
for wl in wavelens:
    img = fits.getdata(data_dir+data_fns[wl])
    i, j = img.shape
    i2, j2 = i//2, j//2
    cutout = (slice(i2-width, i2+width), slice(j2-width, j2+width))
    imgs.append(img)
    err = fits.getdata(data_dir+err_fns[wl])
    errs.append(err)
print("Cutout shape:", imgs[0].shape)

t0 = datetime.datetime.now()
with open("./log.log", 'a') as f:
    f.write(f"Cutout shape: {imgs[0].shape}\n")
    f.write(f"started at {t0}\n")
result = fit.fit_full_image(imgs, errs, get_instrument(wavelens), src_fn,
    [10, 20, 1.80], ((0, None), (18, 25), (1, 2.5)), chisq=False)
t1 = datetime.datetime.now()
with open("./log.log", 'a') as f:
    f.write(f"finished at {t1}\n")
    f.write(f"took {(t1-t0).total_seconds()/60.} minutes\n")


print('done')
with open('./test1.pkl', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(result, f)
print('saved')
with open("./log.log", 'a') as f:
    f.write(f"written at {datetime.datetime.now()}\n")
