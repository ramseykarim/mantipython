from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.nddata.utils import Cutout2D
from astropy.wcs import WCS
import datetime
import sys

from ..physics.greybody import Greybody
from ..physics.dust import Dust
from ..physics.instrument import get_instrument

data_dir = "/n/sgraraid/filaments/Perseus/Herschel/processed/1342190326/"
def fn_gen(band_stub, offset=""):
    if offset:
        offset = "-plus{:06d}".format(offset)
    return f"{data_dir}{band_stub}-image-remapped-conv{offset}.fits"
data_fns = {
    160: fn_gen("PACS160um", 45),
    250: fn_gen("SPIRE250um"),
    350: fn_gen("SPIRE350um"),
    500: fn_gen("SPIRE500um"),
}

result_dir = "/n/sgraraid/filaments/Perseus/Herschel/results/"
result_fn = "full-1.5.1-Per1-pow-1000-0.1-1.80.fits"
result_path = f"{result_dir}{result_fn}"

frame_key = {}
with fits.open(result_path) as hdul:
    for i in range(1, len(hdul)):
        frame_key[hdul[i].header['EXTNAME']] = i
    w = WCS(hdul[1].header)
    T, N = (hdul[frame_key[k]].data for k in ('T', 'N(H2)'))
    wavelens = sorted(list(data_fns.keys()))
    mant_diffs = [hdul[frame_key[f'diff{b}']].data for b in wavelens]
    mant_bands = [hdul[frame_key[f'BAND{b}']].data for b in wavelens]
mant_models = [d+b for d, b in zip(mant_diffs, mant_bands)]

if False:
    center = tuple(x//2 for x in mant_models[0].shape)
    width = 20
    c2d = Cutout2D(T, (center[1], center[0]), (width, width), wcs=w)
    cutout = c2d.slices_original
    for i in range(len(wavelens)):
        for l in (mant_diffs, mant_bands, mant_models):
            l[i] = l[i][cutout]
    T, N = T[cutout], N[cutout]
    w = c2d.wcs

all_things = {f'mant_diff{b}': d for b, d in zip(wavelens, mant_diffs)}
all_things.update({f'BAND{b}': d for b, d in zip(wavelens, mant_bands)})
all_things.update({f'manticore{b}': d for b, d in zip(wavelens, mant_models)})
all_things.update({'T': T, 'N': N})


plt.figure()
for i, m in enumerate(mant_models):
    plt.subplot(221 + i)
    plt.title(f"{wavelens[i]}")
    plt.imshow(m, origin='lower', vmax=np.nanmedian(m)+2*np.nanstd(m))
plt.gcf().canvas.set_window_title("Observations")
plt.figure()
for i, m in enumerate([T, N]):
    if i:
        m = np.log10(m)
    plt.subplot(121 + i)
    plt.title("N" if i else "T")
    plt.imshow(m, origin='lower', vmax=np.nanmedian(m)+2*np.nanstd(m))
    plt.colorbar()
plt.gcf().canvas.set_window_title("Model parameters")

mantipy_result = np.full((len(wavelens), T.size), np.nan)
herschel = get_instrument(wavelens)
dust = Dust(beta=1.80)
for i, t, n in zip(range(T.size), T.flat, N.flat):
    if np.isnan(t) | np.isnan(n):
        continue
    else:
        mantipy_result[:, i] = [d.detect(Greybody(t, np.log10(n), dust)) for d in herschel]
mantipy_result = mantipy_result.reshape(len(wavelens), *(T.shape))
mantipy_result = [mantipy_result[i] for i in range(len(wavelens))]

plt.figure()
for i, m in enumerate(mantipy_result):
    all_things[f'py{wavelens[i]}'] = m
    plt.subplot(221 + i)
    plt.title(f"{wavelens[i]}")
    plt.imshow(m, origin='lower', vmax=np.nanmedian(m)+2*np.nanstd(m))
plt.gcf().canvas.set_window_title("mantipython results")

plt.figure()
for i, m_py in enumerate(mantipy_result):
    m_core = mant_models[i]
    diff = m_py - m_core
    all_things[f'pycorediff{wavelens[i]}'] = diff
    difffrac = diff / m_core
    all_things[f'pycorediff_frac{wavelens[i]}'] = difffrac
    plt.subplot(221 + i)
    plt.title(f"{wavelens[i]}")
    m = difffrac
    plt.imshow(m, origin='lower', vmax=np.nanmedian(m)+2*np.nanstd(m))
plt.gcf().canvas.set_window_title("final diffs")

phdu = fits.PrimaryHDU()
phdu.header['DATE'] = (datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(), "File creation date")
phdu.header['CREATOR'] = ("Ramsey: {}".format(str(__file__)), "FITS file creator")
phdu.header['HISTORY'] = "This is a comparison on Per1 of manticore vs mantipython"
phdu.header['HISTORY'] = "manticore: {:s}".format(result_fn)
phdu.header.update(w.to_header())
hdu_list = [phdu]
for k, v in all_things.items():
    ihdu = fits.ImageHDU(data=v, header=fits.Header())
    ihdu.header.update(w.to_header())
    ihdu.header['EXTNAME'] = k
    ihdu.header['BUNIT'] = "MJy/sr unless K or cm-2. unitless if frac."
    if "py" in k:
        ihdu.header['HISTORY'] = "Product of mantipython (Ramsey)"
    elif "mant" in k:
        ihdu.header['HISTORY'] = "Product of manticore (Kevin)"
    hdu_list.append(ihdu)
hdulnew = fits.HDUList(hdu_list)
hdulnew.writeto("/home/rkarim/Research/mantipython/compare_manticore_2.fits", overwrite=True)
# plt.show()

print('done')
