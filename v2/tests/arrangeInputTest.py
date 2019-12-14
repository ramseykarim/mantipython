import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

"""
Test suite for rearranging the Herschel data and placing in memmap
"""

def print_extensions(hdul):
    # print extension names
    for i, hdu in enumerate(hdul):
        if i == 0:  # skip PHDU
            continue
        print(f"{i}: {hdu.header['EXTNAME']}")


# load data from an old manticore run on L723
test_file = "full-1.5-L723-pow-1000-0.1-2.10.fits"
with fits.open(test_file) as hdul:
    images = []
    wavelens = []
    original_shape = tuple(hdul[1].header[f'NAXIS{k}'] for k in (2, 1))
    for i in range(12, len(hdul), 2):
        images.append(hdul[i].data.ravel())
        wavelens.append(int(hdul[i].header['EXTNAME'].replace("BAND", "")))

data = np.moveaxis(np.stack(images), 0, -1)

# plot loaded data
def confirm_data_exists():
    for i in range(len(images)):
        plt.subplot(221 + i)
        plt.imshow(data[:, i].reshape(original_shape), origin='lower')
        plt.title(wavelens[i])
    plt.show()

# np.memmap('data.dat', dtype=np.float, mode='r', shape=(size, size))

"""
for cache size, do:
    sudo lshw -C memory
"""

if __name__ == "__main__":
    print(data._type_)
    print(data.shape)
