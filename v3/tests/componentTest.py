import numpy as np
import matplotlib.pyplot as plt

from ..src.mpy_utils import wl_micron_hz
from ..src import dust
from ..src import greybody

"""
Test the pieces that go into fit.py
"""

def dustTest():
    beta2 = dust.Dust(beta=2.0)
    freqs = np.linspace(wl_micron_hz(600), wl_micron_hz(50), 200, dtype=np.float)
    plt.plot(freqs, beta2(freqs))
    plt.show()

def greybodyTest():
    beta2 = dust.Dust(beta=2.0)
    gb = greybody.Greybody(15, 1e22, beta2)
    freqs = np.linspace(wl_micron_hz(600), wl_micron_hz(50), 200, dtype=np.float)
    plt.plot(freqs, gb.radiate(freqs))
    plt.show()

if __name__ == "__main__":
    dustTest()
    greybodyTest()
