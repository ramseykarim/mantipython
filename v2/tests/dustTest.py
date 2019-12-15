import numpy as np
import matplotlib.pyplot as plt

from ..src import dust
# Use python -m v2.tests.dustTest

"""
test suite for dust opacity functions
"""
__author__ = "Ramsey Karim"


def make_power_law():
    beta2 = dust.create_kappa_function(beta=2.0)
    freqs = np.arange(800, 1100, dtype=np.float)*1e9
    plt.plot(freqs, beta2(freqs))
    plt.show()

if __name__ == "__main__":
    make_power_law()
