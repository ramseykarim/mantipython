import numpy as np

from .mpy_utils import cst, H2mass

"""
Greybody
Author: Ramsey Karim
Represents a source model based on a uniform dust cloud.
Implements a RADIATE function that returns flux density at argument frequencies
Needs a dust model (Dust) as input
"""

def B(nu, T):
    # make sure:
    # >> nu [FREQUENCY] in HERTZ
    # >> T [TEMPERATURE] in KELVIN
    exponential = np.exp(cst.h * nu / (cst.k * T)) - 1
    out_front = 2 * cst.h * nu**3 / cst.c**2
    total_value = out_front / exponential
    # CONVERT to MJy/sr
    return total_value * 1e20

class Greybody:
    """
    Single component Greybody
    """
    def __init__(self, temperature, column_density, dust_model):
        self.T = temperature
        self.N = column_density
        self.dust = dust_model

    def radiate(self, nu):
        if not hasattr(nu, "__iter__"):
            nu = (nu,)
        if not isinstance(nu, np.ndarray):
            nu = np.array(nu)
        cross_section = self.dust(nu) * H2mass * self.N
        source = B(nu, self.T)
        # returns a nu-sized array
        return source * (1 - np.exp(-tau))

    def __repr__(self):
        s = "({:.1f}K/{:.1E}/{:s})".format(self.T, self.N, str(self.dust))
        return f"<Greybody:{s}>"

    def __str__(self):
        s = "{:.1f}".format(self.T))
        return f"gb({s})"
