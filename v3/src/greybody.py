import numpy as np

from .mpy_utils import cst

"""
Greybody
Author: Ramsey Karim
Represents a source model based on a uniform dust cloud.
Implements a RADIATE function that returns flux density at argument frequencies
Needs a dust model (Dust) as input
"""

MJysr = 1e20

def B(nu, T):
    # make sure:
    # >> nu [FREQUENCY] in HERTZ
    # >> T [TEMPERATURE] in KELVIN
    expm1 = np.exp(cst.h * nu / (cst.k * T)) - 1
    out_front = 2 * cst.h * nu**3 / cst.c**2
    total_value = out_front / expm1
    # CONVERT from SI/sr to MJy/sr (1e26 / 1e6)
    return total_value * MJysr


def dB_dT(nu, T):
    # same rules as above
    # first derivative of B wrt T
    # see pg 135 of my notebook
    hv = cst.h * nu
    expm1 = np.exp(hv / (cst.k * T)) - 1
    out_front = hv * nu / (cst.c * T)
    out_front = (2/cst.k) * (out_front**2) * (expm1+1)
    total_value = out_front / (expm1**2)
    return total_value * MJysr


class Greybody:
    """
    Single component Greybody
    """
    def __init__(self, temperature, tau160, dust_model):
        self.T = temperature
        self.tau160 = 10**tau160 # arg as LOG10
        self.dust = dust_model

    def radiate(self, nu):
        if not hasattr(nu, "__iter__"):
            nu = (nu,)
        if not isinstance(nu, np.ndarray):
            nu = np.array(nu)
        # Tau can still "act like" column density, if
        #   Dust is used instead of TauOpacity
        tau = self.dust(nu) * self.tau160
        source = B(nu, self.T)
        # returns a nu-sized array
        return source * (1 - np.exp(-tau))

    def __repr__(self):
        s = "({:.1f}K/{:.1E}/{:s})".format(self.T, self.tau160, str(self.dust))
        return f"<Greybody:{s}>"

    def __str__(self):
        s = "{:.1f}".format(self.T)
        return f"gb({s})"
