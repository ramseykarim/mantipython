import numpy as np
from .mpy_utils import cst, H2mass

"""
Implements functions for a source model based on a dust cloud divided up into
uniform, symmetrically arranged sections.
Implements a RADIATE function that returns flux density at argument frequencies
Needs a dust model (from .dust) as input
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

def create_radiate_function(nu):
    """
    Takes in frequency nu in Hertz
    """
    if not hasattr(nu, "__iter__"):
        nu = (nu,)
    if not isinstance(nu, np.ndarray):
        nu = np.array(nu)
    # left off here
