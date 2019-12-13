import numpy as np
import scipy.constants as cst
from scipy.interpolate import UnivariateSpline

"""
Dust represents a dust opacity model.
Implements a KAPPA function that returns opacity (cm2/g) at
    argument frequencies.
Dust object is also callable as KAPPA
"""
__author__ = "Ramsey Karim"
