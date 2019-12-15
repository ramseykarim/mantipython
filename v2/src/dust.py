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

k0_default = 0.1 # Standard in literature
nu0_default = 1000*1e9 # Traditional default is 1000 GHz
micron_meters = 1e-6



"""
The following code was taken directly from v1
It is the kappa function creation, which will now be the main element
of Dust
"""

def create_kappa_function(*args, **kwargs):
    # First, parse arguments and set up variables
    k0, nu0, beta = None, None, None
    table = None
    if len(args) == 3:
        k0, nu0, beta = args
    elif 'beta' in kwargs:
        beta = kwargs['beta']
        k0 = kwargs.get('k0', None)
        if k0 is None:
                k0 = kwargs.get('kappa0', k0_default)
        nu0 = kwargs.get('nu0', nu0_default)
    if len(args) == 1 or len(args) == 2:
        table = args[0]
    # Next, run through available variables and set up kappa function
    # First, is beta defined?
    if beta is None:
        if table is None:
            raise RuntimeError("Can't init dust model with no information.")
        else:
            powerlaw = fit_power_law(table)
    else:
        powerlaw = init_power_law(k0, nu0, beta)
    # Second, is table defined?
    if table is None:
        spline, table_limits = None, None
    else:
        spline = init_table(table)
        table_limits = (np.min(table[:, 0]), np.max(table[:, 0]))
    # Fit this whole thing together
    # All kappa functions should check for crazy input wavelength/frequency
    # Powerlaw only just yields powerlaw
    # Table is only applied within limits (or else spline will
    #    do strange things)
    errmsg = "That's a really far-out wavelength; are your units right?"
    if spline is None:
        def kappa(nu):
            if np.any(nu > cst.c/(1e-5*micron_meters)) or np.any(nu < cst.c/(5e5*micron_meters)):
                raise RuntimeError(errmsg+"::"+str(cst.c/nu/micron_meters))
            return powerlaw(nu)
    else:
        spline_filter = lambda x: (x > table_limits[0]) & (x < table_limits[1])
        def kappa(nu):
            if np.any(nu > cst.c/(1e-5*micron_meters)) or np.any(nu < cst.c/(5e5*micron_meters)):
                raise RuntimeError(errmsg+"::"+str(cst.c/nu/micron_meters))
            results = np.zeros(nu.shape)
            in_spline = spline_filter(nu)
            results[in_spline] = spline(nu[in_spline])
            results[~in_spline] = powerlaw(nu[~in_spline])
            return results
    # Set the kappa function; callable with frequency in Hertz
    return kappa

def init_table(table):
    # returns spline fit to a table
    # return value is callable as function of frequency in Hz
    nu_column, k_column = table[:, 0], table[:, 1]
    sorted_nu = np.argsort(nu_column)
    nu_column = nu_column[sorted_nu]
    k_column = k_column[sorted_nu]
    spline = UnivariateSpline(nu_column, k_column, s=0) # no smoothing
    return spline

def init_power_law(k0, nu0, beta):
    # returns power law function, callable as function of frequency in Hz
    def powerlaw(nu):
        return k0 * (nu / nu0)**beta
    return powerlaw

def fit_power_law(table):
    # fits a power law to the 100-1000 micron (FIR) part of the table
    nu_column, k_column = table[:, 0], table[:, 1]
    sorted_nu = np.argsort(nu_column)
    nu_column = nu_column[sorted_nu]
    k_column = k_column[sorted_nu]
    fir = (nu_column > cst.c/(1000*micron_meters)) & (nu_column < cst.c/(100*micron_meters))
    beta_fit, k0_fit = np.polyfit(np.log(nu_column/nu0_default),
        np.log(k_column), deg=1)
    k0_fit = np.exp(k0_fit)
    return init_power_law(k0_fit, nu0_default, beta_fit)
