import numpy as np
from scipy.interpolate import UnivariateSpline

from .mpy_utils import cst

"""
Dust
Various ways to represent dust opacity
"""
__author__ = "Ramsey Karim"


# Constants that ONLY AFFECT DUST
# (Kevin's values are found in solve.cc and manticore.h)

# mass of H2 molecule in grams
# NIST: https://webbook.nist.gov/cgi/inchi/InChI%3D1S/H
H_mass_amu = 1.00794
H2mass = 2.75 * (cst.m_u * H_mass_amu * 1e3) # kg->g
# I asked Kevin where 2.75 comes from (1/20/20):
# "standard" metallicity-corrected conversion we adopted for the N(H2) maps
# an empirical factor supposed to correspond to observations of these regions


k0_default = 0.1 # Standard in literature
nu0_default = 1000*1e9 # Hz; Traditional default is 1000 GHz
micron_meters = 1e-6 # microns * micron_meters = meters
nu0_160 = cst.c / (160 * micron_meters) # frequency in Hz


# Much simpler opacity equation. Power law slope only,
#   normalization dealt with in Greybody
# Can be instantiated and called same as Dust
class TauOpacity:

    def __init__(self, beta=2.0):
        # beta unitless
        # nu0 is fixed at 160micron
        self.beta = beta
        self._short = "beta={:.1f}".format(beta)
        self._text = "<TauOpac:{:s}>".format(self._short)

    def tau(self, nu):
        # nu in Hz
        return (nu / nu0_160)**self.beta

    def dtau_dbeta_helper(self, nu):
        return np.log(nu / nu0_160)

    def __call__(self, x):
        return self.tau(x)

    def __repr__(self):
        return self._text

    def __str__(self):
        return self._short


class ConstantOpacity:

    def __init__(self):
        self._short = "cst"
        self._text = "<ConstantOpac:{:s}>".format(self._short)

    def tau(self, nu):
        return np.ones_like(nu)

    def __call__(self, x):
        return self.tau(x)

    def __repr__(self):
        return self._text

    def __str__(self):
        return self._short


# Formal dust, in kappa (cm2/g)
# This will still work with Greybody (1/20/20)
# The Greybody.t160 parameter will effectively be column density N(H2)
class Dust:

    def __init__(self, *args, **kwargs):
        """
        Represents a dust opacity model.
        Implements a KAPPA function that returns opacity (cm2/g) at
            argument frequencies.
        Dust object is also callable as KAPPA
        Can be constructed with:
            k0, nu0, beta as floats; this creates a power law
            an Nx2 array; the first column is assumed to be frequencies
                in Hz, the second to be kappa values in cm2/g
                This initializes a spline interpolation.
                Values outside the given range will be assigned the power law
                fit to the FIR (100-1000) part of the table.
            Both (use kwargs for k0, nu0, beta): power law will take over for
                table values outside table range
        """
        self._text = ""
        self._short = ""
        # First, parse arguments and set up variables
        if len(args) == 3:
            k0, nu0, beta = args
        elif 'beta' in kwargs:
            beta = kwargs['beta']
            k0 = kwargs.get('k0', None)
            if k0 is None:
                    k0 = kwargs.get('kappa0', k0_default)
            nu0 = kwargs.get('nu0', nu0_default)
        else:
            k0, nu0, beta = None, None, None
        if len(args) == 1 or len(args) == 2:
            table = args[0]
        else:
            table = None
        # Next, run through available variables and set up kappa function
        # First, is beta defined?
        if beta is None:
            if table is None:
                raise RuntimeError("Can't init dust model with no information.")
            else:
                powerlaw = fit_power_law(table)
                self._text += "fit to table"
        else:
            powerlaw = init_power_law(k0, nu0, beta)
            self._text += "{:.1f}(v/{:.0f}GHz)^{:.1f} cm2/g".format(
                k0, nu0*1e-9, beta
            )
        # Second, is table defined?
        if table is None:
            spline, table_limits = None, None
            self._short = "beta={:.1f}".format(beta)
        else:
            spline = init_table(table)
            table_limits = (np.min(table[:, 0]), np.max(table[:, 0]))
            if len(args) == 2:
                self._short = args[1]
            else:
                self._short = "table"
            self._text = self._short + " + " + self._text + " outside"
        self._text = "<Dust:"+self._text+">"
        # Set the kappa function; callable with frequency in Hertz
        self.kappa = init_kappa(powerlaw, spline, table_limits)

    def __call__(self, x):
        # Now applies the H2mass to the result of kappa (1/20/20)
        return self.kappa(x) * H2mass

    def __repr__(self):
        return self._text

    def __str__(self):
        return self._short


def init_kappa(powerlaw, spline, limits):
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
        spline_filter = lambda x: (x > limits[0]) & (x < limits[1])
        def kappa(nu):
            if np.any(nu > cst.c/(1e-5*micron_meters)) or np.any(nu < cst.c/(5e5*micron_meters)):
                raise RuntimeError(errmsg+"::"+str(cst.c/nu/micron_meters))
            results = np.zeros(nu.shape)
            in_spline = spline_filter(nu)
            results[in_spline] = spline(nu[in_spline])
            results[~in_spline] = powerlaw(nu[~in_spline])
            return results
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
