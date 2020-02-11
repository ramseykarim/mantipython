import numpy as np

from .mpy_utils import cst, arg_as_array

"""
Greybody
Represents a source model based on a uniform dust cloud.
Implements a RADIATE function that returns flux density at argument frequencies
Implements a DRADIATE function for first derivatives of two or three
    GREYBODY parameters. DRADIATE only works if dust.TauOpacity is the
    dust model.
Needs a dust model (Dust) as input
"""
__author__ = "Ramsey Karim"

MJysr = 1e20
log10 = np.log(10.)

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
    out_front = hv * nu / (cst.c * T) # h v**2 / cT
    out_front = (2/cst.k) * (out_front**2) * (expm1+1)
    total_value = out_front / (expm1**2)
    return total_value * MJysr


class Greybody:
    """
    Single component Greybody
    """
    def __init__(self, temperature, tau160, dust_model, p=3):
        self.T = temperature*1. # make sure *everything* is a float....
        # Bug (1/28/20): had a problem with 10**tau vs 10.**tau,
        #  int infected(?) float and caused np.exp(-tau) to fail.
        self.tau160 = 10.**tau160 # arg as LOG10
        self.dust = dust_model
        # nparams only matters for taking derivatives
        self.nparams = p

    def radiate(self, nu):
        nu = arg_as_array(nu)
        # Tau can still "act like" column density, if
        #   Dust is used instead of TauOpacity
        tau = self.dust(nu) * self.tau160
        source = B(nu, self.T)
        # returns a nu-sized array
        return source * (1 - np.exp(-tau))

    def dradiate(self, nu):
        nu = arg_as_array(nu)
        # Return a (2 or 3, *nu.shape) array for T, tau(, beta)
        # self.dust(nu) just returns (nu/nu0)**beta
        tau = self.dust(nu) * self.tau160
        exptau = np.exp(-tau)
        dradiate_dT = dB_dT(nu, self.T) * (1 - exptau)
        # we need the derivative of LOG10(tau160)
        dradiate_dtau_base = B(nu, self.T) * exptau * tau
        if self.nparams == 3:
            return np.array([
                dradiate_dT,
                dradiate_dtau_base*log10,
                dradiate_dtau_base*self.dust.dtau_dbeta_helper(nu)
            ])
        elif self.nparams == 2:
            return np.array([
                dradiate_dT,
                dradiate_dtau_base*log10,
            ])
        else:
            raise NotImplementedError(f"Derivative for {self.nparams} fit parameters.")

    def __call__(self, x):
        # Calls radiate by default
        return self.radiate(x)

    def __repr__(self):
        s = "({:.1f}K/{:.1E}/{:s})".format(self.T, self.tau160, str(self.dust))
        return f"<Greybody:{s}>"

    def __str__(self):
        s = "{:.1f}".format(self.T)
        return f"gb({s})"
