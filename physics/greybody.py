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
    def __init__(self, temperature, normalization, dust_model, p=3):
        """
        :param temperature: temperature in K
        :param normalization: either tau160 or kappa0 depending on the
            dust model class passed as dust_model. GIVEN IN LOG10!
        :param dust_model: either TauOpacity or Dust instance, dust model to use
        :param p: number of parameters, only used if taking derivatives
        """
        self.T = temperature*1. # make sure *everything* is a float....
        # Bug (1/28/20): had a problem with 10**tau vs 10.**tau,
        #  int infected(?) float and caused np.exp(-tau) to fail.
        self.normalization = 10.**normalization # arg as LOG10
        self.dust = dust_model
        # nparams only matters for taking derivatives
        self.nparams = p

    def tau(self, nu):
        # Convenience function for calculating optical depth
        nu = arg_as_array(nu)
        return self.dust(nu) * self.normalization

    def radiate(self, nu, expntau=None):
        """
        Calculate emission in MJy/sr at frequencies given by nu array
        """
        source = B(nu, self.T)
        # norm is either tau160 or column density, depending if
        #   Dust or TauOpacity is used
        if expntau is None:
            # (for MultiGreybody: option to pass in tau)
            expntau = np.exp(-self.tau(nu))
        # returns a nu-sized array
        return source * (1 - expntau)

    def dradiate(self, nu):
        # Return a (2 or 3, *nu.shape) array for T, tau(, beta)
        # self.dust(nu) just returns (nu/nu0)**beta
        tau = self.tau(nu)
        exptau = np.exp(-tau)
        dradiate_dT = dB_dT(nu, self.T) * (1 - exptau)
        # we need the derivative of LOG10(normalization)
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
        s = "({:.1f}K/{:.1E}/{:s})".format(self.T, self.normalization, str(self.dust))
        return f"<Greybody:{s}>"

    def __str__(self):
        s = "{:.1f}".format(self.T)
        return f"gb({s})"



class ThinGreybody:
    """
    Single component Greybody, but OPTICALLY THIN

    This class was created on March 16, 2021 to allow me to do math with
    the linear optically thin approximation.

    Most of the class is copied from Greybody. This class does not support
    any partial derivatives.

    The "optically thin" approximation just means that, instead of using the
    normalization (C) to do:
        tau = C * dust.tau
    and then putting that tau into (1 - exp(-tau)) to multiply against the BB,
    instead we use C*dust.tau directly in place of (1 - exp(-tau)) and multiply
    C*dust.tau directly against the BB. If C = 1 and dust.tau = Constant, this
    "ThinGreybody" can be used to model an optically thick blackbody.
    Theoretically, you could use dust.tau = dust.ConstantOpacity to bypass
    the automatic handling of wavelength dependance and put whatever you want
    as C to be multiplied against the BB

    So really, this isn't necessarily optically thin, it's the "optically thin
    approximation"
    """
    def __init__(self, temperature, normalization, dust_model, p=3):
        """
        :param temperature: temperature in K
        :param normalization: either tau160 or kappa0 depending on the
            dust model class passed as dust_model. GIVEN IN LOG10!
        :param dust_model: either TauOpacity or Dust instance, dust model to use
        :param p: number of parameters, only used if taking derivatives
        """
        self.T = temperature*1. # make sure *everything* is a float....
        # Bug (1/28/20): had a problem with 10**tau vs 10.**tau,
        #  int infected(?) float and caused np.exp(-tau) to fail.
        self.normalization = 10.**normalization # arg as LOG10
        self.dust = dust_model
        try:
            assert "Dust" not in repr(self.dust)
        except:
            raise RuntimeError("Can't use a kappa model in this ThinGreybody.")
        # nparams only matters for taking derivatives
        self.nparams = p

    def tau(self, nu):
        # Convenience function for calculating optical depth
        nu = arg_as_array(nu)
        return self.dust(nu) * self.normalization

    def radiate(self, nu, expntau=None):
        """
        Calculate emission in MJy/sr at frequencies given by nu array
        """
        source = B(nu, self.T)
        # Only works with TauOpacity, not kappa
        # returns a nu-sized array
        return source * self.tau(nu)

    def dradiate(self, nu):
        raise NotImplementedError(f"Derivative for ThinGreybody.")

    def __call__(self, x):
        # Calls radiate by default
        return self.radiate(x)

    def __repr__(self):
        s = "({:.1f}K/{:.1E}/{:s})".format(self.T, self.normalization, str(self.dust))
        return f"<ThinGreybody:{s}>"

    def __str__(self):
        s = "{:.1f}".format(self.T)
        return f"tgb({s})"



class MultiGreybody:

    def __init__(self, temperatures, normalizations, dust_models, p=None):
        """
        See Greybody documentation for description of these parameters.
        These are the same arguments, but passed as lists.
        It is assumed that the 0-index is the most background layer and the
            last element is the most foreground layer.
        :param temperatures: list of temperatures
        :param normalizations: list of normalizations
        :param dust_models: list of Dust or TauOpacity instances
        :param p: not used in this class
        """
        if not len(temperatures) == len(normalizations) == len(dust_models):
            txt = f"T: {len(temperatures)}, norm: {len(normalizations)}, dust: {len(dust_models)}"
            raise RuntimeError("Mismatched numbers of parameters: " + txt)
        self.greybodies = [Greybody(t, norm, dust, p=None) for t, norm, dust in zip(temperatures, normalizations, dust_models)]

    def radiate(self, nu):
        # Loops thru the greybody layers and does radiative transfer thru each
        total_emission = np.zeros_like(nu)
        for gb in self.greybodies:
            expntau = np.exp(-gb.tau(nu))
            total_emission = total_emission*expntau + gb.radiate(nu, expntau=expntau)
        return total_emission

    def dradiate(self, nu):
        raise NotImplementedError("No derivatives for MultiGreybody yet.")

    def __call__(self, x):
        return self.radiate(x)

    def __repr__(self):
        s = "->".join([gb.__repr__() for gb in self.greybodies])
        return f"Multi[{s}]"

    def __str__(self):
        s = ",".join([gb.__str__() for gb in self.greybodies])
        return f"multi[{s}]"
