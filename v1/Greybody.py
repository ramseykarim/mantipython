import numpy as np
from mpy_utils import cst, H2mass

"""
Greybody
Author: Ramsey Karim
Represents a source model based on a dust cloud divided up into uniform,
    symmetrically arranged sections.
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

    def __init__(self, temperatures, column_densities, dust_models):
        """
        Make sure temperatures, column_densities, and dust_models are the
            same size
        The order should be outermost-to-innermost. Generally, hottest to
            coldest.
        """
        if hasattr(temperatures, '__iter__'):
            self.Ts = temperatures
            assert hasattr(column_densities, '__iter__')
            self.Ns = column_densities
            assert hasattr(dust_models, '__iter__')
            self.dusts = dust_models
        else:
            self.Ts = [temperatures]
            assert not hasattr(column_densities, '__iter__')
            self.Ns = [column_densities]
            assert not hasattr(dust_models, '__iter__')
            self.dusts = [dust_models]

    def attributes(self):
        # Return a list of tuples of (T, N, dust) for each layer,
        #  hottest to coldest
        # Useful if combining greybodies into a larger one
        return list(zip(self.Ts, self.Ns, self.dusts))

    def radiate(self, nu):
        if not hasattr(nu, "__iter__"):
            nu = (nu,)
        if not isinstance(nu, np.ndarray):
            nu = np.array(nu)
        taus, sources = [], []
        for T, N, kappa in self.attributes():
            cross_sections = kappa(nu) * H2mass
            taus.append(N * cross_sections)
            sources.append(B(nu, T))
        taus = taus + taus[-2::-1]
        sources = sources + sources[-2::-1]
        cumulative_tau, cumulative_source = 0., 0.
        for t, S in zip(taus, sources):
            cumulative_source += S * (1 - np.exp(-t)) * np.exp(-cumulative_tau)
            cumulative_tau += t
        # This is a frequency-sized array
        return cumulative_source

    def __repr__(self):
        s = []
        for T, N, kappa in self.attributes():
            s.append("({:.1f}K/{:.1E}/{:s})".format(T, N, str(kappa)))
        return "<Greybody:" + "&".join(s) + ">"

    def __str__(self):
        s = []
        for T in self.Ts:
            s.append("{:.1f}".format(T))
        return "gb({:s})".format("/".join(s))
