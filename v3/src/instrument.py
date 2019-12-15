import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad
QUAD_ARGS = {'limit': 512, 'epsrel': 1e-6}

from .mpy_utils import cst, H_WL, H_stub, valid_wavelength
from .greybody import B

"""
Instrument
Author: Ramsey Karim
Represents a single filter/detector combination on some instrument.
Implements a DETECT function that returns an estimated flux density at a
    stated reference wavelength using a calculated filter width.
Needs access to a filter profile.
The GREYBODY passed to DETECT must implement a RADIATE function.
"""

from .computer_config import bandpass_directory, p_RIMO

"""
The argument "wl" should ALWAYS be an integer in microns
"""

@valid_wavelength
def bandpass_filename(wl):
    return f"{bandpass_directory}{H_stub(wl)}_fromManticore.dat"

@valid_wavelength
def bandpass_center(wl):
    return cst.c/(wl*1e-6)

SPIRE_gamma = -0.85*2
SPIRE_Omegaeff = {
    # SPIRE effective beam area for spectral index -1
    # units: arcsec^2
	250: 469.35,
	350: 831.27,
	500: 1804.31,
}


@valid_wavelength
def open_bandpass(wl):
    # Open the bandpasses I copied from manticore
    data = np.genfromtxt(bandpass_filename(wl))
    nu, weight = data[:, 0], data[:, 1]
    nu_sorted = np.argsort(nu)
    nu = nu[nu_sorted]
    weight = weight[nu_sorted]
    return nu, weight


class Instrument:

    def __init__(self, band_wavelength):
        # Gather relevant information about the band and set up useful things
        #  like a spline fit to the filter profile
        self.wavelength = band_wavelength
        self.name = H_stub(band_wavelength)
        self.freq_array, self.response_array = open_bandpass(band_wavelength)
        self.center = bandpass_center(band_wavelength)
        if 'SPIRE' in self.name:
            self.response_array *= beam_size(band_wavelength, self.freq_array)

        self.response = UnivariateSpline(self.freq_array,
            self.response_array, s=0)

        self.freq_limits = (np.min(self.freq_array), np.max(self.freq_array))
        self.filter_width = self.calculate_width()

    def integrate_response(self, f):
        # f is a function of frequency
        # f will be integrated over filter response
        f_response_array = self.response_array * f(self.freq_array)
        return np.trapz(f_response_array, x=self.freq_array)

    def calculate_width(self):
        # Copying Kevin's method from manticore
        # integrates response(nu)/nu
        return self.integrate_response(np.reciprocal) * self.center

    def detect(self, gb):
        # Replicating manticore method
        # gb must have RADIATE method (Greybody)
        return self.integrate_response(gb.radiate) / self.filter_width

    def __repr__(self):
        return "<Instrument: {:s} band>".format(self.name)

    def __str__(self):
        return self.name


@valid_wavelength
def beam_size(wl, nu):
    if wl < 200:
        return None
    # Beam size per frequency nu
    nu0 = bandpass_center(wl)
    beam0 = SPIRE_Omegaeff[wl]
    return beam0 * (nu / nu0)**SPIRE_gamma


def get_Herschel():
    return [Instrument(wl) for wl in H_WL]
