import numpy as np
from mpy_utils import cst, H_WL, H_stubs
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad
QUAD_ARGS = {'limit': 512, 'epsrel': 1e-6}
from Greybody import B

"""
Instrument
Author: Ramsey Karim
Represents a single filter/detector combination on some instrument.
Implements a DETECT function that returns an estimated flux density at a
    stated reference wavelength using a calculated filter width.
Needs access to a filter profile.
The GREYBODY passed to DETECT must implement a RADIATE function.
"""

from computer_config import bandpass_directory, p_RIMO
spire_stub = lambda x: "Herschel_SPIRE.P"+x+"W_ext.dat"
kevin_bp_fn = lambda b: "{}_fromManticore.dat".format(b)
bandpass_files_kevin = {
    # Copied these from manticore
	"PACS160um": kevin_bp_fn("PACS160um"),
	"SPIRE250um": kevin_bp_fn("SPIRE250um"),
	"SPIRE350um": kevin_bp_fn("SPIRE350um"),
	"SPIRE500um": kevin_bp_fn("SPIRE500um"),
}
bandpass_files_SVO = {
    # Downloaded these from the SVO filter service
    # These may be lacking further correction
    # i.e. bolometer response, transmission, or whatever
	"PACS160um": "Herschel_Pacs.red.dat",
	"SPIRE250um": spire_stub("S"),
	"SPIRE350um": spire_stub("M"),
	"SPIRE500um": spire_stub("L"),
}

bandpass_centers = {
	# Angstroms, converted to Hz
	# "PACS160um": cst.c/(1539451.3*1e-10),
	# "SPIRE250um": cst.c/(2471245.1*1e-10),
	# "SPIRE350um": cst.c/(3467180.4*1e-10),
	# "SPIRE500um": cst.c/(4961067.7*1e-10),
	"PACS160um": cst.c/(160*1e-6),
	"SPIRE250um": cst.c/(250*1e-6),
	"SPIRE350um": cst.c/(350*1e-6),
	"SPIRE500um": cst.c/(500*1e-6),
}

SPIRE_gamma = -0.85*2
SPIRE_Omegaeff = {
    # SPIRE effective beam area for spectral index -1
    # units: arcsec^2
	"SPIRE250um": 469.35,
	"SPIRE350um": 831.27,
	"SPIRE500um": 1804.31,
}


def open_SVO_bandpass(stub):
    # Open SVO bandpasses
    data = np.genfromtxt(bandpass_directory + bandpass_files_SVO[stub])
    wavelengths_Angst, weight = data[:, 0], data[:, 1]
    nu = cst.c/(wavelengths_Angst*1e-10)
    nu_sorted = np.argsort(nu)
    nu = nu[nu_sorted]
    weight = weight[nu_sorted]
    return nu, weight


def open_kevin_bandpass(stub):
    # Open the bandpasses I copied from manticore
    data = np.genfromtxt(bandpass_directory + bandpass_files_kevin[stub])
    nu, weight = data[:, 0], data[:, 1]
    nu_sorted = np.argsort(nu)
    nu = nu[nu_sorted]
    weight = weight[nu_sorted]
    return nu, weight


def open_bandpass(stub, version='kevin'):
    # Open either kind of bandpass file
    if version == 'kevin':
        return open_kevin_bandpass(stub)
    elif version == 'SVO':
        return open_SVO_bandpass(stub)
    else:
        raise RuntimeError("{:s} not a recognized source.".format(version))

class Instrument:

    def __init__(self, band_stub):
        # Gather relevant information about the band and set up useful things
        #  like a spline fit to the filter profile
        self.name = band_stub
        self.freq_range, self.response_range = open_bandpass(band_stub)
        self.center = bandpass_centers[band_stub]
        self.beam = gen_beam_function(band_stub)
        self.response = UnivariateSpline(self.freq_range,
            self.response_range, s=0)
        self.freq_limits = (np.min(self.freq_range), np.max(self.freq_range))
        self.filter_width = self.calculate_width()

    def integrate_response(self, f):
        # f is a function of frequency
        # f will be integrated over filter response
        if "SPIRE" in self.name:
            # EXTENDED SOURCE (SPIRE only)
            integrand = lambda nu: self.response(nu) * self.beam(nu) * f(nu)
        else:
            # PACS
            integrand = lambda nu: self.response(nu) * f(nu)
        # SCIPY.QUAD takes a very long time (significant fraction of a second)
        # result = quad(integrand, *self.freq_limits,
        #     **QUAD_ARGS)[0]
        result = alt_integrate(integrand, *self.freq_limits, size=200)
        # The difference tends to be ~5e-5 to 1e-6
        # print("ERR: {:.2E}".format(abs(result - result2)/result))
        return result

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

def alt_integrate(f, a, b, size=100):
    x = np.linspace(a, b, size)
    return (x[1] - x[0])*np.sum(f(x))

def calculate_ccf(detector, T, beta, nu0=1e12):
    tau = lambda nu: (nu/nu0)**beta
    flux = lambda nu: (1 - np.exp(-tau(nu)))*B(nu, T)
    integrand = lambda nu: flux(nu)*detector.response(nu)
    integrated_flux = quad(integrand, *detector.freq_limits,
        **QUAD_ARGS)[0]
    meas_flux = integrated_flux / detector.filter_width
    true_flux = flux(detector.center)
    return meas_flux / true_flux

def gen_beam_function(band_stub):
    if "SPIRE" not in band_stub:
        return None
    # Yield a function that calculates beam size for frequency
    nu0 = bandpass_centers[band_stub]
    beam0 = SPIRE_Omegaeff[band_stub]
    def beam_f(nu):
        return beam0 * (nu / nu0)**SPIRE_gamma
    return beam_f

def get_Herschel():
    return [Instrument(H_stubs[wl]) for wl in H_WL]

def gen_data_filename(band_stub, imgerr, stub_override=None, stub_append=""):
    if stub_override is None:
        stub_override = "remapped-conv"
    else:
        stub_override = stub_override.strip('-')
    if stub_append:
        stub_append = stub_append.strip('-')
        stub_append = '-'+stub_append
    if imgerr[0] == 'i' and band_stub[0] == 'P':
        stub_prepend = "-plus045"
    elif imgerr[0] == 'e':
        stub_prepend = "-plus05.0pct"
    else:
        stub_prepend = ""
    return f"{band_stub}{stub_prepend}-{imgerr}-{stub_override}{stub_append}.fits"

def gen_data_filenames(**kwargs):
    imgerr_stubs = ("image", "error")
    return_list = []
    for band_stub in bandpass_centers:
        return_list.append(tuple(gen_data_filename(band_stub, ie, **kwargs) for ie in imgerr_stubs))
    return tuple(return_list)
