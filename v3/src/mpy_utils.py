import numpy as np
import scipy.constants as cst

# Constants

# Kevin's values
cst.c = 2.99792458e8
H2mass = 1.6737237e-24 * 2.75 # g


# scipy value
# H2mass = 2.75 * cst.m_p * 1e3 * 1.001  # g (mass of H2 molecule in grams)

BINS = 32

meters_micron = 1e6
f_hz_meters = lambda hz: cst.c / hz
# so for NU_hz = frequency in hertz
# LAMBDA_micron = f_hz_meters(NU_hz) * meters_micron
# and NU_hz = f_hz_meters( LAMBDA_micron / meters_micron )
f_hz_micron = lambda hz: f_hz_meters(hz) * meters_micron
wl_micron_hz = lambda um: f_hz_meters(um / meters_micron)

H_WL = [70, 100, 160, 250, 350, 500]


def valid_wavelength(func_to_decorate):
    def decorated_function(*args):
        wl = args[0]
        if wl not in H_WL:
            raise RuntimeError(f"wavelenth {wl} is not supported")
        return func_to_decorate(*args)
    return decorated_function


@valid_wavelength
def H_stub(wl):
    if wl < 200:
        return f"PACS{wl}um"
    else:
        return f"SPIRE{wl}um"
