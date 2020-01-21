import numpy as np
import scipy.constants as cst


BINS = 32

meters_micron = 1e6
f_hz_meters = lambda hz: cst.c / hz
# so for NU_hz = frequency in hertz
# LAMBDA_micron = f_hz_meters(NU_hz) * meters_micron
# and NU_hz = f_hz_meters(LAMBDA_micron) * meters_micron
# Same function can be used.
f_hz_micron = lambda hz: f_hz_meters(hz) * meters_micron

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
