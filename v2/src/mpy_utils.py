import scipy.constants as cst
"""
Global constants and utility functions for mantipython
"""
__author__ = "Ramsey Karim"

# Constants

# Kevin's values
cst.c = 2.99792458e8 # m / s
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
