import numpy as np
import matplotlib.pyplot as plt

from ..physics import mpy_utils
from ..physics import dust
from ..physics import greybody
from ..physics import instrument

cst = mpy_utils.cst

"""
Test the pieces that go into fit.py
"""

def dustTest():
    beta2 = dust.Dust(beta=2.0)
    freqs = np.linspace(mpy_utils.f_hz_micron(600), mpy_utils.f_hz_micron(50), 200, dtype=np.float)
    plt.plot(mpy_utils.f_hz_micron(freqs), beta2(freqs))
    plt.title("Dust opacity for beta = 2")
    plt.xlabel("Wavelength (micron)")
    plt.ylabel("Kappa (cm2/g) * m_H2 (g)")
    plt.show()

def tauTest():
    beta2 = dust.TauOpacity(2)
    freqs = np.linspace(mpy_utils.f_hz_micron(600), mpy_utils.f_hz_micron(50), 200, dtype=np.float)
    plt.plot(mpy_utils.f_hz_micron(freqs), beta2(freqs))
    plt.title("Dust opacity for beta = 2")
    plt.xlabel("Wavelength (micron)")
    plt.ylabel("Optical depth (tau)")
    plt.show()

def dusttauTest():
    kappa = dust.Dust(beta=2.0, k0=0.05625, nu0=750*1e9)
    N1Av = 1.1e21
    tau = dust.TauOpacity(2.0)
    tau160 = kappa(dust.nu0_160)*N1Av
    freqs = np.linspace(mpy_utils.f_hz_micron(600), mpy_utils.f_hz_micron(50), 200, dtype=np.float)
    plt.plot(mpy_utils.f_hz_micron(freqs), kappa(freqs)*N1Av, label='Kappa')
    plt.plot(mpy_utils.f_hz_micron(freqs), tau(freqs)*tau160, label='Tau')
    plt.plot(mpy_utils.f_hz_micron(freqs), tau(freqs)*tau160/(kappa(freqs)*N1Av), label='Ratio')
    plt.title("Dust opacity for $\\beta$ = 2")
    plt.xlabel("Wavelength (micron)")
    plt.ylabel("Optical depth")
    plt.xscale('log'), plt.yscale('log')
    plt.legend()
    plt.show()

def greybodyTest():
    beta2 = dust.TauOpacity(2.0)
    gb = greybody.Greybody(15, -2, beta2)
    freqs = np.linspace(mpy_utils.f_hz_micron(600), mpy_utils.f_hz_micron(50), 200)
    plt.plot(mpy_utils.f_hz_micron(freqs), gb.radiate(freqs))
    plt.title("Greybody at T=15K, $\\tau_{160}$=0.01, $\\beta$=2 dust")
    plt.xlabel("Wavelength (micron)")
    plt.ylabel("Flux (MJy/sr)")
    plt.show()

def jacobianTest():
    beta2 = dust.TauOpacity(2.0)
    gb = greybody.Greybody(15, -2, beta2, p=3)
    freqs = np.linspace(mpy_utils.f_hz_micron(600), mpy_utils.f_hz_micron(50), 200, dtype=np.float64)
    wl = mpy_utils.f_hz_micron(freqs)
    jacs = gb.dradiate(freqs)
    labels = ['X = T', 'X = $\\tau_{160}$', 'X = $\\beta$']
    for i in range(3):
        plt.plot(wl, jacs[i], label=labels[i])
    plt.legend(shadow=True)
    plt.title("Greybody at T=15K, $\\tau_{160}$=0.01, $\\beta$=2 dust")
    plt.xlabel("Wavelength (micron)")
    plt.ylabel("$\\partial$Flux / $\\partial$X (MJy/sr / [X])")
    plt.show()

def instrumentTest():
    colors = ['violet', 'blue', 'green', 'yellow', 'orange', 'red']
    for i, wl in enumerate(mpy_utils.H_WL):
        d = instrument.Instrument(wl)
        plt.plot(mpy_utils.f_hz_micron(d.freq_array), d.response_array/d.filter_width, color=colors[i], label=str(wl))
        plt.plot([mpy_utils.f_hz_micron(d.center)]*2, [0, np.max(d.response_array/d.filter_width)], '--', color=colors[i], label="_nolabel_")
    plt.legend()
    plt.title("Herschel bandpasses")
    plt.xlabel("Wavelength (micron)")
    plt.ylabel("Transmission curves (unitless)")
    plt.show()

def instrumentFilterTest():
    detector = instrument.Instrument(160)
    center_micron = 1e6*cst.c/detector.center
    width_micron = 1e6*detector.filter_width*cst.c/(detector.center**2)
    width_hz = detector.filter_width/1e11
    # TODO: FIND SOURCES IN DOCUMENTATION FOR THESE
    assert round(center_micron, 1) == round(160.0, 1)
    assert round(width_micron, 1) == round(30.2, 1)
    assert round(width_hz, 2) == round(3.54, 2)


def instrument_calculate_ccf(detector, T, beta, nu0=1e12):
    tau = lambda nu: (nu/nu0)**beta
    flux = lambda nu: (1 - np.exp(-tau(nu)))*greybody.B(nu, T)
    meas_flux = detector.integrate_response(flux) / detector.filter_width
    true_flux = flux(detector.center)
    return meas_flux / true_flux

def instrumentColorCorrectionTest1():
    # Test the color correction to other temperatures
    # Retains the nu^-1 pipeline assumption
    # TODO: FIND SOURCES IN DOCUMENTATION FOR THESE
    detector = instrument.Instrument(160)
    temps = [5, 10, 50, 100, 1000, 10000]
    ccfs_1 = [4.278, 1.184, 1.010, 1.042, 1.072, 1.074]
    ccfs_2 = [4.279, 1.185, 1.010, 1.043, 1.072, 1.075]
    beta = 0
    print("Columns: (1) calculated | (2,3) from literature")
    for i, T in enumerate(temps):
        result = instrument_calculate_ccf(detector, T, beta)
        print("T({:5.0f}K): {:.3f} | {:.3f} {:.3f}".format(
            T, result, ccfs_1[i], ccfs_2[i]
        ))
        mean_ccf = np.mean([ccfs_1[i], ccfs_2[i]])
        err_ccf = abs(result - mean_ccf)/mean_ccf
        assert err_ccf < 0.01
    print("(CCF1) Deviations less than 1%")

def instrumentColorCorrectionTest2():
    # Test color correction to other spectral indices
    detector = instrument.Instrument(160)
    temps = [10, 100, 10, 100]
    betas = [2, 2, 1, 1]
    ccfs_1 = [1.009, 1.175, 1.083, 1.098]
    ccfs_2 = [1.009, 1.176, 1.083, 1.075]
    print("Columns: (1) calculated, (2,3) from literature")
    for i, T in enumerate(temps):
        result = instrument_calculate_ccf(detector, T, betas[i], nu0=cst.c/(1e-6))
        print("T({:3.0f}K)/beta={:.0f}: {:.3f} | {:.3f} {:.3f}".format(
            T, betas[i], result, ccfs_1[i], ccfs_2[i]
        ))
        mean_ccf = np.mean([ccfs_1[i], ccfs_2[i]])
        err_ccf = abs(result - mean_ccf)/mean_ccf
        assert err_ccf < 0.01
    print("(CCF2) Deviations less than 1%")

def derivativeGOFTest():
    src1 = greybody.Greybody(14, -2, dust.TauOpacity(2.0))
    herschel = instrument.get_all_Herschel()
    obs = [d.detect(src1) for d in herschel]
    err = [0.05*o for o in obs]
    src2 = greybody.Greybody(14.0, -2.00, dust.TauOpacity(1.90))
    jac = [2*(d.detect(src2.radiate) - o)*d.detect(src2.dradiate)/(e*e) for d, o, e in zip(herschel, obs, err)]
    print(type(jac), len(jac), len(jac[0]))
    jac = np.sum(jac, axis=0)
    print(jac)

if __name__ == "__main__":
    derivativeGOFTest()
