import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cst
from astropy.io.fits import getdata
from astropy.io.fits import open as fitsopen
import inspect

def get_f_name():
    return str(inspect.stack()[1][3]).upper()

"""
Dust tests
"""
from Dust import Dust

dtest_stub = "<DUST>"
def test_power_law():
    k0, nu0, beta = 0.2, 999*1e9, 2
    d1 = Dust(k0, nu0, beta)
    d2 = Dust(k0=k0, nu0=nu0, beta=beta)
    nu_range = np.exp(np.linspace(np.log(cst.c/(1000*1e-6)), np.log(cst.c/(100*1e-6)), 100))
    assert (np.sum(np.abs(d1(nu_range) - d2(nu_range)))/np.sum(d1(nu_range))) < 1e-10
    print(d1.__repr__(), "|||", str(d1))
    print(d2.__repr__(), "|||", str(d2))
    print("-----------{} PASSED [{}]-----------".format(dtest_stub, get_f_name()))
    print()


def test_table():
    k0, nu0, beta = 0.1, 1000*1e9, 2
    d1 = Dust(beta=beta)
    nu_range = np.exp(np.linspace(np.log(cst.c/(1000*1e-6)), np.log(cst.c/(100*1e-6)), 100))
    k_range = d1(nu_range)
    table = np.stack([nu_range, k_range], axis=1)
    assert table.shape == (100, 2)
    d2 = Dust(table, "PWR")
    nu_range = np.exp(np.linspace(np.log(cst.c/(1500*1e-6)), np.log(cst.c/(50*1e-6)), 100))
    assert (np.sum(np.abs(d1(nu_range) - d2(nu_range)))/np.sum(d1(nu_range))) < 1e-10
    print(d1.__repr__(), "|||", str(d1))
    print(d2.__repr__(), "|||", str(d2))
    print("-----------{} PASSED [{}]-----------".format(dtest_stub, get_f_name()))
    print()


def test_broken_power():
    beta = 2
    d1 = Dust(beta=beta)
    nu_range = np.exp(np.linspace(np.log(cst.c/(1000*1e-6)), np.log(cst.c/(100*1e-6)), 100))
    nu_lims = (np.min(nu_range), np.max(nu_range))
    k_range = d1(nu_range)
    table = np.stack([nu_range, k_range], axis=1)
    assert table.shape == (100, 2)
    d2 = Dust(table)
    d3 = Dust(table, beta=3, k0=d1(1000*1e9))
    nu_range = np.exp(np.linspace(np.log(cst.c/(1500*1e-6)), np.log(cst.c/(50*1e-6)), 100))
    assert np.all(d3(nu_range[nu_range < nu_lims[0]]) < d2(nu_range[nu_range < nu_lims[0]]))
    assert np.all(d3(nu_range[nu_range > nu_lims[1]]) > d2(nu_range[nu_range > nu_lims[1]]))
    print(d1.__repr__(), "|||", str(d1))
    print(d2.__repr__(), "|||", str(d2))
    print(d3.__repr__(), "|||", str(d3))
    print("-----------{} PASSED [{}]-----------".format(dtest_stub, get_f_name()))
    print()
    return
    plt.plot(nu_range, d2(nu_range), '--', label='fitted')
    plt.plot(nu_range, d3(nu_range), '--', label='broken')
    plt.xscale('log'), plt.yscale('log')
    for lim in nu_lims:
        plt.plot([lim, lim], plt.ylim(), '-.', linewidth=1, color='k')
    plt.legend()
    plt.show()

tests_dust = [test_power_law, test_table, test_broken_power]

"""
Greybody tests
"""
from Greybody import Greybody, H2mass

gtest_stub = "<GREYBODY>"
def test_simple_clouds():
    print("H2mass:", H2mass)
    print("Kevins:", 1.6737237e-24*2.75)
    nu_range = np.exp(np.linspace(np.log(cst.c/(1500*1e-6)), np.log(cst.c/(50*1e-6)), 100))
    c1 = Greybody(15, 1e20, Dust(beta=2))
    print("ATTRIB:", c1.attributes())
    print(c1.__repr__(), "|||", str(c1))
    c2 = Greybody(10, 5e21, Dust(beta=1.5))
    for c in (c1, c2):
        plt.plot(1e6*cst.c/nu_range, c.radiate(nu_range), '--',
            label="component:{:s}".format(str(c.dusts[0])))
    c1 = Greybody(15, 1e20/2, Dust(beta=2))
    c3 = Greybody(*zip(*(c1.attributes() + c2.attributes())))
    print("ATTRIB:", c3.attributes())
    print(c3.__repr__(), "|||", str(c3))
    plt.plot(1e6*cst.c/nu_range, c3.radiate(nu_range), '--',
        label="two-component")
    for beta in np.arange(1.5, 2.5, 0.2):
        c1 = Greybody(15, 3e20, Dust(beta=beta))
        plt.plot(1e6*cst.c/nu_range, c1.radiate(nu_range),
            label="{:.1f}".format(beta))
    plt.xscale('log'), plt.yscale('log')
    plt.ylim([1e-3, plt.ylim()[1]])
    plt.legend()
    plt.clf()
    print("-----------{} PASSED [{}]-----------".format(gtest_stub, get_f_name()))
    print()
    return
    plt.show()

tests_greybody = [test_simple_clouds]

"""
Instrument test_simple_clouds
"""

from Instrument import Instrument
from scipy.integrate import quad

itest_stub = "<INSTRUMENT>"
def debug_instrument():
    detector = Instrument("PACS160um")
    print("CREATED OBJECT")
    limits = (np.min(detector.freq_range), np.max(detector.freq_range))
    print("Freq lim",
        tuple(1e6*cst.c/x for x in limits)
    )
    plt.plot(1e6*cst.c/detector.freq_range, detector.response_range, 'x')
    nu_range = np.linspace(*limits, detector.freq_range.size*5)
    plt.fill(1e6*cst.c/nu_range, detector.response(nu_range), '-', alpha=0.1)
    plt.yscale('log')
    result = quad(detector.response, *limits, limit=100, epsrel=1e-6)
    print("QUAD", result)
    dnu = nu_range[1] - nu_range[0]
    alt_result = dnu * np.sum(detector.response(nu_range))
    print("SUM", alt_result)
    print("aggreement: ", 100*abs(result[0] - alt_result)/alt_result)

def test_instrument_filter():
    detector = Instrument("PACS160um")
    print(detector.__repr__(), "|||",  str(detector))
    center_micron = 1e6*cst.c/detector.center
    width_micron = 1e6*detector.filter_width*cst.c/(detector.center**2)
    width_hz = detector.filter_width/1e11
    assert round(center_micron, 1) == round(160.0, 1)
    assert round(width_micron, 1) == round(30.2, 1)
    assert round(width_hz, 2) == round(3.54, 2)
    print("-----------{} PASSED [{}]-----------".format(itest_stub, get_f_name()))
    print()
    # Values found in tables in PACS literature

from Instrument import calculate_ccf

def test_color_correction_1():
    # Test the color correction to other temperatures
    # Retains the nu^-1 pipeline assumption
    detector = Instrument("PACS160um")
    temps = [5, 10, 50, 100, 1000, 10000]
    ccfs_1 = [4.278, 1.184, 1.010, 1.042, 1.072, 1.074]
    ccfs_2 = [4.279, 1.185, 1.010, 1.043, 1.072, 1.075]
    beta = 0
    print("Columns: (1) calculated | (2,3) from literature")
    for i, T in enumerate(temps):
        result = calculate_ccf(detector, T, beta)
        print("T({:5.0f}K): {:.3f} | {:.3f} {:.3f}".format(
            T, result, ccfs_1[i], ccfs_2[i]
        ))
        mean_ccf = np.mean([ccfs_1[i], ccfs_2[i]])
        err_ccf = abs(result - mean_ccf)/mean_ccf
        assert err_ccf < 0.01
    print("Deviations less than 1%")
    print("-----------{} PASSED [{}]-----------".format(itest_stub, get_f_name()))
    print()

def test_color_correction_2():
    # Test color correction to other spectral indices
    detector = Instrument("PACS160um")
    temps = [10, 100, 10, 100]
    betas = [2, 2, 1, 1]
    ccfs_1 = [1.009, 1.175, 1.083, 1.098]
    ccfs_2 = [1.009, 1.176, 1.083, 1.075]
    print("Columns: (1) calculated, (2,3) from literature")
    for i, T in enumerate(temps):
        result = calculate_ccf(detector, T, betas[i], nu0=cst.c/(1e-6))
        print("T({:3.0f}K)/beta={:.0f}: {:.3f} | {:.3f} {:.3f}".format(
            T, betas[i], result, ccfs_1[i], ccfs_2[i]
        ))
        mean_ccf = np.mean([ccfs_1[i], ccfs_2[i]])
        err_ccf = abs(result - mean_ccf)/mean_ccf
        assert err_ccf < 0.01
    print("Deviations less than 1%")
    print("-----------{} PASSED [{}]-----------".format(itest_stub, get_f_name()))
    print()

from Instrument import bandpass_centers

def test_instrument_detect():
    # Test linking a detector and a greybody
    c1 = Greybody(15, 1e21, Dust(beta=2))
    print(c1.__repr__())
    for b in bandpass_centers:
        print("{:s}: {:.3f} MJy/sr".format(b, Instrument(b).detect(c1)))
    print("-----------{} PASSED [{}]-----------".format(itest_stub, get_f_name()))
    print()

tests_instrument = [test_instrument_filter, test_color_correction_1,
    test_color_correction_2, test_instrument_detect
]


"""
manticore consistency tests
"""

from mpy_utils import get_manticore_info

mtest_stub = "<MANTICORE>"
manticore_soln_2p = "../../T4-absdiff-Per1J-plus045-pow-1000-0.1-1.80.fits"

def test_manticore_retrieval_2p():
    # Test pixel value retrieval from FITS file
    info_dict = get_manticore_info(manticore_soln_2p, 568-1, 524-1)
    print(f"Pixel: {568-1}, {524-1}")
    for k in info_dict:
        if k[0] == 'N':
            print("{}: {:.1E}".format(k, info_dict[k]), end=", ")
        else:
            print("{}: {:.2f}".format(k, info_dict[k]), end=", ")
    print()
    assert abs(info_dict['Tc'] - 13.5998)/13.5998 < 1e-3
    assert abs(info_dict['Nc']/1e21 - 4.39137)/4.39137 < 1e-3
    info_dict = get_manticore_info(manticore_soln_2p, ((568-1, 524-1), (551-1, 307-1)))
    print(info_dict['Tc'])
    print("-----------{} PASSED [{}]-----------".format(mtest_stub, get_f_name()))
    print()

def test_manticore_2p_single_pixel():
    info_dict = get_manticore_info(manticore_soln_2p, 634-1, 503-1)
    Tc, Nc = (info_dict[x] for x in ("Tc", "Nc"))
    obs_flux = [info_dict[x] for x in ("obs160", "obs250", "obs350", "obs500")]
    err_flux = [info_dict[x] for x in ("err160", "err250", "err350", "err500")]
    mod_flux = [info_dict[x] for x in ("mod160", "mod250", "mod350", "mod500")]
    mc = Greybody(Tc, Nc, Dust(beta=1.80))
    mc_flux = [Instrument(b).detect(mc) for b in bandpass_centers]
    print(Tc, Nc)
    print([(a-b)/b for a, b in zip(mc_flux, mod_flux)])
    assert all(abs(a-b)/b < 1e-3 for a, b in zip(mc_flux, mod_flux))
    print("-----------{} PASSED [{}]-----------".format(mtest_stub, get_f_name()))
    print()
    return
    nu_range = np.exp(np.linspace(np.log(cst.c/(1500*1e-6)), np.log(cst.c/(50*1e-6)), 100))
    plt.plot(1e6*cst.c/nu_range, mc.radiate(nu_range), '--', label=str(mc))
    hwl = [160, 250, 350, 500]
    plt.errorbar(hwl, obs_flux, yerr=err_flux, fmt='o', label='observed')
    plt.plot(hwl, mod_flux, 'x', label='model')
    plt.plot(hwl, mc_flux, '>', label='python')
    plt.xlabel("wavelength (micron)")
    plt.ylabel("flux (MJy/sr)")
    plt.title(f"Pixel {568-1}, {524-1} SED")
    plt.xscale('log')
    plt.legend()
    plt.show()

def test_manticore_2p_many_pixels():
    detectors = [Instrument(b) for b in bandpass_centers]
    dust = Dust(beta=1.80)
    Ts, Ns  = [], []
    diffs = []
    for i in range(100, 900, 100):
        for j in range(100, 900, 100):
            info_dict = get_manticore_info(manticore_soln_2p, i, j)
            Tc, Nc = (info_dict[x] for x in ("Tc", "Nc"))
            if np.isnan(Tc) or np.isnan(info_dict['obs160']):
                continue
            # obs_flux = [info_dict[x] for x in ("obs160", "obs250", "obs350", "obs500")]
            # err_flux = [info_dict[x] for x in ("err160", "err250", "err350", "err500")]
            mod_flux = [info_dict[x] for x in ("mod160", "mod250", "mod350", "mod500")]
            mc = Greybody(Tc, Nc, dust)
            mc_flux = [d.detect(mc) for d in detectors]
            flux_diffs = [(a-b)/b for a, b in zip(mc_flux, mod_flux)]
            assert all(abs(a-b)/b < 1e-4 for a, b in zip(mc_flux, mod_flux))
            diffs.append(flux_diffs)
            Ts.append(Tc)
            Ns.append(Nc)
    print("-----------{} PASSED [{}]-----------".format(mtest_stub, get_f_name()))
    print()
    return
    diffs_sorted = list(zip(*diffs))
    plt.figure()
    for i, b in enumerate([160, 250, 350, 500]):
        plt.subplot(221 + i)
        plt.plot(Ts, diffs_sorted[i], '.', label=str(b))
        plt.xlabel("T")
    plt.show()


manticore_soln_3p = "../../T4-absdiff-Per1J-3param-plus045-plus05.0pct-cpow-1000-0.1-2.10hpow-1000-0.1-1.80-bcreinit-Th15.95-Nh5E19,2E22.fits"

def test_manticore_3p_single_pixel():
    pi, pj = 582, 270
    info_dict = get_manticore_info(manticore_soln_3p, pi-1, pj-1)
    Tc, Nc, Th, Nh = (info_dict[x] for x in ("Tc", "Nc", "Th", "Nh"))
    obs_flux = [info_dict[x] for x in ("obs160", "obs250", "obs350", "obs500")]
    err_flux = [info_dict[x] for x in ("err160", "err250", "err350", "err500")]
    mod_flux = [info_dict[x] for x in ("mod160", "mod250", "mod350", "mod500")]
    mc = Greybody([Th, Tc], [Nh, Nc], [Dust(beta=1.80), Dust(beta=2.10)])
    mc_flux = [Instrument(b).detect(mc) for b in bandpass_centers]
    print(mc.__repr__())
    print(mod_flux)
    print(mc_flux)
    print([(a-b)/b for a, b in zip(mc_flux, mod_flux)])
    print("-----------{} PASSED [{}]-----------".format(mtest_stub, get_f_name()))
    print()
    return
    nu_range = np.exp(np.linspace(np.log(cst.c/(1500*1e-6)), np.log(cst.c/(50*1e-6)), 100))
    plt.plot(1e6*cst.c/nu_range, mc.radiate(nu_range), '--', label=str(mc))
    hwl = [160, 250, 350, 500]
    plt.errorbar(hwl, obs_flux, yerr=err_flux, fmt='o', label='observed')
    plt.plot(hwl, mod_flux, 'x', label='model')
    plt.plot(hwl, mc_flux, '>', label='python')
    plt.xlabel("wavelength (micron)")
    plt.ylabel("flux (MJy/sr)")
    plt.title(f"Pixel {pi-1}, {pj-1} SED")
    plt.xscale('log')
    plt.legend()
    plt.show()

mask_fn = "../filament_mask_syp.fits"
def test_manticore_2p_mask_pixels():
    m = getdata(mask_fn)
    soln = fitsopen(manticore_soln_3p)
    info_dict = get_manticore_info(soln, m)
    Tcdata = np.full(soln[1].data.shape, np.nan)
    Tcdata[m.astype(bool)] = info_dict['Tc']
    plt.imshow(Tcdata, origin='lower')
    plt.show()

from mpy_utils import gen_CHAIN_dict
def test_get_CHAIN():
    desktop_soln_path = "/n/sgraraid/filaments/data/TEST4/Per/testregion1342190326JS/T4-absdiff-Per1J-3param-plus045-plus05.0pct-cpow-1000-0.1-2.10hpow-1000-0.1-1.80-bcreinit-Th15.95-Nh5E19,2E22.fits"
    for i in range(6):
        try:
            gen_CHAIN_dict(desktop_soln_path, chain=i)
        except Exception as e:
            print(f"CHAIN {i} {e.__class__.__name__}: {e}")
    print('done')


from mpy_utils import get_obs, get_err
from Instrument import get_Herschel
from mantipyfit import fit_source_2p, fit_source_3p, ITER

def test_mp_fit_2p():
    info_dict = get_manticore_info(manticore_soln_2p, 582-1, 270-1)
    Tc, Nc = (info_dict[x] for x in ("Tc", "Nc"))
    obs, err = get_obs(info_dict), get_err(info_dict)
    herschel = get_Herschel()
    dust = Dust(beta=1.80)
    result = fit_source_2p(obs, err, herschel, dust)
    print("--->", result)
    print("manticore found: T{:.4f}, N{:.4f}".format(Tc, np.log10(Nc)))
    print("i =", ITER['a'])

def test_mp_fit_3p():
    info_dict = get_manticore_info(manticore_soln_3p, 582-1, 270-1)
    Tc, Nc, Th, Nh = (info_dict[x] for x in ("Tc", "Nc", "Th", "Nh"))
    obs, err = get_obs(info_dict), get_err(info_dict)
    herschel = get_Herschel()
    dust = [Dust(beta=2.10), Dust(beta=1.80)]
    result = fit_source_3p(obs, err, herschel, dust, Th=15.95)
    print("--->", result)
    print("manticore found: T{:.4f}, N{:.4f},N{:.4f}".format(
        Tc, np.log10(Nc), np.log10(Nh)
    ))
    print("i =", ITER['a'])




tests_manticore = [test_manticore_retrieval_2p,
    test_manticore_2p_single_pixel, test_manticore_3p_single_pixel,
    test_manticore_2p_mask_pixels
]


"""
imported package tests
"""
mptest_stub = "<IMPORTANT PACKAGES>"
from mantipyfit import goodness_of_fit_f_2p, goodness_of_fit_f_3p

def test_multiprocessing():
    import multiprocessing as mproc
    print(mproc.cpu_count())

def test_corner():
    import corner
    samples = np.random.normal(size=(300, 2))
    corner.corner(samples, labels=['x', 'y'], truths=[0, 0])
    plt.show()

def test_emcee():
    import corner
    import emcee
    info_dict = get_manticore_info(manticore_soln_2p, 582-1, 270-1)
    nominal = [info_dict[x] for x in ("Tc", "Nc") if x in info_dict]
    for i in (1,):
        nominal[i] = np.log10(nominal[i])
    dust = Dust(beta=1.80)
    obs, err = get_obs(info_dict), get_err(info_dict)
    herschel = get_Herschel()
    dof = 2.
    arguments = (dust, obs, err, herschel, dof)
    def lnprob(x, *args):
        return -1.*goodness_of_fit_f_2p(x, *args)
    nwalkers, ndim = 10, 2
    p0 = np.concatenate([
        np.random.normal(scale=3, size=(nwalkers, 1)) + 10,
        np.random.normal(scale=1.5, size=(nwalkers, 1)) + 21
    ], axis=1)
    print(p0)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=arguments)
    sampler.run_mcmc(p0, 150)
    print(sampler.chain.shape)
    samples = sampler.chain[:, 70:, :].reshape((-1, ndim))
    print(samples.shape)
    # plt.figure()
    # for i in range(2):
    #     plt.subplot(211+i)
    #     for j in range(nwalkers):
    #         plt.plot(sampler.chain[j, :, i])
    fig = corner.corner(samples, labels=['Tc', 'Nc'],
        truths=nominal,)# range=[(0, 15), (18, 23.5)])
    plt.show()

def test_threading():
    from concurrent.futures import ThreadPoolExecutor
    list1, list2 = dict(), dict()
    def thread_function(index, writeto1=list1, writeto2=list2):
        x = index+1
        writeto1[index] = x
        x = x**3
        writeto2[index] = x
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(thread_function, range(200))
    print("1", list1)
    print("2", list2)

from Instrument import gen_data_filenames
from mantipyfit import fit_full_image
from mpy_utils import WCS, save_fits
def test_full_image_fit():
    print("ONLY GOOD ON DESKTOP")
    raise RuntimeWarning("THIS HAS ALREADY BEEN RUN!")
    """
    2 PARAM:
    This worked, and for -crop6, found almost exactly the same solutions
    for all but 3 of nearly 1500 pixels
    """
    per_path = "/n/sgraraid/filaments/data/TEST4/Per/testregion1342190326JS/"
    obs_maps, err_maps = [], []
    w = None
    for pair in gen_data_filenames(stub_append="-crop6"):
        if w is None:
            w = WCS(getdata(per_path+pair[0], header=True)[1])
        obs_maps.append(getdata(per_path+pair[0]))
        err_maps.append(getdata(per_path+pair[1]))
    ## PLOT DATA
    # axes = plt.subplots(ncols=2, nrows=4, sharex=True, sharey=True)[1]
    # for i, o, e in zip(range(4), obs_maps, err_maps):
    #     axes[i, 0].imshow(o, origin='lower')
    #     axes[i, 1].imshow(e, origin='lower')
    # plt.show()
    ##
    # 2-param fit; x is [T, log10(N)]
    dust = [Dust(beta=1.80), Dust(beta=2.10)]
    src_fn = lambda x: Greybody([15.95, x[0]], [10**x[1], 10**x[2]], dust)
    x0 = [10, 20, 22]
    bounds = ((0, None), (18, 25), (18, 25))
    herschel = get_Herschel()
    solution = fit_full_image(obs_maps, err_maps, herschel, src_fn, x0, bounds, dof=1.)
    # This works; now write up the "save fits" bit from "prepare_bc" & "spectralidx_vs_temp"
    write_data_dict = {
        ("Tc", "K"): solution[0],
        ("Nh", "cm-2"): 10**solution[1],
        ("Nc", "cm-2"): 10**solution[2],
    }
    save_name = per_path+"mantipyfit_save_test_3p.fits"
    comment = "this is a test"
    save_fits(write_data_dict, w, save_name, comment=comment)
    return


all_tests = tests_dust + tests_greybody + tests_instrument + tests_manticore

if __name__ == "__main__":
    test_full_image_fit()
    # for f in all_tests:
    #     try:
    #         f()
    #     except:
    #         print("!!!!!!!!!! FAILED {:s}".format(str(f.__name__).upper()))
