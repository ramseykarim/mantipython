import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.constants as cst
import mpy_utils as mpu
from Dust import Dust
from Greybody import Greybody
from Instrument import Instrument, get_Herschel, gen_data_filenames
import mantipyfit as mpfit
import corner
import emcee
import pickle
from pathlib import Path

"""
This is where I will run science code on my laptop!
"""

def main():
    mtest_boostrap_proper_error()

def desktop_main():
    mtest_boostrap_proper_error()

"""
Scripts below here
"""
from computer_config import manticore_soln_2p, manticore_soln_3p, mask_fn
import computer_config as cconf

def mtest_2pixel_scatter():
    pij1 = 478-1, 376-1 # looks ok
    pij2 = 479-1, 376-1 # low Nh
    pix_dict = {"ok": pij1, "lowNh": pij2}
    pix_infos = {}
    for k in pix_dict:
        pix_infos[k] = [
            mpu.get_manticore_info(x, *pix_dict[k])
            for x in
            [manticore_soln_2p, manticore_soln_3p]
        ]
    plt.figure()
    linestyles = ['-', '--']
    markerstyles = ['>', 'x']
    dusts = [Dust(beta=2.10), Dust(beta=1.80)]
    nu_range = np.exp(np.linspace(np.log(cst.c/(1500*1e-6)), np.log(cst.c/(50*1e-6)), 100))
    colors = iter(['blue', 'orange', 'red'])
    herschel = get_Herschel()
    for k in pix_infos:
        color = next(colors)
        for i, info_dict in enumerate(pix_infos[k]):
            Ts = [info_dict[x] for x in ("Th", "Tc") if x in info_dict]
            Ns = [info_dict[x] for x in ("Nh", "Nc") if x in info_dict]
            ds = dusts[1-i:]
            cloud = Greybody(Ts, Ns, ds)
            plt.plot(mpu.f_hz_micron(nu_range), cloud.radiate(nu_range),
                linestyle=linestyles[i], color=color,
                label=f"{str(cloud).upper()} ({k})")
            plt.plot([x*1.01 for x in mpu.H_WL],
                [d.detect(cloud) for d in herschel],
                markerstyles[i], label=f"{str(cloud).upper()} ({k})",
                color=color, markersize=8)
            if i == 0:
                obs, err = mpu.get_obs(info_dict), mpu.get_err(info_dict)
                plt.errorbar([x*0.99 for x in mpu.H_WL], obs, yerr=err,
                    fmt='.', color=color,
                    capsize=5, markersize=8)
    plt.legend()
    plt.xscale('log')
    plt.xlabel("wavelength (micron)")
    plt.ylabel("flux (MJy/sr)")
    plt.show()

def mtest_mask_chisq():
    m = mpu.fits.getdata(mask_fn).astype(bool)
    soln = mpu.fits.open(manticore_soln_3p)
    nanmask = ~np.isnan(soln[1].data)
    plt.figure()
    plt.subplot(121)
    plt.imshow((m & nanmask), origin='lower')
    plt.subplot(122)
    plt.imshow((~m & nanmask), origin='lower')
    plt.figure()
    herschel = get_Herschel()
    dusts = [Dust(beta=2.10), Dust(beta=1.80)]
    for mask in (m, ~m):
        info_dict = mpu.get_manticore_info(soln, mask & nanmask)
        Ts = tuple(zip(*[tuple(info_dict[x]) for x in ("Th", "Tc") if x in info_dict]))
        Ns = tuple(zip(*[tuple(info_dict[x]) for x in ("Nh", "Nc") if x in info_dict]))
        observations = tuple(zip(*[tuple(x) for x in mpu.get_obs(info_dict)]))
        errors = tuple(zip(*[tuple(x) for x in mpu.get_err(info_dict)]))
        residuals = mpu.deque()
        for i in range(0, len(Ts), len(Ts)//1000):
            cloud = Greybody(Ts[i], Ns[i], dusts)
            residual = sum((d.detect(cloud) - o)**2 / (e*e) for d, o, e in zip(herschel, observations[i], errors[i]))/1
            residuals.append(residual)
        print('done', end=", ")
        residuals = np.array(residuals)
        hist = mpu.histogram(residuals, x_lim=(0, 3))
        plt.plot(*hist, '-')
    soln.close()
    print()
    plt.show()

def mtest_mask_fit_2p_manticore_agreement():
    dust = Dust(beta=1.80)
    herschel = get_Herschel()
    m = mpu.fits.getdata(mask_fn).astype(bool)
    soln = mpu.fits.open(manticore_soln_2p)
    nanmask = ~np.isnan(soln[1].data)
    info_dict = mpu.get_manticore_info(soln, m&nanmask)
    soln.close()
    Ts = tuple(zip(*[tuple(info_dict[x]) for x in ("Th", "Tc") if x in info_dict]))
    Ns = tuple(zip(*[tuple(info_dict[x]) for x in ("Nh", "Nc") if x in info_dict]))
    observations = tuple(zip(*[tuple(x) for x in mpu.get_obs(info_dict)]))
    errors = tuple(zip(*[tuple(x) for x in mpu.get_err(info_dict)]))
    residuals = mpu.deque()
    values = mpu.deque()
    l = 500
    for i in range(0, len(Ts), len(Ts)//l):
        result = mpfit.fit_source_2p(observations[i], errors[i], herschel, dust)
        residuals.append([abs(mr - pr)/mr for mr, pr in zip([Ts[i][0], np.log10(Ns[i][0])], result)])
        values.append(list(result))
    residuals = np.array(residuals)
    values = np.array(values)
    plt.figure()
    plt.subplot(121)
    plt.plot(values[:, 0], residuals[:, 0], '.')
    plt.title("T")
    plt.yscale('log')
    plt.subplot(122)
    plt.plot(values[:, 1], residuals[:, 1], '.')
    plt.title("N")
    plt.yscale('log')
    plt.show()
    print("{:.1f} calls per fit".format(mpfit.ITER['a']/l))

def mtest_mask_fit_3p_manticore_agreement():
    dust = [Dust(beta=2.10), Dust(beta=1.80)]
    herschel = get_Herschel()
    m = mpu.fits.getdata(mask_fn).astype(bool)
    with mpu.fits.open(manticore_soln_3p) as soln:
        nanmask = ~np.isnan(soln[1].data)
        info_dict = mpu.get_manticore_info(soln, m&nanmask)
    Ts = tuple(zip(*[tuple(info_dict[x]) for x in ("Th", "Tc") if x in info_dict]))
    Ns = tuple(zip(*[tuple(info_dict[x]) for x in ("Nh", "Nc") if x in info_dict]))
    observations = tuple(zip(*[tuple(x) for x in mpu.get_obs(info_dict)]))
    errors = tuple(zip(*[tuple(x) for x in mpu.get_err(info_dict)]))
    residuals = mpu.deque()
    values = mpu.deque()
    l = 10
    for i in range(0, len(Ts), len(Ts)//l):
        result = mpfit.fit_source_3p(observations[i], errors[i], herschel, dust, Th=15.95)
        nres = [abs(np.log10(mr) - pr) for pr, mr in zip(result[1:], Ns[i])]
        residuals.append([abs(Ts[i][1] - result[0])]+nres)
        values.append(list(result))
        print(".", end="")
    print()
    print("{:.1f} calls per fit".format(mpfit.ITER['a']/l))
    residuals = np.array(residuals)
    values = np.array(values)
    plt.figure()
    plt.subplot(131)
    plt.plot(values[:, 0], residuals[:, 0], '.')
    plt.title("Tc")
    plt.subplot(132)
    plt.plot(values[:, 1], 10**residuals[:, 1], '.')
    plt.title("Nh")
    plt.yscale('log')
    plt.subplot(133)
    plt.plot(values[:, 2], 10**residuals[:, 2], '.')
    plt.title("Nc")
    plt.yscale('log')
    plt.show()


def mtest_selected_pixels_error():
    good_list, name_list, coords, color_list = zip(*mpu.PIXELS_OF_INTEREST)
    info_dict = mpu.get_manticore_info(manticore_soln_3p, coords)
    nu_range = np.exp(np.linspace(np.log(cst.c/(1500*1e-6)), np.log(cst.c/(50*1e-6)), 100))
    wl_range = 1e6*cst.c/nu_range
    herschel = get_Herschel()
    dusts = [Dust(beta=1.80), Dust(beta=2.10)]
    err_names = ('dTc', 'dNh', 'dNc')
    plt.figure(figsize=(8, 4.5))
    plt.subplot(111)
    for i, name in enumerate(name_list):
        if i != 0:
            continue
        # plt.subplot(331 + i)
        label = name + " (" + ("good" if good_list[i] else "bad") + ")"
        print(label)
        fmt = "o" if good_list[i] else "x"
        # plot points+error, original manticore fits
        obs = [x[i] for x in mpu.get_obs(info_dict)]
        err = [x[i] for x in mpu.get_err(info_dict)]
        # err[0] = err[0]/2
        plt.errorbar(mpu.H_WL, obs, yerr=err, fmt=fmt, capsize=6,
            color=color_list[i], label=label)
        cloud = Greybody([info_dict[x][i] for x in ("Th", "Tc")],
            [info_dict[x][i] for x in ('Nh', 'Nc')], dusts
        )
        plt.plot(wl_range, cloud.radiate(nu_range), '-', color=color_list[i],
            label=str(cloud))
        Tcf, Nhf, Ncf = mpfit.fit_source_3p(obs, err,
            herschel, dusts, Th=info_dict['Th'][i])
        Ncf, Nhf = (10**x for x in (Ncf, Nhf))
        cloudf = Greybody([info_dict['Th'][i], Tcf],
            [Nhf, Ncf], dusts)
        plt.plot(wl_range, cloudf.radiate(nu_range), '--', color=color_list[i],
            label=str(cloudf))
        p_sets, p_errs = mpfit.bootstrap_errors(obs, err, herschel,
            dusts, niter=50, fit_f=mpfit.fit_source_3p,
            dof=1, Th=info_dict['Th'][i])
        manticore_errors = tuple(info_dict[x][i] for x in err_names)
        for x in zip(err_names, manticore_errors, p_errs):
            print("{}: manticore({:.2E}), python({:.2E})".format(*x))
        title = "dTc({0:.2f}|{3:.2f}) / dNh({1:.2E}|{4:.2E}) / dNc({2:.2E}|{5:.2E})".format(
            *manticore_errors, *p_errs
        )
        # plt.title(title)
        nominal = [info_dict[x][i] for x in ("Tc", "Nh", "Nc")]
        print("nominal>> Tc:{:.2f}, Nh:{:.2E}, Nc:{:.2E}".format(*nominal))
        print("fitted >> Tc:{:.2f}, Nh:{:.2E}, Nc:{:.2E}".format(Tcf, Nhf, Ncf))
        for s in p_sets:
            cloudf = Greybody([info_dict['Th'][i], s[0]], s[1:], dusts)
            plt.plot(wl_range, cloudf.radiate(nu_range), '-', alpha=0.15,
                color='grey')
            print(">>> Tc:{:5.2f}, Nh:{:.2E}, Nc:{:.2E}".format(*s))
        plt.xscale('log')
        plt.legend()
        print()
    plt.show()


def mtest_corner_2p_boostrap_single_pixel():
    pixel_index = 0
    niter = 600
    good, name, coords, color = mpu.PIXELS_OF_INTEREST[pixel_index]
    print(name)
    info_dict = mpu.get_manticore_info(manticore_soln_2p, *coords)

    herschel = get_Herschel()
    dusts = Dust(beta=1.80)
    nu_range = np.exp(np.linspace(np.log(cst.c/(1500*1e-6)), np.log(cst.c/(50*1e-6)), 100))
    wl_range = 1e6*cst.c/nu_range

    obs, err = mpu.get_obs(info_dict), mpu.get_err(info_dict)
    #### HALF PACS ERROR
    # err[0] = err[0]/2
    nominal = [info_dict[x] for x in ("Tc", "Nc") if x in info_dict]

    Tcf, Ncf = mpfit.fit_source_2p(obs, err, herschel, dusts)
    print("nominal>> Tc:{:5.2f}, Nc:{:.2E}".format(*nominal))
    print("fitted >> Tc:{:5.2f}, Nc:{:.2E}".format(Tcf, 10**Ncf))
    p_sets, p_errs = mpfit.bootstrap_errors(obs, err, herschel,
        dusts, niter=niter, fit_f=mpfit.fit_source_2p, dof=2)

    # CREATE SED PLOT
    plt.figure(figsize=(10, 7))
    for s in p_sets:
        cloudf = Greybody(s[0], s[1], dusts)
        plt.plot(wl_range, cloudf.radiate(nu_range), '-', alpha=0.1,
            color='grey')
    cloud = Greybody(*nominal, dusts)
    plt.plot(wl_range, cloud.radiate(nu_range), '-', color=color,
        label=str(cloud))
    cloudf = Greybody(Tcf, 10**Ncf, dusts)
    plt.plot(wl_range, cloudf.radiate(nu_range), '--', color=color,
        label=str(cloudf))
    plt.errorbar(mpu.H_WL, obs, yerr=err, fmt='.', capsize=6,
        color=color, label=name)
    plt.xscale('log')
    plt.legend()
    plt.show()
    return

    params = list(zip(*p_sets))
    Tcs, Ncs = map(np.array, params)
    Ncs = np.log10(Ncs)
    nominal[1] = np.log10(nominal[1])
    print("log:")
    print("nominal>> Tc:{:5.2f}, Nc:{:5.2f}".format(*nominal))
    print("fitted >> Tc:{:5.2f}, Nc:{:5.2f}".format(Tcf, Ncf))
    labels = ['Tc', 'log(Nh)', 'log(Nc)']
    params = np.stack([Tcs, Ncs], axis=1)
    fig = corner.corner(params, labels=labels, truths=nominal,
        range=[(13, 14.8), (21.8, 22.0)])
    fig.set_size_inches((10, 10))
    plt.title("BOOTSTRAP (n={:d})".format(niter))
    plt.show()

def mtest_corner_3p_boostrap_single_pixel():
    pixel_index = 6
    niter = 500
    good, name, coords, color = mpu.PIXELS_OF_INTEREST[pixel_index]
    info_dict = mpu.get_manticore_info(manticore_soln_3p, *coords)

    herschel = get_Herschel()
    dusts = [Dust(beta=1.80), Dust(beta=2.10)]
    nu_range = np.exp(np.linspace(np.log(cst.c/(1500*1e-6)), np.log(cst.c/(50*1e-6)), 100))
    wl_range = 1e6*cst.c/nu_range

    obs, err = mpu.get_obs(info_dict), mpu.get_err(info_dict)
    #### HALF PACS ERROR
    err[0] = err[0]/2
    Th = info_dict['Th']
    nominal = [info_dict[x] for x in ("Tc", "Nh", "Nc") if x in info_dict]
    Tcf, Nhf, Ncf = mpfit.fit_source_3p(obs, err, herschel, dusts, Th=Th)
    print("nominal>> Tc:{:5.2f}, Nh:{:.2E}, Nc:{:.2E}".format(*nominal))
    print("fitted >> Tc:{:5.2f}, Nh:{:.2E}, Nc:{:.2E}".format(Tcf, 10**Nhf, 10**Ncf))
    p_sets, p_errs = mpfit.bootstrap_errors(obs, err, herschel,
        dusts, niter=niter, fit_f=mpfit.fit_source_3p, dof=1)

    plt.figure(figsize=(10, 7))
    for s in p_sets:
        cloudf = Greybody([Th, s[0]], s[1:], dusts)
        plt.plot(wl_range, cloudf.radiate(nu_range), '-', alpha=0.1,
            color='grey')
    cloud = Greybody([Th, nominal[0]], nominal[1:], dusts)
    plt.plot(wl_range, cloud.radiate(nu_range), '-', color=color,
        label=str(cloud))
    cloudf = Greybody([Th, Tcf], [10**Nhf, 10**Ncf], dusts)
    plt.plot(wl_range, cloudf.radiate(nu_range), '--', color=color,
        label=str(cloudf))
    plt.errorbar(mpu.H_WL, obs, yerr=err, fmt='.', capsize=6,
        color=color, label=name)
    plt.xscale('log')
    plt.legend()
    plt.show()

    params = list(zip(*p_sets))
    Tcs, Nhs, Ncs = map(np.array, params)
    Nhs, Ncs = np.log10(Nhs), np.log10(Ncs)
    nominal[1] = np.log10(nominal[1])
    nominal[2] = np.log10(nominal[2])
    print("log:")
    print("nominal>> Tc:{:5.2f}, Nh:{:5.2f}, Nc:{:5.2f}".format(*nominal))
    print("fitted >> Tc:{:5.2f}, Nh:{:5.2f}, Nc:{:5.2f}".format(Tcf, Nhf, Ncf))
    labels = ['Tc', 'log(Nh)', 'log(Nc)']
    params = np.stack([Tcs, Nhs, Ncs], axis=1)
    fig = corner.corner(params, labels=labels, truths=nominal,
        range=[(3, 16), (20, 21.6), (21.5, 23.5)])
    fig.set_size_inches((10, 10))
    plt.title("BOOTSTRAP (n={:d})".format(niter))
    plt.show()
    return


def mtest_grid_to_single_pixel():
    pixel_index = 6
    good, name, coords, color = mpu.PIXELS_OF_INTEREST[pixel_index]
    info_dict = mpu.get_manticore_info(manticore_soln_2p, *coords)
    # nu_range = np.exp(np.linspace(np.log(cst.c/(1500*1e-6)), np.log(cst.c/(50*1e-6)), 100))
    # wl_range = 1e6*cst.c/nu_range
    herschel = get_Herschel()
    dof = 2.
    dusts = Dust(beta=1.80) # [Dust(beta=1.80), Dust(beta=2.10)]
    obs, err = mpu.get_obs(info_dict), mpu.get_err(info_dict)
    nominal = [info_dict[x] for x in ('Tc', 'Nc')]
    nominal[1] = np.log10(nominal[1])
    print("T:{:5.2f}, N:{:5.2f}".format(*nominal))
    Tclim, Nclim = (14, 17), (21.2, 21.8)
    ex = (*Tclim, *Nclim)
    aspect = (Tclim[1] - Tclim[0]) / (Nclim[1] - Nclim[0])
    Tcrange = np.linspace(*Tclim, 40)
    Ncrange = np.linspace(*Nclim, 40)
    Tcgrid, Ncgrid = np.meshgrid(Tcrange, Ncrange, indexing='xy')
    gofgrid = np.empty(Tcgrid.size)
    for i, pvec in enumerate(zip(Tcgrid.ravel(), Ncgrid.ravel())):
        gof = mpfit.goodness_of_fit_f_2p(pvec, dusts, obs, err, herschel, dof)
        gofgrid[i] = gof
    gofgrid = np.log10(gofgrid.reshape(Tcgrid.shape))
    plt.imshow(gofgrid, origin='lower', extent=ex, aspect=aspect)
    plt.show()

def mtest_3dgrid_to_single_pixel_WRITEOUT():
    pixel_index = 6
    good, name, coords, color = mpu.PIXELS_OF_INTEREST[pixel_index]
    info_dict = mpu.get_manticore_info(manticore_soln_3p, *coords)
    # nu_range = np.exp(np.linspace(np.log(cst.c/(1500*1e-6)), np.log(cst.c/(50*1e-6)), 100))
    # wl_range = 1e6*cst.c/nu_range
    herschel = get_Herschel()
    dof = 1.
    dusts = [Dust(beta=1.80), Dust(beta=2.10)]
    obs, err = mpu.get_obs(info_dict), mpu.get_err(info_dict)
    nominal = [info_dict[x] for x in ('Tc', 'Nh', 'Nc')]
    Th = info_dict['Th']
    nominal[1] = np.log10(nominal[1])
    nominal[2] = np.log10(nominal[2])
    print("Tc:{:5.2f}, Nh:{:5.2f}, Nc:{:5.2f}".format(*nominal))
    Tclim, Nclim = (0, 16), (19., 21.)
    Nhlim = (20., 21.5)
    # ex = (*Tclim, *Nclim)
    # aspect = (Tclim[1] - Tclim[0]) / (Nclim[1] - Nclim[0])
    Tcrange = np.arange(*Tclim, 0.05)
    Nhrange = np.arange(*Nhlim, 0.05)
    Ncrange = np.arange(*Nclim, 0.05)
    Tcgrid, Nhgrid, Ncgrid = np.meshgrid(Tcrange, Nhrange, Ncrange, indexing='ij')
    print(Tcgrid.shape, Tcgrid.size)
    if int(input("Ready? Type '1'")) != 1:
        print("quitting...")
        return
    gofgrid = np.empty(Tcgrid.size)
    for i, pvec in enumerate(zip(Tcgrid.ravel(), Nhgrid.ravel(), Ncgrid.ravel())):
        gof = mpfit.goodness_of_fit_f_3p(pvec, dusts, obs, err, herschel, Th, dof)
        gofgrid[i] = gof
    gofgrid = np.log10(gofgrid.reshape(Tcgrid.shape))
    # plt.imshow(gofgrid, origin='lower', extent=ex, aspect=aspect)
    # plt.show()
    # NOTE :::
    # grid_file_p6_lowNc.pkl goes from Nc:19-21, and _p6.pkl goes from Nc:21:22.5
    # _p1.pkl goes from Tc(0,16) Nh(20, 21.5) Nc(21, 22.5)
    # [original].pkl goes from Tc(4,16) ??? ??? (check github)
    with open("grid_file_p6_lowNc.pkl", 'wb') as pfl:
        pickle.dump(gofgrid, pfl)
    print("Written and finished")
    return

def mtest_3dgrid_to_single_pixel_PLOT():
    with open("grid_file_p6.pkl", 'rb') as pfl:
        Xs_grid_hiNc = pickle.load(pfl)
    with open("grid_file_p6_lowNc.pkl", 'rb') as pfl:
        Xs_grid_lowNc = pickle.load(pfl)
    Xs_grid = np.concatenate([Xs_grid_lowNc, Xs_grid_hiNc], axis=2)
    del Xs_grid_hiNc, Xs_grid_lowNc
    Tclim, Nclim = (0, 16), (19., 22.5)
    Nhlim = (20., 21.5)
    Tcrange = np.arange(*Tclim, 0.05)
    Nhrange = np.arange(*Nhlim, 0.05)
    Ncrange = np.arange(*Nclim, 0.05)
    print(Xs_grid.shape)
    print(Tcrange.shape, Nhrange.shape, Ncrange.shape)
    print(Xs_grid.ptp(), Xs_grid.min(), Xs_grid.max())
    from mayavi import mlab
    src = mlab.pipeline.scalar_field(-Xs_grid)
    mlab.pipeline.iso_surface(src, contours=[-1.5, -1, -.5],
        colormap='cool', opacity=0.2, vmin=-3, vmax=0)
    mlab.pipeline.iso_surface(src, contours=[0., 0.1, 0.2, 0.3, .5],
        opacity=0.8, vmin=0, vmax=.5, colormap='hot')
    # mlab.pipeline.volume(src, vmin=0, vmax=0.5)
    # mlab.pipeline.iso_surface(src, contours=[2,], opacity=0.3)
    # mlab.pipeline.iso_surface(src, contours=[1,], opacity=0.4)
    # mlab.pipeline.iso_surface(src, contours=[0.5,], opacity=0.5)
    mlab.axes(extent=[0, Xs_grid.shape[0], 0, Xs_grid.shape[1], 0, Xs_grid.shape[2]],
        ranges=[*Tclim, *Nhlim, *Nclim], nb_labels=6,
        xlabel="Tc", ylabel='Nh', zlabel="Nc")
    mlab.show()

def mtest_emcee_2p():
    pixel_index = 0
    niter, burn = 200, 70
    good, name, coords, color = mpu.PIXELS_OF_INTEREST[pixel_index]
    info_dict = mpu.get_manticore_info(manticore_soln_2p, *coords)
    nominal = [info_dict[x] for x in ("Tc", "Nc") if x in info_dict]
    for i in (1,):
        nominal[i] = np.log10(nominal[i])

    dust = Dust(beta=1.80)
    nu_range = np.exp(np.linspace(np.log(cst.c/(1500*1e-6)), np.log(cst.c/(50*1e-6)), 100))
    wl_range = 1e6*cst.c/nu_range
    obs, err = mpu.get_obs(info_dict), mpu.get_err(info_dict)
    # err[0] = err[0]/2
    herschel = get_Herschel()
    dof = 2.

    arguments = (dust, obs, err, herschel, dof)
    def lnposterior(x):
        T, N = x
        if T<0 or T>30 or N<17 or N>26:
            return -np.inf
        else:
            return -1.*mpfit.goodness_of_fit_f_2p(x, *arguments)
    nwalkers, ndim = 10, 2
    p0 = np.concatenate([
        np.random.normal(scale=3, size=(nwalkers, 1)) + 10,
        np.random.normal(scale=1.5, size=(nwalkers, 1)) + 21
    ], axis=1)
    badTmask = ((p0[:, 0]<0)|(p0[:, 0]>30))
    p0[badTmask, 0] = np.random.normal(scale=.5, size=p0[badTmask, 0].shape) + 10
    badNmask = ((p0[:, 1]<17)|(p0[:, 1]>26))
    p0[badNmask, 1] = np.random.normal(scale=.3, size=p0[badNmask, 1].shape) + 21
    print(nominal)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnposterior)
    sampler.run_mcmc(p0, niter+burn)
    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))

    plt.figure(figsize=(10, 7))
    for s in sampler.chain[:, -20:, :].reshape((-1, ndim)):
        cloudf = Greybody(s[0], 10**s[1], dust)
        plt.plot(wl_range, cloudf.radiate(nu_range), '-', alpha=0.1,
            color='grey')
    cloud = Greybody(nominal[0], 10**nominal[1], dust)
    plt.plot(wl_range, cloud.radiate(nu_range), '-', color=color,
        label=str(cloud))
    plt.errorbar(mpu.H_WL, obs, yerr=err, fmt='.', capsize=6,
        color=color, label=name)
    plt.xscale('log')
    plt.legend()
    plt.show()

    ### CHAIN PLOT
    # plt.figure()
    # for i in range(2):
    #     plt.subplot(211+i)
    #     for j in range(nwalkers):
    #         plt.plot(sampler.chain[j, :, i])
    #     plt.plot(np.arange(samples.shape[0])*sampler.chain.shape[1]/samples.shape[0],
    #         samples[:, i], '--', color='k', linewidth=3)
    fig = corner.corner(samples, labels=['Tc', 'Nc'], truths=nominal,
        range=[(13, 14.8), (21.8, 22.0)])
    fig.set_size_inches((10, 10))
    plt.title("emcee (n={:d})".format(niter*nwalkers))
    plt.show()
    return


def mtest_emcee_3p():
    # standard sampling, with a couple nifty plots
    # saves PDF of the corner plot
    # Currently also plots data
    pixel_index = 6
    niter, burn = 800, 400
    good, name, coords, color = mpu.PIXELS_OF_INTEREST[pixel_index]
    info_dict = mpu.get_manticore_info(manticore_soln_3p, *coords)
    nominal = [info_dict[x] for x in ("Tc", "Nh", "Nc") if x in info_dict]
    print("NOMINAL", nominal)
    print("nom err", [info_dict[x] for x in ("dTc", "dNh", "dNc") if x in info_dict])
    for i in (1, 2):
        nominal[i] = np.log10(nominal[i])
    print("NOMINAL", nominal)
    dust = [Dust(beta=1.80), Dust(beta=2.10)]
    nu_range = np.exp(np.linspace(np.log(cst.c/(1500*1e-6)), np.log(cst.c/(50*1e-6)), 100))
    wl_range = 1e6*cst.c/nu_range
    obs, err = mpu.get_obs(info_dict), mpu.get_err(info_dict)
    ## HALF PACS ERROR
    # err[0] = err[0]/2
    herschel = get_Herschel()
    dof = 2.
    Th = info_dict['Th']
    arguments = (dust, obs, err, herschel, Th, dof)
    def lnposterior(x):
        T, N1, N2 = x
        if T<0 or T>Th or N1<17 or N1>26 or N2<17 or N2>26:
            return -np.inf
        else:
            return -1.*mpfit.goodness_of_fit_f_3p(x, *arguments)
    nwalkers, ndim = 60, 3
    p0 = np.concatenate([
        np.random.normal(scale=3, size=(nwalkers, 1)) + 10,
        np.random.normal(scale=1.5, size=(nwalkers, 2)) + 21
    ], axis=1)
    print(p0.shape)
    badTmask = ((p0[:, 0]<0)|(p0[:, 0]>Th))
    p0[badTmask, 0] = np.random.normal(scale=.5, size=p0[badTmask, 0].shape) + 10
    badNmask = ((p0[:, 1]<17)|(p0[:, 1]>26))
    p0[badNmask, 1] = np.random.normal(scale=.3, size=p0[badNmask, 1].shape) + 21
    badNmask = ((p0[:, 2]<17)|(p0[:, 2]>26))
    p0[badNmask, 2] = np.random.normal(scale=.3, size=p0[badNmask, 2].shape) + 21
    print(p0)
    print("NOMINAL", nominal)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnposterior)
    sampler.run_mcmc(p0, niter+burn)
    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
    plt.figure(figsize=(10, 7))
    for s in sampler.chain[:, -20:, :].reshape((-1, ndim)):
        cloudf = Greybody([Th, s[0]], [10**x for x in s[1:]], dust)
        plt.plot(wl_range, cloudf.radiate(nu_range), '-', alpha=0.1,
            color='grey')
    cloud = Greybody([Th, nominal[0]], [10**x for x in nominal[1:]], dust)
    plt.plot(wl_range, cloud.radiate(nu_range), '-', color=color,
        label=str(cloud))
    plt.errorbar(mpu.H_WL, obs, yerr=err, fmt='.', capsize=6,
        color=color, label=name)
    plt.xscale('log')
    plt.ylim((0, 230))
    plt.legend()
    plt.show()

    ### CHAIN PLOT
    plt.figure(figsize=(16, 9))
    for i in range(3):
        plt.subplot(311+i)
        for j in range(nwalkers):
            plt.plot(sampler.chain[j, :, i])
        # plt.plot(np.arange(samples.shape[0])*sampler.chain.shape[1]/samples.shape[0],
        #     samples[:, i], '--', color='k', linewidth=3)
    plt.show()

    # simulated_data = mpu.deque() # plot fluxes in corner plot
    # for s in samples:
    #     cloudf = Greybody([Th, s[0]], [10**x for x in s[1:]], dust)
    #     simulated_data.append([d.detect(cloudf) for d in herschel])
    # a = np.stack(simulated_data, axis=0)
    # big_data = np.concatenate([samples, a], axis=1)
    # print(big_data.shape)
    # fig = corner.corner(big_data, labels=['Tc', 'Nh', 'Nc',
    #     'PACS160um', 'SPIRE250um', 'SPIRE350um', 'SPIRE500um'],
    #     truths=nominal+obs,
    #     # range=[(3, 16), (20, 21.6), (21.5, 23.5)])
    #     range=[(0, 16), (19.5, 21.6), (17., 23.5),
    #         (0, 250), (0, 250), (0, 250), (0, 250)])

    fig = corner.corner(samples, labels=['Tc', 'Nh', 'Nc'], truths=nominal,
        # range=[(3, 16), (20, 21.6), (21.5, 23.5)])
        range=[(0, 16), (19.5, 21.6), (17., 23.5)])
    fig.set_size_inches((10, 10))
    plt.title("emcee (n={:d})".format(niter*nwalkers))

    # fig.set_size_inches((20, 20))
    # plt.savefig("/home/ramsey/Desktop/test_corner.pdf")
    plt.show()
    return

def mtest_manyemcee():
    chainnum = 0
    info_dict = mpu.gen_CHAIN_dict(manticore_soln_3p, chain=chainnum)
    info_dict_2p = mpu.gen_CHAIN_dict(manticore_soln_2p, chain=chainnum)
    interp_Th = mpu.fill_T_chain((info_dict['Nc'] < 3e21), info_dict_2p['Tc'])
    dust_gen = lambda : [Dust(beta=1.80), Dust(beta=2.10)]
    for index in range(1):
        print(index, end=" ")
        fname = "./emcee_imgs/chain{:02d}_samples2_{:02d}.pkl".format(chainnum, index)
        mpu.emcee_3p(index, info_dict, chainnum=chainnum,
            niter=800, burn=100, nwalkers=60,
            instrument=get_Herschel(), dust=dust_gen(),
            goodnessoffit=mpfit.goodness_of_fit_f_3p,
            fig_fname="no plot", samples_fname=fname,
            Th=interp_Th[index])
    print('finished')

def mtest_manygrids_test():
    info_dict = mpu.gen_CHAIN_dict(manticore_soln_3p)
    dust_gen = lambda : [Dust(beta=1.80), Dust(beta=2.10)]
    Tclim, Nclim = (0, 16), (19, 22.5)
    Nhlim = (20, 21.5)
    dT, dN = 0.1, 0.05  # good resolution
    # dT, dN = 1, 0.5
    Tcrange = np.arange(*Tclim, dT)
    Nhrange = np.arange(*Nhlim, dN)
    Ncrange = np.arange(*Nclim, dN)
    Tcgrid, Nhgrid, Ncgrid = np.meshgrid(Tcrange, Nhrange, Ncrange, indexing='ij')
    for i in range(2, 35):
        gofgrid = np.empty(Tcgrid.size)
        with open('./emcee_imgs/logf.log', 'a') as wf:
            wf.write("{:d} !\n".format(i))
        mpu.grid_3d(i, info_dict, instrument=get_Herschel(), dust=dust_gen(),
            goodnessoffit=mpfit.goodness_of_fit_f_3p,
            Tcgrid=Tcgrid, Nhgrid=Nhgrid, Ncgrid=Ncgrid,
            empty_grid=gofgrid)
    print("finished")

def mtest_hidefgrid():
    Tclim, Nhlim, Nclim, dT, dN = mpu.LIMS_hidef_00
    # dT, dN = 0.3, 0.2
    Tcrange = np.arange(*Tclim, dT)
    Nhrange = np.arange(*Nhlim, dN)
    Ncrange = np.arange(*Nclim, dN)
    ranges = (Tcrange, Nhrange, Ncrange)
    Tcgrid, Nhgrid, Ncgrid = np.meshgrid(Tcrange, Nhrange, Ncrange, indexing='ij')
    print(Tcgrid.shape)
    print(Tcgrid.size)
    info_dict = mpu.gen_CHAIN_dict(manticore_soln_3p)
    index = 0
    gofgrid = np.empty(Tcgrid.size)
    with open("./emcee_imgs/logf.log", 'a') as wf:
        wf.write("beginning hi def grid\n")
    fname = "./emcee_imgs/grid1_00_HIDEF.pkl"
    mpu.grid_3d(index, info_dict, instrument=get_Herschel(),
        dust=[Dust(beta=1.80), Dust(beta=2.10)],
        goodnessoffit=mpfit.goodness_of_fit_f_3p,
        Tcgrid=Tcgrid, Nhgrid=Nhgrid, Ncgrid=Ncgrid,
        empty_grid=gofgrid,
        fname_override=fname)
    with open("./emcee_imgs/logf.log", 'a') as wf:
        wf.write("finished hi def grid\n")
    return

"""
######################## IMPORTANT ###########################################
"""
def mtest_write_manygrids():
    chainnum = 5
    Tclim, Nhlim, Nclim, dT, dN = mpu.LIMS_grid3
    Tcrange, Nhrange, Ncrange = mpu.genranges((Tclim, Nhlim, Nclim), (dT, dN))
    Tcgrid, Nhgrid, Ncgrid = mpu.gengrids((Tcrange, Nhrange, Ncrange))
    logf = "./emcee_imgs/logf_c{:1d}.log".format(chainnum)
    Path(logf).touch()
    print(Tcgrid.shape)
    print(Tcgrid.size)
    info_dict = mpu.gen_CHAIN_dict(manticore_soln_3p, chain=chainnum)
    for index in range(0, len(info_dict['Th']), 1):
        gofgrid = np.empty(Tcgrid.size)
        with open(logf, 'a') as wf:
            wf.write("beginning grid {:d}..\n".format(index))
        fname = "./emcee_imgs/chain{:02d}_grid3_{:02d}.pkl".format(chainnum, index)
        mpu.grid_3d(index, info_dict, instrument=get_Herschel(),
            dust=[Dust(beta=1.80), Dust(beta=2.10)],
            goodnessoffit=mpfit.goodness_of_fit_f_3p,
            Tcgrid=Tcgrid, Nhgrid=Nhgrid, Ncgrid=Ncgrid,
            empty_grid=gofgrid, chainnum=chainnum,
            fname_override=fname)
        with open(logf, 'a') as wf:
            wf.write("..finished {:d}\n".format(index))
    return


def mtest_renderhidefgrids():
    Tclim, Nhlim, Nclim, dT, dN = mpu.LIMS_grid2
    #Tclim, Nhlim, Nclim, dT, dN = mpu.LIMS_hidef_00
    ranges = mpu.genranges((Tclim, Nhlim, Nclim), (dT, dN))
    grids = mpu.gengrids(ranges)
    info_dict = mpu.gen_CHAIN_dict(manticore_soln_3p)
    from mayavi import mlab
    for index in [0]:
        fname = "./emcee_imgs/grid2_{:02d}.pkl".format(index)
        #savename = "./emcee_imgs/gridimg2_{:02d}.png".format(index)
        #fname = "./emcee_imgs/grid1_00_HIDEF.pkl"
        mpu.render_grid(index, info_dict, fname=fname,
            grids=grids, ranges=ranges, more_contours=False,
            focalpoint_nominal=False, mlab=mlab)
    return


"""
######################## IMPORTANT ###########################################
"""
def mtest_rendergrid():
    chainnum = 0
    from mayavi import mlab
    Tclim, Nhlim, Nclim, dT, dN = mpu.LIMS_grid2
    ranges = mpu.genranges((Tclim, Nhlim, Nclim), (dT, dN))
    grids = mpu.gengrids(ranges)
    info_dict = mpu.gen_CHAIN_dict(manticore_soln_3p, chain=chainnum)
    info_dict_2p = mpu.gen_CHAIN_dict(manticore_soln_2p, chain=chainnum)
    dusts = [Dust(beta=1.80), Dust(beta=2.10)]
    herschel = get_Herschel()
    Th, dof = info_dict['Th'][0], 1.
    Tscale = 4
    interp_Th = mpu.fill_T_chain((info_dict['Nc'] < 3e21), info_dict_2p['Tc'])
    irange = [0,0,] + list(range(len(info_dict['Th'])))
    for index in [0,]:
        fname = "./emcee_imgs/chain{:02d}_grid2_{:02d}.pkl".format(chainnum, index)
        print("opening ", fname)
        nominal = [info_dict[x][index] for x in mpu.P_LABELS]
        for x in (1, 2):
            nominal[x] = np.log10(nominal[x])
        chi_sq = info_dict['chi_sq'][index]
        print("-> Xs manticore: ", chi_sq)
        obs = [x[index] for x in mpu.get_obs(info_dict)]
        err = [x[index] for x in mpu.get_err(info_dict)]
        chi_sq = mpfit.goodness_of_fit_f_3p(nominal, dusts, obs, err, herschel, Th, dof)
        print("-> Xs calculated:", chi_sq)
        savename = "./emcee_imgs/chain{:02d}_gridimg3_{:02d}.png".format(chainnum, index)
        savename = None # also check KWarg
        try:
            with open('./emcee_imgs/chain{:02d}_samples1_{:02d}.pkl'.format(chainnum, index), 'rb') as pfl:
                mc_points = pickle.load(pfl)
            spk = {'color':(0.588, 0.090, 0.588), 'opacity':0.05, 'scale_factor':0.02}
        except FileNotFoundError:
            mc_points, spk = None, None
        mpu.render_grid(index, info_dict, fname=fname, savename=None,
            grids=grids, ranges=ranges, more_contours=True, Tscale=Tscale,
            focalpoint_nominal=True, mlab=mlab, noshow=True,
            scatter_points=mc_points, scatter_points_kwargs=spk)
        arbitrarily_complex = True # add more sample points
        if arbitrarily_complex:
            with open('./emcee_imgs/chain{:02d}_samples2_{:02d}.pkl'.format(chainnum, index), 'rb') as pfl:
                mc_points = pickle.load(pfl)
            spk = {'color':(0.419, 0.764, 0.074), 'opacity':0.05, 'scale_factor':0.02}
            mlab.points3d(mc_points[:, 0]/Tscale, mc_points[:, 1], mc_points[:, 2], **spk)
        cold_dominated_result = mpfit.fit_source_2p(obs, err, herschel, dusts[1])
        cold_dominated_result[0] /= Tscale
        cold_dom_T, cold_dom_N = ([cold_dominated_result[j] for x in range(2)] for j in range(2))
        mlab.plot3d(cold_dom_T, [Nhlim[0], Nhlim[1]], cold_dom_N,
            colormap='flag', tube_radius=.05, opacity=.3,)
        hot_dominated_result = mpfit.fit_source_1p(obs, err, herschel, dusts[0], Th=Th)
        hot_dominated_result[0] -= np.log10(2)
        mlab.plot3d([x/Tscale for x in Tclim], [hot_dominated_result[0] for x in range(2)], Nclim,
            colormap='flag', tube_radius=.05, opacity=.3,)
        # mlab.plot3d([x/Tscale for x in Tclim[::-1]], [hot_dominated_result[0] for x in range(2)], Nclim,
        #     colormap='flag', tube_radius=.05, opacity=.3,)
        hot_dominated_result = mpfit.fit_source_1p(obs, err, herschel, dusts[0], Th=interp_Th[index])
        hot_dominated_result[0] -= np.log10(2)
        mlab.plot3d([x/Tscale for x in Tclim[::-1]], [hot_dominated_result[0] for x in range(2)], Nclim,
            colormap='flag', tube_radius=.05, opacity=.3,)
        if info_dict['Nc'][index] < 3e21:
            print("This point would be masked OUT and left as single-component")
        else:
            print("This point would be masked IN and solved for two-component")
        if info_dict_2p['Nc'][index] < 3e21:
            print("This point would be masked OUT via single-T N")
        else:
            print("This point would be masked IN via single-T N")
        mlab.view(focalpoint=[nominal[0]/Tscale]+nominal[1:])
        mlab.show()
    return


"""
######################## IMPORTANT ###########################################
"""
def mtest_plot_params():
    plot_kwargs = {'color': 'green',
        'linewidth': 1, 'linestyle':'-', 'label':None,
        'marker':'^'}
    colors = ('green', 'blue', 'orange', 'navy', 'violet', 'firebrick')
    axes = None
    for i in range(6):
        plot_kwargs.update({'color': colors[i], 'label':f'{i}' })
        info_dicts = tuple(mpu.gen_CHAIN_dict(soln, chain=i) for soln in (manticore_soln_2p, manticore_soln_3p))
        axes = mpu.plot_parameters_across_filament(info_dicts, **plot_kwargs, axes=axes)
    axes['Tc 3p'].legend(loc='center right')
    axes['Tc 3p'].set_ylim([7.5, 14.5])
    axes['Nc 3p'].set_ylim([0, 2e22])
    focus_on_cold = False
    tie_N_limits = False
    if focus_on_cold and tie_N_limits:
        axes['Nh 3p'].set_ylim(axes['Nc 2p'].get_ylim())
    elif tie_N_limits:
        lim = [0, 1e22]
        axes['Nc 2p'].set_ylim(lim)
        axes['Nc 3p'].set_ylim(lim)
        axes['Nh 3p'].set_ylim(lim)
        # axes['Nc 2p'].set_ylim(axes['Nh 3p'].get_ylim())
    plt.show()
    return


def mtest_invenstigate_chain_errors():
    i=0
    info_dict = mpu.gen_CHAIN_dict(manticore_soln_3p, chain=i)
    if 1:
        axes = plt.subplots(nrows=3, ncols=2, sharex=True)[1].flatten('F')
        for i, l in enumerate(('Nh', 'err160', 'obs160', 'Tc', 'Nc',)):
            axes[i].plot(info_dict[l])
            axes[i].set_ylabel(l)
        plt.show()
        return
    else:
        axes = plt.subplots(nrows=4, ncols=2, sharex=True)[1].flatten('F')
        obserr = mpu.get_obs(info_dict) + mpu.get_err(info_dict)
        labels = [str(x) for x in [160, 250, 350, 500]]
        labels = ['o'+x for x in labels]+['e'+x for x in labels]
        for oe, ax, l in zip(obserr, axes, labels):
            ax.plot(oe)
            ax.set_title(l)
        plt.show()
        return

def mtest_carry_Th_over():
    i=0
    info_dict_3p = mpu.gen_CHAIN_dict(manticore_soln_3p, chain=i)
    edge_mask = info_dict_3p['Nc'] < 3e21
    c_edge_mask = edge_mask.copy()
    info_dict = mpu.gen_CHAIN_dict(manticore_soln_2p, chain=i)
    print(edge_mask)
    # plt.plot(info_dict['Tc'])
    newT = info_dict['Tc'].copy()
    newT[~edge_mask] = np.nan
    workingT = newT.copy()
    xs = np.arange(newT.size)
    def calcavg(xarr, pos, Tarr, mask):
        weights = mpu.gaussian_1d(xarr, pos, 10)[mask]
        weighted = weights*Tarr[mask]
        return np.sum(weighted)/np.sum(weights)
    while not np.all(edge_mask):
        first = 0
        while edge_mask[first]:
            first += 1
        last = len(edge_mask) - 1
        while edge_mask[last]:
            last -= 1
        avgT1 = calcavg(xs, first, workingT, edge_mask)
        if first != last:
            avgT2 = calcavg(xs, last, workingT, edge_mask)
        workingT[first] = avgT1
        edge_mask[first] = True
        if first != last:
            workingT[last] = avgT2
            edge_mask[last] = True
    herschel = get_Herschel()
    dusts = [Dust(beta=1.80), Dust(beta=2.10)]
    old_py_solution = {p: info_dict_3p[p].copy() for p in mpu.P_LABELS}
    old_Th = info_dict_3p['Th'][0]
    new_py_solution = {p: info_dict_3p[p].copy() for p in mpu.P_LABELS}
    for x in xs:
        if c_edge_mask[x]:
            for j, p in enumerate(mpu.P_LABELS):
                old_py_solution[p][x] = np.nan
                new_py_solution[p][x] = np.nan
            continue
        Th = workingT[x]
        obs = [o[x] for o in mpu.get_obs(info_dict)]
        err = [o[x] for o in mpu.get_err(info_dict)]
        old_py = mpfit.fit_source_3p(obs, err, herschel, dusts, Th=old_Th)
        new_py = mpfit.fit_source_3p(obs, err, herschel, dusts, Th=Th)
        for j, p in enumerate(mpu.P_LABELS):
            oldpyj = old_py[j] if j == 0 else 10**old_py[j]
            newpyj = new_py[j] if j == 0 else 10**new_py[j]
            old_py_solution[p][x] = oldpyj
            new_py_solution[p][x] = newpyj
    plt.figure()
    ax = plt.subplot(221)
    ax.plot(info_dict['Tc'], '--', label='2p', color='blue')
    ax.plot(info_dict_3p['Th'], '--', label='3p', color='red')
    ax.plot(workingT, label='new', color='green')
    ax.legend()
    ax.set_title('Th')
    ax = plt.subplot(223)
    ax.plot(info_dict['Tc'], '--', label='2p', color='blue')
    ax.plot(info_dict_3p['Tc'], '--', label='3p', color='red')
    ax.plot(new_py_solution['Tc'], label='new', color='green')
    ax.plot(old_py_solution['Tc'], '-.', label='old py', color='grey')
    ax.legend()
    ax.set_title('Tc')
    ax = plt.subplot(222)
    # ax.plot(info_dict['Nc'], '--', label='2p', color='blue')
    ax.plot(info_dict_3p['Nh'], '--', label='3p', color='red')
    ax.plot(new_py_solution['Nh'], label='new', color='green')
    ax.plot(old_py_solution['Nh'], '-.', label='old py', color='grey')
    ax.legend()
    ax.set_title('Nh')
    ax = plt.subplot(224)
    ax.plot(info_dict['Nc'], '--', label='2p', color='blue')
    ax.plot(info_dict_3p['Nc'], '--', label='3p', color='red')
    ax.plot(new_py_solution['Nc'], label='new', color='green')
    ax.plot(old_py_solution['Nc'], '-.', label='old py', color='grey')
    ax.legend()
    ax.set_title('Nc')
        # plt.plot(workingT, '--', color='red')
    plt.show()
    # info_dicts = (mpu.gen_CHAIN_dict(soln) for soln in manticore_soln_2p, manticore_soln_3p)


def mtest_write_boundary_solutions():
    print("ONLY GOOD ON DESKTOP")
    raise RuntimeWarning("THIS HAS ALREADY BEEN RUN!")
    cropnum = "-crop6"
    obs_maps, err_maps = [], []
    w = None
    for pair in gen_data_filenames(stub_append=cropnum):
        if w is None:
            w = mpu.WCS(mpu.fits.getdata(cconf.per1+pair[0], header=True)[1])
        obs_maps.append(mpu.fits.getdata(cconf.per1+pair[0]))
        err_maps.append(mpu.fits.getdata(cconf.per1+pair[1]))
    # standard fixed parameters
    dusth = Dust(beta=1.80)
    dustc = Dust(beta=2.10)
    Th = 15.95
    Tc0, Nh0, Nc0 = 10, 20, 22
    Tbound, Nbound = (0, None), (18, 25)
    herschel = get_Herschel()
    first_args = (obs_maps, err_maps, herschel)
    # 2 parameter COLD BETA (2.10); x = (Tc, Nc)
    src_fn = lambda x: Greybody(x[0], 10**x[1], dustc)
    solution, chisq = mpfit.fit_full_image(*first_args, src_fn,
        [Tc0, Nc0], (Tbound, Nbound), dof=2., chisq=True)
    write_data_dict = {
        ("Tc (2)", "K"): solution[0],
        ("Nc (2)", "cm-2"): 10**solution[1],
        ("chisq (2)", "Xs/dof"): chisq,
    }
    # now 1 parameter HOT BETA, fixed Th=15.95; x = (Nh,)
    src_fn = lambda x: Greybody(Th, 10**x[0], dusth)
    solution, chisq = mpfit.fit_full_image(*first_args, src_fn,
        [Nh0,], (Nbound,), dof=3., chisq=True)
    write_data_dict.update({
        ("Nh (1)", "cm-2"): 10**solution[0],
        ("chisq (1)", "Xs/dof"): chisq,
    })
    # now 3 parameter, fixed Th; x = (Tc, Nh, Nc); Tc>7
    src_fn = lambda x: Greybody([Th, x[0]], [10**x[1], 10**x[2]], [dusth, dustc])
    solution, chisq = mpfit.fit_full_image(*first_args, src_fn,
        [Tc0, Nh0, Nc0], ((7, None), Nbound, Nbound), dof=1., chisq=True)
    write_data_dict.update({
        ("Tc (3)", "K"): solution[0],
        ("Nh (3)", "cm-2"): 10**solution[1],
        ("Nc (3)", "cm-2"): 10**solution[2],
        ("chisq (3)", "Xs/dof"): chisq,
    })
    save_name = cconf.per1 + f"mantipyfit_boundary_solutions{cropnum}.fits"
    comment = "(2) beta=2.10; (1) beta=1.80, Th=15.95; (3) Tc>7"
    for k, v in write_data_dict.items():
        print(k, '-->', v.shape)
    try:
        mpu.save_fits(write_data_dict, w, save_name, comment=comment)
        print("worked!")
        return 0
    except:
        print("whoops, somethin happened")
        return write_data_dict


def mtest_init_conditions_sensitivity():
    print("ONLY GOOD ON DESKTOP")
    # raise RuntimeWarning("THIS HAS ALREADY BEEN RUN!")
    cropnum = "-crop6"
    obs_maps, err_maps = [], []
    write_data_dict = dict()
    w = None
    for pair in gen_data_filenames(stub_append=cropnum):
        if w is None:
            w = mpu.WCS(mpu.fits.getdata(cconf.per1+pair[0], header=True)[1])
        obs_maps.append(mpu.fits.getdata(cconf.per1+pair[0]))
        err_maps.append(mpu.fits.getdata(cconf.per1+pair[1]))
    # standard fixed parameters
    dusth = Dust(beta=1.80)
    dustc = Dust(beta=2.10)
    Th = 15.95
    Tc0, Nh0, Nc0 = 10, 20, 22
    Tbound, Nbound = (0, None), (18, 25)
    herschel = get_Herschel()
    first_args = (obs_maps, err_maps, herschel)
    # 3 parameter, fixed Th; x = (Tc, Nh, Nc); Tc>7
    # lowT/highN init conditions
    src_fn = lambda x: Greybody([Th, x[0]], [10**x[1], 10**x[2]], [dusth, dustc])
    solution, chisq = mpfit.fit_full_image(*first_args, src_fn,
        [8, 21.5, 22], ((0, None), Nbound, Nbound), dof=1., chisq=True)
    write_data_dict.update({
        ("A) Tc (init 8)", "K"): solution[0],
        ("A) Nh (init 3E21)", "cm-2"): 10**solution[1],
        ("A) Nc (init 1E22)", "cm-2"): 10**solution[2],
        ("A) chisq (T8)", "Xs/dof"): chisq,
    })
    # 3 parameter, fixed Th; x = (Tc, Nh, Nc); Tc>7
    # highT/lowN init conditions
    src_fn = lambda x: Greybody([Th, x[0]], [10**x[1], 10**x[2]], [dusth, dustc])
    solution, chisq = mpfit.fit_full_image(*first_args, src_fn,
        [14, 20.5, 20], ((0, None), Nbound, Nbound), dof=1., chisq=True)
    write_data_dict.update({
        ("B) Tc (init 14)", "K"): solution[0],
        ("B) Nh (init 3E20)", "cm-2"): 10**solution[1],
        ("B) Nc (init 1E20)", "cm-2"): 10**solution[2],
        ("B) chisq (T14)", "Xs/dof"): chisq,
    })
    save_name = cconf.per1 + f"mantipyfit_init_conditions_reasonable{cropnum}.fits"
    comment = "beta_c=2.10; beta_h=1.80; Th=15.95"
    for k, v in write_data_dict.items():
        print(k, '-->', v.shape)
    try:
        mpu.save_fits(write_data_dict, w, save_name, comment=comment)
        print("worked!")
        return 0
    except:
        print("whoops, somethin happened")
        return write_data_dict


def mtest_boostrap():
    chainnum = 6
    index = 28
    niter = 500
    info_dict = mpu.gen_CHAIN_dict(manticore_soln_3p, chain=chainnum)
    herschel = get_Herschel()
    dusts = [Dust(beta=1.80), Dust(beta=2.10)]
    obs = [x[index] for x in mpu.get_obs(info_dict)]
    err = [x[index] for x in mpu.get_err(info_dict)]
    #### HALF PACS ERROR
    # err[0] = err[0]/2
    Th = info_dict['Th'][index]
    Tscale = 4
    point_size = 0.2
    # get manticore answer
    nominal = [info_dict[x][index] for x in ("Tc", "Nh", "Nc") if x in info_dict]
    for x in (1, 2):
        nominal[x] = np.log10(nominal[x])
    # get python answer
    Tcf, Nhf, Ncf = mpfit.fit_source_3p(obs, err, herschel, dusts, Th=Th)
    print("manticore  >> Tc:{:5.2f}, Nh:{:.2E}, Nc:{:.2E}".format(*nominal))
    print("mantipyfit >> Tc:{:5.2f}, Nh:{:.2E}, Nc:{:.2E}".format(Tcf, 10**Nhf, 10**Ncf))
    # get bootstrap samples
    boostrap_samples, boostrap_errors = mpfit.bootstrap_errors(obs, err, herschel,
        dusts, niter=niter, fit_f=mpfit.fit_source_3p, dof=1, Th=Th, verbose=True)
    boostrap_samples = np.array(boostrap_samples) # rows are realizations, cols are Tc logNh logNc
    boostrap_samples[:, 1:] = np.log10(boostrap_samples[:, 1:])
    # get emcee samples
    emcee_samples = mpu.emcee_3p(index, info_dict, chainnum=chainnum,
        dust=dusts, instrument=herschel, goodnessoffit=mpfit.goodness_of_fit_f_3p,
        niter=200, burn=300, nwalkers=50, fig_fname="no plot", samples_fname="no save",
        Th=Th)
    nhmod = lambda a, b: np.log10(10**b + 10**a)
    ncmod = lambda a, b: b - a
    mod_bs_samp = boostrap_samples.copy()
    mod_bs_samp[:, 1] = nhmod(boostrap_samples[:, 1], boostrap_samples[:, 2])
    mod_bs_samp[:, 2] = ncmod(boostrap_samples[:, 1], boostrap_samples[:, 2])
    mod_mc_samp = emcee_samples.copy()
    mod_mc_samp[:, 1] = nhmod(emcee_samples[:, 1], emcee_samples[:, 2])
    mod_mc_samp[:, 2] = ncmod(emcee_samples[:, 1], emcee_samples[:, 2])
    mod_nominal = nominal.copy()
    mod_nominal[1] = nhmod(nominal[1], nominal[2])
    mod_nominal[2] = ncmod(nominal[1], nominal[2])
    lims = [0, 16, 21, 23, -3, 3]
    from mayavi import mlab
    fig = mpu.render_points(mod_bs_samp, nominal=mod_nominal, Tscale=Tscale, mlab=mlab,
        scale_factor=0.05, color=(0.219, 0.588, 0.192), lims=lims, noshow=True,
        ax_labels=("Tc", "sum", "ratio Nc/Nh"))
    mpu.render_points(mod_mc_samp, figure=fig, Tscale=Tscale, mlab=mlab,
        scale_factor=0.05, color=(0.588, 0.090, 0.588), lims=lims,
        focalpoint=[8, 20, 0], ax_labels=("Tc", "sum", "ratio Nc/Nh"))
    return


def mtest_boostrap_proper_error():
    chainnum = 0
    index = 33
    # the errors in _staterr are JUST the off-the-archive map err
    info_dict = mpu.gen_CHAIN_dict(cconf.manticore_soln_3p_staterr, chain=chainnum)
    mpu.adjust_uncertainties(info_dict,
        lambda o, e: np.sqrt(e**2 + (o*1.5/100.)**2), # add 1.5%
        exception=(lambda k: "160" in k)) # to SPIRE
    mpu.adjust_uncertainties(info_dict,
        lambda o, e: np.sqrt(e**2 + (o*5/100.)**2), # add 5%
        exception=(lambda k: "160" not in k)) # to PACS
    # nu_range = np.exp(np.linspace(np.log(cst.c/(1500*1e-6)), np.log(cst.c/(50*1e-6)), 100))
    # wl_range = 1e6*cst.c/nu_range
    herschel = get_Herschel()
    dusts = [Dust(beta=1.80), Dust(beta=2.10)]
    obs = [x[index] for x in mpu.get_obs(info_dict)]
    err = [x[index] for x in mpu.get_err(info_dict)]
    Th = info_dict['Th'][index]
    Tscale = 4
    point_size = 0.2
    # get manticore answer
    nominal = [info_dict[x][index] for x in ("Tc", "Nh", "Nc")]
    for x in (1, 2):
        nominal[x] = np.log10(nominal[x])
    # get python answer
    Tcf, Nhf, Ncf = mpfit.fit_source_3p(obs, err, herschel, dusts, Th=Th)
    print("manticore  >> Tc:{:5.2f}, Nh:{:.2E}, Nc:{:.2E}".format(*nominal))
    print("mantipyfit >> Tc:{:5.2f}, Nh:{:.2E}, Nc:{:.2E}".format(Tcf, Nhf, Ncf))
    # set up correlated error
    def spire_correlated_err(o, e):
        # first draw from statistical uncertanties already present
        result = np.random.normal(loc=o, scale=e)
        # now draw from 4% distribution and add to each spire band
        result[1:] *= np.random.normal(loc=1, scale=0.04)
        return result
    """
    WE NEED TO TEST THIS (wrote this at midnight on Jul 22 2019)
    """
    # bootstrap samples
    bsargs = (obs, err, herschel, dusts,)
    bskwargs = dict(niter=50, fit_f=mpfit.fit_source_3p, dof=1, Th=Th,
        verbose=True, samples_only=True, method='SLSQP')
    def clean_bootstrap_results(array):
        array = np.array(array)
        array[:, 1:] = np.log10(array[:, 1:])
        return array
    # 1: 1.5% SPIRE + 4% correlated
    boostrap_samples1 = mpfit.bootstrap_errors(*bsargs, **bskwargs,
        perturbation_f=spire_correlated_err) # rows are realizations, cols are Tc logNh logNc
    boostrap_samples1 = clean_bootstrap_results(boostrap_samples1)
    # 2: 1.5% SPIRE
    boostrap_samples2 = mpfit.bootstrap_errors(*bsargs, **bskwargs)
    boostrap_samples2 = clean_bootstrap_results(boostrap_samples2)
    # 3: 5.5% SPIRE
    mpu.reset_uncertainties(info_dict)
    mpu.adjust_uncertainties(info_dict,
        lambda o, e: np.sqrt(e**2 + (o*5.5/100.)**2), # add 5.5%
        exception=(lambda k: "160" in k)) # to SPIRE
    mpu.adjust_uncertainties(info_dict,
        lambda o, e: np.sqrt(e**2 + (o*5/100.)**2), # add 5%
        exception=(lambda k: "160" not in k)) # to PACS
    boostrap_samples3 = mpfit.bootstrap_errors(*bsargs, **bskwargs)
    boostrap_samples3 = clean_bootstrap_results(boostrap_samples3)
    from mayavi import mlab
    magenta = (0.588, 0.090, 0.588)
    green = (0.219, 0.588, 0.192)
    blue = (0.478, 0.725, 0.780)
    pkwargs = dict(Tscale=Tscale, mlab=mlab, scale_factor=0.05, opacity=0.2)
    fig = mpu.render_points(boostrap_samples1, color=green,
        noshow=True, setup_ax=False, **pkwargs)
    fig = mpu.render_points(boostrap_samples2, color=blue, figure=fig,
        noshow=True, setup_ax=False, **pkwargs)
    fig = mpu.render_points(boostrap_samples3, color=magenta, figure=fig,
        ax_labels=("Tc", "Nh", "Nc"), nominal=nominal,
        focalpoint_nominal=True, **pkwargs)
    # get emcee samples
    # emcee_samples = mpu.emcee_3p(index, info_dict, chainnum=chainnum,
    #     dust=dusts, instrument=herschel, goodnessoffit=mpfit.goodness_of_fit_f_3p,
    #     niter=200, burn=300, nwalkers=40, fig_fname="no plot", samples_fname="no save",
    #     Th=Th)
    # fig = mpu.render_points(emcee_samples, color=magenta, figure=fig,
    #     ax_labels=("Tc", "Nh", "Nc"), nominal=nominal, **pkwargs)
    return

if __name__ == "__main__":
    print('nothing here')
