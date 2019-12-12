import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from datetime import datetime, timezone
import scipy.constants as cst
from collections import deque
import emcee
import corner
import pickle
import matplotlib.pyplot as plt
import sys

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

H_WL = [160, 250, 350, 500]
H_stubs = {
    160: "PACS160um",
    250: "SPIRE250um",
    350: "SPIRE350um",
    500: "SPIRE500um",
}

MANTICORE_KEYS = [
    "Tc", "dTc", "Nc", "dNc",
    "Th", "dTh", "Nh", "dNh",
    "obs160", "obs250", "obs350", "obs500",
    "err160", "err250", "err350", "err500",
    "mod160", "mod250", "mod350", "mod500",
    "chi_sq"
]
# --------
# indices for all these are:
MANTICORE_INDICES_3p = [
    1, 2, 3, 4,
    5, 6, 7, 8,
    15, 17, 19, 21,
    16, 18, 20, 22,
    11, 12, 13, 14,
    9,
]
MANTICORE_INDICES_2p = [
    1, 2, 3, 4,
    None, None, None, None,
    11, 13, 15, 17,
    12, 14, 16, 18,
    7, 8, 9, 10,
    5,
]

PIXELS_OF_INTEREST = (
    (1, 'B1', (551-1, 307-1), 'dodgerblue'),
    (1, 'NGC 1333', (590-1, 520-1), 'midnightblue'),
    (1, 'L1455', (353-1, 554-1), 'forestgreen'),
    (0, 'L1455', (465-1, 513-1), 'olive'), # dNh larger than Nh
    (0, 'L1455', (425-1, 401-1), 'darkmagenta'),
    (0, 'B1', (587-1, 264-1), 'salmon'), # dNc ~ 0.5*Nc
    (0, 'B1', (587-1, 260-1), 'firebrick'), # Tc ~ 4 K
    (0, 'B1', (587, 260), 'deeppink'), # Tc ~ 6 K
)

PIXEL_CHAINS = (
    ((554-1, 271-1), (588-1, 305-1)), # 0 just southwest of B1 (diag NW)
    ((507-1, 321-1), (529-1, 343-1)), # 1 more SW of B1 than previous (diag NW)
    ((587-1, 483-1), (587-1, 552-1)), # 2 south of NGC 1333 (horiz W)
    ((381-1, 671-1), (347-1, 705-1)), # 3 just southeast of L1448 (diag SW)
    ((365-1, 512-1), (390-1, 537-1)), # 4 close to L1455 (diag NW)
    ((389-1, 427-1), (416-1, 454-1)), # 5 near L1455, up towards B1 (diag NW)
    ((642-1, 234-1), (677-1, 234-1)), # 6 past B1, towards B1-E (vertical N)
)

def gen_CHAIN_dict(source, chain=0):
    start, end = PIXEL_CHAINS[chain]
    # ALL chains should head west!
    if (end[0] > start[0]) and (end[1] > start[1]):
        # diagonal cut, heading northwest
        coord_fn = lambda coord: tuple(x+1 for x in coord)
        length = end[0] - start[0] + 1
    elif (end[1] > start[1]) and end[0] == start[0]:
        # horizontal cut, heading west
        coord_fn = lambda coord: (coord[0], coord[1]+1)
        length = end[1] - start[1] + 1
    elif (end[1] > start[1]) and (end[0] < start[0]):
        # diagonal cut, heading southwest
        coord_fn = lambda coord: (coord[0]-1, coord[1]+1)
        length = end[1] - start[1] + 1
    elif (end[0] > start[0]) and (end[1] == start[1]):
        # vertical cut, heading north
        coord_fn = lambda coord: (coord[0]+1, coord[1])
        length = end[0] - start[0] + 1
    else:
        # ill-defined
        raise RuntimeError("gen_CHAIN_dict not prepared for this!")
    coords = []
    current_coord = start
    count, max_count = 0, 200 # none of these is 200 elements long
    while count < length:
        coords.append(current_coord)
        current_coord = coord_fn(current_coord)
        count += 1
        if count > max_count:
            raise RuntimeError("max interations reached")
    assert all(x==y for x, y in zip(coords[-1], end))
    return get_manticore_info(source, tuple(coords))

LIMS_hidef_00 = (
    (11.645157051086425, 13.245157051086426,), # Tc
    (20.694322967529295, 20.8943229675293,), # Nh
    (20.65826873779297, 21.058268737792968,), # Nc
    0.01, 0.0025) # dT, dN (arange)

LIMS_grid1 = ((0, 16.), (20, 21.5), (19, 22.5), 0.1, 0.05) # Tc,Nh,Nc,dT,dN (arange)
LIMS_grid2 = ((4, 16.01), (20, 21.21), (19, 22.41), 0.05, 0.025) # same ^
LIMS_grid3 = ((4, 16.), (20, 21.25), (19, 22.5), 0.1, 0.05) # Tc,Nh,Nc,dT,dN (arange)
LIMS_grid4 = ((4, 16.), (19.5, 21.25), (20, 22.5), 0.1, 0.05) # Tc,Nh,Nc,dT,dN (arange)

def genranges(lims, differentials):
    # lims: Tc, Nh, Nc
    dT, dN = differentials
    return [np.arange(*l, d) for l, d in zip(lims, (dT, dN, dN))]

def gengrids(ranges):
    # output of genranges
    return np.meshgrid(*ranges, indexing='ij')

P_LABELS = ('Tc', 'Nh', 'Nc')
PE_LABELS = ('dTc', 'dNh', 'dNc')
OBS_LABELS = ("obs160", "obs250", "obs350", "obs500")
ERR_LABELS = ("err160", "err250", "err350", "err500")
ORIG_ERR_LABELS = ("Xerr160", "Xerr250", "Xerr350", "Xerr500")
MOD_LABELS = ("mod160", "mod250", "mod350", "mod500")

def get_obs(info_dict):
    return [info_dict[x] for x in OBS_LABELS]

def get_err(info_dict):
    return [info_dict[x] for x in ERR_LABELS]

def get_mod(info_dict):
    return [info_dict[x] for x in MOD_LABELS]

def adjust_uncertainties(info_dict, f, exception=None):
    """
    f should be a function of (obs, err) for a given band
    f should return an array the same shape as obs and err
        e.g. f = lambda obs, err: sqrt(err**2 + (obs*0.05)**2)
        would add 5% of the flux to the error
    exception should take the obs key and return True/False
    if exception(key) returns true, does NOT modify that error
        e.g. exception = lambda key: "160" in key
        would leave PACS error untouched
    you'll have to do anything more complex manually
    this will write backups of original errors to ORIG_ERR_LABELS keys
        but it won't overwrite existing backups,
        in case you're layering modifications
    """
    for o_key, e_key, xe_key in zip(OBS_LABELS, ERR_LABELS, ORIG_ERR_LABELS):
        if exception is not None and exception(o_key):
            continue
        if xe_key not in info_dict: # unless there already is a backup
            info_dict[xe_key] = info_dict[e_key]
        info_dict[e_key] = f(info_dict[o_key], info_dict[e_key])


def reset_uncertainties(info_dict):
    # undo adjustments made to uncertainties
    for e_key, xe_key in zip(ERR_LABELS, ORIG_ERR_LABELS):
        if xe_key in info_dict: # if there is a backup; if it has been modified
            info_dict[e_key] = info_dict[xe_key]
            del info_dict[xe_key]


def get_manticore_info(source, *args):
    # i is the ROW, which is Y in FITS
    # zero indexed, so subtract 1 from FITS coordinates
    return_dict = dict()

    if isinstance(source, str):
        hdul = fits.open(source)
    else:
        hdul = source

    if len(args) == 2:
        mask = (args[0], args[1])
    elif isinstance(args[0], tuple):
        mask = tuple(zip(*args[0]))
    elif isinstance(args[0], np.ndarray):
        mask = args[0].astype(bool)
    else:
        msg = ["Unknown call signature ({:d} args):".format(len(args))]
        msg += [str(a) for a in args]
        raise RuntimeError(" ".join(msg))

    indices = MANTICORE_INDICES_3p if len(hdul) > 20 else MANTICORE_INDICES_2p

    for k, idx in zip(MANTICORE_KEYS, indices):
        if idx is not None:
            return_dict[k] = hdul[idx].data[mask]
    # but the last four (mod) need to be added to the obs
    for n in range(4):
        # obs is 8-11, so n+8
        # mod is 16-19, so n+16
        return_dict[MANTICORE_KEYS[n+16]] += return_dict[MANTICORE_KEYS[n+8]]
    if isinstance(source, str):
        hdul.close()
    return return_dict


def save_fits(data, wcs_info, save_name, comment=""):
    # data should be a dictionary
    #   from tuple (string extension_name, string units)
    #   to ndarray data_array
    phdu = fits.PrimaryHDU()
    header = fits.Header()
    header.update(wcs_info.to_header())
    header['COMMENT'] = comment
    header['CREATOR'] = (f"Ramsey: {str(__file__)}", "FITS file creator")
    header['OBJECT'] = ("per_04_nom", "Target name")
    header['DATE'] = (datetime.now(timezone.utc).astimezone().isoformat(), "File creation date")
    hdu_list = []
    for ext_name, ext_units in data:
        ihdu = fits.ImageHDU(data[(ext_name, ext_units)], header=header)
        ihdu.header['EXTNAME'] = ext_name
        ihdu.header['BUNIT'] = (ext_units, 'Data unit')
        hdu_list.append(ihdu)
    hdu_list.insert(0, phdu)
    hdul = fits.HDUList(hdu_list)
    try:
        hdul.writeto(save_name)
        print(f"WRITTEN to {save_name}")
    except OSError:
        hdul.writeto(save_name, overwrite=True)
        print(f"WRITTEN TO {save_name} (overwriting existing)")


prep_arr = lambda a, b: np.array([a, b]).T.flatten()
def histogram(x, x_lim=None):
    if x_lim is None:
        x_lim = (np.min(x), np.max(x))
    dhist, dedges = np.histogram(x.ravel(), bins=BINS, range=x_lim)
    histx, histy = prep_arr(dedges[:-1], dedges[1:]), prep_arr(dhist, dhist)
    return histx, histy


def emcee_3p(index, info_dict, chainnum=0,
        dust=None, instrument=None, goodnessoffit=None,
        niter=800, burn=400, nwalkers=60,
        fig_fname=None, samples_fname=None,
        Th=None):
    # to avoid saving png image, fig_name="no plot"
    # to avoid saving sample array, samples_name="no save"
    ndim = 3
    p_labels = ('Tc', 'Nh', 'Nc')
    nominal = [info_dict[x][index] for x in p_labels]
    if Th is None:
        Th = info_dict['Th'][index]
    for i in (1, 2):
        nominal[i] = np.log10(nominal[i])
    if dust is None:
        raise RuntimeError("dust!")
    if not isinstance(dust, list):
        dust = dust()
    if instrument is None:
        raise RuntimeError("instrument!")
    if not isinstance(instrument, list):
        instrument = instrument()
    if goodnessoffit is None:
        raise RuntimeError("goodnessoffit!")
    obs = [x[index] for x in get_obs(info_dict)]
    err = [x[index] for x in get_err(info_dict)]
    dof = 1.
    arguments = (dust, obs, err, instrument, Th, dof)
    Tlo, Thi, Nlo, Nhi = 0., Th, 18., 25.
    Tcenter, Ncenter = 10, 21
    def lnposterior(x):
        T, N1, N2 = x
        if T<Tlo or T>Th or N1<Nlo or N1>Nhi or N2<Nlo or N2>Nhi:
            return -np.inf
        else:
            return -1*goodnessoffit(x, *arguments)
    p0 = np.concatenate([
        np.random.normal(scale=3, size=(nwalkers, 1)) + Tcenter,
        np.random.normal(scale=1.5, size=(nwalkers, 2)) + Ncenter
    ], axis=1)
    badTmask = ((p0[:, 0]<Tlo)|(p0[:, 0]>Thi))
    p0[badTmask, 0] = np.random.normal(scale=.5, size=p0[badTmask, 0].shape) + Tcenter
    badNmask = ((p0[:, 1]<Nlo)|(p0[:, 1]>Nhi))
    p0[badNmask, 1] = np.random.normal(scale=.3, size=p0[badNmask, 1].shape) + Ncenter
    badNmask = ((p0[:, 2]<Nlo)|(p0[:, 2]>Nhi))
    p0[badNmask, 2] = np.random.normal(scale=.3, size=p0[badNmask, 2].shape) + Ncenter
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnposterior)
    sampler.run_mcmc(p0, niter+burn)
    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
    if fig_fname != "no plot":
        fig = corner.corner(samples, labels=p_labels, truths=nominal,
            range=[(0, 16), (19.5, 21.6), (17., 23.5)],)
        fig.set_size_inches((10, 10))
        plt.title("pixel #{:02d}, n={:d}".format(index, niter*nwalkers))
        if fig_fname:
            fname = fig_fname
        else:
            fname = "./emcee_imgs/chain{:02d}_corner1_{:02d}.pdf".format(chainnum, index)
        plt.savefig(fname)
    if samples_fname != "no save":
        if samples_fname:
            fname = samples_fname
        else:
            fname = "./emcee_imgs/chain{:02d}_samples1_{:02d}.pkl".format(chainnum, index)
        with open(fname, 'wb') as pfl:
            pickle.dump(samples, pfl)
    return samples

def grid_3d(index, info_dict,
    chainnum=0, dust=None, instrument=None,
    goodnessoffit=None,
    Tcgrid=None, Nhgrid=None, Ncgrid=None,
    empty_grid=None, fname_override=None):
    p_labels = ('Tc', 'Nh', 'Nc')
    nominal = [info_dict[x][index] for x in p_labels]
    Th = info_dict['Th'][index]
    for i in (1, 2):
        nominal[i] = np.log10(nominal[i])
    if dust is None:
        raise RuntimeError("dust!")
    if not isinstance(dust, list):
        dust = dust()
    if instrument is None:
        raise RuntimeError("instrument!")
    if not isinstance(instrument, list):
        instrument = instrument()
    if goodnessoffit is None:
        raise RuntimeError("goodnessoffit!")
    obs = [x[index] for x in get_obs(info_dict)]
    err = [x[index] for x in get_err(info_dict)]
    dof = 1.
    arguments = (dust, obs, err, instrument, Th, dof)
    own_grid = (empty_grid is None)
    if own_grid:
        Tclim, Nclim = (0, 16), (19, 22.5)
        Nhlim = (20, 21.5)
        dT, dN = 0.1, 0.05
        Tcrange, Nhrange, Ncrange = genranges((Tclim, Nhlim, Nclim), (dT, dN))
        Tcgrid, Nhgrid, Ncgrid = gengrids(Tcrange, Nhrange, Ncrange)
        empty_grid = np.empty(Tcgrid.size)
    for i, pvec in enumerate(zip(Tcgrid.ravel(), Nhgrid.ravel(), Ncgrid.ravel())):
        gof = goodnessoffit(pvec, *arguments)
        empty_grid[i] = gof
    empty_grid = np.log10(empty_grid.reshape(Tcgrid.shape))
    if fname_override is not None:
        fname = fname_override
    else:
        fname = "./emcee_imgs/chain{:02d}_grid1_{:02d}.pkl".format(chainnum, index)
    with open(fname, 'wb') as pfl:
        pickle.dump(empty_grid, pfl)
    return empty_grid


def render_points(points, nominal=None, Tscale=4, focalpoint_nominal=False,
    color=(0.219, 0.588, 0.192), opacity=0.4, scale_factor=0.1,
    mlab=None, noshow=False, figure="main",
    fig_size=(1200, 1050), nominal_point_size=0.2,
    lims=[0, 20, 18, 23, 18, 23], ext=None, ax_labels=("Tc", "Nh", "Nc"),
    focalpoint=[10., 20.75, 20.75], setup_ax=True):
    # assumes the points are Tc logNh logNc within reasonable bounds
    focalpoint[0] /= Tscale
    if mlab is None:
        raise RuntimeError("Please pass the mlab module as kwarg 'mlab'")
    fig = mlab.figure(figure=figure, size=fig_size)
    mlab.points3d(points[:, 0]/Tscale, points[:, 1], points[:, 2],
        color=color, opacity=opacity, scale_factor=scale_factor,)
    if nominal is not None:
        if nominal[2] > 100:
            nominal = [nominal[0]/Tscale, np.log10(nominal[1]), np.log10(nominal[2])]
        else:
            nominal = [nominal[0]/Tscale, *nominal[1:]]
        if focalpoint_nominal:
            focalpoint = nominal
        mlab.points3d(*([x] for x in nominal),
            colormap='flag', mode='axes', scale_factor=nominal_point_size, line_width=4)
        for x in ("x", "y", "z"):
            pts = mlab.points3d(*([x] for x in nominal),
                colormap='flag', mode='axes', scale_factor=nominal_point_size, line_width=4)
            eval("pts.glyph.glyph_source._trfm.transform.rotate_{:s}(45)".format(x))
    if setup_ax:
        if ext is None:
            ext = [x if i > 1 else x/Tscale for i, x in enumerate(lims)]
        mlab.axes(ranges=lims, extent=ext,
            nb_labels=5, xlabel=ax_labels[0],
            ylabel=ax_labels[1], zlabel=ax_labels[2])
        mlab.view(azimuth=45., elevation=92., distance=9.,
            focalpoint=focalpoint)
    if not noshow:
        mlab.show()
    return fig


def render_grid(index, info_dict, fname=None, savename=None,
    gofgrid=None, grids=None, ranges=None,
    fig_size=(1200, 1050), Tscale=4,
    more_contours=False, point_size=0.2, focalpoint_nominal=False,
    scatter_points=None, scatter_points_kwargs={},
    mlab=None, noshow=False):
    # mayavi rendering of some contours over chi squared surface
    if mlab is None:
        raise RuntimeError("Please pass the mlab module as kwarg 'mlab'")
    if grids is None:
        raise RuntimeError("Can't just leave grids blank these days")
    Tcgrid, Nhgrid, Ncgrid = grids
    if ranges is None:
        raise RuntimeError("Can't just leave ranges blank these days")
    Tcrange, Nhrange, Ncrange = ranges
    if (fname is None) and (gofgrid is None):
        # gofgrid, if given, is assumed to be multiplied by -1 and un-logged already
        raise RuntimeError("Give filename of grid pickle or the actual grid")
    nominal = [info_dict[x][index] for x in P_LABELS]
    for x in (1, 2):
        nominal[x] = np.log10(nominal[x])
    chi_sq = info_dict['chi_sq'][index]
    if gofgrid is None:
        with open(fname, 'rb') as pfl:
            gofgrid = -1*(10**pickle.load(pfl))
    fig = mlab.figure(figure="main", size=fig_size)
    src = mlab.pipeline.scalar_field(Tcgrid/Tscale, Nhgrid, Ncgrid, gofgrid)
    if more_contours:
        ####### Grey Xs=100 contour
        mlab.pipeline.iso_surface(src, contours=[-100,],
            opacity=0.1, vmin=-101, vmax=-100, colormap='gist_yarg')
        ####### Blue Xs~5 contours
        mlab.pipeline.iso_surface(src, contours=[-5, -3],
            colormap='cool', opacity=0.15, vmin=-8, vmax=-2)
    ####### Red/Yellow Xs~1 contours
    mlab.pipeline.iso_surface(src, contours=[-1.5, -1, -.5],
        colormap='hot', opacity=0.25, vmin=-2, vmax=-.3)
    ####### Axes
    mlab.axes(ranges=sum(([x.min(), x.max()] for x in (Tcrange, Nhrange, Ncrange)), []),
        extent=sum(([x.min(), x.max()] for x in (Tcrange/Tscale, Nhrange, Ncrange)), []),
        nb_labels=5, xlabel="Tc", ylabel='Nh', zlabel="Nc")
    ####### Title
    mlab.title("pt({:02d}) [Tc: {:04.1f}, Nh: {:4.1f}, Nc: {:4.1f}], ChiSq: {:6.2f}".format(
        index, *nominal, chi_sq),
        size=0.25, height=.9)
    nominal[0] /= Tscale # since the grid is rescaled
    ####### Manticore solution point
    mlab.points3d(*([x] for x in nominal),
        colormap='flag', mode='axes', scale_factor=point_size, line_width=4)
    for x in ("x", "y", "z"):
        pts = mlab.points3d(*([x] for x in nominal),
            colormap='flag', mode='axes', scale_factor=point_size, line_width=4)
        eval("pts.glyph.glyph_source._trfm.transform.rotate_{:s}(45)".format(x))
    ####### Scatter additional points
    if scatter_points is not None:
        scatter_points_kwargs['color'] = scatter_points_kwargs.get('color', (0.219, 0.588, 0.192))
        scatter_points_kwargs['opacity'] = scatter_points_kwargs.get('opacity', 0.4)
        scatter_points_kwargs['scale_factor'] = scatter_points_kwargs.get('scale_factor', 0.1)
        mlab.points3d(scatter_points[:, 0]/Tscale, scatter_points[:, 1], scatter_points[:, 2],
            **scatter_points_kwargs)
    ####### Favorable camera angle
    if focalpoint_nominal:
        focalpoint = nominal
    else:
        focalpoint = [10./Tscale, 20.75, 20.75]
    mlab.view(azimuth=45., elevation=92., distance=9.,
        focalpoint=focalpoint)
    if savename is not None:
        mlab.savefig(savename, figure=fig, magnification=1)
        mlab.clf()
    elif not noshow:
        mlab.show()
    return

def plot_parameters_across_filament(info_dicts,
        i0=0, size=None, axes=None, **plot_kwargs):
    # plot parameters from info_dicts [i0:i0+size]
    # each parameter gets an axis; should be 6 with Xs
    # info_dicts is (2p, 3p)
    # axis_labels = list(p+' 3p' for p in P_LABELS) + list(p+' 2p' for p in P_LABELS if 'h' not in p) + ['chi_sq']
    axis_labels = ['Tc 3p', 'Tc 2p', 'Nh 3p', 'chi_sq', 'Nc 3p', 'Nc 2p']
    if axes is None:
        # switch "F" to "C" for different axis stacking!
        axes_list = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(12, 9))[1].flatten('F')
        axes = {label: ax for label, ax in zip(axis_labels, axes_list)}
    if size is None:
        size = len(info_dicts[0]['Tc'])
    i1 = i0+size
    impact_parameter = range(-size//2, size//2)
    # print("+++++", len(impact_parameter), size)
    for ax_label in axes:
        ax = axes[ax_label]
        if len(ax_label.split()) == 1:
            # chi_sq
            chi_sqs = (idict[ax_label] for idict in info_dicts)
            for chi_sq, m, l in zip(chi_sqs, ('$2$', '$3$'), ('--', '-')):
                ax.plot(impact_parameter, chi_sq[i0:i1], marker=m, linestyle=l,
                    **{k:v for k, v in plot_kwargs.items() if k not in ('marker', 'linestyle')})
            ax.set_xlabel("impact parameter")
        else:
            # all other parameters
            p, l = ax_label.split()
            ax.errorbar(impact_parameter, info_dicts[int(l=='3p')][p][i0:i1],
                yerr=info_dicts[int(l=='3p')]['d'+p][i0:i1], capsize=2,
                **plot_kwargs)
        ax.set_ylabel(ax_label)
    return axes


def plot_stacked_SEDs(parameter_array, x_plot_array, src_fn,
    x_array=None, color='k', ax=None):
    if x_array is None:
        x_array = x_plot_array
    if ax is None:
        plt.figure()
        ax = plt.subplot(111)
    # plot all parameter sets
    for p_set in parameter_array:
        src = src_fn(p_set)
        ax.plot(x_plot_array, src.radiate(x_array), '-', alpha=0.1,
            color='grey')
    return


def gaussian_1d(x, mu, sigma):
    # exactly what it looks like
    return np.exp(-(((x-mu)/sigma)**2)/2.)

def calc_avg_1d(xarr, pos, Tarr, mask, gauss_sigma):
    # gaussian sum over nearby "edge" pixels
    # mask is True where "edge" (False where filament)
    weights = gaussian_1d(xarr, pos, gauss_sigma)[mask]
    weighted = weights*Tarr[mask]
    return np.sum(weighted)/np.sum(weights)

def fill_T_chain(mask_N, T_array, kernel_width=10):
    # 1D Th fill designed for a single chain of pixels
    # mask_N is the result of masking on column density
    #   should be True when "off-filament" or "edge"
    # T_array is the single-component T result
    # kernel_width is std dev of gaussian used for avg
    mask_N, T_array = mask_N.copy(), T_array.copy()
    T_array[~mask_N] = np.nan
    x_array = np.arange(T_array.size)
    first, last = 0, len(T_array)-1
    while mask_N[first]:
        first += 1
    while mask_N[last]:
        last -= 1
    while not np.all(mask_N):
        # if first == last, then one operation is redundant but safe
        avg_first = calc_avg_1d(x_array, first, T_array, mask_N, kernel_width)
        avg_last = calc_avg_1d(x_array, last, T_array, mask_N, kernel_width)
        T_array[first] = avg_first
        T_array[last] = avg_last
        mask_N[first] = True
        mask_N[last] = True
        first += 1
        last -= 1
    return T_array
