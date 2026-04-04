"""
Visualization tools for uv-fitting.

- UV-distance plots with data/model comparison and rebinning
- Import model visibilities back into a measurement set for CASA imaging
- Clean image generation via CASA tclean and plotting
"""

import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib.patches import Ellipse


def _setup_matplotlib():
    mpl.rc("font", family="serif", size=15)
    mpl.rc("xtick.major", size=7, width=1.5)
    mpl.rc("ytick.major", size=7, width=1.5)
    mpl.rc("xtick.minor", size=4, width=1.0)
    mpl.rc("ytick.minor", size=4, width=1.0)
    mpl.rc("axes", linewidth=2.0)
    mpl.rc("xtick", direction="in", top=True)
    mpl.rc("ytick", direction="in", right=True)


def uvbin(re, im, weights, uvdist, bin_edges, use_std=True):
    """Bin visibility data by uv-distance.

    Parameters
    ----------
    re, im : array-like
        Real and imaginary parts of visibility.
    weights : array-like
        Visibility weights.
    uvdist : array-like
        uv-distance for each point.
    bin_edges : array-like
        Bin edge values (monotonically increasing).
    use_std : bool
        If True, error bars are standard error of the mean.
        If False, error bars are 1/sqrt(total_weight).

    Returns
    -------
    bin_uvdist : ndarray
        Mean uv-distance in each bin.
    bin_count : ndarray
        Number of points per bin.
    bin_re, bin_re_err : ndarray
        Binned real visibility and error.
    bin_im, bin_im_err : ndarray
        Binned imaginary visibility and error.
    """
    nbins = len(bin_edges) - 1
    bin_uvdist = np.zeros(nbins)
    bin_weights = np.zeros(nbins)
    bin_count = np.zeros(nbins, dtype=int)
    uv_intervals = []

    for i in range(nbins):
        interval = np.where((uvdist >= bin_edges[i]) &
                            (uvdist < bin_edges[i + 1]))
        bin_count[i] = len(interval[0])

        if bin_count[i] != 0:
            bin_uvdist[i] = uvdist[interval].mean()
            bin_weights[i] = np.sum(weights[interval])
        else:
            bin_uvdist[i] = 0.5 * (bin_edges[i] + bin_edges[i + 1])

        uv_intervals.append(interval)

    bin_re = np.zeros(nbins)
    bin_re_err = np.zeros(nbins)
    for i in range(nbins):
        if bin_count[i] != 0:
            bin_re[i] = (np.sum(re[uv_intervals[i]] *
                                weights[uv_intervals[i]]) / bin_weights[i])
            if use_std:
                bin_re_err[i] = (np.std(re[uv_intervals[i]]) /
                                 np.sqrt(bin_count[i]))
            else:
                bin_re_err[i] = 1.0 / np.sqrt(bin_weights[i])

    bin_im = np.zeros(nbins)
    bin_im_err = np.zeros(nbins)
    for i in range(nbins):
        if bin_count[i] != 0:
            bin_im[i] = (np.sum(im[uv_intervals[i]] *
                                weights[uv_intervals[i]]) / bin_weights[i])
            if use_std:
                bin_im_err[i] = (np.std(im[uv_intervals[i]]) /
                                 np.sqrt(bin_count[i]))
            else:
                bin_im_err[i] = 1.0 / np.sqrt(bin_weights[i])

    return bin_uvdist, bin_count, bin_re, bin_re_err, bin_im, bin_im_err


def _make_bin_edges(uvdist, n_bins=15, scale='log'):
    """Create bin edges for uv-distance binning.

    Parameters
    ----------
    uvdist : array-like
        uv-distance values.
    n_bins : int
        Number of bins.
    scale : {'log', 'linear'}
        Binning scale.

    Returns
    -------
    bin_edges : ndarray
    """
    uv_min = uvdist[uvdist > 0].min() if np.any(uvdist > 0) else 1.0
    uv_max = uvdist.max()

    if scale == 'log':
        return np.logspace(np.log10(uv_min), np.log10(uv_max), n_bins + 1)
    else:
        return np.linspace(uv_min, uv_max, n_bins + 1)


def plot_uv(dvis, mvis=None, n_bins=15, scale='log', use_std=True,
            outpath=None, fname='uvplot.png', unit_mJy=True,
            show_im=True, show_log_x=True, mvis_samples=None,
            show_stats=True, fit_stats=None):
    """Plot real and imaginary parts of visibility vs uv-distance.

    Parameters
    ----------
    dvis : Visibility
        Data visibility object.
    mvis : complex ndarray, optional
        Model visibility (same length as data).  If None, only data is plotted.
    n_bins : int
        Number of uv-distance bins.
    scale : {'log', 'linear'}
        Scale for bin edges.
    use_std : bool
        Error bar calculation method.
    outpath : str, optional
        Directory for output.  If None, plot is shown.
    fname : str
        Output filename.
    unit_mJy : bool
        If True, plot in mJy.  If False, plot in Jy.
    show_im : bool
        If True, show imaginary panel.
    show_log_x : bool
        If True, use log scale on x-axis.
    mvis_samples : ndarray, optional
        Array of shape (n_samples, n_vis) containing model visibilities
        drawn from the MCMC posterior.  Used to plot a shaded 16-84%
        credible region around the best-fit model.
    show_stats : bool
        If True and ``fit_stats`` is provided, annotate the plot with
        goodness-of-fit statistics.
    fit_stats : dict, optional
        Fit statistics dict with keys ``chi2``, ``dof``, ``redchi2``,
        ``bic``.  Passed through from ``MCMCResult.fit_stats``.

    Returns
    -------
    fig : matplotlib Figure
    """
    _setup_matplotlib()

    data_uvdist = dvis.uvdist
    bin_edges = _make_bin_edges(data_uvdist, n_bins=n_bins, scale=scale)

    data_uvbin, data_count, data_re, data_ere, data_im, data_eim = \
        uvbin(dvis.re, dvis.im, dvis.wgt, data_uvdist, bin_edges, use_std)

    model_re = model_im = model_ere = model_eim = None
    if mvis is not None:
        _, _, model_re, model_ere, model_im, model_eim = \
            uvbin(mvis.real, mvis.imag, dvis.wgt, data_uvdist, bin_edges,
                  use_std)

    # --- Posterior uncertainty from MCMC samples ---
    model_re_lo = model_re_hi = model_im_lo = model_im_hi = None
    if mvis_samples is not None:
        n_samples = mvis_samples.shape[0]
        model_re_binned = np.zeros((n_samples, n_bins))
        model_im_binned = np.zeros((n_samples, n_bins))
        for s in range(n_samples):
            sv = mvis_samples[s]
            _, _, re_b, _, im_b, _ = \
                uvbin(sv.real, sv.imag, dvis.wgt, data_uvdist,
                       bin_edges, use_std=False)
            model_re_binned[s, :] = re_b
            model_im_binned[s, :] = im_b
        model_re_lo = np.percentile(model_re_binned, 16, axis=0)
        model_re_hi = np.percentile(model_re_binned, 84, axis=0)
        model_im_lo = np.percentile(model_im_binned, 16, axis=0)
        model_im_hi = np.percentile(model_im_binned, 84, axis=0)

    data_mask = data_count > 0
    data_x = data_uvbin / 1e3  # convert to kilo-lambda

    scale_fac = 1e3 if unit_mJy else 1.0
    unit_label = 'mJy' if unit_mJy else 'Jy'

    if show_im:
        fig = plt.figure(figsize=(6, 7))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
    else:
        fig = plt.figure(figsize=(6, 4))
        ax1 = fig.add_subplot(111)

    # Real part
    ax1.errorbar(data_x[data_mask], data_re[data_mask] * scale_fac,
                 yerr=data_ere[data_mask] * scale_fac,
                 color='k', fmt='.', ecolor='dimgrey', ms=8, capsize=2,
                 label='Data (Re)')
    if model_re is not None:
        model_mask = data_count > 0
        ax1.plot(data_x[model_mask], model_re[model_mask] * scale_fac,
                 c='r', lw=2, zorder=10, label='Model (Re)')
        if model_re_lo is not None:
            ax1.fill_between(data_x[model_mask],
                             model_re_lo[model_mask] * scale_fac,
                             model_re_hi[model_mask] * scale_fac,
                             color='red', alpha=0.2, zorder=9)
    ax1.axhline(y=0.0, ls='--', color='grey')
    ax1.set_ylabel(f'Re ({unit_label})')
    ax1.legend(frameon=True, loc='lower left')
    ax1.minorticks_on()

    # --- Fit statistics annotation ---
    if show_stats and fit_stats is not None and mvis is not None:
        txt = (f"DOF = {fit_stats['dof']}\n"
               f"$\\chi^2$ = {fit_stats['chi2']:.1f}\n"
               f"$\\chi^2_\\nu$ = {fit_stats['redchi2']:.3f}\n"
               f"BIC = {fit_stats['bic']:.1f}")
        ax1.text(0.97, 0.97, txt, transform=ax1.transAxes,
                 ha='right', va='top', fontsize=11, family='monospace',
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    if show_log_x:
        ax1.set_xscale('log')

    if show_im:
        ax2.plot(data_x[data_mask], data_im[data_mask] * scale_fac, 'o',
                 color='none', mec='C1', ms=4, label='Data (Im)')
        if model_im is not None:
            ax2.plot(data_x[model_mask], model_im[model_mask] * scale_fac,
                     '--', c='C0', lw=2, zorder=10, label='Model (Im)')
            if model_im_lo is not None:
                ax2.fill_between(data_x[model_mask],
                                 model_im_lo[model_mask] * scale_fac,
                                 model_im_hi[model_mask] * scale_fac,
                                 color='C0', alpha=0.2, zorder=9)
        ax2.set_ylabel(f'Im ({unit_label})')
        ax2.set_xlabel(r'uv-distance (k$\lambda$)')
        ax2.legend(frameon=True)
        ax2.minorticks_on()
        if show_log_x:
            ax2.set_xscale('log')
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.subplots_adjust(hspace=0.1)
    else:
        ax1.set_xlabel(r'uv-distance (k$\lambda$)')

    plt.tight_layout()

    if outpath is not None:
        os.makedirs(outpath, exist_ok=True)
        save_path = os.path.join(outpath, fname)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"UV plot saved to {save_path}")
        return fig
    else:
        return fig


def import_model_to_ms(msfile, u, v, mvis, wle, suffix='model',
                       make_resid=False, datacolumn='DATA'):
    """Import model visibilities into a measurement set for CASA imaging.

    Copies the original MS, replaces the DATA column with model visibilities.
    Optionally creates a residual MS (data - model).

    Parameters
    ----------
    msfile : str
        Path to the original measurement set (must end in ``.ms``).
    u, v : array-like
        Model uv coordinates in wavelengths.
    mvis : array-like, complex
        Model visibility in Jy.
    wle : float
        Wavelength in meters.
    suffix : str
        Suffix for the output MS (e.g. ``'model'`` -> ``name_model.ms``).
    make_resid : bool
        If True, also create a residual MS.
    datacolumn : str
        Column to read original data from.
    """
    if not msfile.endswith('.ms'):
        raise ValueError("MS name must end in '.ms'")

    ms_name = msfile[:-3]  # strip .ms
    tb_tool = __import__('casatools', fromlist=['table']).table()

    # --- Write model visibilities into a copy of the MS ---
    model_ms = f"{ms_name}.{suffix}.ms"
    os.system(f'rm -rf {model_ms}')
    os.system(f'cp -r {msfile} {model_ms}')

    # Read original data and flags
    from galfit_uv.export import _getvarcol_safe, _getvarcol_flag
    tb_tool.open(model_ms)
    data = _getvarcol_safe(tb_tool, datacolumn)
    flag = _getvarcol_flag(tb_tool, 'FLAG')
    tb_tool.close()

    # data shape: (n_pol, n_rows)
    n_pol = data.shape[0]
    n_rows_data = data.shape[-1]

    # Get unflagged mask
    if flag.ndim == 2:
        unflagged = ~np.any(flag, axis=0)
    else:
        unflagged = ~flag
    unflagged = unflagged[:n_rows_data]

    # Trim model visibilities to match n_rows
    n_model = len(mvis)
    n_use = min(n_rows_data, n_model)

    # Construct complex data array: same visibility in both polarizations
    mdl_array = np.zeros((n_pol, n_rows_data), dtype=np.complex128)
    mdl_array[:, unflagged[:n_use]] = mvis[:n_use]

    # Write back
    tb_tool.open(model_ms, nomodify=False)
    # Need to reconstruct the full DATA shape including channels.
    # Read the shape to know how to expand.
    original_data = np.squeeze(tb_tool.getcol(datacolumn))
    # original_data may be (n_pol, n_chan, n_rows) or (n_pol, n_rows)
    if original_data.ndim == 3:
        n_chan = original_data.shape[1]
        full_model = np.zeros_like(original_data, dtype=np.complex128)
        # Divide model flux equally among channels
        for ichan in range(n_chan):
            full_model[:, ichan, :] = mdl_array / n_chan
    elif original_data.ndim == 2:
        full_model = mdl_array.copy()
    else:
        full_model = mdl_array.copy()

    tb_tool.putcol(datacolumn, full_model)
    tb_tool.flush()
    tb_tool.close()
    print(f"Model MS written to {model_ms}")

    # --- Optionally create residual MS ---
    if make_resid:
        resid_ms = f"{ms_name}.resid.ms"
        os.system(f'rm -rf {resid_ms}')
        os.system(f'cp -r {msfile} {resid_ms}')

        tb_tool.open(resid_ms, nomodify=False)
        original_data = tb_tool.getcol(datacolumn)
        # Replace unflagged data with residuals
        if original_data.ndim == 3:
            n_chan = original_data.shape[1]
            resid_data = original_data.copy()
            for ichan in range(n_chan):
                resid_data[:, ichan, unflagged[:n_use]] -= \
                    mvis[:n_use] / n_chan
        elif original_data.ndim == 2:
            resid_data = original_data.copy()
            resid_data[:, unflagged[:n_use]] -= mvis[:n_use]
        else:
            resid_data = original_data.copy()

        tb_tool.putcol(datacolumn, resid_data)
        tb_tool.flush()
        tb_tool.close()
        print(f"Residual MS written to {resid_ms}")


def clean_image(msfile, u, v, mvis, wle,
                outdir='.', suffix='visfit',
                cell=None, imsize=None, niter=500,
                threshold='0.0mJy', gain=0.1,
                weighting='natural', robust=0.5,
                deconvolver='hogbom', stokes='I',
                source_size=None, fov_pad=10,
                verbose=True):
    """Run CASA tclean on data, model, and residual MS files.

    Creates model and residual MS copies via ``import_model_to_ms``, runs
    ``tclean`` on each with identical gridding, and returns the restored
    images along with beam and grid metadata.

    Parameters
    ----------
    msfile : str
        Original measurement set path (must end in ``.ms``).
    u, v : array-like
        Model uv coordinates in wavelengths.
    mvis : array-like, complex
        Model visibility in Jy.
    wle : float
        Wavelength in meters.
    outdir : str
        Directory for tclean output images.
    suffix : str
        Suffix for MS copies (model -> ``{name}.{suffix}.ms``,
        residual -> ``{name}.resid.{suffix}.ms``).
    cell : str or None
        Pixel size string (e.g. ``'0.5arcsec'``).  Auto-derived from UV
        coverage when None.
    imsize : list or None
        Image size ``[nx, ny]`` in pixels.  Auto-derived when None.
    niter : int
        Maximum clean iterations per image.
    threshold : str
        Clean threshold (e.g. ``'0.0mJy'``).
    gain : float
        Loop gain for each minor iteration.
    weighting : str
        UV weighting scheme (``'natural'``, ``'uniform'``, ``'briggs'``).
    robust : float
        Briggs robustness parameter (only used when weighting='briggs').
    deconvolver : str
        Deconvolution algorithm (e.g. ``'hogbom'``, ``'clark'``).
    stokes : str
        Stokes parameter to image (e.g. ``'I'``).
    source_size : float or None
        Measured source size in arcsec (e.g. Gaussian sigma from the fit).
        When provided, the FOV is set to ``fov_pad * source_size``.
        When None, the FOV is derived from the UV coverage.
    fov_pad : float
        Multiplier applied to ``source_size`` to set the image FOV.
        Ignored when ``source_size`` is None.
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict
        ``'data_image'`` — restored clean image (Jy/beam), 2-D ndarray
        ``'model_image'`` — model clean image, 2-D ndarray
        ``'resid_image'`` — residual clean image, 2-D ndarray
        ``'data_residual'`` — tclean residual for data, 2-D ndarray
        ``'beam_info'`` — ``{'bmaj': float, 'bmin': float, 'bpa': float}``
                            (arcsec, arcsec, deg)
        ``'cellsize'`` — pixel size in arcsec
        ``'imsize'`` — ``[nx, ny]`` pixel count
        ``'outdir'`` — output directory path
    """
    from casatasks import tclean
    from casatools import image as ia_tool

    # -- Step 1: create model and residual MS copies --
    if verbose:
        print("clean_image: creating model/residual MS copies ...")
    import_model_to_ms(msfile, u, v, mvis, wle,
                       suffix=suffix, make_resid=True,
                       datacolumn='CORRECTED_DATA')

    ms_name = msfile[:-3]  # strip .ms
    model_ms = f"{ms_name}.{suffix}.ms"
    resid_ms = f"{ms_name}.resid.ms"

    # -- Step 2: determine cell and imsize --
    u_max = max(np.max(np.abs(u)), np.max(np.abs(v)))
    if cell is None:
        cellsize_rad = 1.0 / (4.0 * u_max)
        cellsize_arcsec = cellsize_rad * 206265.0
        cell = f'{cellsize_arcsec:.4f}arcsec'
    else:
        # Parse user-supplied cell string to extract arcsec value
        cellsize_arcsec = float(cell.replace('arcsec', '').replace('"', ''))

    beam_fwhm_arcsec = 206265.0 / u_max

    if imsize is None:
        if source_size is not None:
            fov = fov_pad * source_size
        else:
            fov = max(10.0 * beam_fwhm_arcsec, 5.0)
        npix = int(2 ** np.ceil(np.log2(fov / cellsize_arcsec)))
        npix = max(npix, 64)
        imsize = [npix, npix]

    if verbose:
        print(f"clean_image: cell={cell}, imsize={imsize}, "
              f"niter={niter}, weighting={weighting}")

    os.makedirs(outdir, exist_ok=True)

    # Common tclean keyword arguments (identical for all three images)
    tclean_kw = dict(
        cell=cell,
        imsize=imsize,
        niter=niter,
        threshold=threshold,
        gain=gain,
        weighting=weighting,
        robust=robust,
        deconvolver=deconvolver,
        stokes=stokes,
        restoringbeam='',
    )

    # -- Step 3-5: run tclean on each MS --
    ms_list = [
        (msfile,   'data_clean'),
        (model_ms, 'model_clean'),
        (resid_ms, 'resid_clean'),
    ]

    ia = ia_tool()
    results = {}
    beam = None
    for ms_path, imagename in ms_list:
        full_imagename = os.path.join(outdir, imagename)
        if verbose:
            print(f"clean_image: tclean {ms_path} -> {full_imagename}")
        tclean(vis=ms_path, imagename=full_imagename, **tclean_kw)

        # Load restored image
        img_path = f'{full_imagename}.image'
        ia.open(img_path)
        img_data = ia.getchunk()        # shape: (1, 1, ny, nx)
        if beam is None:
            beam = ia.restoringbeam()   # dict with 'major', 'minor', 'positionangle'
        ia.done()
        img = img_data.squeeze()        # -> (ny, nx)
        results[imagename] = img

        # Load tclean residual
        resid_path = f'{full_imagename}.residual'
        ia.open(resid_path)
        resid_data = ia.getchunk()
        ia.done()
        results[f'{imagename}_resid'] = resid_data.squeeze()

    ia.close()

    # -- Step 7: extract beam info --
    bmaj = beam['major']['value']       # arcsec
    bmin = beam['minor']['value']
    bpa = beam['positionangle']['value']  # deg

    beam_info = {
        'bmaj': float(bmaj),
        'bmin': float(bmin),
        'bpa': float(bpa),
    }

    if verbose:
        print(f"clean_image: beam {bmaj:.3f}\" x {bmin:.3f}\", "
              f"PA={bpa:.1f} deg")

    return {
        'data_image':    results['data_clean'],
        'model_image':   results['model_clean'],
        'resid_image':   results['resid_clean'],
        'data_residual': results['data_clean_resid'],
        'beam_info':     beam_info,
        'cellsize':      cellsize_arcsec,
        'imsize':        imsize,
        'outdir':        outdir,
    }


def plot_clean_images(data_image, model_image, resid_image,
                      beam_info, cellsize,
                      imsize=None, outpath=None, fname='clean_images.png',
                      unit_mJy=True, crosshairs=True):
    """Three-panel figure: cleaned data, cleaned model, residual.

    Parameters
    ----------
    data_image : ndarray
        Restored clean image of the data (Jy/beam).
    model_image : ndarray
        Restored clean image of the model (Jy/beam).
    resid_image : ndarray
        Restored clean image of the residual (Jy/beam).
    beam_info : dict
        ``{'bmaj': float, 'bmin': float, 'bpa': float}`` in arcsec/deg.
    cellsize : float
        Pixel size in arcsec.
    imsize : list or None
        ``[nx, ny]`` pixel count.  Inferred from array shape if None.
    outpath : str or None
        Directory for output image.  If None, plot is shown.
    fname : str
        Output filename.
    unit_mJy : bool
        If True, display in mJy/beam.
    crosshairs : bool
        Draw crosshairs at image center.

    Returns
    -------
    fig : matplotlib Figure
    """
    _setup_matplotlib()

    ny, nx = data_image.shape
    if imsize is None:
        imsize = [nx, ny]

    half_x = nx * cellsize / 2.0
    half_y = ny * cellsize / 2.0
    extent = [-half_x, half_x, -half_y, half_y]

    scale_fac = 1e3 if unit_mJy else 1.0
    unit_label = 'mJy/beam' if unit_mJy else 'Jy/beam'

    images = [
        (data_image,  'Data'),
        (model_image, 'Model'),
        (resid_image, 'Residual'),
    ]
    cmaps = ['inferno', 'inferno', 'RdBu_r']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (img, title), cmap in zip(axes, images, cmaps):
        img_scaled = img * scale_fac

        # Compute RMS for contours
        rms = np.std(img_scaled[np.isfinite(img_scaled)])

        im = ax.imshow(img_scaled, origin='lower', extent=extent,
                        cmap=cmap, interpolation='nearest')

        # Contour levels
        if 'Residual' in title:
            levels = np.array([-3, -2, -1, 1, 2, 3]) * rms
        else:
            levels = np.array([3, 5, 10, 20, 50]) * rms
        levels = levels[levels != 0]
        if len(levels) > 0:
            ax.contour(img_scaled, origin='lower', extent=extent,
                       levels=levels, colors='cyan', linewidths=0.5)

        # Beam ellipse in bottom-left corner
        bmaj = beam_info['bmaj']
        bmin = beam_info['bmin']
        bpa = beam_info['bpa']
        # Position beam in the bottom-left corner
        beam_x = extent[0] + 0.15 * (extent[1] - extent[0])
        beam_y = extent[2] + 0.15 * (extent[3] - extent[2])
        ell = Ellipse(xy=(beam_x, beam_y), width=bmaj, height=bmin,
                       angle=bpa, facecolor='none', edgecolor='white',
                       linewidth=1.5)
        ax.add_patch(ell)

        # Crosshairs
        if crosshairs:
            ax.axhline(0, color='grey', ls=':', lw=0.5, alpha=0.7)
            ax.axvline(0, color='grey', ls=':', lw=0.5, alpha=0.7)

        ax.set_xlabel(r'Offset (arcsec)')
        if ax == axes[0]:
            ax.set_ylabel(r'Offset (arcsec)')
        ax.set_title(title)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=unit_label)

    plt.tight_layout()

    if outpath is not None:
        os.makedirs(outpath, exist_ok=True)
        save_path = os.path.join(outpath, fname)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Clean image plot saved to {save_path}")
        return fig
    else:
        return fig
