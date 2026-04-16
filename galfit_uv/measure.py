"""
Cube measurement pipeline for ALMA spectral cubes.

Provides source masking, spectrum extraction, line detection, and
nested-sampling fitting of spectral line profiles.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from matplotlib.patches import Ellipse
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import mad_std, sigma_clipped_stats
from astropy.visualization import AsinhStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from scipy import ndimage

from galfit_uv.lineprofiles import (
    Gaussian, Gaussian_DoublePeak, Gaussian_DoublePeak_Asymmetric,
)

try:
    from spectral_cube import SpectralCube
except ImportError:
    SpectralCube = None

try:
    import dynesty
    from dynesty import NestedSampler
    from dynesty import utils as dyfunc
except ImportError:
    dynesty = None
    NestedSampler = None
    dyfunc = None

try:
    from scipy.integrate import trapezoid as trapz
except ImportError:
    from scipy.integrate import trapz


__all__ = [
    "source_mask", "source_mask_snr", "field_mask",
    "extract_spectrum", "detect_source", "quick_measure",
    "plot_detection", "plot_nondetection", "fit_dynesty",
    "calculate_w50", "Plot_Map", "Plot_Beam",
    "plot_1d_spectrum", "plot_circular_aperture",
    "plot_mask_contour", "compare_source_masks",
]


def quick_measure(cube, freq_line, freq_range=None, nbeam=1, offset=None, vrange=400*u.km/u.s,
                  field_radius=22, fit=False, fit_model='gaussian', nlive=500, dlogz=0.1,
                  progress=True, use_continuum=True, title=None, map_vrange=[-1, 5],
                  flux_unit='mJy', detect=False, mask_method='circular', nsigma=2,
                  return_spectrum_data=True):
    '''
    Make quick measurements.

    Parameters
    ----------
    cube : SpectralCube
        The spectral cube.
    freq_line : Quantity (units: GHz)
        The rest frequency of the line.
    freq_range : array-like, optional
        The frequency range [min, max] in GHz.
    nbeam : int (default: 2)
        The number of beams to extract the spectrum.
    offset : tuple, optional
        The (position_angle, separation) offset in degrees and arcseconds.
    vrange : Quantity (default: 400 km/s)
        The velocity range for non-detection upper limit calculation.
    field_radius : float (default: 22)
        The radius of the field in arcsec.
    fit : bool (default: False)
        Whether to fit the spectrum.
    fit_model : str (default: 'gaussian')
        The model to use for fitting: 'gaussian' or 'double_peak'.
    nlive : int (default: 500)
        Number of live points for nested sampling.
    dlogz : float (default: 0.1)
        Stopping criterion based on log evidence increment.
    progress : bool (default: True)
        Whether to show progress bar during nested sampling.
    use_continuum : bool (default: True)
        Whether to fit the continuum.
    title : str, optional
        The title for the plot.
    map_vrange : list (default: [-1, 5])
        The vmin and vmax for the map in units of sky RMS.
    flux_unit : str (default: 'mJy')
        The unit of the flux.
    detect : bool (default: False)
        Whether to plot detection or non-detection.
    mask_method : str (default: 'circular')
        The source mask method: 'circular' or 'snr'.
    nsigma : float (default: 2)
        The SNR threshold for SNR masking.
    return_spectrum_data : bool (default: True)
        Whether to return spectrum data in results.

    Returns
    -------
    res : dict or measurement results
        If return_spectrum_data=False: returns the result from plot_detection or plot_nondetection.
        If return_spectrum_data=True: returns dict with 'measurements', 'spectrum_data', and 'detection_plots' keys.
    '''
    if SpectralCube is None:
        raise ImportError("spectral-cube is required for quick_measure. "
                          "Install it with: pip install galfit-uv[measure]")

    fig = plt.figure(figsize=(12, 12))
    ax0 = fig.add_axes([0.05, 0.05, 0.9, 0.4])
    ax1 = fig.add_axes([0.05, 0.53, 0.4, 0.4])
    ax2 = fig.add_axes([0.46, 0.53, 0.4, 0.4])

    detect_result = detect_source(
        cube, axs=[ax1, ax2], nbeam=nbeam, offset=offset, field_radius=field_radius,
        title=title, freq_line=freq_line, freq_range=freq_range,
        map_vrange=map_vrange, flux_unit=flux_unit, mask_method=mask_method,
        nsigma=nsigma, return_spectrum_data=True
    )
    axs = detect_result['axs']
    spectrum_data = detect_result['spectrum_data']

    axs[1].legend(loc='upper right', fontsize=14)
    axs[1].set_xlabel(f'Frequency (GHz)', fontsize=24)
    axs[1].set_ylabel(f'Flux ({flux_unit})', fontsize=24)
    axs[1].yaxis.set_label_position("right")

    if detect:
        res = plot_detection(
            spectrum_data, fit=fit, fit_model=fit_model, nlive=nlive, dlogz=dlogz,
            progress=progress, ax=ax0, flux_unit=flux_unit, use_continuum=use_continuum)
    else:
        res = plot_nondetection(
            spectrum_data, vrange=vrange, ax=ax0, flux_unit=flux_unit)

    if return_spectrum_data:
        return {
            'measurements': res,
            'spectrum_data': spectrum_data,
            'detection_plots': axs
        }
    else:
        return res


def detect_source(cube, axs=None, nbeam=1, field_radius=20, offset=None,
                  title=None, freq_line=None,
                  freq_range=None, map_vrange=[-1, 5], contour_levels=[-1, 1, 3, 5],
                  flux_unit='mJy', mask_method='circular', nsigma=2, return_spectrum_data=True):
    '''
    Detect the source in the cube and extract the spectrum.

    Parameters
    ----------
    cube : SpectralCube
        The spectral cube.
    axs : list of matplotlib.axes.Axes, optional
        The axes to plot on. If None, creates new axes.
    nbeam : int (default: 1)
        The number of beams to extract the spectrum (for circular mask).
    field_radius : float (default: 20)
        The radius of the field mask in arcsec.
    offset : tuple, optional
        The (position_angle, separation) offset in degrees and arcseconds.
    title : str, optional
        The title for the plot.
    freq_line : Quantity, optional
        The rest frequency of the line.
    freq_range : array-like, optional
        The frequency range [min, max] in GHz.
    map_vrange : list (default: [-1, 5])
        The vmin and vmax for the map in units of sky RMS.
    contour_levels : list (default: [-1, 1, 3, 5])
        The contour levels in units of sky RMS.
    flux_unit : str (default: 'mJy')
        The unit of the flux.
    mask_method : str (default: 'circular')
        The source mask method: 'circular' or 'snr'.
    nsigma : float (default: 2)
        The SNR threshold for SNR masking.
    return_spectrum_data : bool (default: True)
        Whether to return spectrum data dictionary.

    Returns
    -------
    axs : list of matplotlib.axes.Axes or dict
        If return_spectrum_data=False: returns the axes with the plots.
        If return_spectrum_data=True: returns dict with 'axs' and 'spectrum_data' keys.
    '''
    if SpectralCube is None:
        raise ImportError("spectral-cube is required for detect_source. "
                          "Install it with: pip install galfit-uv[measure]")

    # Select the appropriate source mask function
    if mask_method == 'snr':
        mask_src = source_mask_snr(cube, nbeam=nbeam, nsigma=nsigma, offset=offset,
                                   freq_range=freq_range)
    elif mask_method == 'circular':
        mask_src = source_mask(cube, nbeam=nbeam, offset=offset)
    else:
        raise ValueError(f"mask_method must be 'circular' or 'snr', got '{mask_method}'")
    mask_fld = field_mask(cube, field_radius)
    mask_bkg = mask_fld & ~mask_src

    if np.sum(mask_bkg) == 0:
        raise ValueError("The background mask is empty!")

    # Check the frequency axis unit
    if not cube.spectral_axis.unit.is_equivalent(u.GHz):
        cube = cube.with_spectral_unit(
            u.GHz,
            velocity_convention='radio',
            rest_value=cube.header['RESTFRQ'] * u.Hz)

    if freq_range is not None:
        slab = cube.spectral_slab(freq_range[0] * u.GHz, freq_range[1] * u.GHz)
        m0 = slab.moment(order=0)
    else:
        m0 = cube.moment(order=0)

    # Prepare plotting the image
    skyrms = mad_std(m0.value[mask_bkg], ignore_nan=True)
    vmin = map_vrange[0] * skyrms
    vmax = map_vrange[1] * skyrms
    norm = ImageNormalize(stretch=AsinhStretch(), vmin=vmin, vmax=vmax)

    # Get the 1D spectrum using the source mask
    spc_x, spc_f = extract_spectrum(cube, mask_src=mask_src, perbeam=False)
    spc_x = spc_x.to(u.GHz)
    spc_f = spc_f.to(flux_unit)
    levels = [i*skyrms for i in contour_levels]
    contour_dict={"map": m0.value, "levels": levels, "kws": {"colors": "k", "linewidths": 2}}
    sigma = sigma_clipped_stats(spc_f.value, sigma=3)[2]

    # Convert to velocity axis using freq_line (if provided)
    if freq_line is not None:
        spc_x_vel = spc_x.to(u.km/u.s, equivalencies=u.doppler_radio(freq_line))
    else:
        spc_x_vel = None

    # Sort all spectral arrays by velocity in ASCENDING order
    # This ensures velocity increases from left to right in plots
    if spc_x_vel is not None:
        idx = np.argsort(spc_x_vel.value)  # Sort by velocity to make it ascending
    else:
        idx = np.argsort(spc_x.value)  # Fall back to frequency if no velocity

    spc_x = spc_x[idx]              # Frequency (will be descending)
    spc_f = spc_f[idx]              # Flux (aligned)
    if spc_x_vel is not None:
        spc_x_vel = spc_x_vel[idx]  # Velocity (NOW ASCENDING!)

    # Build spectrum data dictionary with sorted arrays
    spectrum_data = {
        'mask_src': mask_src,
        'spc_x': spc_x,                    # GHz for display (sorted)
        'spc_x_vel': spc_x_vel,            # km/s for calculations (sorted)
        'spc_f': spc_f,                    # Sorted flux
        'flux_unit': flux_unit,
        'nbeam': nbeam,
        'offset': offset,
        'mask_method': mask_method,
        'nsigma': nsigma,
        'rms': sigma * spc_f.unit,
        'spectral_unit': str(spc_x.unit),
        'freq_line': freq_line,            # Store for reference
    }

    # RMS per beam
    f_ppx = cube[:, cube.shape[1]//2, cube.shape[2]//2].to('mJy/beam').value
    if freq_range is not None:
        # Use sorted spc_x for filtering
        fltr = (spc_x < freq_range[0] * u.GHz) | (spc_x > freq_range[1] * u.GHz)
        x = spc_x[fltr]
        f_ppx = f_ppx[fltr]
    rms_ppx = sigma_clipped_stats(f_ppx, sigma=3)[2] * 1e3  # uJy/beam
    txtList = [f'RMS per beam\n{rms_ppx:.0f}' + r' ($\mu$Jy)']

    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        fig.subplots_adjust(wspace=0.05)
        plain = False
    else:
        plain = True

    ax = axs[0]
    Plot_Map(m0, norm=norm, contour_dict=contour_dict, ax=ax, imshow_interpolation='none',
             xlim=[field_radius, -field_radius], ylim=[-field_radius, field_radius])

    # Plot circular aperture for circular mask, or contour for SNR mask
    if mask_method == 'snr':
        plot_mask_contour(m0, mask_src, ax, linewidths=1.5)
    else:  # circular
        plot_circular_aperture(m0, nbeam, ax, offset=offset, color='r', fill=False)

    if title is not None:
        ax.set_title(title, fontsize=24)

    ax = axs[1]
    # Use sorted arrays for plotting
    plot_1d_spectrum(spc_x, spc_f, ax=ax, color='k', fill=False, lw=1.5)

    ax.axhline(y=0, color='k', ls='--')
    ax.axhspan(ymin=-sigma, ymax=sigma, color='gray', alpha=0.5)
    ax.yaxis.tick_right()

    if freq_range is not None:
        # Use sorted arrays for filtering and plotting
        fltr = (spc_x > freq_range[0]*u.GHz) & (spc_x < freq_range[1]*u.GHz)
        plot_1d_spectrum(
            spc_x[fltr], spc_f[fltr], ax=ax, dx=np.diff(spc_x.value).mean(),
            color='yellow', fill=True, lw=1.5, alpha=0.5)
        ax.axvspan(freq_range[0], freq_range[1], edgecolor='gray', facecolor='none', ls='--', alpha=0.5)

        nu_mean = np.average(spc_x[fltr].value, weights=spc_f[fltr].value)
        txtList.append(r'$\langle \nu \rangle$: ' + f'{nu_mean:.3f} GHz')

        # Add filtered spectrum data (using sorted arrays)
        spectrum_data['spc_x_filtered'] = spc_x[fltr]
        spectrum_data['spc_f_filtered'] = spc_f[fltr]
        spectrum_data['freq_range'] = freq_range

        # Also compute filtered velocity axis (from sorted spc_x)
        if freq_line is not None:
            spc_x_filtered_vel = spc_x[fltr].to(u.km/u.s, equivalencies=u.doppler_radio(freq_line))
            spectrum_data['spc_x_filtered_vel'] = spc_x_filtered_vel
        else:
            spectrum_data['spc_x_filtered_vel'] = None
    else:
        spectrum_data['spc_x_filtered'] = None
        spectrum_data['spc_f_filtered'] = None
        spectrum_data['freq_range'] = None
        spectrum_data['spc_x_filtered_vel'] = None

    if freq_line is not None:
        ax.axvline(x=freq_line.value, color='r', ls='--', lw=2, label='Line center')

    ax.set_ylim(-3*sigma, None)
    ax.minorticks_on()

    ax.text(0.05, 0.95, '\n'.join(txtList), fontsize=16,
            transform=ax.transAxes, ha='left', va='top')

    if not plain:
        axs[1].legend(loc='upper right', fontsize=14)
        axs[1].set_xlabel(f'Frequency (GHz)', fontsize=24)
        axs[1].set_ylabel(f'Flux ({spc_f.unit})', fontsize=24)
        axs[1].yaxis.set_label_position("right")

    if return_spectrum_data:
        return {'axs': axs, 'spectrum_data': spectrum_data}
    else:
        return axs


def plot_detection(spectrum_data, fit=False, fit_model='gaussian', nlive=500,
                   dlogz=0.1, progress=True, ax=None, flux_unit='mJy',
                   use_continuum=False):
    '''
    Plot detection spectrum and fit models.

    Parameters
    ----------
    spectrum_data : dict
        Pre-extracted spectrum data from detect_source(). Must contain:
        - 'spc_x': spectral axis in GHz (Quantity)
        - 'spc_x_vel': velocity axis in km/s (Quantity or None)
        - 'spc_f': flux values (Quantity)
        - 'spc_x_filtered': filtered spectral axis (Quantity or None)
        - 'spc_x_filtered_vel': filtered velocity axis (Quantity or None)
        - 'spc_f_filtered': filtered flux values (Quantity or None)
        - 'freq_range': frequency range used (list or None)
        - 'rms': RMS of spectrum (Quantity)
        - 'spectral_unit': Unit of spectral axis (str)
        - 'flux_unit': Flux unit (str)
        - 'freq_line': Rest frequency of the line (Quantity or None)
    fit : bool
        Whether to fit the spectrum
    fit_model : str
        Model to fit ('gaussian', 'double_peak', or 'double_peak_asym')
    nlive : int
        Number of live points for nested sampling
    dlogz : float
        Stopping criterion for nested sampling
    progress : bool
        Whether to show progress bar
    ax : matplotlib Axes, optional
        Axes to plot on
    flux_unit : str
        Flux unit for plotting
    use_continuum : bool
        Whether to use continuum subtraction

    Returns
    -------
    dict
        Fitting results if fit=True, None otherwise
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Validate spectrum_data has all required fields
    required_fields = ['spc_x', 'spc_x_vel', 'spc_f', 'rms', 'spectral_unit',
                       'flux_unit', 'spc_x_filtered', 'spc_x_filtered_vel',
                       'spc_f_filtered', 'freq_range', 'freq_line']
    missing_fields = [f for f in required_fields if f not in spectrum_data]
    if missing_fields:
        raise ValueError(f"spectrum_data missing required fields: {missing_fields}")

    # Extract data from spectrum_data
    spc_x = spectrum_data['spc_x']
    spc_x_vel = spectrum_data['spc_x_vel']
    spc_f = spectrum_data['spc_f'].to(flux_unit)
    rms = spectrum_data['rms'].to(flux_unit)
    rms_val = rms.value

    # Use pre-computed velocity axis for calculations and plotting
    if spc_x_vel is not None:
        spc_x_calc = spc_x_vel
        spc_x_plot = spc_x_vel  # Plot in velocity when available
    else:
        spc_x_calc = spc_x
        spc_x_plot = spc_x  # Fall back to frequency if no velocity

    plot_1d_spectrum(spc_x_plot, spc_f, ax=ax, color='k', fill=False, lw=1.5, zorder=4, label='Extracted spectrum')

    txtList = []

    # Only use pre-computed filtered data from spectrum_data
    if spectrum_data['spc_x_filtered'] is not None:
        spc_x_filtered = spectrum_data['spc_x_filtered']
        spc_x_filtered_vel = spectrum_data['spc_x_filtered_vel']
        spc_f_filtered = spectrum_data['spc_f_filtered']

        # Plot in velocity when available
        spc_x_filtered_plot = spc_x_filtered_vel if spc_x_filtered_vel is not None else spc_x_filtered
        plot_1d_spectrum(spc_x_filtered_plot, spc_f_filtered, ax=ax,
                         dx=np.diff(spc_x_filtered_plot.value).mean(),
                         color='yellow', fill=True, lw=1.5, alpha=0.5, label='Summed channels')

        # Use velocity units for integration
        spc_x_filtered_calc = spc_x_filtered_vel if spc_x_filtered_vel is not None else spc_x_filtered

        dv = np.abs(np.diff(spc_x_filtered_calc).mean())
        nchan = len(spc_x_filtered)
        width = nchan * dv
        flux_sum = np.sum(spc_f_filtered * dv).to(u.Jy*u.km/u.s)
        nu_mean = np.average(spc_x_filtered.value, weights=spc_f_filtered.value)

        # Calculate RMS from complement region
        if spc_x_vel is not None:
            mask_complement = ~((spc_x_vel > spc_x_filtered_vel.min()) &
                               (spc_x_vel < spc_x_filtered_vel.max()))
        else:
            mask_complement = ~((spc_x > spc_x_filtered.min()) &
                               (spc_x < spc_x_filtered.max()))

        rms_in_range = mad_std(spc_f[mask_complement])
        flux_sum_err = (rms_in_range * np.sqrt(width * dv)).to(u.Jy*u.km/u.s)

        txtList.append(f'Flux (sum): {flux_sum.value:.3f}' + r'$\pm$' +
                      f'{flux_sum_err.value:.3f}' + r' $\mathrm{(Jy\,km\,s^{-1})}$')
        txtList.append(f'Width (sum): {nchan} * {dv.to(u.km/u.s).value:.0f}' +
                      r' $\mathrm{(km\,s^{-1})}$')
        txtList.append(r'$\langle \nu \rangle$: ' + f'{nu_mean:.3f} GHz')
    else:
        width = None
        flux_sum = None
        flux_sum_err = None
        nu_mean = None

    if fit:
        # Set up prior bounds based on data (use velocity axis when available)
        flux_max = np.percentile(spc_f.value, 99.9)
        vmin, vmax = spc_x_calc.value.min(), spc_x_calc.value.max()

        # Set up prior bounds based on model type
        if fit_model == 'gaussian':
            prior_bounds = {
                'a': (0, flux_max * 2),      # amplitude
                'b': (vmin, vmax),           # mean (velocity center)
                'c': (0, abs(vmax - vmin) / 2)  # stddev
            }
        elif fit_model == 'double_peak':
            prior_bounds = {
                'ag': (0, flux_max * 2),     # peak amplitude
                'ac': (0, flux_max),         # central flux
                'v0': (vmin, vmax),          # center velocity
                'sigma': (0, 500),            # core sigma
                'w': (0, 500)                 # half-width
            }
        elif fit_model == 'double_peak_asym':
            prior_bounds = {
                'ag_left': (0, flux_max * 2),  # left peak amplitude
                'ag_right': (0, flux_max * 2),  # right peak amplitude
                'ac': (0, flux_max),            # central flux
                'v0': (vmin, vmax),             # center velocity
                'sigma': (0, 500),               # core sigma
                'w_left': (0, 500),              # left half-width
                'w_right': (0, 500)              # right half-width
            }
        else:
            raise ValueError(f"Unsupported fit_model: {fit_model}. Must be 'gaussian', 'double_peak', or 'double_peak_asym'")

        # Run dynesty fitting
        fit_result = fit_dynesty(
            x=spc_x_calc.value,
            y=spc_f.value,
            yerr=rms_val,
            model_type=fit_model,
            prior_bounds=prior_bounds,
            nlive=nlive,
            dlogz=dlogz,
            plot=False,  # We'll plot separately
            progress=progress
        )

        # Extract results for plotting and return dictionary
        # Note: flux from fit_dynesty is in mJy*km/s, need to convert to Jy*km/s
        flux_fit = fit_result['derived']['flux_int'] * 1e-3 * u.Jy*u.km/u.s

        # Use std for errors (single value instead of asymmetric)
        # Initialize variables to None for both models
        if fit_model == 'gaussian':
            w50 = fit_result['derived']['w50'] * spc_x_calc.unit
            w50_err = fit_result['derived']['w50_err'] * spc_x_calc.unit
            vc = fit_result['params']['b'] * spc_x_calc.unit  # 'b' is the mean parameter
            vc_err = fit_result['params_err']['b'] * spc_x_calc.unit
        elif fit_model == 'double_peak':
            w50 = fit_result['derived']['w50'] * spc_x_calc.unit
            w50_err = fit_result['derived']['w50_err'] * spc_x_calc.unit
            vc = fit_result['params']['v0'] * spc_x_calc.unit  # 'v0' is the center velocity
            vc_err = fit_result['params_err']['v0'] * spc_x_calc.unit
        elif fit_model == 'double_peak_asym':
            w50 = fit_result['derived']['w50'] * spc_x_calc.unit
            w50_err = fit_result['derived']['w50_err'] * spc_x_calc.unit
            vc = fit_result['params']['v0'] * spc_x_calc.unit  # 'v0' is the center velocity
            vc_err = fit_result['params_err']['v0'] * spc_x_calc.unit

        flux_fit_err = fit_result['derived']['flux_int_err'] * 1e-3 * u.Jy*u.km/u.s

        # Plot best fit and uncertainties
        x_plot = np.linspace(spc_x_calc.value.min(), spc_x_calc.value.max(), int(10*len(spc_x_calc)))

        if fit_model == 'gaussian':
            y_fit = Gaussian(x_plot,
                             fit_result['params']['a'],
                             fit_result['params']['b'],
                             fit_result['params']['c'])
            ax.plot(x_plot, y_fit, lw=2, color='r', label='Nested Sampling Fit')
        elif fit_model == 'double_peak':
            y_fit = Gaussian_DoublePeak(x_plot,
                                        fit_result['params']['ag'],
                                        fit_result['params']['ac'],
                                        fit_result['params']['v0'],
                                        fit_result['params']['sigma'],
                                        fit_result['params']['w'])
            ax.plot(x_plot, y_fit, lw=2, color='r', label=f'Nested Sampling Fit ({fit_model})')
        elif fit_model == 'double_peak_asym':
            y_fit = Gaussian_DoublePeak_Asymmetric(x_plot,
                                                     fit_result['params']['ag_left'],
                                                     fit_result['params']['ag_right'],
                                                     fit_result['params']['ac'],
                                                     fit_result['params']['v0'],
                                                     fit_result['params']['sigma'],
                                                     fit_result['params']['w_left'],
                                                     fit_result['params']['w_right'])
            ax.plot(x_plot, y_fit, lw=2, color='r', label=f'Nested Sampling Fit ({fit_model})')

        # Plot uncertainty samples using weighted posterior samples
        samples = fit_result['samples']
        for i in np.random.randint(0, len(samples), 100):
            if fit_model == 'gaussian':
                y_samp = Gaussian(x_plot, *samples[i, :])
            elif fit_model == 'double_peak':
                y_samp = Gaussian_DoublePeak(x_plot, *samples[i, :])
            elif fit_model == 'double_peak_asym':
                y_samp = Gaussian_DoublePeak_Asymmetric(x_plot, *samples[i, :])
            ax.plot(x_plot, y_samp, lw=1, color='r', alpha=0.08)

        # Update text list with results
        txtList.append(f"Flux: {flux_fit.value:.3f} ± {flux_fit_err.value:.3f} " + r'$\mathrm{(Jy\,km\,s^{-1})}$')
        txtList.append(f"W50: {w50.value:.0f} ± {w50_err.value:.0f} ({w50.unit})")
        txtList.append(f"$v_0$: {vc.value:.0f} ± {vc_err.value:.0f} ({vc.unit})")
        txtList.append(f"logZ: {fit_result['logz']:.2f} ± {fit_result['logzerr']:.2f}")

    else:
        w50 = None
        flux_fit = None
        vc = None
        w50_err = None
        flux_fit_err = None
        vc_err = None

    txt = '\n'.join(txtList)
    ax.text(0.05, 0.95, txt, fontsize=14, transform=ax.transAxes, va='top', ha='left')

    ax.legend(loc='upper right', fontsize=14)
    ax.axhline(y=0, color='gray', ls='--')
    ax.minorticks_on()
    ax.set_xlabel(f'Velocity ({spc_x_plot.unit})', fontsize=24)
    ax.set_ylabel(f'Flux ({spc_f.unit})', fontsize=24)

    ax.set_ylim(-3*rms_val, flux_max*1.2)

    res = {
        'flux_sum': flux_sum,
        'flux_sum_err': flux_sum_err,
        'width': width,
        'nu_mean': nu_mean,
        'flux_fit': flux_fit,
        'w50': w50,
        'vc': vc,
        'flux_fit_err': flux_fit_err,
        'w50_err': w50_err,
        'vc_err': vc_err,
        'ax': ax}
    return res


def plot_nondetection(spectrum_data, vrange=400*u.km/u.s, ax=None,
                      flux_unit='mJy', nsigma=3):
    '''
    Plot non-detection spectrum and compute upper limits.

    Parameters
    ----------
    spectrum_data : dict
        Pre-extracted spectrum data from detect_source(). Must contain:
        - 'spc_x': spectral axis in GHz (Quantity)
        - 'spc_x_vel': velocity axis in km/s (Quantity or None)
        - 'spc_f': flux values (Quantity)
        - 'rms': RMS of spectrum (Quantity)
        - 'spectral_unit': Unit of spectral axis (str)
    vrange : Quantity
        Velocity range for upper limit calculation
    ax : matplotlib Axes, optional
        Axes to plot on
    flux_unit : str
        Flux unit for plotting
    nsigma : float
        Number of sigma for upper limit

    Returns
    -------
    dict
        Upper limit results
    '''
    # Validate spectrum_data has all required fields
    required_fields = ['spc_x', 'spc_x_vel', 'spc_f', 'rms', 'spectral_unit']
    missing_fields = [f for f in required_fields if f not in spectrum_data]
    if missing_fields:
        raise ValueError(f"spectrum_data missing required fields: {missing_fields}")

    # Extract data from spectrum_data
    spc_x = spectrum_data['spc_x']
    spc_x_vel = spectrum_data['spc_x_vel']
    spc_f = spectrum_data['spc_f'].to(flux_unit)
    rms = spectrum_data['rms'].to(flux_unit)

    # Use pre-computed velocity axis for calculations and plotting
    if spc_x_vel is not None:
        spc_x_calc = spc_x_vel
        spc_x_plot = spc_x_vel  # Plot in velocity when available
    else:
        spc_x_calc = spc_x
        spc_x_plot = spc_x  # Fall back to frequency if no velocity

    dv = np.abs(np.diff(spc_x_calc).mean())
    flux_up = (nsigma * rms * np.sqrt(vrange * dv)).to(u.Jy*u.km/u.s)  # Proper upper limit

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    ax.step(spc_x_plot.value, spc_f.value, lw=1.5, color='k')
    ax.axhspan(ymin=-rms.value, ymax=rms.value, color='gray', alpha=0.5)

    txt = fr'{nsigma}$\sigma=$' + f'{flux_up.value:.3f} ' + r'$\mathrm{Jy\,km\,s^{-1}}$'
    ax.text(0.05, 0.95, txt, fontsize=14, transform=ax.transAxes, va='top', ha='left')

    ax.axhline(y=0, color='gray', ls='--')
    ax.minorticks_on()
    ax.set_xlabel(f'Velocity ({spc_x_plot.unit})', fontsize=24)
    ax.set_ylabel(f'Flux ({spc_f.unit})', fontsize=24)

    res = {'rms': rms,
           'flux_up': flux_up,
           'ax': ax}
    return res


def plot_circular_aperture(m0, nbeam, ax, offset=None, **kwargs):
    '''
    Plot a circular aperture on the axis.
    '''
    ra  = m0.header['OBSRA'] * u.deg
    dec = m0.header['OBSDEC'] * u.deg
    c = SkyCoord(ra, dec)

    if offset is not None:
        pa = offset[0] * u.deg
        sep = offset[1] * u.arcsec
        c = c.directional_offset_by(pa, sep)

    xs, ys = m0.wcs.world_to_pixel(c)
    xs = (xs - m0.shape[1] // 2) * m0.header['CDELT1'] * 3600
    ys = (ys - m0.shape[0] // 2) * m0.header['CDELT2'] * 3600

    radius = m0.header['BMAJ'] * 3600 * nbeam

    if 'color' not in kwargs:
        kwargs['color'] = 'r'

    if 'fill' not in kwargs:
        kwargs['fill'] = False

    if ('linestyle' not in kwargs) & ('ls' not in kwargs):
        kwargs['ls'] = '--'

    if ('linewidth' not in kwargs) & ('lw' not in kwargs):
        kwargs['lw'] = 2

    ircle1 = plt.Circle((xs, ys), radius, **kwargs)
    ax.add_artist(ircle1)
    return ircle1


def plot_mask_contour(m0, mask, ax, **kwargs):
    '''
    Plot a contour showing the masked region boundary.

    Parameters
    ----------
    m0 : Projection or 2D array with WCS
        The moment map with WCS information.
    mask : 2D boolean array
        The mask to plot as a contour.
    ax : matplotlib.axes.Axes
        The axis to plot on.
    **kwargs : dict
        Additional keyword arguments for matplotlib.pyplot.contour().

    Returns
    -------
    quad : matplotlib.contour.QuadContourSet
        The contour object.
    '''
    # Calculate extent from m0 WCS (reuse pattern from Plot_Map)
    header = m0.wcs.to_header()
    ra0 = (-m0.shape[1] // 2 - 0.5) * header["CDELT1"] * 3600
    ra1 = (m0.shape[1] // 2 + 0.5) * header["CDELT1"] * 3600
    dec0 = (-m0.shape[0] // 2 - 0.5) * header["CDELT2"] * 3600
    dec1 = (m0.shape[0] // 2 + 0.5) * header["CDELT2"] * 3600
    extent = [ra0, ra1, dec0, dec1]

    # Set default styling
    if 'colors' not in kwargs:
        kwargs['colors'] = 'r'
    if 'linestyles' not in kwargs and 'ls' not in kwargs:
        kwargs['linestyles'] = '--'
    if 'linewidths' not in kwargs and 'lw' not in kwargs:
        kwargs['linewidths'] = 1.5

    # Plot contour
    quad = ax.contour(mask, levels=[0.5], extent=extent, **kwargs)

    return quad


def compare_source_masks(cube, nbeam=2, nsigma=2, freq_range=None, offset=None):
    '''
    Compare circular and SNR-based source masks side by side.

    Parameters
    ----------
    cube : SpectralCube
        The spectral cube.
    nbeam : int (default: 2)
        The number of beams for circular mask.
    nsigma : float (default: 2)
        The SNR threshold for SNR mask.
    freq_range : array-like, optional
        The frequency range [min, max] in GHz.
    offset : tuple, optional
        The (position_angle, separation) offset in degrees and arcseconds.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    masks : dict
        Dictionary with 'circular' and 'snr' mask arrays.
    '''
    if SpectralCube is None:
        raise ImportError("spectral-cube is required for compare_source_masks. "
                          "Install it with: pip install galfit-uv[measure]")

    # Generate both masks
    mask_circular = source_mask(cube, nbeam=nbeam, offset=offset)
    mask_snr = source_mask_snr(cube, nbeam=nbeam, nsigma=nsigma, freq_range=freq_range, offset=offset)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot circular mask
    axes[0].imshow(mask_circular, origin='lower', cmap='Reds')
    axes[0].set_title(f'Circular Mask (nbeam={nbeam})\n{np.sum(mask_circular)} pixels')
    axes[0].set_xlabel('X (pixels)')
    axes[0].set_ylabel('Y (pixels)')

    # Plot SNR mask
    axes[1].imshow(mask_snr, origin='lower', cmap='Reds')
    axes[1].set_title(f'SNR Mask (≥{nsigma}σ)\n{np.sum(mask_snr)} pixels')
    axes[1].set_xlabel('X (pixels)')
    axes[1].set_ylabel('Y (pixels)')

    plt.tight_layout()

    masks = {'circular': mask_circular, 'snr': mask_snr}
    return fig, masks


def source_mask(cube, nbeam, offset=None):
    '''
    Generate the source mask.
    '''
    m0 = cube.moment(order=0)
    ra  = cube.header['OBSRA'] * u.deg
    dec = cube.header['OBSDEC'] * u.deg
    c = SkyCoord(ra, dec)

    if offset is not None:
        pa = offset[0] * u.deg
        sep = offset[1] * u.arcsec
        c = c.directional_offset_by(pa, sep)

    xs, ys = m0.wcs.world_to_pixel(c)
    bmaj = np.abs(cube.header['BMAJ'] / cube.header['CDELT1'])
    xx, yy = np.meshgrid(np.arange(m0.shape[1]), np.arange(m0.shape[0]))
    mask = np.sqrt((xx - xs)**2 + (yy - ys)**2) < bmaj * nbeam
    return mask


def source_mask_snr(cube, nbeam, nsigma=2, freq_range=None, offset=None):
    '''
    Generate the source mask based on SNR threshold with connected segment filtering.

    Creates a mask where pixels in the moment 0 map are above nsigma times the
    background RMS noise, but only keeps connected segments whose center is
    within the circular aperture defined by nbeam and offset.

    Parameters
    ----------
    cube : SpectralCube
        The spectral cube.
    nbeam : int
        The number of beams for the circular aperture. Segments whose center
        is outside this aperture will be discarded.
    nsigma : float (default: 2)
        The SNR threshold for masking.
    freq_range : array-like, optional (default: None)
        The frequency range [min, max] in GHz for creating the moment map.
    offset : tuple, optional (default: None)
        The (position_angle, separation) offset in degrees and arcseconds.

    Returns
    -------
    mask : 2D boolean array
        The source mask where True indicates pixels above the SNR threshold
        that belong to segments with centers inside the circular aperture.
    '''
    # Convert cube to GHz units if needed (for freq_range compatibility)
    if not cube.spectral_axis.unit.is_equivalent(u.GHz):
        cube_ghz = cube.with_spectral_unit(
            u.GHz,
            velocity_convention='radio',
            rest_value=cube.header['RESTFRQ'] * u.Hz)
    else:
        cube_ghz = cube

    # Create moment 0 map from cube (with optional freq_range filtering)
    if freq_range is not None:
        slab = cube_ghz.spectral_slab(freq_range[0] * u.GHz, freq_range[1] * u.GHz)
        m0 = slab.moment(order=0)
    else:
        m0 = cube_ghz.moment(order=0)

    # Create field mask to get background region
    mask_fld = field_mask(cube, radius=20)

    # Calculate background RMS using mad_std on background pixels
    skyrms = mad_std(m0.value[mask_fld], ignore_nan=True)

    # Apply SNR threshold
    mask_snr = m0.value > (nsigma * skyrms)

    # Handle NaN values
    mask_snr = mask_snr & ~np.isnan(m0.value)

    # Create circular aperture mask to filter segments
    mask_aperture = source_mask(cube, nbeam, offset=offset)

    # Find connected segments (components) above SNR threshold
    labeled_mask, num_features = ndimage.label(mask_snr)

    # Build final mask by keeping segments whose center is in the aperture
    mask_final = np.zeros_like(mask_snr, dtype=bool)

    if num_features > 0:
        # Get the center of each labeled segment
        # ndimage.center_of_mass returns (y, x) coordinates
        centers = ndimage.center_of_mass(mask_snr, labeled_mask,
                                         range(1, num_features + 1))

        # For each segment, check if its center is inside the circular aperture
        for i, (cy, cx) in enumerate(centers):
            # Convert to integer coordinates for mask indexing
            ix, iy = int(round(cx)), int(round(cy))

            # Check if the center is within the circular aperture
            if 0 <= iy < mask_aperture.shape[0] and 0 <= ix < mask_aperture.shape[1]:
                if mask_aperture[iy, ix]:
                    # Keep this entire segment
                    mask_final = mask_final | (labeled_mask == (i + 1))

    return mask_final


def field_mask(cube, radius=20):
    '''
    Generate the mask of the field to avoid the noisy edge.

    Parameters
    ----------
    cube : SpectralCube
        The spectral cube.
    radius : float
        The radius of the field mask in arcsec.

    Returns
    -------
    mask : 2D boolean array
        The mask of the field.
    '''
    m0 = cube.moment(order=0)
    pixsc = np.abs(cube.header['CDELT1'] * 3600)
    npix = radius / pixsc
    xx, yy = np.meshgrid(np.arange(m0.shape[1]), np.arange(m0.shape[0]))
    mask = (xx - m0.shape[1] / 2)**2 + (yy - m0.shape[0] / 2)**2 < npix**2
    return mask


def extract_spectrum(cube, mask_src, perbeam=True):
    '''
    Extract a spectrum from the cube using a provided mask.

    Parameters
    ----------
    cube : SpectralCube
        The spectral cube.
    mask_src : 2D boolean array
        The source mask where True indicates pixels to include.
    perbeam : bool (default: True)
        If True, return flux per beam. If False, return total flux.

    Returns
    -------
    spc_x : Quantity
        The spectral axis (velocity or frequency).
    spc_f : Quantity
        The extracted flux values.
    '''
    cube_msk = cube.with_mask(mask_src)
    spc_f = cube_msk.sum(axis=(1,2))
    spc_x = cube_msk.spectral_axis

    if not perbeam:
        b2p = (4 * np.log(2)) / (np.pi * (cube.header['BMAJ'] * cube.header['BMIN'])) * np.abs(cube.header['CDELT1'] * cube.header['CDELT2'])
        spc_f *= b2p * u.Unit('beam')
    return spc_x, spc_f


def Plot_Map(mom, cmap="viridis", norm=None, ax=None, imshow_interpolation="none",
             contour_dict={}, vpercentile=98, beam_on=True,
             beam_kws={}, plain=False, xlim=None, ylim=None):
    """
    Plot the moment maps.
    """
    if ax is None:
        fig = plt.figure()
        ax  = plt.gca()

    x = mom.value
    if norm is None:
        fltr = np.logical_not(np.isnan(x))
        vperc = np.atleast_1d(vpercentile)
        if len(vperc) == 1:
            prc1 = 100-vperc
            prc2 = vperc
        elif len(vperc) == 2:
            prc1, prc2 = vperc
        else:
            raise ValueError("The length of vpercentile ({0}) is at most 2!".format(vperc))
        vmin = np.percentile(x[fltr], prc1)
        vmax = np.percentile(x[fltr], prc2)
        norm = Normalize(vmin=vmin, vmax=vmax)
    #-> Make relative coordinates
    header = mom.wcs.to_header()
    ra0 = (-mom.shape[1] // 2 - 0.5) * header["CDELT1"] * 3600
    ra1 = (mom.shape[1] // 2 + 0.5) * header["CDELT1"] * 3600
    dec0 = (-mom.shape[0] // 2 - 0.5) * header["CDELT2"] * 3600
    dec1 = (mom.shape[0] // 2 + 0.5) * header["CDELT2"] * 3600
    extent = [ra0, ra1, dec0, dec1]
    #-> Plot the image
    cmap = get_cmap(cmap)
    cmap.set_bad(color='w', alpha=1.)
    im = ax.imshow(x, cmap=cmap, extent=extent, norm=norm, origin="lower", interpolation=imshow_interpolation)
    if not xlim is None:
        ax.set_xlim(xlim)
    if not ylim is None:
        ax.set_ylim(ylim)
    #-> Draw the contour
    contour_data = contour_dict.get("map", None)
    if not contour_data is None:
        levels = contour_dict.get("levels", 5)
        contour_kws = contour_dict.get("kws", {})
        ax.contour(contour_data, levels, extent=extent, **contour_kws)
    #-> Plot the beam
    if beam_on:
        bmaj = mom.beam.major.to(u.arcsec).value
        bmin = mom.beam.minor.to(u.arcsec).value
        bpa  = mom.beam.pa.to(u.degree).value
        Plot_Beam(ax, bmaj, bmin, bpa, **beam_kws)
    #-> Tune the axis
    if not plain:
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', length=8, width=1., labelsize=18)
        ax.tick_params(axis='both', which='minor', length=5, width=1., labelsize=18)
        ax.set_xlabel('R. A. Offset (")', fontsize=24)
        ax.set_ylabel('Decl. Offset (")', fontsize=24)
    return (ax, im)


def Plot_Beam(ax, bmaj, bmin, bpa, **ellipse_kws):
    """
    Plot the beam.

    Parameters
    ----------
    ax : axis object
        The figure axis.
    bmaj : float
        The beam major axis, units: arcsec.
    bmin : float
        The beam minor axis, units: arcsec.
    bpa : float
        The beam position angle, units: degree.
    **ellipse_kws : dict
        The keywords for Ellipse.  The default color is white and the default
        position is on the lower-left corner.
    """
    if ellipse_kws.get("zorder", None) is None:
        ellipse_kws["zorder"] = 5
    if ellipse_kws.get("color", None) is None:
        ellipse_kws["color"] = "white"
    if ellipse_kws.get("xy", None) is None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        bx = xlim[0] - 0.1 * (xlim[0] - xlim[1])
        by = ylim[0] + 0.1 * (ylim[1] - ylim[0])
        ellipse_kws["xy"] = (bx, by)
    ax.add_artist(Ellipse(width=bmin, height=bmaj, angle=-bpa, **ellipse_kws))


def plot_1d_spectrum(x, y, ax=None, dx=None, **kwargs):
    '''
    Properly plot the 1D spectrum with plt.stairs.

    Parameters
    ----------
    x : 1D `numpy.array` or `Quantity`
        The x-axis data.
    y : 1D `numpy.array` or `Quantity`
        The y-axis data.
    ax : `matplotlib.Axes` (optional)
        The axis to plot the spectrum.
    dx : float (optional)
        The x-axis bin width.
    **kwargs : dict
        The keywords for `pyplot.stairs`.

    Returns
    -------
    StepPatch : `StepPatch`
    '''
    if isinstance(x, u.Quantity):
        x = x.value

    if isinstance(y, u.Quantity):
        y = y.value

    if ax is None:
        fig, ax = plt.subplots()

    if dx is None:
        dx = np.diff(x).mean()

    if np.isnan(dx):
        raise ValueError("The dx is NaN!")

    edges = np.concatenate([[x[0]-dx], x]) + dx/2
    return ax.stairs(y, edges, **kwargs)


def calculate_w50(x, y):
    '''
    Calculate W50 (width at 50% of peak) from a line profile.

    This function finds the peak of the profile, then determines the width
    at half-maximum by finding where the profile crosses 50% of the peak value.

    Parameters
    ----------
    x : array-like
        Velocity or frequency axis (any units).
    y : array-like
        Flux or profile values.

    Returns
    -------
    w50 : float
        The width at 50% of peak, in the same units as x.
    '''
    x = np.asarray(x)
    y = np.asarray(y)

    # Find peak value
    peak = np.max(y)

    # Half-maximum level
    half_max = 0.5 * peak

    # Find where profile is above half-maximum
    above_half = y >= half_max

    if not np.any(above_half):
        # Profile never reaches half-maximum
        return np.nan

    # Find indices where profile crosses above half-max
    indices = np.where(above_half)[0]

    # Width is the difference between first and last index above half-max
    # Interpolate for better precision
    if len(indices) == 1:
        # Only one point above half-max (unlikely but possible)
        w50 = 0.0
    else:
        # Use linear interpolation at the boundaries for better precision
        # Left boundary: find x where y crosses half_max
        i_left = indices[0]
        if i_left > 0:
            # Linear interpolation between i_left-1 and i_left
            x1, x2 = x[i_left-1], x[i_left]
            y1, y2 = y[i_left-1], y[i_left]
            if y2 != y1:
                x_left = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
            else:
                x_left = x1
        else:
            x_left = x[i_left]

        # Right boundary: find x where y crosses below half_max
        i_right = indices[-1]
        if i_right < len(x) - 1:
            # Linear interpolation between i_right and i_right+1
            x1, x2 = x[i_right], x[i_right+1]
            y1, y2 = y[i_right], y[i_right+1]
            if y2 != y1:
                x_right = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
            else:
                x_right = x1
        else:
            x_right = x[i_right]

        w50 = x_right - x_left

    # Return absolute value to ensure width is always positive
    return np.abs(w50)


def fit_dynesty(x, y, yerr, model_type='gaussian', prior_bounds=None,
                nlive=500, dlogz=0.1, sample='rwalk', plot=True, ax=None,
                progress=True, rstate=None):
    '''
    Fit spectral line profiles using dynesty nested sampling.

    Parameters
    ----------
    x : array-like
        Velocity or frequency axis.
    y : array-like
        Flux values.
    yerr : float or array-like
        Flux uncertainties (RMS).
    model_type : str (default: 'gaussian')
        Type of model to fit: 'gaussian', 'double_peak', or 'double_peak_asym'.
    prior_bounds : dict, optional
        Dictionary of parameter bounds (lower, upper) tuples.
        Keys should match parameter names.
    nlive : int (default: 500)
        Number of live points for nested sampling.
    dlogz : float (default: 0.1)
        Stopping criterion based on log evidence increment.
    plot : bool (default: True)
        Whether to plot the fitting results.
    ax : matplotlib.axes.Axes, optional
        The axis to plot on. If None and plot=True, creates new figure.
    progress : bool (default: True)
        Whether to show progress bar during sampling.
    rstate : numpy.random.Generator, optional
        Random state for reproducibility.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'params': dict of median parameter values
        - 'params_err': dict of parameter uncertainties (upper, lower)
        - 'samples': posterior samples array
        - 'logz': log evidence
        - 'logzerr': evidence error
        - 'model_type': model type used
        - 'derived': dict of derived parameters (flux_int, w50)
        - 'ax': matplotlib axis (if plotted)
    '''
    if dynesty is None:
        raise ImportError("dynesty is required for fit_dynesty. "
                          "Install it with: pip install galfit-uv[measure]")

    x = np.asarray(x)
    y = np.asarray(y)

    # Select model function and parameter names
    if model_type == 'gaussian':
        model_func = Gaussian
        param_names = ['a', 'b', 'c']  # amplitude, mean, stddev
        default_bounds = {
            'a': (0, np.max(y) * 2),
            'b': (np.min(x), np.max(x)),
            'c': (0, np.abs(np.max(x) - np.min(x)) / 2)
        }
    elif model_type == 'double_peak':
        model_func = Gaussian_DoublePeak
        param_names = ['ag', 'ac', 'v0', 'sigma', 'w']  # symmetric double peak
        default_bounds = {
            'ag': (0, np.max(y) * 2),        # peak amplitude
            'ac': (0, np.max(y)),             # central flux
            'v0': (np.min(x), np.max(x)),     # center velocity
            'sigma': (0, 500),                # core sigma
            'w': (0, 500)                     # half-width of central parabola
        }
    elif model_type == 'double_peak_asym':
        model_func = Gaussian_DoublePeak_Asymmetric
        param_names = ['ag_left', 'ag_right', 'ac', 'v0', 'sigma', 'w_left', 'w_right']
        default_bounds = {
            'ag_left': (0, np.max(y) * 2),    # left peak amplitude
            'ag_right': (0, np.max(y) * 2),   # right peak amplitude
            'ac': (0, np.max(y)),             # central flux
            'v0': (np.min(x), np.max(x)),     # center velocity
            'sigma': (0, 500),                # core sigma (shared)
            'w_left': (0, 500),               # left half-width
            'w_right': (0, 500)               # right half-width
        }
    else:
        raise ValueError(f"model_type must be 'gaussian', 'double_peak', or 'double_peak_asym', got '{model_type}'")

    # Use provided bounds or defaults
    if prior_bounds is None:
        prior_bounds = default_bounds
    else:
        # Merge with defaults for any missing parameters
        for key in default_bounds:
            if key not in prior_bounds:
                prior_bounds[key] = default_bounds[key]

    # Extract bounds in parameter order
    bounds = [prior_bounds[p] for p in param_names]
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])

    # Define prior transform (uniform within bounds)
    def prior_transform(u):
        """Transform from unit cube to parameter space."""
        return lower_bounds + (upper_bounds - lower_bounds) * u

    # Define log-likelihood function
    def loglike(theta):
        """Gaussian log-likelihood."""
        model = model_func(x, *theta)
        return -0.5 * np.sum(((y - model) / yerr)**2)

    # Initialize random state
    if rstate is None:
        rstate = np.random.default_rng()

    # Run nested sampling
    if progress:
        print(f"Running dynesty nested sampling for {model_type} model...")

    sampler = NestedSampler(loglike, prior_transform, len(param_names),
                           nlive=nlive, sample=sample, rstate=rstate)
    sampler.run_nested(dlogz=dlogz, print_progress=progress)

    # Extract results
    results = sampler.results

    # Use weight resampling to get properly weighted posterior samples
    # This gives us samples that reflect the true posterior distribution
    samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
    posterior_samples = dyfunc.resample_equal(samples, weights)

    logz = float(results.logz[-1])       # Log evidence (final value)
    logzerr = float(results.logzerr[-1]) # Evidence error (final value)

    # Use the last sampled point (highest likelihood by definition) as best-fit
    # In dynesty, the final sample has the highest log-likelihood
    theta_best = results.samples[-1]
    params = {}
    params_err = {}

    # Compute std of posterior distribution for errors
    posterior_transposed = posterior_samples.T
    for i, name in enumerate(param_names):
        params[name] = theta_best[i]
        params_err[name] = np.std(posterior_transposed[i])

    # Calculate derived parameters
    derived = {}

    # Use best-fit for value, std for error
    y_best_fit = model_func(x, *theta_best)
    derived['flux_int'] = np.abs(trapz(y_best_fit, x))  # Use abs for robustness
    derived['w50'] = calculate_w50(x, y_best_fit)

    # Calculate flux_int and W50 for each sample to get error
    w50_samples = []
    flux_int_samples = []
    for sample in posterior_samples:
        y_model = model_func(x, *sample)

        flux_int = np.abs(trapz(y_model, x))  # Use abs for robustness
        flux_int_samples.append(flux_int)

        w50 = calculate_w50(x, y_model)
        if not np.isnan(w50):
            w50_samples.append(w50)
    derived['flux_int_err'] = np.std(flux_int_samples)
    derived['w50_err'] = np.std(w50_samples) if len(w50_samples) > 0 else 0.0

    # Plot results if requested
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Data is sorted by ascending velocity (primary analysis axis)
        # Frequency is in descending order as a consequence
        # Use as-is without re-sorting to maintain consistency

        # Plot data
        ax.step(x, y, color='k', lw=2, label='Data', where='mid')

        # Plot best fit (evaluate on original x, not sorted)
        theta_best = [params[name] for name in param_names]
        y_fit = model_func(x, *theta_best)
        ax.plot(x, y_fit, color='r', lw=2, label='Best Fit')

        # Plot uncertainty samples
        # Use the properly weighted posterior samples
        n_samples = 100
        sample_indices = np.random.choice(len(posterior_samples), min(n_samples, len(posterior_samples)), replace=False)
        for idx in sample_indices:
            y_samp = model_func(x, *posterior_samples[idx, :])
            ax.plot(x, y_samp, color='r', lw=1, alpha=0.08)

        # Add labels and formatting
        ax.set_xlabel('Velocity')
        ax.set_ylabel('Flux')
        ax.legend(loc='upper right')
        ax.minorticks_on()

        # Add parameter text
        param_text = "Best-fit parameters:\n"
        for name in param_names:
            param_text += f"  {name}: {params[name]:.3g} ± {params_err[name]:.3g}\n"

        if 'w50' in derived:
            param_text += f"  W50: {derived['w50']:.3g} ± {derived['w50_err']:.3g}\n"

        ax.text(0.05, 0.95, param_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Prepare result dictionary
    result = {
        'params': params,
        'params_err': params_err,
        'samples': posterior_samples,  # Return weighted posterior samples
        'logz': logz,
        'logzerr': logzerr,
        'model_type': model_type,
        'derived': derived,
        'sampler_results': results,  # Return raw sampler.results for corner plots
        'ax': ax if plot else None
    }

    return result
