#!/usr/bin/env python
"""
Demonstration of the galfit_uv measure module — detection and line profile fitting.

Uses real ALMA data (GQC J0054-4955 CO(3-2) line cube) to show:
  1. Loading a spectral cube with spectral_cube
  2. Source detection with SNR masking
  3. Spectrum extraction from the cube
  4. Nested-sampling line profile fitting with dynesty (DoublePeak model)
  5. Non-detection upper limit calculation
  6. Saving diagnostic figures

Data
----
The cube is a primary-beam-corrected line image from ALMA Band 3:
  GQC_J0054-4955_co32_init_line_freq.image.pbcor.fits
  56 channels x 100 x 100 pixels, ~2.2 MB

Source info:
  Name:       GQC J0054-4955
  Redshift:   z = 2.2512
  Line:       CO(3-2), rest freq = 345.796 GHz
  Observed:   ~106.36 GHz
"""
import os
from pathlib import Path

# Import the package (not submodules) so that __init__.py sets
# OMP_NUM_THREADS etc. before numpy is loaded.
import galfit_uv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import astropy.units as u

from spectral_cube import SpectralCube
from galfit_uv.measure import (
    quick_measure, detect_source, source_mask, source_mask_snr,
    extract_spectrum, plot_1d_spectrum, plot_detection, plot_nondetection,
    fit_dynesty, calculate_w50, compare_source_masks,
)

DEMO_DIR = Path(__file__).resolve().parent
FIGS_DIR = DEMO_DIR / 'figs'
FIGS_DIR.mkdir(exist_ok=True)

CUBE_FILE = str(DEMO_DIR / 'data' /
                'GQC_J0054-4955_co32_init_line_freq.image.pbcor.fits')

# Source properties
FREQ_REST = 345.796   # GHz, CO(3-2) rest frequency
REDSHIFT = 2.2512
FREQ_LINE = FREQ_REST / (1 + REDSHIFT)  # ~106.36 GHz
LS_KM = 299792.458    # speed of light in km/s

# ------------------------------------------------------------------ #
# 1.  Load the spectral cube                                          #
# ------------------------------------------------------------------ #
print("=" * 60)
print("1. Loading spectral cube")
print("=" * 60)

cube = SpectralCube.read(CUBE_FILE)
print(f"  Shape: {cube.shape}  (nchan, ny, nx)")
print(f"  Spectral axis: {cube.spectral_axis.min():.4f} -- "
      f"{cube.spectral_axis.max():.4f}  ({cube.spectral_axis.unit})")
print(f"  Channel width: {np.diff(cube.spectral_axis[:2])[0]:.4e}")
beam = cube.beam
print(f"  Beam: {beam.major:.4f} x {beam.minor:.4f}, "
      f"PA = {beam.pa:.1f}")
print()

# ------------------------------------------------------------------ #
# 2.  Compare circular vs SNR source masks                           #
# ------------------------------------------------------------------ #
print("=" * 60)
print("2. Comparing source masks")
print("=" * 60)

fig, masks = compare_source_masks(
    cube, nbeam=1, nsigma=2,
    freq_range=[106.27, 106.6],
)
fig.savefig(str(FIGS_DIR / 'measure_mask_comparison.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Circular mask pixels: {masks['circular'].sum()}")
print(f"  SNR mask pixels:      {masks['snr'].sum()}")
print(f"  Saved: figs/measure_mask_comparison.png")
print()

# ------------------------------------------------------------------ #
# 3.  Source detection (full quick_measure with fit)                   #
# ------------------------------------------------------------------ #
print("=" * 60)
print("3. Running quick_measure (detection + DoublePeak fit)")
print("=" * 60)

res = quick_measure(
    cube,
    freq_line=FREQ_LINE * u.GHz,
    freq_range=[106.27, 106.6],
    mask_method='snr',
    nsigma=2,
    field_radius=22,
    title='GQC J0054-4955',
    detect=True,
    fit=True,
    fit_model='double_peak',
    nlive=500,
    dlogz=0.01,
    use_continuum=False,
)

plt.savefig(str(FIGS_DIR / 'measure_quick_detection.png'),
            dpi=150, bbox_inches='tight')
plt.close(plt.gcf())
print(f"  Saved: figs/measure_quick_detection.png")
print()

# ------------------------------------------------------------------ #
# 4.  Print detection results                                         #
# ------------------------------------------------------------------ #
print("=" * 60)
print("4. Detection results")
print("=" * 60)

measurements = res['measurements']
spectrum_data = res['spectrum_data']

print(f"  Flux (sum):    {measurements['flux_sum']:.4f}")
print(f"  Flux (fit):    {measurements['flux_fit']:.4f} "
      f"± {measurements['flux_fit_err']:.4f}  Jy km/s")
print(f"  W50:           {measurements['w50']:.0f} "
      f"± {measurements['w50_err']:.0f}  km/s")
print(f"  v_center:      {measurements['vc']:.0f} "
      f"± {measurements['vc_err']:.0f}  km/s")
print(f"  RMS:           {spectrum_data['rms']:.3f}")
print()

# ------------------------------------------------------------------ #
# 5.  Calculate measured frequency and redshift                      #
# ------------------------------------------------------------------ #
print("=" * 60)
print("5. Measured frequency and redshift")
print("=" * 60)

vc = measurements['vc']
vc_err = measurements['vc_err']
freq_measured = FREQ_LINE * (1 - vc / (LS_KM * u.km / u.s))
freq_measured_err = np.abs(FREQ_LINE * (vc_err / (LS_KM * u.km / u.s)))
z_measured = FREQ_REST / freq_measured - 1
z_measured_err = (1 + z_measured) * (freq_measured_err / freq_measured)

print(f"  Predicted freq:  {FREQ_LINE:.5f} GHz")
print(f"  Measured freq:   {freq_measured:.5f} ± {freq_measured_err:.5f} GHz")
print(f"  Input redshift:  {REDSHIFT:.4f}")
print(f"  Measured z:      {z_measured:.5f} ± {z_measured_err:.5f}")
print()

# ------------------------------------------------------------------ #
# 6.  Frequency confirmation plot                                     #
# ------------------------------------------------------------------ #
print("=" * 60)
print("6. Saving frequency confirmation plot")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))
spc_x = spectrum_data['spc_x']
spc_f = spectrum_data['spc_f']

fm_val = freq_measured if np.isscalar(freq_measured) else freq_measured.value
fm_err = freq_measured_err if np.isscalar(freq_measured_err) else freq_measured_err.value

ax.step(spc_x.value, spc_f.value, where='mid', lw=2, color='k',
        label='Spectrum')
ax.axvline(x=fm_val, color='r', ls='--', lw=2,
           label=f'Measured: {fm_val:.3f} ± {fm_err:.3f} GHz')
ax.axvspan(fm_val - fm_err, fm_val + fm_err,
           color='r', alpha=0.2, label='±1σ uncertainty')

if spectrum_data['spc_x_filtered'] is not None:
    ax.step(spectrum_data['spc_x_filtered'].value,
            spectrum_data['spc_f_filtered'].value,
            where='mid', lw=2, color='yellow', alpha=0.5,
            label='Line range')

rms_mjy = spectrum_data['rms'].to(u.mJy).value
ax.axhspan(ymin=-rms_mjy, ymax=rms_mjy, color='gray', alpha=0.3,
           label='±1σ RMS')

ax.set_xlabel('Frequency (GHz)', fontsize=16)
ax.set_ylabel(f'Flux ({spc_f.unit})', fontsize=16)
ax.set_title('GQC J0054-4955 — Frequency Confirmation', fontsize=18)
ax.legend(loc='upper right', fontsize=12)
ax.minorticks_on()
ax.grid(True, alpha=0.3)

fig.savefig(str(FIGS_DIR / 'measure_freq_confirmation.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: figs/measure_freq_confirmation.png")
print()

# ------------------------------------------------------------------ #
# 7.  Non-detection upper limit example                               #
# ------------------------------------------------------------------ #
print("=" * 60)
print("7. Non-detection upper limit example")
print("=" * 60)

res_nondet = quick_measure(
    cube,
    freq_line=FREQ_LINE * u.GHz,
    freq_range=[106.2, 106.6],
    mask_method='circular',
    nsigma=2,
    nbeam=1,
    field_radius=22,
    title='GQC J0054-4955 (non-det.)',
    detect=False,
    fit=False,
    vrange=400 * u.km / u.s,
)

plt.savefig(str(FIGS_DIR / 'measure_nondetection.png'),
            dpi=150, bbox_inches='tight')
plt.close(plt.gcf())

nondet_meas = res_nondet['measurements']
print(f"  3σ upper limit: {nondet_meas['flux_up']:.4f} Jy km/s")
print(f"  Saved: figs/measure_nondetection.png")
print()

# ------------------------------------------------------------------ #
# 8.  Summary                                                        #
# ------------------------------------------------------------------ #
print("=" * 60)
print("Summary")
print("=" * 60)
print(f"  Detection flux:  {measurements['flux_fit']:.4f} ± "
      f"{measurements['flux_fit_err']:.4f} Jy km/s")
print(f"  Line width W50:  {measurements['w50']:.0f} ± "
      f"{measurements['w50_err']:.0f} km/s")
print(f"  Measured z:      {z_measured:.5f} ± {z_measured_err:.5f}")
print(f"  Non-det limit:   {nondet_meas['flux_up']:.4f} Jy km/s (3σ)")
print()
print("Done. All figures saved to:", FIGS_DIR)
