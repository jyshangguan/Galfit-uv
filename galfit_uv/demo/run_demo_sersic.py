#!/usr/bin/env python
"""
Demonstration of the galfit_uv package — Sersic model.

Uses real ALMA data (GQC J0054-4955) to show:
  1. Exporting visibility data from a measurement set
  2. Building and fitting a single Sersic profile model
  3. Plotting data vs model
  4. Diagnostic plots (chains, corner)
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

from galfit_uv.export import export_vis, save_uvtable
from galfit_uv.models import make_model_fn
from galfit_uv.fit import fit_mcmc
from galfit_uv.plot import plot_uv, import_model_to_ms, clean_image, plot_clean_images

DEMO_DIR = Path(__file__).resolve().parent
FIGS_DIR = DEMO_DIR / 'figs'
FIGS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------ #
# 1.  Export visibility data from MS                                  #
# ------------------------------------------------------------------ #
print("=" * 60)
print("1. Exporting visibility data from MS")
print("=" * 60)

MS_FILE = str(DEMO_DIR / 'data' / 'GQC_J0054-4955_avg.ms')

dvis, wle = export_vis(MS_FILE, verbose=True)
print(f"\nExtracted {len(dvis.u)} visibility points")
print(f"Wavelength: {wle*1e3:.3f} mm")
print(f"uv range: {dvis.uvdist.min()/1e3:.0f} -- {dvis.uvdist.max()/1e3:.0f} klambda")
print(f"Total flux (zero-baseline): {np.abs(dvis.vis[0])*1e3:.2f} mJy")

# Save as uvplot-compatible text file
uvtxt = str(DEMO_DIR / 'GQC_J0054-4955_uv.txt')
save_uvtable(dvis.u, dvis.v, dvis.vis, dvis.wgt, uvtxt, wle=wle)
print()

# ------------------------------------------------------------------ #
# 2.  Build model and set up fit                                       #
# ------------------------------------------------------------------ #
print("=" * 60)
print("2. Setting up Sersic profile model")
print("=" * 60)

model_fn, param_info = make_model_fn(['sersic'])
labels = param_info['labels']
n_params = param_info['n_params']

print(f"Profiles: {param_info['profiles']}")
print(f"Parameters ({n_params}): {labels}")

OUTPATH = str(DEMO_DIR / 'fit_output_sersic')
print()

# ------------------------------------------------------------------ #
# 3.  Run MCMC fit                                                    #
# ------------------------------------------------------------------ #
print("=" * 60)
print("3. Running MCMC fit")
print("=" * 60)

result = fit_mcmc(
    dvis, model_fn, param_info,
    max_steps=5000,
    burnin=2500,
    nwalk_factor=5,
    outpath=OUTPATH,
    seed=42,
    n_workers=32,
)
print()

# ------------------------------------------------------------------ #
# 4.  Plot data vs model                                              #
# ------------------------------------------------------------------ #
print("=" * 60)
print("4. Plotting data vs model")
print("=" * 60)

uv = (dvis.u, dvis.v)
mvis = model_fn(result.bestfit, uv)

# Draw 100 random posterior samples for model uncertainty band
idx = np.random.choice(len(result.samples), size=100, replace=False)
mvis_samples = np.array([model_fn(result.samples[j], uv) for j in idx])

# Single UV plot: real-only, log x-axis, with fit statistics
fig = plot_uv(dvis, mvis, n_bins=15, scale='log', outpath=str(FIGS_DIR),
              fname='uvplot_sersic.png', show_im=False, show_log_x=True,
              mvis_samples=mvis_samples,
              fit_stats=result.fit_stats)
plt.close(fig)

print()

# ------------------------------------------------------------------ #
# 5.  Copy diagnostic plots to figs/                                  #
# ------------------------------------------------------------------ #
print("=" * 60)
print("5. Organizing output plots")
print("=" * 60)

import shutil
diag_files = ['corner_plot.png']
for fname in diag_files:
    src = os.path.join(OUTPATH, fname)
    dst = str(FIGS_DIR / fname.replace('.png', '_sersic.png'))
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  Copied {fname} -> {dst}")

# Copy chain plots
chains_src = os.path.join(OUTPATH, 'chains.png')
if os.path.exists(chains_src):
    shutil.copy2(chains_src, str(FIGS_DIR / 'chains_sersic.png'))
    print(f"  Copied chains_sersic.png")

print()

# ------------------------------------------------------------------ #
# 6.  Print best-fit summary                                          #
# ------------------------------------------------------------------ #
print("=" * 60)
print("6. Best-fit results")
print("=" * 60)

re_samples = result.samples
for i, lbl in enumerate(labels):
    q16, q50, q84 = np.percentile(re_samples[:, i], [16, 50, 84])
    print(f"  {lbl:>20s} = {q50:+.4f}  (+{q84-q50:.4f}, -{q50-q16:.4f})")

print()
print("Done. All outputs saved to:", DEMO_DIR)

# ------------------------------------------------------------------ #
# 7.  Clean images and plot                                            #
# ------------------------------------------------------------------ #
print("=" * 60)
print("7. Running tclean and plotting clean images")
print("=" * 60)

clean = clean_image(MS_FILE, dvis.u, dvis.v, mvis, wle,
                    outdir=str(DEMO_DIR / 'clean_output_sersic'),
                    source_size=result.bestfit[labels.index('Re')],
                    niter=500, verbose=True)

fig = plot_clean_images(clean['data_image'], clean['model_image'],
                        clean['resid_image'], clean['beam_info'],
                        clean['cellsize'], imsize=clean['imsize'],
                        outpath=str(FIGS_DIR), fname='clean_images_sersic.png')
plt.close(fig)

print(f"Beam: {clean['beam_info']['bmaj']:.3f}\" x "
      f"{clean['beam_info']['bmin']:.3f}\", PA={clean['beam_info']['bpa']:.1f} deg")
print()
print("Done. All outputs saved to:", DEMO_DIR)
