# galfit_uv — ALMA Visibility Fitting Package

## Overview

galfit_uv is a Python package for fitting parametric surface-brightness models to
ALMA interferometric visibility data.  It extracts visibilities from CASA
measurement sets, fits models via MCMC (emcee), and produces UV plots, clean
images, and diagnostic figures.

## Module Map

| Module | Purpose |
|--------|---------|
| `galfit_uv/export.py` | Read MS columns (`export_vis`, `save_uvtable`, `Visibility`) |
| `galfit_uv/models.py` | Profile functions and Hankel transforms (`vis_point`, `vis_gaussian`, `vis_sersic`, `make_model_fn`) |
| `galfit_uv/fit.py` | MCMC sampler with emcee (`fit_mcmc`, `MCMCResult`) |
| `galfit_uv/plot.py` | UV plots, MS import, tclean imaging (`plot_uv`, `import_model_to_ms`, `clean_image`, `plot_clean_images`) |
| `galfit_uv/lineprofiles.py` | Spectral line profile functions (`Gaussian`, `Gaussian_DoublePeak`, `Gaussian_DoublePeak_Asymmetric`) |
| `galfit_uv/measure.py` | Cube measurement pipeline (`source_mask`, `extract_spectrum`, `detect_source`, `quick_measure`, `fit_dynesty`) |

## Environment

- Python 3 with CASA conda environment (`casatools`, `casatasks`).
- Key dependencies: `numpy`, `scipy`, `emcee`, `dill`, `matplotlib`, `corner`, `astropy`.
- Optional `[measure]` dependencies: `spectral-cube`, `dynesty` (for cube measurement pipeline).
- Import `galfit_uv` (not submodules directly) so `__init__.py` sets threading env vars before numpy loads.

## Units Convention

- Internal visibility values: **Jy** (complex).
- Flux parameters in model functions: **mJy** (converted to Jy internally).
- Size parameters (`sigma`, `Re`): **arcsec**.
- UV coordinates (`u`, `v`): **wavelengths** (lambda).
- Wavelength (`wle`): **meters**.
- Display (plots): **mJy** on y-axis, **kilo-lambda** on x-axis.

## Critical Gotchas

1. **`import_model_to_ms` datacolumn**: When feeding into `clean_image`, the
   internal call uses `datacolumn='CORRECTED_DATA'` because tclean defaults to
   `datacolumn='corrected'`.  If calling `import_model_to_ms` manually for
   tclean, pass `datacolumn='CORRECTED_DATA'`.

2. **Variable-shape MS columns**: SPWs with different channel counts produce
   variable-shape columns.  Handled via `tb.getvarcol` in `export.py` — do not
   bypass with `tb.getcol`.

3. **Numpy threading**: `__init__.py` limits all numpy BLAS threads to 1
   (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, etc.) to avoid contention with
   multiprocessing workers.  Do not override these env vars.

4. **Multiprocessing with closures**: `fit_mcmc` uses a dill-based pool
   (`_DillPool`) instead of stdlib `multiprocessing.Pool` so that closures
   (e.g. `model_fn` from `make_model_fn`) can be serialized to workers.
   Setting `n_workers=1` disables parallelism for debugging.

5. **Priors**: `make_model_fn` defines default prior ranges and scales for all
   parameters via `_DEFAULT_PRIORS`.  Flux uses Jeffreys (log) prior; all others
   use uniform (linear).  Inclination additionally gets a `sin(incl)` prior.
   Override defaults via the `priors` dict; fix parameters via the `fixed` dict.
   Prior scales are `'log'` (Jeffreys, log-space init) or `'linear'` (uniform).

6. **Label format**: Single-component models use bare names (`flux`, `Re`, `n`,
   `incl`, `PA`, `dx`, `dy`).  Multi-component models prefix intrinsic params
   with `profile:` (e.g. `sersic:flux`, `point:flux`) and use bare names for
   tied shared params.

## Quick Reference

| Function | Description |
|----------|-------------|
| `Visibility(u, v, vis, wgt)` | Container for visibility data |
| `export_vis(msfile)` | Extract averaged visibilities from MS |
| `save_uvtable(u, v, vis, wgt, filename)` | Write uvplot-compatible ASCII table |
| `vis_point(theta, uv)` | Point source visibility model |
| `vis_gaussian(theta, uv)` | Gaussian visibility model |
| `vis_sersic(theta, uv)` | Sersic profile visibility model |
| `hankel_transform(profile, centers, edges, rho)` | Numerical J1 Hankel transform |
| `make_model_fn(profiles)` | Build combined multi-component model function |
| `fit_mcmc(dvis, model_fn, ...)` | Run MCMC fit with emcee |
| `plot_uv(dvis, mvis)` | Plot data vs model UV comparison |
| `import_model_to_ms(msfile, u, v, mvis, wle)` | Write model vis into MS copy |
| `clean_image(msfile, u, v, mvis, wle)` | Run tclean on data/model/residual |
| `plot_clean_images(data, model, resid, beam)` | Three-panel clean image figure |
| `Gaussian(x, a, b, c)` | Gaussian line profile |
| `Gaussian_DoublePeak(x, ag, ac, v0, sigma, w)` | Double-horn profile (Tiley+ 2016) |
| `Gaussian_DoublePeak_Asymmetric(x, ag_l, ag_r, ac, v0, sigma, w_l, w_r)` | Asymmetric double-horn profile |
| `source_mask(cube, nbeam, offset)` | Circular aperture mask |
| `source_mask_snr(cube, nbeam, nsigma, ...)` | SNR threshold mask with segment filtering |
| `field_mask(cube, radius)` | Field boundary mask |
| `extract_spectrum(cube, mask_src, perbeam)` | Extract 1D spectrum from cube |
| `detect_source(cube, ...)` | Detect source and extract spectrum |
| `quick_measure(cube, freq_line, ...)` | Top-level: detect + plot + optional fit |
| `fit_dynesty(x, y, yerr, model_type, ...)` | Nested sampling fit with dynesty |
| `calculate_w50(x, y)` | Width at 50% of peak |
| `Plot_Map(mom, ...)` | Moment map visualization |
| `Plot_Beam(ax, bmaj, bmin, bpa)` | Beam ellipse overlay |
| `plot_1d_spectrum(x, y, ...)` | Step plot for spectra |


## Development

The working directory for developing new features of this package is `dev/` which is not tracked by git. Create one if it does not exist. Use the following strategy for the development.
- Create a `develop_note.md` file, if it does not exist, to record all the features and functions added.
- Create a `problem_note.md` file, if it does not exist, to record the existing problems or commonly happened errors.
- Create a `tasks_note.md` file, if it does not exist, to record the current tasks and action items. Check all the items when they are done.
- Keep updating the above files. Everytime when planning a new task, always read these files first to understand the status.
- The user may use the `prompt.md` file to record the prompts.

### Completeness check

After the development of a key feature, always remember to check the following,
- Update the unittests so that the new features are tested and the old tests are still working.
- Update the skills to be consistent with the modified code.
- Update the docs so that the new features are updated.