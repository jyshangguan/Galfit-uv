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

## Environment

- Python 3 with CASA conda environment (`casatools`, `casatasks`).
- Key dependencies: `numpy`, `scipy`, `emcee`, `dill`, `matplotlib`, `corner`, `astropy`.
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
