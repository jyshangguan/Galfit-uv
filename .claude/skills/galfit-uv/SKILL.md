---
name: galfit_uv
description: >
  This skill should be used when the user asks to "fit ALMA visibility data",
  "run UV fitting", "fit a Gaussian model to visibilities", "use galfit_uv",
  "export visibilities from measurement set", "run MCMC on ALMA data",
  "make clean images from galfit_uv results", or "plot UV data vs model".
---

# galfit_uv Workflow

Execute the full galfit_uv pipeline: export visibilities from a CASA measurement
set, build a parametric model, fit with MCMC, plot the UV comparison, and
optionally generate clean images.

## Prerequisites

- CASA conda environment is active (provides `casatools`, `casatasks`).
- The galfit_uv package is importable (`import galfit_uv`).
- The measurement set (`.ms`) exists at the path the user provides.

**IMPORTANT — Import order:** `import galfit_uv` **must** appear before any `import numpy`
(or other library that imports numpy).  The package's `__init__.py` sets environment
variables that limit numpy's internal BLAS threads to 1, preventing numpy from
spawning threads on every CPU core.  If numpy is imported first, it locks in the
default thread count and each MCMC worker will compete for cores, making
multi-worker fits extremely slow.

## Workflow

### Step 1 — Export visibilities

```python
from galfit_uv.export import export_vis, save_uvtable

dvis, wle = export_vis('/path/to/source.ms', verbose=True)
```

- Returns `Visibility` object (`.u`, `.v`, `.vis`, `.wgt`) and wavelength in meters.
- Default: reads `DATA` column, time-bins at 10 s.  Use `timebin=None` to skip binning.
- Optionally save an ASCII table for external tools:

```python
save_uvtable(dvis.u, dvis.v, dvis.vis, dvis.wgt, 'output_uv.txt', wle=wle)
```

### Step 2 — Build model

```python
from galfit_uv.models import make_model_fn

model_fn, param_info = make_model_fn(['sersic', 'point'])
```

`make_model_fn` returns a callable `model_fn(theta, uv) -> complex ndarray` and
a `param_info` dict with labels, priors, fixed info, and free/free-only slices.

**Available profiles** (use 1 or 2):

| Profile | Intrinsic params |
|---------|-----------------|
| `'point'` | flux (mJy) |
| `'gaussian'` | flux (mJy), sigma (arcsec) |
| `'sersic'` | flux (mJy), Re (arcsec), n (index) |

Shared geometry params: `incl` (deg), `PA` (deg), `dx` (arcsec), `dy` (arcsec).

**Tie flags** (all default `True`):
- `tie_center` — tie dx, dy between components
- `tie_incl` — tie inclination
- `tie_pa` — tie position angle

**Label format:**
- Single component: bare names — `['flux', 'Re', 'n', 'incl', 'PA', 'dx', 'dy']`
- Multi-component: intrinsic params prefixed with `profile:`, tied shared params bare:
  `['sersic:flux', 'sersic:Re', 'sersic:n', 'point:flux', 'incl', 'PA', 'dx', 'dy']`

**Default priors** (built into `make_model_fn`):

| Parameter | Default range | Scale | Notes |
|-----------|--------------|-------|-------|
| `flux` | (0.1, 100) mJy | log | Jeffreys prior |
| `Re` | (0, 5) arcsec | linear | |
| `sigma` | (0, 5) arcsec | linear | |
| `n` | (0.3, 8) | linear | Sersic index |
| `incl` | (0, 90) deg | linear | sin(incl) prior |
| `PA` | (-90, 90) deg | linear | |
| `dx`, `dy` | (-5, 5) arcsec | linear | |

Override with `priors` dict (keys in `"profile:param"` format):

```python
model_fn, param_info = make_model_fn(
    ['sersic'],
    priors={'flux': (1.0, 50.0), 'dx': (-0.5, 0.5)},
)
```

Fix parameters with `fixed` dict:

```python
model_fn, param_info = make_model_fn(
    ['sersic', 'point'],
    fixed={'sersic:n': 1.0},
)
```

### Step 3 — Run MCMC

```python
from galfit_uv.fit import fit_mcmc

result = fit_mcmc(
    dvis, model_fn, param_info,
    max_steps=5000, burnin=2500, nwalk_factor=5,
    outpath='./fit_output', seed=42, n_workers=32,
)
```

- Default priors are set by `make_model_fn` — no manual `p_ranges`/`p_lo`/`p_hi` needed.
- Optionally provide `p_init` (free params only) to center walkers:
  `p_init=np.array([10.0, 0.5, 1.0, ...])`.  Walkers initialized with 0.1% perturbation.
- `nwalk_factor`: number of walkers = `nwalk_factor * n_free`.
- `burnin`: steps to discard; should be >= the autocorrelation time.
- `n_workers`: parallel workers; set to 1 for debugging.
- **Ensure `galfit_uv` was imported before numpy** (see Prerequisites), otherwise
  numpy's multi-threaded BLAS will consume all cores and the parallel MCMC will be
  orders of magnitude slower.
- Returns `MCMCResult` with `.bestfit`, `.samples`, `.labels`, `.outpath`, `.fit_stats`.

**Post-fit statistics** (automatically computed and stored in `result.fit_stats`):
- `chi2`, `dof`, `redchi2`, `bic`, `ndata`, `n_free`
- Can also be computed manually with `compute_fit_stats`:

```python
from galfit_uv.fit import compute_fit_stats
stats = compute_fit_stats(dvis, model_fn, result.bestfit, n_free_params=len(result.free_labels))
```

**Output files** (in `outpath`):
- `fit_results.fits` — multi-extension FITS with data, model, best-fit params, samples, fit statistics
- `corner_plot.png` — posterior corner plot (free params only)
- `chains.png` — walker chain plots with autocorrelation time

### Step 4 — Plot UV comparison

```python
from galfit_uv.plot import plot_uv

uv = (dvis.u, dvis.v)
mvis = model_fn(result.bestfit, uv)

# Posterior uncertainty band (draw ~100 samples)
idx = np.random.choice(len(result.samples), size=100, replace=False)
mvis_samples = np.array([model_fn(result.samples[j], uv) for j in idx])

plot_uv(dvis, mvis, n_bins=15, scale='log', outpath='./figs',
         fname='uvplot.png', mvis_samples=mvis_samples)
```

- Default: log-scale x-axis (kilo-lambda), mJy y-axis, 15 bins, shows imaginary panel.
- Set `show_im=False` to show only the real part.
- Set `scale='linear'` and `show_log_x=False` for linear binning.
- `mvis_samples` draws a shaded 16-84% credible region.

### Step 5 — Clean images (optional)

```python
from galfit_uv.plot import clean_image, plot_clean_images

clean = clean_image(
    '/path/to/source.ms', dvis.u, dvis.v, mvis, wle,
    outdir='./clean_output',
    source_size=result.bestfit[param_info['labels'].index('sersic:Re')],
    niter=500,
)

plot_clean_images(
    clean['data_image'], clean['model_image'], clean['resid_image'],
    clean['beam_info'], clean['cellsize'], imsize=clean['imsize'],
    outpath='./figs', fname='clean_images.png',
)
```

- `clean_image` internally calls `import_model_to_ms` with
  `datacolumn='CORRECTED_DATA'` (required for tclean).
- `source_size` sets the FOV to `fov_pad * source_size` (default `fov_pad=10`).
- Omit `source_size` to derive FOV from UV coverage.
- Returns dict with `data_image`, `model_image`, `resid_image`, `beam_info`,
  `cellsize`, `imsize`.
- `plot_clean_images` produces a 3-panel figure (data / model / residual).

## Interpreting Results

**Checklist after a fit:**

1. **Acceptance fraction**: should be 0.2-0.5 per walker.  Very low values
   indicate poor mixing; try wider initial draw ranges or reparameterize.
2. **Autocorrelation time** (`tau` in chain plots): burn-in should be several
   times tau.  If tau is large relative to `max_steps`, run longer.
3. **Corner plot**: look for unimodal, roughly Gaussian posteriors.  Strong
   bimodality or elongated degeneracies suggest the model is under-constrained.
4. **UV plot**: the model curve should pass through the binned data points
   within error bars.  Systematic offsets at short or long baselines indicate
   missing flux or wrong source geometry.
5. **Imaginary part**: should scatter around zero for a centrosymmetric source.
   Persistent non-zero imaginary signal suggests complex visibility structure
   not captured by the model.
6. **Clean images**: residual panel should show noise only (no coherent
   structure).  Contours in the residual beyond +/-3 sigma indicate a poor fit.

## When to Add a Second Component

Start with a single profile (`['gaussian']` or `['sersic']`).  Add a second
component (e.g. `['gaussian', 'point']`) when:

- The UV plot shows excess flux at short baselines not captured by a single
  Gaussian (suggests an extended + compact structure).
- The residual clean image shows coherent structure at the source position.
- The single-component fit has systematically high chi-squared residuals.

Common two-component models:
- `['gaussian', 'point']` — extended emission + compact nucleus
- `['sersic', 'point']` — host galaxy + AGN core
- `['gaussian', 'gaussian']` — two emission components with different sizes
