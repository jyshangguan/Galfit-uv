# galfit-uv

Parametric UV visibility fitting for ALMA data. galfit-uv extracts visibilities from CASA measurement sets, fits surface-brightness models (point source, Gaussian, Sersic) via MCMC using numerical Hankel transforms, and produces UV plots, clean images, and diagnostic figures. The package supports multi-component models with configurable geometry tying and fixed-parameter constraints.

## Features

- **Visibility export** from CASA measurement sets with variable-shape column handling and time binning
- **Parametric profiles**: point source, Gaussian, and generalized Sersic
- **MCMC fitting** with emcee, Jeffreys and sin(incl) priors, and automatic convergence checks
- **Multi-component models** (up to 2 profiles) with configurable geometry tying
- **Clean imaging** via CASA tclean — data, model, and residual side by side
- **Diagnostic plots**: UV visibility comparison, corner plots, and walker chains

## Installation

Requires a CASA conda environment (provides `casatools` and `casatasks`):

```bash
conda activate alma
pip install -e /path/to/Galfit-uv
```

**Dependencies:**

| Package | Source |
|---------|--------|
| `numpy`, `scipy` | pip / conda |
| `emcee`, `dill` | pip |
| `matplotlib`, `corner` | pip |
| `astropy` | pip / conda |
| `casatools`, `casatasks` | CASA conda environment (not pip-installable) |

> **Note:** Always import `galfit_uv` (not submodules directly) so that `__init__.py` can set threading environment variables before numpy loads.

## Quick Start

A complete four-step workflow: export, model, fit, and plot.

```python
import galfit_uv

# 1. Export visibilities from a measurement set
dvis, wle = galfit_uv.export_vis('target.ms', verbose=True)

# 2. Build a model (e.g., single Sersic profile)
model_fn, param_info = galfit_uv.make_model_fn(['sersic'])

# 3. Fit with MCMC
result = galfit_uv.fit_mcmc(
    dvis, model_fn, param_info,
    max_steps=5000, burnin=2500, nwalk_factor=5,
    outpath='./fit_output', seed=42,
)

# 4. Plot data vs model
uv = (dvis.u, dvis.v)
mvis = model_fn(result.bestfit, uv)
galfit_uv.plot_uv(dvis, mvis, outpath='./figs', fname='uvplot.png')
```

> **Demo data:** The measurement set used in the examples (GQC J0054-4955) can be downloaded from: <https://drive.google.com/file/d/1TPZQvP-7wc5Kk169Gh6IgiYAUoaug2bb/view?usp=sharing>

## Available Profiles

| Profile | Intrinsic parameters | Description |
|---------|---------------------|-------------|
| `point` | flux (mJy) | Unresolved source |
| `gaussian` | flux (mJy), sigma (arcsec) | Gaussian surface brightness |
| `sersic` | flux (mJy), Re (arcsec), n | Generalized Sersic profile |

All profiles share geometry parameters: `incl`, `PA`, `dx`, `dy`.

Multi-component models (e.g., `['sersic', 'point']`) support configurable geometry tying via `tie_center`, `tie_incl`, `tie_pa` flags, and fixed parameters via the `fixed` dict.

## Units Convention

| Quantity | Internal units | Display units |
|----------|---------------|---------------|
| Visibility amplitudes | Jy (complex) | mJy |
| Flux parameters | mJy | mJy |
| Size parameters (sigma, Re) | arcsec | arcsec |
| UV coordinates (u, v) | wavelengths (lambda) | kilo-lambda |
| Wavelength (wle) | meters | mm |

## Next Steps

- **[Examples](examples.md)** — Walk through fitting real ALMA data (GQC J0054-4955) with two different models
- **[API Reference](api.md)** — Full function signatures, parameters, and return values
