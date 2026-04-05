# Galfit-uv

Parametric UV visibility fitting for ALMA data. Extracts visibilities from
CASA measurement sets, fits surface-brightness models (point source, Gaussian,
Sersic) via MCMC (emcee) with numerical Hankel transforms, and produces UV
plots, clean images, and diagnostic figures.

## Installation

Requires a CASA conda environment (provides `casatools` and `casatasks`):

```bash
conda activate alma
pip install -e /path/to/Galfit-uv
```

## Quick Start

```python
import galfit_uv

# 1. Export visibilities from a measurement set
dvis, wle = galfit_uv.export_vis('target.ms', verbose=True)

# 2. Build a model (e.g., Sersic profile)
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

## Available Profiles

| Profile   | Intrinsic params       |
|-----------|----------------------|
| `point`   | flux (mJy)           |
| `gaussian`| flux (mJy), sigma (arcsec) |
| `sersic`  | flux (mJy), Re (arcsec), n |

Shared geometry: `incl`, `PA`, `dx`, `dy`.

Multi-component models (e.g., `['sersic', 'point']`) support configurable
geometry tying via `tie_center`, `tie_incl`, `tie_pa` flags, and fixed
parameters via the `fixed` dict.

## Demo Data

The measurement set used in the examples (GQC J0054-4955, CO(3-2), z~2.5) can be
downloaded from: https://drive.google.com/file/d/1TPZQvP-7wc5Kk169Gh6IgiYAUoaug2bb/view?usp=sharing

## Dependencies

- `numpy`, `scipy`, `emcee`, `dill`, `matplotlib`, `corner`, `astropy`
- `casatools`, `casatasks` — from the CASA conda environment (not pip-installable)

## Citation

If you use `galfit_uv` in your research, please cite:

> Long, F. et al. 2021, ApJ, 915, 131 — [ADS](https://ui.adsabs.harvard.edu/abs/2021ApJ...915..131L/abstract)

## Acknowledgements

Development of this code was assisted by [Claude Code](https://claude.com/claude-code) and [GLM-5](https://bigmodel.cn).

## License

Internal use. See source code for details.

## Documentation

Full documentation is hosted at: https://jyshangguan.github.io/Galfit-uv/
