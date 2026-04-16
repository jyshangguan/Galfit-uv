# UV Fitting API Reference

Function signatures for the UV visibility fitting workflow: export, model, fit, and plot.

---

## Classes

### `Visibility`

**Module:** `galfit_uv.export`

Container for visibility data extracted from a measurement set.

```python
Visibility(u, v, vis, wgt)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `u` | array-like | u coordinates in wavelengths |
| `v` | array-like | v coordinates in wavelengths |
| `vis` | array-like, complex | Visibility values in Jy |
| `wgt` | array-like | Visibility weights |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `uvdist` | ndarray | UV distance in wavelengths (`sqrt(u^2 + v^2)`) |
| `re` | ndarray | Real part of visibilities |
| `im` | ndarray | Imaginary part of visibilities |

---

### `MCMCResult`

**Module:** `galfit_uv.fit`

Dataclass container for MCMC fit results.

```python
@dataclass
class MCMCResult:
    bestfit: np.ndarray           # Median of post-burn-in posterior (n_params,)
    samples: np.ndarray           # Flattened post-burn-in samples (n_samples, n_params)
    all_samples: np.ndarray       # Full chain (n_steps, n_walkers, n_free_params)
    labels: list                  # Parameter label strings
    param_info: dict              # From make_model_fn
    acceptance_fraction: np.ndarray  # Per-walker acceptance rate
    burnin: int = 0               # Burn-in steps discarded
    outpath: str = '.'            # Output directory
    free_labels: list = None      # Labels of free params only
    fixed: dict = None            # {label: value} of fixed params
    fit_stats: dict = None        # {chi2, dof, redchi2, bic, ndata, n_free}
```

---

## Export Functions

### `export_vis`

**Module:** `galfit_uv.export`

Extract averaged visibility data from a CASA measurement set.

```python
export_vis(msfile, datacolumn='DATA', timebin=10.0, verbose=False)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `msfile` | str | required | Path to measurement set (must end in `.ms`) |
| `datacolumn` | str | `'DATA'` | Column to read (`'DATA'`, `'CORRECTED_DATA'`, etc.) |
| `timebin` | float or None | `10.0` | Time-bin width in seconds. `None` or 0 disables binning |
| `verbose` | bool | `False` | Print column shapes for debugging |

**Returns:** `(galfit_uv.Visibility, float)` — visibility data and average wavelength in meters.

**Details:**
- Handles variable-shape columns via `tb.getvarcol`.
- Averages polarizations using weights.
- Time-bins per baseline (ANTENNA1, ANTENNA2) within each time bin.
- Converts u,v from meters to wavelengths using per-row CHAN_FREQ.

---

### `save_uvtable`

**Module:** `galfit_uv.export`

Save visibility data as a uvplot-compatible ASCII table.

```python
save_uvtable(u, v, vis, wgt, filename, wle=None, uv_units='lambda')
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `u`, `v` | array-like | required | UV coordinates |
| `vis` | array-like, complex | required | Visibility values in Jy |
| `wgt` | array-like | required | Visibility weights |
| `filename` | str | required | Output file path |
| `wle` | float | `None` | Wavelength in meters (written to header) |
| `uv_units` | str | `'lambda'` | Units of u, v: `'lambda'` or `'m'` |

---

## Model Functions

### `vis_point`

**Module:** `galfit_uv.models`

Point source visibility model.

```python
vis_point(theta, uv, has_geometry=False)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `theta` | array-like | `[flux_mJy]` or `[flux_mJy, incl, PA, dx, dy]` if `has_geometry` |
| `uv` | tuple | `(u, v)` in wavelengths |
| `has_geometry` | bool | Whether theta includes geometry parameters |

**Returns:** `complex ndarray` — visibility in Jy.

---

### `vis_gaussian`

**Module:** `galfit_uv.models`

Gaussian surface-brightness visibility model (Hankel transform).

```python
vis_gaussian(theta, uv, has_geometry=False)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `theta` | array-like | `[flux_mJy, sigma]` or `[flux_mJy, sigma, incl, PA, dx, dy]` if `has_geometry` |
| `uv` | tuple | `(u, v)` in wavelengths |
| `has_geometry` | bool | Whether theta includes geometry parameters |

**Returns:** `complex ndarray` — visibility in Jy.

---

### `vis_sersic`

**Module:** `galfit_uv.models`

Sersic profile visibility model (Hankel transform).

```python
vis_sersic(theta, uv, has_geometry=False)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `theta` | array-like | `[flux_mJy, Re, n]` or `[flux_mJy, Re, n, incl, PA, dx, dy]` if `has_geometry` |
| `uv` | tuple | `(u, v)` in wavelengths |
| `has_geometry` | bool | Whether theta includes geometry parameters |

**Returns:** `complex ndarray` — visibility in Jy.

---

### `hankel_transform`

**Module:** `galfit_uv.models`

Compute visibility via numerical Hankel (J1) transform of a surface-brightness profile.

```python
hankel_transform(intensity_profile, bin_centers, bin_edges, rho)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `intensity_profile` | ndarray | Surface brightness at bin centers (Jy/arcsec^2) |
| `bin_centers` | ndarray | Radial bin centers in arcsec |
| `bin_edges` | ndarray | Radial bin outer edges in arcsec |
| `rho` | ndarray | UV distance in cycles/arcsec |

**Returns:** `complex ndarray` — visibility in Jy.

---

### `make_model_fn`

**Module:** `galfit_uv.models`

Build a combined multi-component visibility model function with default priors,
fixed-parameter support, and configurable geometry tying.

```python
make_model_fn(profiles, tie_center=True, tie_incl=True, tie_pa=True,
              priors=None, fixed=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `profiles` | list of str | required | Profile names: `'point'`, `'gaussian'`, `'sersic'`. 1 or 2 components |
| `tie_center` | bool | `True` | Tie dx, dy between components |
| `tie_incl` | bool | `True` | Tie inclination between components |
| `tie_pa` | bool | `True` | Tie position angle between components |
| `priors` | dict or None | `None` | Prior overrides. Keys in `"profile:param"` format, values `(lo, hi)` or `(lo, hi, scale)` |
| `fixed` | dict or None | `None` | Fixed parameter values. Keys in `"profile:param"` or bare label format |

**Returns:** `(model_fn, param_info)`

- `model_fn(theta, uv) -> complex ndarray` — combined model.
- `param_info`: dict with keys:

| Key | Type | Description |
|-----|------|-------------|
| `n_params` | int | Total number of parameters |
| `labels` | list | Parameter label strings |
| `profiles` | list | Profile names |
| `tie_center` | bool | Center tying flag |
| `tie_incl` | bool | Inclination tying flag |
| `tie_pa` | bool | PA tying flag |
| `p_ranges` | list | `[(lo, hi), ...]` for all params |
| `p_scales` | list | `['log'\|'linear', ...]` for all params |
| `p_ranges_info` | list | `[(lo, hi, scale), ...]` for all params |
| `fixed` | dict or None | `{label: value}` of fixed params |
| `fixed_indices` | list | Indices of fixed params |
| `fixed_values` | list | Values of fixed params |
| `free_labels` | list | Labels of free params only |
| `free_indices` | list | Indices of free params |
| `n_free` | int | Number of free params |
| `free_p_ranges` | list | `[(lo, hi), ...]` for free params |
| `free_p_scales` | list | `['log'\|'linear', ...]` for free params |

**Label format:**
- Single component: bare names — `['flux', 'Re', 'n', 'incl', 'PA', 'dx', 'dy']`
- Multi-component (tied): `['sersic:flux', 'sersic:Re', 'sersic:n', 'point:flux', 'incl', 'PA', 'dx', 'dy']`
- Multi-component (untied): `['sersic:flux', ..., 'sersic:incl', 'point:flux', ..., 'point:incl', 'PA', 'dx', 'dy']`

**Default priors** (`_DEFAULT_PRIORS`):

| Parameter | (lo, hi, scale) | Notes |
|-----------|-----------------|-------|
| `flux` | (0.1, 100.0, `'log'`) | Jeffreys prior |
| `Re` | (0.0, 5.0, `'linear'`) | |
| `sigma` | (0.0, 5.0, `'linear'`) | |
| `n` | (0.3, 8.0, `'linear'`) | Sersic index |
| `incl` | (0.0, 90.0, `'linear'`) | sin(incl) prior |
| `PA` | (-90.0, 90.0, `'linear'`) | |
| `dx` | (-5.0, 5.0, `'linear'`) | |
| `dy` | (-5.0, 5.0, `'linear'`) | |

**Raises:** `ValueError` if >2 profiles, unknown profile name, ambiguous bare key in multi-component, or log-scale prior with lo <= 0.

---

## Fitting Functions

### `compute_fit_stats`

**Module:** `galfit_uv.fit`

Compute goodness-of-fit statistics for visibility data.  Each complex visibility
point contributes 2 independent real DOF (real and imaginary parts).

```python
compute_fit_stats(dvis, model_fn, theta_full, n_free_params)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `dvis` | Visibility | required | Data visibility object |
| `model_fn` | callable | required | `model_fn(theta, uv) -> complex ndarray` |
| `theta_full` | ndarray | required | Full parameter vector (all params, including any fixed) |
| `n_free_params` | int | required | Number of free (non-fixed) parameters |

**Returns:** `dict`

| Key | Type | Description |
|-----|------|-------------|
| `chi2` | float | Sum of weighted squared residuals |
| `dof` | int | `2 * N_vis - n_free_params` |
| `redchi2` | float | `chi2 / dof` |
| `bic` | float | `chi2 + n_free * log(2 * N_vis)` |
| `ndata` | int | `2 * N_vis` |
| `n_free` | int | Number of free parameters |

---

### `LogProbability`

**Module:** `galfit_uv.fit`

Picklable log-probability function for emcee.  Supports three prior types:
`sin(incl)` for inclination, Jeffreys (`1/theta`) for `'log'`-scale params,
uniform for `'linear'`.  When fixed parameters are set, `__call__` accepts a
free-only theta vector and reconstructs the full vector before evaluation.

```python
LogProbability(model_fn, u, v, vis, wgt, p_ranges_info, labels,
               fixed_indices=None, fixed_values=None, full_ndim=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_fn` | callable | required | `model_fn(theta, uv) -> complex ndarray` |
| `u`, `v` | ndarray | required | UV coordinates in wavelengths |
| `vis` | ndarray | required | Data visibilities (complex, Jy) |
| `wgt` | ndarray | required | Visibility weights |
| `p_ranges_info` | list | required | `[(lo, hi, scale), ...]` for all params |
| `labels` | list | required | Parameter label strings |
| `fixed_indices` | list or None | `None` | Indices of fixed params |
| `fixed_values` | list or None | `None` | Values of fixed params |
| `full_ndim` | int or None | `None` | Full parameter dimension (auto-derived if None) |

---

### `fit_mcmc`

**Module:** `galfit_uv.fit`

Run MCMC fitting using emcee.  All prior configuration is derived from
`param_info` (set by `make_model_fn`).

```python
fit_mcmc(dvis, model_fn, param_info, p_init=None,
         max_steps=5000, burnin=1000, nwalk_factor=5,
         outpath='.', seed=None, n_workers=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dvis` | Visibility | required | Data visibility object |
| `model_fn` | callable | required | `model_fn(theta, uv) -> complex ndarray` |
| `param_info` | dict | required | From `make_model_fn` |
| `p_init` | array-like | `None` | Initial values (free params only). Walkers perturbed 0.1% around these |
| `max_steps` | int | `5000` | Maximum MCMC steps per walker |
| `burnin` | int | `1000` | Steps to discard |
| `nwalk_factor` | int | `5` | Walkers = `nwalk_factor * n_free` |
| `outpath` | str | `'.'` | Output directory for files |
| `seed` | int | `None` | Random seed |
| `n_workers` | int | `None` | Parallel workers. `None` uses Pool; `1` disables |

**Returns:** `MCMCResult`

**Details:**
- Uses dill-based `_DillPool` for multiprocessing (handles closures).
- **Import order matters:** `import galfit_uv` must come before any `import numpy` in the
  user's script.  Otherwise numpy's multi-threaded BLAS will consume all CPU cores,
  making parallel MCMC (`n_workers > 1`) extremely slow.
- Prior types: `sin(incl)` for inclination, Jeffreys (`1/theta`) for `'log'`-scale params, uniform for `'linear'`.
- Walker init: random from prior, or 0.1% perturbation around `p_init` if provided.
- `p_init` values outside prior bounds raise `ValueError`.
- Convergence check via integrated autocorrelation time (stops early if converged).
- Saves: `fit_results.fits`, `corner_plot.png`, `chains.png`.

---

## Plotting Functions

### `plot_uv`

**Module:** `galfit_uv.plot`

Plot real (and optional imaginary) visibility vs UV distance.

```python
plot_uv(dvis, mvis=None, n_bins=15, scale='log', use_std=True,
        outpath=None, fname='uvplot.png', unit_mJy=True,
        show_im=True, show_log_x=True, mvis_samples=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dvis` | Visibility | required | Data visibility |
| `mvis` | complex ndarray | `None` | Model visibility (same length as data) |
| `n_bins` | int | `15` | Number of UV-distance bins |
| `scale` | str | `'log'` | Binning scale: `'log'` or `'linear'` |
| `use_std` | bool | `True` | Error bars: `True` = std of mean, `False` = 1/sqrt(weight) |
| `outpath` | str | `None` | Output directory. `None` shows plot interactively |
| `fname` | str | `'uvplot.png'` | Output filename |
| `unit_mJy` | bool | `True` | Convert to mJy for display |
| `show_im` | bool | `True` | Show imaginary panel |
| `show_log_x` | bool | `True` | Log scale on x-axis |
| `mvis_samples` | ndarray | `None` | Posterior samples `(n_samples, n_vis)` for credible band |

**Returns:** `matplotlib.figure.Figure`

---

### `import_model_to_ms`

**Module:** `galfit_uv.plot`

Import model visibilities into a measurement set copy for CASA imaging.

```python
import_model_to_ms(msfile, u, v, mvis, wle, suffix='model',
                   make_resid=False, datacolumn='DATA')
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `msfile` | str | required | Path to original MS (must end in `.ms`) |
| `u`, `v` | array-like | required | Model UV coordinates in wavelengths |
| `mvis` | array-like, complex | required | Model visibility in Jy |
| `wle` | float | required | Wavelength in meters |
| `suffix` | str | `'model'` | Suffix for output MS (`{name}.{suffix}.ms`) |
| `make_resid` | bool | `False` | Also create a residual MS |
| `datacolumn` | str | `'DATA'` | Column to read original data from |

**Note:** When using for tclean, pass `datacolumn='CORRECTED_DATA'` (tclean reads the `corrected` column by default).

---

### `clean_image`

**Module:** `galfit_uv.plot`

Run CASA tclean on data, model, and residual MS files with identical gridding.

```python
clean_image(msfile, u, v, mvis, wle,
            outdir='.', suffix='galfit_uv',
            cell=None, imsize=None, niter=500,
            threshold='0.0mJy', gain=0.1,
            weighting='natural', robust=0.5,
            deconvolver='hogbom', stokes='I',
            source_size=None, fov_pad=10,
            verbose=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `msfile` | str | required | Original MS path |
| `u`, `v` | array-like | required | Model UV coordinates in wavelengths |
| `mvis` | array-like, complex | required | Model visibility in Jy |
| `wle` | float | required | Wavelength in meters |
| `outdir` | str | `'.'` | Directory for tclean output |
| `suffix` | str | `'galfit_uv'` | Suffix for MS copies |
| `cell` | str or None | `None` | Pixel size (e.g. `'0.5arcsec'`). Auto-derived if None |
| `imsize` | list or None | `None` | Image size `[nx, ny]`. Auto-derived if None |
| `niter` | int | `500` | Maximum clean iterations |
| `threshold` | str | `'0.0mJy'` | Clean threshold |
| `gain` | float | `0.1` | Loop gain |
| `weighting` | str | `'natural'` | UV weighting: `'natural'`, `'uniform'`, `'briggs'` |
| `robust` | float | `0.5` | Briggs robustness (for `weighting='briggs'`) |
| `deconvolver` | str | `'hogbom'` | Deconvolution algorithm |
| `stokes` | str | `'I'` | Stokes parameter |
| `source_size` | float or None | `None` | Source size in arcsec; sets FOV = `fov_pad * source_size` |
| `fov_pad` | float | `10` | FOV multiplier when `source_size` is set |
| `verbose` | bool | `True` | Print progress |

**Returns:** `dict`

| Key | Type | Description |
|-----|------|-------------|
| `data_image` | ndarray (2-D) | Restored clean image of data (Jy/beam) |
| `model_image` | ndarray (2-D) | Restored clean image of model |
| `resid_image` | ndarray (2-D) | Restored clean image of residual |
| `data_residual` | ndarray (2-D) | tclean residual for data |
| `beam_info` | dict | `{'bmaj': float, 'bmin': float, 'bpa': float}` (arcsec, deg) |
| `cellsize` | float | Pixel size in arcsec |
| `imsize` | list | `[nx, ny]` |
| `outdir` | str | Output directory path |

**Details:**
- Internally calls `import_model_to_ms` with `datacolumn='CORRECTED_DATA'`.
- Runs tclean on three MS copies: original data, model, residual.
- Cell size defaults to `1/(4*u_max)` radians; image size is next power of 2.

---

### `plot_clean_images`

**Module:** `galfit_uv.plot`

Three-panel figure: cleaned data, cleaned model, residual.

```python
plot_clean_images(data_image, model_image, resid_image,
                  beam_info, cellsize,
                  imsize=None, outpath=None, fname='clean_images.png',
                  unit_mJy=True, crosshairs=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_image` | ndarray | required | Restored clean image of data (Jy/beam) |
| `model_image` | ndarray | required | Restored clean image of model |
| `resid_image` | ndarray | required | Restored clean image of residual |
| `beam_info` | dict | required | `{'bmaj': float, 'bmin': float, 'bpa': float}` |
| `cellsize` | float | required | Pixel size in arcsec |
| `imsize` | list | `None` | `[nx, ny]`. Inferred from array if None |
| `outpath` | str | `None` | Output directory. `None` shows plot |
| `fname` | str | `'clean_images.png'` | Output filename |
| `unit_mJy` | bool | `True` | Display in mJy/beam |
| `crosshairs` | bool | `True` | Draw crosshairs at center |

**Returns:** `matplotlib.figure.Figure`

**Details:**
- Uses `'inferno'` colormap for data/model, `'RdBu_r'` for residual.
- Contour levels: data/model at 3,5,10,20,50 x RMS; residual at +/-3 sigma.
- Draws beam ellipse in bottom-left corner of each panel.
