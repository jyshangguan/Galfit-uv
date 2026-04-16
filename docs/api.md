# API Reference

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
| `uvdist` | ndarray | UV distance in wavelengths (sqrt(u^2 + v^2)) |
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

## Export — `galfit_uv.export`

### `export_vis`

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

## Models — `galfit_uv.models`

### `vis_point`

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

Build a combined multi-component visibility model function with default priors, fixed-parameter support, and configurable geometry tying.

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
| `p_scales` | list | `['log' | 'linear', ...]` for all params |
| `p_ranges_info` | list | `[(lo, hi, scale), ...]` for all params |
| `fixed` | dict or None | `{label: value}` of fixed params |
| `fixed_indices` | list | Indices of fixed params |
| `fixed_values` | list | Values of fixed params |
| `free_labels` | list | Labels of free params only |
| `free_indices` | list | Indices of free params |
| `n_free` | int | Number of free params |
| `free_p_ranges` | list | `[(lo, hi), ...]` for free params |
| `free_p_scales` | list | `['log' | 'linear', ...]` for free params |

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

## Fitting — `galfit_uv.fit`

### `compute_fit_stats`

Compute goodness-of-fit statistics for visibility data. Each complex visibility point contributes 2 independent real DOF (real and imaginary parts).

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

Picklable log-probability function for emcee. Supports three prior types:
`sin(incl)` for inclination, Jeffreys (`1/theta`) for `'log'`-scale params,
uniform for `'linear'`. When fixed parameters are set, `__call__` accepts a
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

Run MCMC fitting using emcee. All prior configuration is derived from `param_info` (set by `make_model_fn`).

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
- Prior types: `sin(incl)` for inclination, Jeffreys (1/theta) for `'log'`-scale params, uniform for `'linear'`.
- Walker init: random from prior, or 0.1% perturbation around `p_init` if provided.
- `p_init` values outside prior bounds raise `ValueError`.
- Convergence check via integrated autocorrelation time (stops early if converged).
- Saves: `fit_results.fits`, `corner_plot.png`, `chains.png`.

---

## Plotting — `galfit_uv.plot`

### `plot_uv`

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

> **Note:** When using for tclean, pass `datacolumn='CORRECTED_DATA'` (tclean reads the `corrected` column by default).

---

### `clean_image`

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
| `cell` | str or None | `None` | Pixel size (e.g., `'0.5arcsec'`). Auto-derived if None |
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
- Contour levels: data/model at 3, 5, 10, 20, 50 x RMS; residual at +/-3 sigma.
- Draws beam ellipse in bottom-left corner of each panel.

---

## Line Profiles — `galfit_uv.lineprofiles`

### `Gaussian`

**Module:** `galfit_uv.lineprofiles`

A Gaussian function.

```python
Gaussian(x, a, b, c)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | array-like | Independent variable |
| `a` | float | Amplitude |
| `b` | float | Mean (center) |
| `c` | float | Standard deviation |

**Returns:** `ndarray` — `a * exp(-0.5 * (x - b)^2 / c^2)`

---

### `Gaussian_DoublePeak`

**Module:** `galfit_uv.lineprofiles`

Gaussian Double Peak function, Eq. (A2) of Tiley et al. (2016, MNRAS, 461, 3494).
Three regions: left half-Gaussian centered at `v0 - w`, a central parabola,
and a right half-Gaussian centered at `v0 + w`.

```python
Gaussian_DoublePeak(x, ag, ac, v0, sigma, w)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | 1D array | Monotonically increasing independent variable |
| `ag` | float | Peak flux of the two half-Gaussians (must be > 0) |
| `ac` | float | Flux at the central velocity (must be > 0) |
| `v0` | float | Center of the profile |
| `sigma` | float | Standard deviation of the half-Gaussian (must be > 0) |
| `w` | float | Half-width of the central parabola (must be > 0) |

**Returns:** `1D ndarray` — concatenated `[left, center, right]` segments.

**Raises:** `ValueError` if any parameter is non-positive.

---

### `Gaussian_DoublePeak_Asymmetric`

**Module:** `galfit_uv.lineprofiles`

Asymmetric Gaussian Double Peak function. Same structure as `Gaussian_DoublePeak`
but allows different peak amplitudes and half-widths for the left and right sides.

```python
Gaussian_DoublePeak_Asymmetric(x, ag_left, ag_right, ac, v0, sigma, w_left, w_right)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | 1D array | Monotonically increasing independent variable |
| `ag_left` | float | Peak flux of the left half-Gaussian (must be > 0) |
| `ag_right` | float | Peak flux of the right half-Gaussian (must be > 0) |
| `ac` | float | Flux at the central velocity (must be > 0) |
| `v0` | float | Center of the profile |
| `sigma` | float | Standard deviation of the half-Gaussian (must be > 0) |
| `w_left` | float | Left half-width of the central parabola (must be > 0) |
| `w_right` | float | Right half-width of the central parabola (must be > 0) |

**Returns:** `1D ndarray` — concatenated `[left, center, right]` segments.

**Raises:** `ValueError` if any parameter is non-positive.

---

## Cube Measurement Pipeline — `galfit_uv.measure`

> **Note:** Functions in this module that accept `SpectralCube` objects require
> `spectral-cube`. Functions that use `fit_dynesty` require `dynesty`.
> Install with: `pip install galfit-uv[measure]`

### `quick_measure`

**Module:** `galfit_uv.measure`

Top-level function: detect source, extract spectrum, plot, and optionally fit.

```python
quick_measure(cube, freq_line, freq_range=None, nbeam=1, offset=None,
              vrange=400*u.km/u.s, field_radius=22, fit=False,
              fit_model='gaussian', nlive=500, dlogz=0.1,
              progress=True, use_continuum=True, title=None,
              map_vrange=[-1, 5], flux_unit='mJy', detect=False,
              mask_method='circular', nsigma=2, return_spectrum_data=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cube` | SpectralCube | required | Input spectral cube |
| `freq_line` | Quantity (GHz) | required | Rest frequency of the line |
| `freq_range` | array-like | `None` | Frequency range `[min, max]` in GHz |
| `nbeam` | int | `1` | Number of beams for circular mask |
| `offset` | tuple | `None` | `(position_angle_deg, separation_arcsec)` |
| `vrange` | Quantity | `400 km/s` | Velocity range for non-detection upper limit |
| `field_radius` | float | `22` | Field mask radius in arcsec |
| `fit` | bool | `False` | Whether to fit the spectrum |
| `fit_model` | str | `'gaussian'` | Model: `'gaussian'`, `'double_peak'`, `'double_peak_asym'` |
| `nlive` | int | `500` | Live points for nested sampling |
| `dlogz` | float | `0.1` | Stopping criterion for log evidence |
| `detect` | bool | `False` | `True` = detection plot, `False` = non-detection |
| `mask_method` | str | `'circular'` | `'circular'` or `'snr'` |
| `nsigma` | float | `2` | SNR threshold for SNR masking |
| `return_spectrum_data` | bool | `True` | Return full spectrum data dict |

**Returns:** `dict` with keys `'measurements'`, `'spectrum_data'`, `'detection_plots'` (if `return_spectrum_data=True`), or the measurement result directly.

---

### `detect_source`

**Module:** `galfit_uv.measure`

Detect the source in the cube, create moment map, extract spectrum.

```python
detect_source(cube, axs=None, nbeam=1, field_radius=20, offset=None,
              title=None, freq_line=None, freq_range=None,
              map_vrange=[-1, 5], contour_levels=[-1, 1, 3, 5],
              flux_unit='mJy', mask_method='circular', nsigma=2,
              return_spectrum_data=True)
```

**Returns:** `dict` with `'axs'` and `'spectrum_data'` keys (if `return_spectrum_data=True`), or the axes list.

---

### `plot_detection`

**Module:** `galfit_uv.measure`

Plot detection spectrum with optional model fitting.

```python
plot_detection(spectrum_data, fit=False, fit_model='gaussian', nlive=500,
               dlogz=0.1, progress=True, ax=None, flux_unit='mJy',
               use_continuum=False)
```

**Returns:** `dict` with keys `flux_sum`, `flux_sum_err`, `width`, `nu_mean`, `flux_fit`, `w50`, `vc`, `flux_fit_err`, `w50_err`, `vc_err`, `ax`.

---

### `plot_nondetection`

**Module:** `galfit_uv.measure`

Plot non-detection spectrum and compute upper limits.

```python
plot_nondetection(spectrum_data, vrange=400*u.km/u.s, ax=None,
                  flux_unit='mJy', nsigma=3)
```

**Returns:** `dict` with keys `rms`, `flux_up`, `ax`.

---

### `source_mask`

**Module:** `galfit_uv.measure`

Generate a circular aperture source mask.

```python
source_mask(cube, nbeam, offset=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `cube` | SpectralCube | Input spectral cube |
| `nbeam` | int | Aperture radius in beam units |
| `offset` | tuple | `(position_angle_deg, separation_arcsec)` |

**Returns:** `2D boolean ndarray`

---

### `source_mask_snr`

**Module:** `galfit_uv.measure`

Generate a source mask based on SNR threshold with connected segment filtering.

```python
source_mask_snr(cube, nbeam, nsigma=2, freq_range=None, offset=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cube` | SpectralCube | required | Input spectral cube |
| `nbeam` | int | required | Circular aperture radius for segment filtering |
| `nsigma` | float | `2` | SNR threshold |
| `freq_range` | array-like | `None` | Frequency range `[min, max]` in GHz |
| `offset` | tuple | `None` | `(position_angle_deg, separation_arcsec)` |

**Returns:** `2D boolean ndarray`

---

### `field_mask`

**Module:** `galfit_uv.measure`

Generate a circular field mask to exclude noisy edges.

```python
field_mask(cube, radius=20)
```

**Returns:** `2D boolean ndarray`

---

### `extract_spectrum`

**Module:** `galfit_uv.measure`

Extract a 1D spectrum from a cube using a source mask.

```python
extract_spectrum(cube, mask_src, perbeam=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cube` | SpectralCube | required | Input spectral cube |
| `mask_src` | 2D bool array | required | Source mask |
| `perbeam` | bool | `True` | `True` = per-beam flux, `False` = total flux |

**Returns:** `(spc_x Quantity, spc_f Quantity)` — spectral axis and flux.

---

### `fit_dynesty`

**Module:** `galfit_uv.measure`

Fit spectral line profiles using dynesty nested sampling.

```python
fit_dynesty(x, y, yerr, model_type='gaussian', prior_bounds=None,
            nlive=500, dlogz=0.1, sample='rwalk', plot=True, ax=None,
            progress=True, rstate=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | array-like | required | Velocity or frequency axis |
| `y` | array-like | required | Flux values |
| `yerr` | float or array-like | required | Flux uncertainties (RMS) |
| `model_type` | str | `'gaussian'` | `'gaussian'`, `'double_peak'`, or `'double_peak_asym'` |
| `prior_bounds` | dict | `None` | `{param_name: (lo, hi), ...}` overrides |
| `nlive` | int | `500` | Live points |
| `dlogz` | float | `0.1` | Stopping criterion |
| `plot` | bool | `True` | Whether to plot results |
| `progress` | bool | `True` | Show progress bar |
| `rstate` | Generator | `None` | Random state for reproducibility |

**Returns:** `dict`

| Key | Type | Description |
|-----|------|-------------|
| `params` | dict | Median parameter values |
| `params_err` | dict | Parameter uncertainties |
| `samples` | ndarray | Posterior samples |
| `logz` | float | Log evidence |
| `logzerr` | float | Evidence error |
| `model_type` | str | Model type used |
| `derived` | dict | Derived parameters: `flux_int`, `flux_int_err`, `w50`, `w50_err` |

---

### `calculate_w50`

**Module:** `galfit_uv.measure`

Calculate W50 (width at 50% of peak) from a line profile with linear interpolation.

```python
calculate_w50(x, y)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | array-like | Velocity or frequency axis |
| `y` | array-like | Flux or profile values |

**Returns:** `float` — width at 50% of peak (same units as x). Returns `NaN` if no point reaches half-maximum.

---

### `plot_1d_spectrum`

**Module:** `galfit_uv.measure`

Plot a 1D spectrum as a step plot using `plt.stairs`.

```python
plot_1d_spectrum(x, y, ax=None, dx=None, **kwargs)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | array-like or Quantity | Spectral axis |
| `y` | array-like or Quantity | Flux values |
| `ax` | Axes | `None` | Axes to plot on |
| `dx` | float | `None` | Bin width (auto-derived if None) |

**Returns:** `matplotlib.patches.StepPatch`

**Raises:** `ValueError` if `dx` is NaN.

---

### `Plot_Map`

**Module:** `galfit_uv.measure`

Plot a moment map with optional contours and beam ellipse.

```python
Plot_Map(mom, cmap="viridis", norm=None, ax=None,
         imshow_interpolation="none", contour_dict={},
         vpercentile=98, beam_on=True, beam_kws={},
         plain=False, xlim=None, ylim=None)
```

**Returns:** `(ax, im)` — axes and image object.

---

### `Plot_Beam`

**Module:** `galfit_uv.measure`

Draw a beam ellipse on an axes.

```python
Plot_Beam(ax, bmaj, bmin, bpa, **ellipse_kws)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `ax` | Axes | Target axes |
| `bmaj` | float | Beam major axis (arcsec) |
| `bmin` | float | Beam minor axis (arcsec) |
| `bpa` | float | Beam position angle (deg) |

---

### `plot_circular_aperture`

**Module:** `galfit_uv.measure`

Draw a circular aperture on a moment map.

```python
plot_circular_aperture(m0, nbeam, ax, offset=None, **kwargs)
```

**Returns:** `plt.Circle` artist.

---

### `plot_mask_contour`

**Module:** `galfit_uv.measure`

Draw a contour showing the mask boundary on a moment map.

```python
plot_mask_contour(m0, mask, ax, **kwargs)
```

**Returns:** `QuadContourSet`

---

### `compare_source_masks`

**Module:** `galfit_uv.measure`

Compare circular and SNR-based source masks side by side.

```python
compare_source_masks(cube, nbeam=2, nsigma=2, freq_range=None, offset=None)
```

**Returns:** `(fig, masks)` — figure and dict with `'circular'` and `'snr'` mask arrays.
