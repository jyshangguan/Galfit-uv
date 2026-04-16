# Cube Measurement API Reference

Function signatures for the cube measurement pipeline: line profiles, source detection,
spectral extraction, and line fitting.

> **Note:** Functions in this module that accept `SpectralCube` objects require
> `spectral-cube`. Functions that use `fit_dynesty` require `dynesty`.
> Install with: `pip install galfit-uv[measure]`

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
