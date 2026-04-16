# Examples

This page walks through fitting real ALMA data with galfit-uv using two models:
a single free-n Sersic and a Sersic(n=1) + point source decomposition.

---

## Cube Measurement Pipeline

The `galfit_uv.measure` module measures emission line properties from ALMA
spectral cubes: source masking, spectrum extraction, line detection, and nested
sampling fitting with dynesty.

> **Install optional dependencies:** `pip install galfit-uv[measure]`

### Data

A primary-beam-corrected CO(3-2) line cube of **GQC J0054-4955** (z = 2.2512):
56 channels x 100 x 100 pixels, channel width ~88 km/s, beam 3.46" x 2.86".

### Source Masking

Two methods: **circular aperture** (simple, for non-detections) and **SNR
threshold** (adapts to emission morphology, for detections).

```python
from galfit_uv.measure import compare_source_masks

fig, masks = compare_source_masks(
    cube, nbeam=1, nsigma=2,
    freq_range=[106.27, 106.6],  # GHz, line emission channels
)
```

![Mask comparison](figs/measure_mask_comparison.png)

### Detection and Line Profile Fitting

`quick_measure` runs the full pipeline: masking, moment-0 map, spectrum
extraction, and optional line fitting.

```python
from galfit_uv.measure import quick_measure
import astropy.units as u

res = quick_measure(
    cube,
    freq_line=106.36 * u.GHz,
    freq_range=[106.27, 106.6],
    mask_method='snr',
    nsigma=2,
    detect=True,
    fit=True,
    fit_model='double_peak',  # or 'gaussian', 'double_peak_asym'
    nlive=500,
    dlogz=0.01,
)
```

Three line profile models are available:

| Model | Description |
|-------|-------------|
| `gaussian` | Simple Gaussian (3 params) |
| `double_peak` | Symmetric double-horn, Tiley+ 2016 (5 params) |
| `double_peak_asym` | Asymmetric double-horn (7 params) |

![Detection and fit](figs/measure_quick_detection.png)

### Results

| Parameter | Value |
|-----------|-------|
| Integrated flux (fit) | 1.98 ± 0.04 Jy km/s |
| W50 (line width) | 630 ± 12 km/s |
| Velocity center | -208 ± 4 km/s |
| Measured redshift | 2.24895 ± 0.00005 |

The summed and fitted fluxes are fully consistent, confirming the Double Peak
model describes the line well.

![Frequency confirmation](figs/measure_freq_confirmation.png)

### Non-detection Upper Limits

For undetected sources, `quick_measure(detect=False)` computes a 3-sigma
upper limit on the integrated line flux:

```python
res = quick_measure(
    cube, freq_line=106.36 * u.GHz,
    mask_method='circular', nbeam=1,
    detect=False, vrange=400 * u.km / u.s,
)
# 3-sigma upper limit: ~0.08 Jy km/s
```

![Non-detection](figs/measure_nondetection.png)

> Full details: see `galfit_uv/demo/report_measure.md` and `galfit_uv/demo/run_demo_measure.py`.

---

## UV Visibility Fitting

**GQC J0054-4955** observed in CO(3-2) at z ~ 2.5.

> **Demo data:** The measurement set can be downloaded from:
> <https://drive.google.com/file/d/1TPZQvP-7wc5Kk169Gh6IgiYAUoaug2bb/view?usp=sharing>

- 1980 visibility points after time-averaging
- UV range: 5 -- 111 kilo-lambda
- Wavelength: 2.817 mm
- Zero-baseline flux: ~19.2 mJy

Two models are compared:

| Model | Components | Free params |
|-------|-----------|-------------|
| **Model A** | Single Sersic (free n) | 7 |
| **Model B** | Sersic(n=1) + Point source | 7 (n fixed) |

## Step 1: Export Visibilities

Extract averaged visibility data from the measurement set.

```python
from galfit_uv.export import export_vis, save_uvtable

dvis, wle = export_vis('data/GQC_J0054-4955_avg.ms', verbose=True)

print(f"Extracted {len(dvis.u)} visibility points")
print(f"Wavelength: {wle*1e3:.3f} mm")
print(f"uv range: {dvis.uvdist.min()/1e3:.0f} -- {dvis.uvdist.max()/1e3:.0f} klambda")
print(f"Total flux (zero-baseline): {abs(dvis.vis[0])*1e3:.2f} mJy")

# Optionally save a uvplot-compatible table
save_uvtable(dvis.u, dvis.v, dvis.vis, dvis.wgt, 'uv_table.txt', wle=wle)
```

Output:

```
Extracted 1980 visibility points
Wavelength: 2.817 mm
uv range: 5 -- 111 klambda
Total flux (zero-baseline): 19.17 mJy
```

## Step 2: Build the Model

**Model A — Single Sersic:**

```python
from galfit_uv.models import make_model_fn

model_fn, param_info = make_model_fn(['sersic'])
# labels = ['flux', 'Re', 'n', 'incl', 'PA', 'dx', 'dy']
```

**Model B — Sersic(n=1) + Point:**

```python
from galfit_uv.models import make_model_fn

model_fn, param_info = make_model_fn(
    ['sersic', 'point'],
    fixed={'sersic:n': 1.0},
)
# labels = ['sersic:flux', 'sersic:Re', 'point:flux', 'incl', 'PA', 'dx', 'dy']
```

## Step 3: Run MCMC

```python
from galfit_uv.fit import fit_mcmc

result = fit_mcmc(
    dvis, model_fn, param_info,
    max_steps=5000,
    burnin=2500,
    nwalk_factor=5,
    outpath='./fit_output',
    seed=42,
)
```

Settings: 35 walkers (7 free params x 5), 5000 steps per walker, 2500 burn-in steps.

Outputs saved to `outpath/`:

- `fit_results.fits` — multi-extension FITS with data, model, samples, fit statistics
- `chains.png` — walker traces with autocorrelation time annotations
- `corner_plot.png` — posterior corner plot

---

## Model A: Single Sersic (free n)

### Results

| Parameter | Best-fit | +1sig | -1sig |
|-----------|---------|-------|-------|
| flux (mJy) | 21.28 | +0.42 | -0.42 |
| Re (arcsec) | 0.365 | +0.03 | -0.03 |
| n | 7.38 | +0.45 | -0.77 |
| incl (deg) | 46.4 | +4.9 | -5.6 |
| PA (deg) | 16.6 | +5.7 | -5.7 |
| dx (arcsec) | 0.011 | +0.009 | -0.009 |
| dy (arcsec) | -0.067 | +0.008 | -0.008 |

**Fit statistics:** chi2/DOF = 10.80, BIC = 42751

The high Sersic index (n ~ 7.4, well above de Vaucouleurs n=4) indicates a very concentrated core with extended wings, while the small effective radius (Re ~ 0.37 arcsec) is consistent with a compact emission region.

### Figures

![UV plot](figs/uvplot_sersic.png)

![Corner plot](figs/corner_plot_sersic.png)

![Walker chains](figs/chains_sersic.png)

![Clean images](figs/clean_images_sersic.png)

---

## Model B: Sersic(n=1) + Point Source

### Results

| Parameter | Best-fit | +1sig | -1sig |
|-----------|---------|-------|-------|
| sersic:flux (mJy) | 7.23 | +0.48 | -0.41 |
| sersic:Re (arcsec) | 1.86 | +0.38 | -0.36 |
| sersic:n | 1.00 | — | — (fixed) |
| point:flux (mJy) | 12.97 | +0.53 | -0.75 |
| incl (deg) | 53.5 | +5.4 | -6.9 |
| PA (deg) | 8.3 | +4.2 | -4.6 |
| dx (arcsec) | 0.012 | +0.008 | -0.009 |
| dy (arcsec) | -0.066 | +0.008 | -0.007 |

**Fit statistics:** chi2/DOF = 10.80, BIC = 42766

The model decomposes the source into an extended exponential disk (~7.2 mJy, Re ~ 1.86 arcsec) plus a dominant unresolved point source (~13 mJy), with a fitted inclination of ~53 degrees.

### Figures

![UV plot](figs/uvplot_2comp.png)

![Corner plot](figs/corner_plot_2comp.png)

![Walker chains](figs/chains_2comp.png)

![Clean images](figs/clean_images_2comp.png)

---

## Model Comparison

| Metric | Model A (Sersic) | Model B (Sersic+Point) |
|--------|-------------------|------------------------|
| Total flux (mJy) | 21.3 | 7.2 + 13.0 = 20.2 |
| Extended size | Re = 0.37 arcsec | Re = 1.86 arcsec |
| Sersic index | n = 7.4 (free) | n = 1.0 (fixed) |
| BIC | 42751 | 42766 |

Both models yield nearly identical reduced chi-squared values. The single Sersic has a slightly lower BIC, but the two-component decomposition provides a more physically interpretable separation between an extended disk and a compact nucleus. The geometry parameters (center offset) are consistent between the two models.
