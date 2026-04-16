"""
galfit_uv - ALMA visibility data fitting package.

Extracts visibilities from measurement sets, fits parametric models
using MCMC (emcee) with Hankel transforms, and provides visualization tools.
Also provides spectral cube measurement and line fitting tools.
"""

import os

# Limit numpy internal threading before any numpy import, to avoid
# contention with multiprocessing workers in the MCMC sampler.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

from galfit_uv.export import export_vis, save_uvtable, Visibility
from galfit_uv.models import (
    vis_point, vis_gaussian, vis_sersic,
    hankel_transform, make_model_fn,
)
from galfit_uv.fit import fit_mcmc, MCMCResult, compute_fit_stats
from galfit_uv.plot import plot_uv, import_model_to_ms, clean_image, plot_clean_images

# Optional: line profile functions (only need numpy)
try:
    from galfit_uv.lineprofiles import (
        Gaussian, Gaussian_DoublePeak, Gaussian_DoublePeak_Asymmetric,
    )
    _HAS_LINEPROFILES = True
except ImportError:
    _HAS_LINEPROFILES = False

# Optional: cube measurement pipeline (needs spectral-cube, dynesty)
try:
    from galfit_uv.measure import (
        fit_dynesty, calculate_w50,
        source_mask, source_mask_snr, field_mask,
        extract_spectrum, detect_source, quick_measure,
        plot_detection, plot_nondetection,
        Plot_Map, Plot_Beam, plot_1d_spectrum,
        plot_circular_aperture, plot_mask_contour, compare_source_masks,
    )
    _HAS_MEASURE = True
except ImportError:
    _HAS_MEASURE = False

__version__ = '0.1.0'

__all__ = [
    # export
    "Visibility", "export_vis", "save_uvtable",
    # models
    "vis_point", "vis_gaussian", "vis_sersic",
    "hankel_transform", "make_model_fn",
    # fit
    "fit_mcmc", "MCMCResult", "compute_fit_stats",
    # plot
    "plot_uv", "import_model_to_ms",
    "clean_image", "plot_clean_images",
]

if _HAS_LINEPROFILES:
    __all__ += [
        "Gaussian", "Gaussian_DoublePeak", "Gaussian_DoublePeak_Asymmetric",
    ]

if _HAS_MEASURE:
    __all__ += [
        "fit_dynesty", "calculate_w50",
        "source_mask", "source_mask_snr", "field_mask",
        "extract_spectrum", "detect_source", "quick_measure",
        "plot_detection", "plot_nondetection",
        "Plot_Map", "Plot_Beam", "plot_1d_spectrum",
        "plot_circular_aperture", "plot_mask_contour", "compare_source_masks",
    ]
