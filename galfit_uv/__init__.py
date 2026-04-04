"""
galfit_uv - ALMA visibility data fitting package.

Extracts visibilities from measurement sets, fits parametric models
using MCMC (emcee) with Hankel transforms, and provides visualization tools.
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
