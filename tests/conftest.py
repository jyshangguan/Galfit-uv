"""Shared fixtures for galfit_uv tests."""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')


@pytest.fixture
def simple_uv():
    """100 random u,v points in wavelengths (kilo-lambda scale)."""
    rng = np.random.default_rng(42)
    u = rng.uniform(10, 1000, 100) * 1e3  # 10-1000 kilo-lambda
    v = rng.uniform(10, 1000, 100) * 1e3
    return u, v


@pytest.fixture
def simple_dvis(simple_uv):
    """Point source Visibility with 1 mJy flux, 100 random uv points."""
    u, v = simple_uv
    flux = 1e-3  # 1 mJy in Jy
    vis = np.full(len(u), flux, dtype=np.complex128)
    wgt = np.ones(len(u))
    from galfit_uv.export import Visibility
    return Visibility(u, v, vis, wgt)


@pytest.fixture
def gaussian_dvis(simple_uv):
    """Gaussian source Visibility, sigma=0.5 arcsec, 1 mJy flux."""
    u, v = simple_uv
    from galfit_uv.models import vis_gaussian
    theta = [1.0, 0.5]  # 1 mJy, 0.5 arcsec sigma
    mvis = vis_gaussian(theta, (u, v), has_geometry=False)
    wgt = np.ones(len(u))
    from galfit_uv.export import Visibility
    return Visibility(u, v, mvis, wgt)


@pytest.fixture
def sersic_dvis(simple_uv):
    """Sersic source Visibility, Re=1 arcsec, n=2, 1 mJy flux."""
    u, v = simple_uv
    from galfit_uv.models import vis_sersic
    theta = [1.0, 1.0, 2.0]  # 1 mJy, 1 arcsec Re, n=2
    mvis = vis_sersic(theta, (u, v), has_geometry=False)
    wgt = np.ones(len(u))
    from galfit_uv.export import Visibility
    return Visibility(u, v, mvis, wgt)
