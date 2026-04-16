"""
Spectral line profile functions for emission line fitting.

Provides Gaussian, Gaussian Double Peak (Tiley et al. 2016), and
asymmetric Gaussian Double Peak profile models.
"""

import numpy as np

__all__ = [
    "Gaussian",
    "Gaussian_DoublePeak",
    "Gaussian_DoublePeak_Asymmetric",
]


def Gaussian(x, a, b, c):
    """
    A Gaussian function.

    Parameters
    ----------
    x : array-like
        The independent variable.
    a : float
        Amplitude.
    b : float
        Mean.
    c : float
        Standard deviation.
    """
    return a * np.exp(-0.5 * (x - b)**2 / c**2)


def Gaussian_DoublePeak(x, ag, ac, v0, sigma, w):
    """
    Gaussian Double Peak function, Eq. (A2) of Tiley et al. (2016, MNRAS, 461, 3494).

    Parameters
    ----------
    x : 1D array
        The variable of the function.  It should be monotonically increasing.
    ag : float
        The peak flux of the two half-Gaussians.  Require ag > 0.
    ac : float
        The flux at the central velocity.  Require ac > 0.
    v0 : float
        The center of the profile.
    sigma : float
        The standard deviation of the half-Gaussian profile.  Require sigma > 0.
    w : float
        The half-width of the central parabola.  Require w > 0.

    Returns
    -------
    y : 1D array
        The Gaussian Double Peak profile.
    """
    if not ((ag > 0) and (ac > 0) and (sigma > 0) and (w > 0)):
        raise ValueError("ag, ac, sigma, and w must all be positive")

    x = np.atleast_1d(x)
    # Left
    vc_l = v0 - w
    fltr_l = x < vc_l
    x_l = x[fltr_l]
    y_l = Gaussian(x_l, ag, vc_l, sigma)
    # Right
    vc_r = v0 + w
    fltr_r = x > vc_r
    x_r = x[fltr_r]
    y_r = Gaussian(x_r, ag, vc_r, sigma)
    # Center
    a = (ag - ac) / w**2
    fltr_c = (x >= vc_l) & (x <= vc_r)
    x_c = x[fltr_c]
    y_c = ac + a * (x_c - v0)**2
    y = np.concatenate([y_l, y_c, y_r])
    return y


def Gaussian_DoublePeak_Asymmetric(x, ag_left, ag_right, ac, v0, sigma, w_left, w_right):
    """
    Asymmetric Gaussian Double Peak function.

    Based on Eq. (A2) of Tiley et al. (2016, MNRAS, 461, 3494), adjusted
    to allow for asymmetric peaks in the double-horn profile.

    Parameters
    ----------
    x : 1D array
        The variable of the function.  It should be monotonically increasing.
    ag_left : float
        The peak flux of the left half-Gaussian.  Require ag_left > 0.
    ag_right : float
        The peak flux of the right half-Gaussian.  Require ag_right > 0.
    ac : float
        The flux at the central velocity.  Require ac > 0.
    v0 : float
        The center of the profile.
    sigma : float
        The standard deviation of the half-Gaussian profile.  Require sigma > 0.
    w_left : float
        The left half-width of the central parabola.  Require w_left > 0.
    w_right : float
        The right half-width of the central parabola.  Require w_right > 0.

    Returns
    -------
    y : 1D array
        The Gaussian Double Peak profile.
    """
    if not ((ag_left > 0) and (ag_right > 0) and (ac > 0)
            and (sigma > 0) and (w_left > 0) and (w_right > 0)):
        raise ValueError("ag_left, ag_right, ac, sigma, w_left, and w_right must all be positive")

    x = np.atleast_1d(x)
    # Left
    vc_l = v0 - w_left
    fltr_l = x < vc_l
    x_l = x[fltr_l]
    y_l = Gaussian(x_l, ag_left, vc_l, sigma)
    # Right
    vc_r = v0 + w_right
    fltr_r = x > vc_r
    x_r = x[fltr_r]
    y_r = Gaussian(x_r, ag_right, vc_r, sigma)
    # Center
    a = (ag_left - ag_right) / (w_left**2 - w_right**2)
    fltr_c = (x >= vc_l) & (x <= vc_r)
    x_c = x[fltr_c]
    y_c = ac + a * (x_c - v0)**2
    y = np.concatenate([y_l, y_c, y_r])
    return y
