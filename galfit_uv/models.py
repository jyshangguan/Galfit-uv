"""
Parametric surface-brightness profiles and Hankel-transform visibility models.

Profiles available:
  - Point source
  - Gaussian
  - Sersic (generalized, with n=1 exponential special case)

Each profile is defined in surface-brightness space (Jy/arcsec^2) and
converted to visibility space via a numerical Hankel (J1) transform over
log-spaced radial bins.
"""

import numpy as np
import scipy.special as sc


# ---- Radial bin setup (shared by all extended models) ----

NBINS = 150
_RIN = 0.1 / 140.0  # innermost bin inner edge in arcsec (~0.7 mas)


def _make_bins(nbins=NBINS, rmin=_RIN):
    """Return (bin_centers, bin_edges, r_edges) in arcsec."""
    b = np.logspace(np.log10(rmin), 2.0, num=nbins)  # outer edges
    a = np.roll(b, 1)
    a[0] = rmin
    cb = 0.5 * (a + b)
    return cb, b


# ---- Geometry helpers ----

def _apply_geometry(u, v, incl_deg, pa_deg, dx_arcsec, dy_arcsec):
    """Project u,v coordinates and compute phase shift.

    Returns
    -------
    rho : ndarray
        De-projected uv-distance in wavelengths, converted to cycles/arcsec.
    shift : complex ndarray
        Phase shift for center offset.
    """
    incl = np.radians(incl_deg)
    PA = 0.5 * np.pi - np.radians(pa_deg)
    dx = dx_arcsec * np.pi / (180.0 * 3600.0)
    dy = dy_arcsec * np.pi / (180.0 * 3600.0)

    uprime = u * np.cos(PA) + v * np.sin(PA)
    vprime = (-u * np.sin(PA) + v * np.cos(PA)) * np.cos(incl)
    rho = np.sqrt(uprime**2 + vprime**2) * np.pi / (180.0 * 3600.0)

    shift = np.exp(-2.0 * np.pi * 1.0j * ((u * -dx) + (v * -dy)))
    return rho, shift


def hankel_transform(intensity_profile, bin_centers, bin_edges, rho):
    """Compute visibility via numerical Hankel (J1) transform.

    Parameters
    ----------
    intensity_profile : ndarray
        Surface brightness at bin centers, in Jy/arcsec^2.
    bin_centers : ndarray
        Radial bin centers in arcsec.
    bin_edges : ndarray
        Radial bin outer edges in arcsec.
    rho : ndarray
        uv-distance in cycles/arcsec.

    Returns
    -------
    vis : complex ndarray
        Visibility in Jy.
    """
    rmin = _RIN
    rbin = np.concatenate([[rmin], bin_edges])
    Ibin = np.concatenate([[0.0], intensity_profile, [0.0]])
    II = Ibin - np.roll(Ibin, -1)
    intensity = np.delete(II, len(bin_edges) + 1)

    jarg = np.outer(2.0 * np.pi * rbin, rho)
    # Handle jarg=0 to avoid nan
    with np.errstate(divide='ignore', invalid='ignore'):
        jinc = sc.j1(jarg) / jarg
    jinc = np.where(np.isfinite(jinc), jinc, 0.5)

    vis = np.dot(2.0 * np.pi * rbin**2 * intensity, jinc)
    return vis


def _normalize_SB(SB, flux_mJy, cb):
    """Normalize surface brightness profile to a total flux.

    Parameters
    ----------
    SB : ndarray
        Un-normalized surface brightness at bin centers.
    flux_mJy : float
        Total integrated flux in mJy.
    cb : ndarray
        Bin centers in arcsec.

    Returns
    -------
    Inu : ndarray
        Surface brightness in Jy/arcsec^2.
    """
    area = np.trapezoid(2.0 * np.pi * SB * cb, cb)
    if area == 0.0:
        return np.zeros(len(cb))
    return flux_mJy * SB / area * 1e-3  # mJy -> Jy


# ---- Profile functions (return un-normalized SB) ----

def _sb_point(cb):
    """Point source surface brightness (delta function)."""
    return np.ones_like(cb)


def _sb_gaussian(cb, sigma):
    """Gaussian surface brightness."""
    return np.exp(-0.5 * (cb / sigma)**2)


def _sb_sersic(cb, Re, n):
    """Sersic surface brightness profile.

    Uses the approximation for b_n from MacArthur et al. (2003).
    """
    if n <= 0:
        raise ValueError(f"Sersic index n must be > 0, got {n}")
    bn = 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n**2)
    x = cb / Re
    with np.errstate(divide='ignore', invalid='ignore'):
        SB = np.exp(-bn * (x**(1.0 / n) - 1.0))
    SB = np.where(np.isfinite(SB), SB, 0.0)
    # Ensure SB is zero at cb=0 if n < 1 (cusp profile)
    return SB


# ---- Visibility model functions ----

def vis_point(theta, uv, has_geometry=False):
    """Point source visibility model.

    Parameters
    ----------
    theta : array-like
        If has_geometry: [flux_mJy, incl, PA, dx, dy]
        If not:          [flux_mJy] (geometry handled externally)
    uv : tuple (u, v)
        uv coordinates in wavelengths.
    has_geometry : bool
        Whether theta includes geometry parameters.

    Returns
    -------
    complex ndarray : visibility in Jy
    """
    u, v = uv
    flux = theta[0] * 1e-3  # mJy -> Jy

    if has_geometry:
        incl_deg, pa_deg, dx, dy = theta[1], theta[2], theta[3], theta[4]
        rho, shift = _apply_geometry(u, v, incl_deg, pa_deg, dx, dy)
    else:
        shift = 1.0

    return flux * shift


def vis_gaussian(theta, uv, has_geometry=False):
    """Gaussian visibility model.

    Parameters
    ----------
    theta : array-like
        If has_geometry: [flux_mJy, sigma, incl, PA, dx, dy]
        If not:          [flux_mJy, sigma]
    uv : tuple (u, v)
        uv coordinates in wavelengths.
    has_geometry : bool
        Whether theta includes geometry parameters.

    Returns
    -------
    complex ndarray : visibility in Jy
    """
    u, v = uv
    flux_mJy = theta[0]
    sigma = theta[1]

    cb, b = _make_bins()
    SB = _sb_gaussian(cb, sigma)
    Inu = _normalize_SB(SB, flux_mJy, cb)

    if has_geometry:
        incl_deg, pa_deg, dx, dy = theta[2], theta[3], theta[4], theta[5]
        rho, shift = _apply_geometry(u, v, incl_deg, pa_deg, dx, dy)
    else:
        rho, shift = None, None

    if rho is None:
        # No geometry: need rho from uv directly
        rho = np.sqrt(u**2 + v**2) * np.pi / (180.0 * 3600.0)
        shift = 1.0

    vis = hankel_transform(Inu, cb, b, rho)
    return vis * shift


def vis_sersic(theta, uv, has_geometry=False):
    """Sersic profile visibility model.

    Parameters
    ----------
    theta : array-like
        If has_geometry: [flux_mJy, Re, n, incl, PA, dx, dy]
        If not:          [flux_mJy, Re, n]
    uv : tuple (u, v)
        uv coordinates in wavelengths.
    has_geometry : bool
        Whether theta includes geometry parameters.

    Returns
    -------
    complex ndarray : visibility in Jy
    """
    u, v = uv
    flux_mJy = theta[0]
    Re = theta[1]
    n = theta[2]

    cb, b = _make_bins()
    SB = _sb_sersic(cb, Re, n)
    Inu = _normalize_SB(SB, flux_mJy, cb)

    if has_geometry:
        incl_deg, pa_deg, dx, dy = theta[3], theta[4], theta[5], theta[6]
        rho, shift = _apply_geometry(u, v, incl_deg, pa_deg, dx, dy)
    else:
        rho, shift = None, None

    if rho is None:
        rho = np.sqrt(u**2 + v**2) * np.pi / (180.0 * 3600.0)
        shift = 1.0

    vis = hankel_transform(Inu, cb, b, rho)
    return vis * shift


# ---- Model factory ----

_PROFILE_REGISTRY = {
    'point': vis_point,
    'gaussian': vis_gaussian,
    'sersic': vis_sersic,
}

# Number of non-geometry parameters per profile
_PROFILE_NPARS = {
    'point': 1,      # flux
    'gaussian': 2,   # flux, sigma
    'sersic': 3,     # flux, Re, n
}

# Geometry parameter names in canonical order
_GEO_PARAMS = ['incl', 'PA', 'dx', 'dy']

# Intrinsic parameter names per profile (in canonical order)
_PROFILE_PARAMS = {
    'point': ['flux'],
    'gaussian': ['flux', 'sigma'],
    'sersic': ['flux', 'Re', 'n'],
}

# Default priors: (lo, hi, scale)
# scale='log' means Jeffreys prior and log-space walker init
_DEFAULT_PRIORS = {
    'flux':  (0.1, 100.0, 'log'),      # mJy, Jeffreys prior
    'Re':    (0.0, 5.0,   'linear'),    # arcsec
    'sigma': (0.0, 5.0,   'linear'),    # arcsec
    'n':     (0.3, 8.0,   'linear'),    # Sersic index
    'incl':  (0.0, 90.0,  'linear'),    # deg
    'PA':    (-90.0, 90.0, 'linear'),   # deg
    'dx':    (-5.0, 5.0,  'linear'),    # arcsec
    'dy':    (-5.0, 5.0,  'linear'),    # arcsec
}

# Profile-specific param names (cannot be bare keys in multi-component)
_INTRINSIC_PARAMS = {'flux', 'Re', 'sigma', 'n'}
# Shared geometry param names (allowed as bare keys when tied)
_SHARED_PARAMS = {'incl', 'PA', 'dx', 'dy'}


# ---- Label / prior helpers ----

def _bare_param_name(label):
    """Strip profile prefix from a label.

    Parameters
    ----------
    label : str
        Parameter label, e.g. ``'sersic:flux'`` or ``'incl'``.

    Returns
    -------
    bare : str
        Bare parameter name, e.g. ``'flux'`` or ``'incl'``.
    profile : str or None
        Profile name if prefixed, e.g. ``'sersic'`` or ``None``.
    """
    if ':' in label:
        profile, bare = label.split(':', 1)
        return bare, profile
    return label, None


def _match_prior_key(key, labels):
    """Resolve a prior/fixed key to matching label indices.

    Parameters
    ----------
    key : str
        ``'profile:param'`` for exact match, or bare ``'param'``.
    labels : list of str
        All parameter labels.

    Returns
    -------
    list of int
        Matching label indices.

    Raises
    ------
    ValueError
        If no labels match or bare key is ambiguous / invalid.
    """
    # Exact match
    if key in labels:
        return [labels.index(key)]

    # Parse key
    bare, profile_from_key = _bare_param_name(key)
    matches = []

    if profile_from_key is not None:
        # "profile:param" — exact match already checked above
        raise ValueError(
            f"Key '{key}' does not match any label. "
            f"Available: {labels}")
    else:
        # Bare key — find all labels with this bare name
        for i, lbl in enumerate(labels):
            b, p = _bare_param_name(lbl)
            if b == bare:
                matches.append(i)

        if len(matches) == 0:
            raise ValueError(
                f"Key '{key}' does not match any label. "
                f"Available: {labels}")

        if len(matches) > 1:
            # Bare key matched multiple labels
            if bare in _INTRINSIC_PARAMS:
                raise ValueError(
                    f"Bare key '{bare}' is ambiguous in multi-component model "
                    f"(matched {len(matches)} labels). "
                    f"Use 'profile:{bare}' format, e.g. "
                    + ", ".join(f"'{labels[i]}'" for i in matches))
            else:
                raise ValueError(
                    f"Bare key '{bare}' matched {len(matches)} labels: "
                    + ", ".join(f"'{labels[i]}'" for i in matches)
                    + ". Use fully qualified 'profile:param' keys.")

        return matches


def _resolve_fixed_key(key, labels):
    """Resolve a fixed-parameter key to exactly one label index.

    Same matching as ``_match_prior_key`` but requires exactly one match.

    Returns
    -------
    int
        Single matching label index.
    """
    indices = _match_prior_key(key, labels)
    if len(indices) != 1:
        raise ValueError(
            f"Fixed key '{key}' resolved to {len(indices)} labels, expected 1. "
            f"Matches: {[labels[i] for i in indices]}")
    return indices[0]


def _parse_prior_value(value, default_scale):
    """Parse a prior override value.

    Parameters
    ----------
    value : tuple
        ``(lo, hi)`` or ``(lo, hi, scale)``.
    default_scale : str
        Default scale if not specified.

    Returns
    -------
    tuple
        ``(lo, hi, scale)``.
    """
    if len(value) == 2:
        return (value[0], value[1], default_scale)
    elif len(value) == 3:
        return (value[0], value[1], value[2])
    else:
        raise ValueError(f"Prior value must be (lo, hi) or (lo, hi, scale), got {value}")


# ---- Model factory ----

def make_model_fn(profiles, tie_center=True, tie_incl=True, tie_pa=True,
                  priors=None, fixed=None):
    """Build a combined visibility model function.

    Parameters
    ----------
    profiles : list of str
        Profile names, e.g. ``['sersic', 'point']``.
        One or two profiles.
    tie_center : bool
        Tie dx, dy between components (default True).
    tie_incl : bool
        Tie inclination between components (default True).
    tie_pa : bool
        Tie position angle between components (default True).
    priors : dict, optional
        Prior overrides.  Keys in ``'profile:param'`` format, values are
        ``(lo, hi)`` or ``(lo, hi, scale)`` tuples.
    fixed : dict, optional
        Fixed parameter values.  Keys in ``'profile:param'`` or bare label
        format, values are floats.

    Returns
    -------
    model_fn : callable
        ``model_fn(theta, uv) -> complex ndarray``
    param_info : dict
        Full parameter configuration including labels, priors, fixed info.
    """
    if len(profiles) > 2:
        raise ValueError("Only 1 or 2 profile components are supported")
    if not all(p in _PROFILE_REGISTRY for p in profiles):
        raise ValueError(f"Unknown profile(s): "
                         f"{[p for p in profiles if p not in _PROFILE_REGISTRY]}")

    priors = priors or {}
    fixed = fixed or {}

    # ---- Build labels ----
    labels = []
    # Track per-component parameter index ranges for the model_fn closure
    param_slices = {}  # label -> (start, end) in full theta

    if len(profiles) == 1:
        profile = profiles[0]
        intrinsic = list(_PROFILE_PARAMS[profile])
        geo = list(_GEO_PARAMS)  # all 4 geo params for single component
        labels = intrinsic + geo
        idx = 0
        for lbl in intrinsic:
            param_slices[lbl] = (idx, idx + 1)
            idx += 1
        for lbl in geo:
            param_slices[lbl] = (idx, idx + 1)
            idx += 1
    else:
        # Two profiles — build labels with `:` prefix for unshared params
        p1, p2 = profiles
        int1 = list(_PROFILE_PARAMS[p1])
        int2 = list(_PROFILE_PARAMS[p2])

        # Intrinsic params always per-component (prefixed)
        for p, ipars in [(p1, int1), (p2, int2)]:
            for par in ipars:
                labels.append(f'{p}:{par}')

        # Geometry params: tied (bare) or untied (prefixed)
        geo_labels = []
        if tie_incl:
            geo_labels.append('incl')
        else:
            geo_labels.extend([f'{p1}:incl', f'{p2}:incl'])
        if tie_pa:
            geo_labels.append('PA')
        else:
            geo_labels.extend([f'{p1}:PA', f'{p2}:PA'])
        if tie_center:
            geo_labels.extend(['dx', 'dy'])
        else:
            geo_labels.extend([f'{p1}:dx', f'{p2}:dx', f'{p1}:dy', f'{p2}:dy'])

        labels.extend(geo_labels)

        # Build param_slices: each component's intrinsic params, then geo
        idx = 0
        for p, ipars in [(p1, int1), (p2, int2)]:
            for par in ipars:
                lbl = f'{p}:{par}'
                param_slices[lbl] = (idx, idx + 1)
                idx += 1
        for lbl in geo_labels:
            param_slices[lbl] = (idx, idx + 1)
            idx += 1

    n_params = len(labels)

    # ---- Build default prior list ----
    p_ranges_info = []
    for lbl in labels:
        bare, prof = _bare_param_name(lbl)
        if bare not in _DEFAULT_PRIORS:
            raise ValueError(f"No default prior for parameter '{bare}'")
        p_ranges_info.append(_DEFAULT_PRIORS[bare])

    # ---- Apply user prior overrides ----
    for key, value in priors.items():
        indices = _match_prior_key(key, labels)
        bare, _ = _bare_param_name(key)
        if bare not in _DEFAULT_PRIORS:
            raise ValueError(f"No default prior for parameter '{bare}'")
        default_scale = _DEFAULT_PRIORS[bare][2]
        parsed = _parse_prior_value(value, default_scale)
        for idx in indices:
            p_ranges_info[idx] = parsed

    # ---- Validate log-scale priors have lo > 0 ----
    for i, (lo, hi, scale) in enumerate(p_ranges_info):
        if scale == 'log' and lo <= 0:
            raise ValueError(
                f"Log-scale prior for '{labels[i]}' requires lo > 0, got lo={lo}")

    # ---- Process fixed parameters ----
    fixed = dict(fixed)  # copy
    fixed_indices = []
    fixed_values = []
    fixed_dict = {}  # label -> value for param_info

    for key, val in fixed.items():
        idx = _resolve_fixed_key(key, labels)
        fixed_indices.append(idx)
        fixed_values.append(float(val))
        fixed_dict[labels[idx]] = float(val)

    fixed_set = set(fixed_indices)
    free_indices = [i for i in range(n_params) if i not in fixed_set]
    free_labels = [labels[i] for i in free_indices]
    n_free = len(free_indices)
    n_fixed = len(fixed_indices)

    # Slice prior info to free parameters only
    free_p_ranges_info = [p_ranges_info[i] for i in free_indices]

    # ---- Build p_ranges (lo, hi only) and p_scales ----
    p_ranges = [(lo, hi) for lo, hi, scale in p_ranges_info]
    p_scales = [scale for lo, hi, scale in p_ranges_info]
    free_p_ranges = [(lo, hi) for lo, hi, scale in free_p_ranges_info]
    free_p_scales = [scale for lo, hi, scale in free_p_ranges_info]

    # ---- Build model_fn closure ----
    if len(profiles) == 1:
        profile = profiles[0]
        vis_fn = _PROFILE_REGISTRY[profile]
        n_prof = _PROFILE_NPARS[profile]

        # For single component, geometry is always the last 4 params
        def model_fn(theta, uv):
            u, v = uv
            incl = theta[param_slices['incl'][0]]
            pa = theta[param_slices['PA'][0]]
            dx = theta[param_slices['dx'][0]]
            dy = theta[param_slices['dy'][0]]
            rho, shift = _apply_geometry(u, v, incl, pa, dx, dy)
            prof_theta = theta[:n_prof]
            if profile == 'point':
                return vis_point(prof_theta, (u, v), has_geometry=False) * shift
            else:
                return _call_extended(vis_fn, prof_theta, rho, shift)

    else:
        p1, p2 = profiles
        fn1 = _PROFILE_REGISTRY[p1]
        fn2 = _PROFILE_REGISTRY[p2]
        n1 = _PROFILE_NPARS[p1]
        n2 = _PROFILE_NPARS[p2]

        # Resolve geometry parameter indices
        # For tied params, use the bare label; for untied, use prefixed
        def _get_geo_indices(param_name, tie_flag):
            if tie_flag:
                # Bare label
                return param_slices[param_name][0]
            else:
                # Return dict of {profile: index}
                return {p: param_slices[f'{p}:{param_name}'][0] for p in profiles}

        incl_idx = _get_geo_indices('incl', tie_incl)
        pa_idx = _get_geo_indices('PA', tie_pa)
        dx_idx = _get_geo_indices('dx', tie_center)
        dy_idx = _get_geo_indices('dy', tie_center)

        # Intrinsic offsets: first component starts at 0, second at n1
        offsets = [0, n1]

        def model_fn(theta, uv):
            u, v = uv

            # Get geometry — each component may have its own or shared
            for comp_i, (fn, n_prof, prof_name) in enumerate(
                    [(fn1, n1, p1), (fn2, n2, p2)]):
                if comp_i == 0:
                    comp_incl = theta[incl_idx] if isinstance(incl_idx, int) else theta[incl_idx[p1]]
                    comp_pa = theta[pa_idx] if isinstance(pa_idx, int) else theta[pa_idx[p1]]
                    comp_dx = theta[dx_idx] if isinstance(dx_idx, int) else theta[dx_idx[p1]]
                    comp_dy = theta[dy_idx] if isinstance(dy_idx, int) else theta[dy_idx[p1]]
                else:
                    comp_incl = theta[incl_idx] if isinstance(incl_idx, int) else theta[incl_idx[p2]]
                    comp_pa = theta[pa_idx] if isinstance(pa_idx, int) else theta[pa_idx[p2]]
                    comp_dx = theta[dx_idx] if isinstance(dx_idx, int) else theta[dx_idx[p2]]
                    comp_dy = theta[dy_idx] if isinstance(dy_idx, int) else theta[dy_idx[p2]]

                rho, shift = _apply_geometry(u, v, comp_incl, comp_pa, comp_dx, comp_dy)

                start = offsets[comp_i]
                prof_theta = theta[start:start + n_prof]

                if comp_i == 0:
                    vis_out = fn(prof_theta, (u, v), has_geometry=False) * shift if prof_name == 'point' \
                        else _call_extended(fn, prof_theta, rho, shift)
                else:
                    vis_out2 = fn(prof_theta, (u, v), has_geometry=False) * shift if prof_name == 'point' \
                        else _call_extended(fn, prof_theta, rho, shift)

            return vis_out + vis_out2

    param_info = {
        'n_params': n_params,
        'labels': labels,
        'profiles': profiles,
        'tie_center': tie_center,
        'tie_incl': tie_incl,
        'tie_pa': tie_pa,
        'p_ranges': p_ranges,
        'p_scales': p_scales,
        'p_ranges_info': p_ranges_info,
        'fixed': fixed_dict if n_fixed > 0 else None,
        'fixed_indices': fixed_indices,
        'fixed_values': fixed_values,
        'free_labels': free_labels,
        'free_indices': free_indices,
        'n_free': n_free,
        'free_p_ranges': free_p_ranges,
        'free_p_scales': free_p_scales,
    }
    return model_fn, param_info


def _call_extended(vis_fn, prof_theta, rho, shift):
    """Call an extended-source vis function with explicit rho and shift."""
    if vis_fn is vis_gaussian:
        flux_mJy, sigma = prof_theta[0], prof_theta[1]
        cb, b = _make_bins()
        SB = _sb_gaussian(cb, sigma)
        Inu = _normalize_SB(SB, flux_mJy, cb)
        return hankel_transform(Inu, cb, b, rho) * shift
    elif vis_fn is vis_sersic:
        flux_mJy, Re, n = prof_theta[0], prof_theta[1], prof_theta[2]
        cb, b = _make_bins()
        SB = _sb_sersic(cb, Re, n)
        Inu = _normalize_SB(SB, flux_mJy, cb)
        return hankel_transform(Inu, cb, b, rho) * shift
    else:
        raise ValueError(f"Unknown extended profile: {vis_fn}")
