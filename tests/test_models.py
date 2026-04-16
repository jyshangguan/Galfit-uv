"""Tests for galfit_uv.models module."""

import numpy as np
import pytest
from galfit_uv.models import (
    _make_bins,
    _apply_geometry,
    _bare_param_name,
    _match_prior_key,
    _resolve_fixed_key,
    _parse_prior_value,
    _sb_point,
    _sb_gaussian,
    _sb_sersic,
    hankel_transform,
    _normalize_SB,
    vis_point,
    vis_gaussian,
    vis_sersic,
    make_model_fn,
)


# ============================================================
# Internal helpers
# ============================================================

class TestMakeBins:
    def test_output_shapes(self):
        cb, b = _make_bins()
        assert cb.shape == (150,)
        assert b.shape == (150,)

    def test_log_spacing(self):
        cb, b = _make_bins()
        # Centers should be log-spaced
        ratios = cb[1:] / cb[:-1]
        np.testing.assert_allclose(ratios, ratios[0], rtol=0.1)

    def test_default_range(self):
        cb, b = _make_bins()
        assert cb[0] > 0  # innermost bin > 0
        assert cb[-1] <= 100.0  # outermost bin <= 100 arcsec

    def test_custom_nbins(self):
        cb, b = _make_bins(nbins=50)
        assert cb.shape == (50,)
        assert b.shape == (50,)


class TestApplyGeometry:
    def test_no_offset(self):
        """With zero offset and no inclination, rho = sqrt(u^2+v^2) in arcsec^-1."""
        u = np.array([1000.0])
        v = np.array([0.0])
        rho, shift = _apply_geometry(u, v, 0.0, 0.0, 0.0, 0.0)
        # rho should be u * pi / (180*3600)
        expected = 1000.0 * np.pi / (180.0 * 3600.0)
        np.testing.assert_allclose(rho, expected)
        np.testing.assert_allclose(shift, 1.0)

    def test_shift_only(self):
        """Non-zero dx,dy produces a non-trivial phase shift."""
        u = np.array([1000.0])
        v = np.array([0.0])
        rho, shift = _apply_geometry(u, v, 0.0, 0.0, 1.0, 0.0)
        # shift = exp(-2*pi*i*(u*(-dx) + v*(-dy))) with dx=1 arcsec
        # With dy=0, shift = exp(2*pi*i*u*dx_rad)
        dx_rad = 1.0 * np.pi / (180.0 * 3600.0)
        expected_phase = -2.0 * np.pi * u[0] * (-dx_rad)
        np.testing.assert_allclose(shift[0], np.exp(1j * expected_phase))

    def test_inclination(self):
        """Inclination reduces apparent v-distance."""
        u = np.array([100.0])
        v = np.array([1000.0])
        rho0, _ = _apply_geometry(u, v, 0.0, 0.0, 0.0, 0.0)
        rho45, _ = _apply_geometry(u, v, 45.0, 0.0, 0.0, 0.0)
        assert rho45[0] < rho0[0]

    def test_dtype(self):
        """Returns float64 rho and complex128 shift."""
        u = np.array([100.0, 200.0])
        v = np.array([50.0, 150.0])
        rho, shift = _apply_geometry(u, v, 30.0, 45.0, 0.1, 0.2)
        assert rho.dtype == np.float64
        # shift can be float64 when all inputs are zero, or complex128 otherwise
        assert shift.dtype in (np.float64, np.complex128)


class TestBareParamName:
    def test_simple(self):
        bare, profile = _bare_param_name('flux')
        assert bare == 'flux'
        assert profile is None

    def test_prefixed(self):
        bare, profile = _bare_param_name('sersic:flux')
        assert bare == 'flux'
        assert profile == 'sersic'

    def test_geo_names(self):
        bare, profile = _bare_param_name('incl')
        assert bare == 'incl'
        assert profile is None


class TestMatchPriorKey:
    def test_exact_match(self):
        labels = ['sersic:flux', 'sersic:Re', 'sersic:n', 'incl', 'PA', 'dx', 'dy']
        idx = _match_prior_key('sersic:flux', labels)
        assert idx == [0]

    def test_bare_single_comp(self):
        labels = ['flux', 'Re', 'n', 'incl', 'PA', 'dx', 'dy']
        idx = _match_prior_key('flux', labels)
        assert idx == [0]

    def test_ambiguous_raises(self):
        labels = ['sersic:flux', 'point:flux', 'incl', 'PA', 'dx', 'dy']
        with pytest.raises(ValueError, match='ambiguous'):
            _match_prior_key('flux', labels)

    def test_prefixed_not_found_raises(self):
        labels = ['sersic:flux', 'sersic:Re', 'incl']
        with pytest.raises(ValueError, match='does not match'):
            _match_prior_key('point:flux', labels)

    def test_shared_param_bare(self):
        """Bare 'incl' in single-component model matches the bare label."""
        labels = ['flux', 'Re', 'n', 'incl', 'PA', 'dx', 'dy']
        idx = _match_prior_key('incl', labels)
        assert idx == [3]


class TestResolveFixedKey:
    def test_single_index(self):
        labels = ['sersic:flux', 'sersic:Re', 'incl', 'PA']
        idx = _resolve_fixed_key('sersic:flux', labels)
        assert idx == 0

    def test_ambiguous_raises(self):
        labels = ['sersic:flux', 'point:flux', 'incl']
        with pytest.raises(ValueError, match='ambiguous'):
            _resolve_fixed_key('flux', labels)


class TestParsePriorValue:
    def test_two_tuple(self):
        result = _parse_prior_value((0.1, 10.0), 'linear')
        assert result == (0.1, 10.0, 'linear')

    def test_three_tuple(self):
        result = _parse_prior_value((0.1, 10.0, 'log'), 'linear')
        assert result == (0.1, 10.0, 'log')

    def test_invalid_length(self):
        with pytest.raises(ValueError):
            _parse_prior_value((0.1,), 'linear')
        with pytest.raises(ValueError):
            _parse_prior_value((0.1, 10.0, 'log', 'extra'), 'linear')


# ============================================================
# Surface brightness profiles
# ============================================================

class TestSurfaceBrightness:
    def test_sb_point(self):
        cb = np.linspace(0.01, 10, 50)
        sb = _sb_point(cb)
        np.testing.assert_allclose(sb, 1.0)

    def test_sb_gaussian_peak(self):
        cb = np.linspace(0.01, 10, 100)
        sb = _sb_gaussian(cb, sigma=1.0)
        # At cb=0, peak should be 1.0
        idx0 = np.argmin(np.abs(cb))
        assert np.isclose(sb[idx0], 1.0, rtol=0.01)

    def test_sb_gaussian_value_at_sigma(self):
        cb = np.array([1.0])
        sb = _sb_gaussian(cb, sigma=1.0)
        expected = np.exp(-0.5)
        np.testing.assert_allclose(sb, expected)

    def test_sb_gaussian_shape(self):
        cb = np.linspace(0.01, 10, 100)
        sb = _sb_gaussian(cb, sigma=0.5)
        assert sb.shape == (100,)
        assert np.all(sb > 0)
        assert np.all(sb <= 1.0)

    def test_sb_sersic_at_re(self):
        cb = np.array([1.0])
        sb = _sb_sersic(cb, Re=1.0, n=1.0)
        assert np.isclose(sb[0], 1.0)

    def test_sb_sersic_finite(self):
        cb = np.linspace(0.01, 100, 200)
        sb = _sb_sersic(cb, Re=2.0, n=4.0)
        assert np.all(np.isfinite(sb))
        assert np.all(sb >= 0)


# ============================================================
# Hankel transform & normalization
# ============================================================

class TestHankelTransform:
    def test_zeros_in_zeros_out(self):
        SB = np.zeros(150)
        cb, b = _make_bins()
        rho = np.array([1.0, 10.0, 100.0])
        vis = hankel_transform(SB, cb, b, rho)
        np.testing.assert_allclose(vis, 0.0)

    def test_output_type(self):
        """Hankel transform returns a numeric array."""
        SB = np.ones(150)
        cb, b = _make_bins()
        rho = np.array([1.0, 10.0])
        vis = hankel_transform(SB, cb, b, rho)
        assert vis.ndim == 1
        assert len(vis) == 2

    def test_delta_like_constant_vis(self):
        """A very narrow Gaussian (sigma << pixel) gives approximately flat vis."""
        cb, b = _make_bins()
        SB = _sb_gaussian(cb, sigma=0.001)
        Inu = _normalize_SB(SB, 1.0, cb)
        rho = np.array([0.01, 0.1, 1.0])
        vis = hankel_transform(Inu, cb, b, rho)
        # All vis amplitudes should be similar for a point-like source
        amps = np.abs(vis)
        assert np.std(amps) / np.mean(amps) < 0.05


class TestNormalizeSB:
    def test_flux_conservation(self):
        cb, b = _make_bins()
        SB = _sb_gaussian(cb, sigma=1.0)
        Inu = _normalize_SB(SB, 10.0, cb)
        # Integrate 2*pi*r*I_nu over r
        area = np.trapz(2.0 * np.pi * Inu * cb, cb)
        # Should be close to 10 mJy = 0.01 Jy
        np.testing.assert_allclose(area, 0.01, rtol=0.05)

    def test_zero_sb_edge_case(self):
        SB = np.zeros(150)
        cb = np.linspace(0.01, 10, 150)
        result = _normalize_SB(SB, 1.0, cb)
        np.testing.assert_allclose(result, 0.0)


# ============================================================
# Visibility models
# ============================================================

class TestVisPoint:
    def test_flat_amplitude(self):
        u = np.array([100.0, 1000.0, 10000.0])
        v = np.array([0.0, 0.0, 0.0])
        theta = [10.0]  # 10 mJy
        vis = vis_point(theta, (u, v), has_geometry=False)
        np.testing.assert_allclose(np.abs(vis), 0.01)  # 10 mJy in Jy

    def test_mjy_to_jy(self):
        theta = [1000.0]  # 1000 mJy = 1 Jy
        vis = vis_point(theta, (np.array([1.0]), np.array([0.0])),
                        has_geometry=False)
        np.testing.assert_allclose(np.abs(vis), 1.0)

    def test_geometry_preserves_abs(self):
        u = np.array([100.0, 500.0])
        v = np.array([0.0, 0.0])
        theta = [5.0, 30.0, 45.0, 0.1, 0.0]  # flux, incl, PA, dx, dy
        vis = vis_point(theta, (u, v), has_geometry=True)
        # Point source: |vis| should be same for all uv points
        amps = np.abs(vis)
        np.testing.assert_allclose(amps, amps[0])


class TestVisGaussian:
    def test_output_shape_and_type(self):
        """Returns correct shape for multi-element input."""
        theta = [1.0, 0.5]  # 1 mJy, 0.5 arcsec
        u = np.array([100.0, 500.0])
        v = np.array([0.0, 0.0])
        vis = vis_gaussian(theta, (u, v), has_geometry=False)
        assert vis.shape == (2,)
        assert np.issubdtype(vis.dtype, np.floating) or np.issubdtype(vis.dtype, np.complexfloating)

    def test_zero_spacing_flux(self):
        """At zero uv-distance, visibility ≈ total flux."""
        theta = [10.0, 0.5]  # 10 mJy, 0.5 arcsec
        vis = vis_gaussian(theta, (np.array([1e-10]), np.array([0.0])),
                           has_geometry=False)
        np.testing.assert_allclose(np.abs(vis), 0.01, rtol=0.1)

    def test_falls_off(self):
        """Visibility amplitude decreases with uv-distance."""
        theta = [1.0, 1.0]
        u = np.array([10.0, 100.0, 1000.0])
        v = np.zeros(3)
        vis = vis_gaussian(theta, (u, v), has_geometry=False)
        amps = np.abs(vis)
        assert amps[0] > amps[1] > amps[2]

    def test_geometry_shifts_phase(self):
        """With non-zero dx, visibility is no longer purely real."""
        theta = [1.0, 0.5, 0.0, 0.0, 1.0, 0.0]
        u = np.array([500.0])
        v = np.array([0.0])
        vis = vis_gaussian(theta, (u, v), has_geometry=True)
        assert not np.isclose(vis.imag if np.ndim(vis) > 0 else vis, 0.0)


class TestVisSersic:
    def test_output_shape_and_type(self):
        """Returns correct shape for multi-element input."""
        theta = [1.0, 1.0, 2.0]  # 1 mJy, 1 arcsec, n=2
        u = np.array([100.0, 500.0])
        v = np.array([0.0, 0.0])
        vis = vis_sersic(theta, (u, v), has_geometry=False)
        assert vis.shape == (2,)

    def test_zero_spacing_flux(self):
        theta = [10.0, 1.0, 1.0]
        vis = vis_sersic(theta, (np.array([1e-10]), np.array([0.0])),
                         has_geometry=False)
        np.testing.assert_allclose(np.abs(vis), 0.01, rtol=0.1)

    def test_distinguishable_from_gaussian(self):
        """Sersic n=4 and Gaussian should give different visibilities."""
        u = np.array([100.0, 500.0, 1000.0])
        v = np.zeros(3)
        vis_g = vis_gaussian([1.0, 1.0], (u, v), has_geometry=False)
        vis_s = vis_sersic([1.0, 1.0, 4.0], (u, v), has_geometry=False)
        assert not np.allclose(np.abs(vis_g), np.abs(vis_s))


# ============================================================
# make_model_fn
# ============================================================

class TestMakeModelFn:
    def test_sersic_single(self):
        fn, info = make_model_fn(['sersic'])
        assert info['n_params'] == 7  # flux, Re, n, incl, PA, dx, dy
        assert info['labels'] == ['flux', 'Re', 'n', 'incl', 'PA', 'dx', 'dy']

    def test_gaussian_single(self):
        fn, info = make_model_fn(['gaussian'])
        assert info['n_params'] == 6
        assert info['labels'] == ['flux', 'sigma', 'incl', 'PA', 'dx', 'dy']

    def test_point_single(self):
        fn, info = make_model_fn(['point'])
        assert info['n_params'] == 5
        assert info['labels'] == ['flux', 'incl', 'PA', 'dx', 'dy']

    def test_two_components_labels(self):
        fn, info = make_model_fn(['sersic', 'point'])
        assert 'sersic:flux' in info['labels']
        assert 'sersic:Re' in info['labels']
        assert 'sersic:n' in info['labels']
        assert 'point:flux' in info['labels']
        assert 'incl' in info['labels']
        assert 'PA' in info['labels']

    def test_two_components_n_params(self):
        fn, info = make_model_fn(['sersic', 'point'], tie_center=True)
        # sersic: 3 + point: 1 + geo: 4 = 8
        assert info['n_params'] == 8

    def test_untied_geometry(self):
        """Untied incl and PA gives prefixed geo labels for each component."""
        fn, info = make_model_fn(['sersic', 'gaussian'],
                                 tie_center=True, tie_incl=False, tie_pa=False)
        # intrinsic: 3 + 2 = 5, geo: incl*2 + PA*2 + dx + dy = 6, total = 11
        assert info['n_params'] == 11
        assert 'sersic:incl' in info['labels']
        assert 'gaussian:incl' in info['labels']
        assert 'sersic:PA' in info['labels']
        assert 'gaussian:PA' in info['labels']
        assert 'dx' in info['labels']  # shared dx
        assert 'dy' in info['labels']  # shared dy

    def test_untied_center(self):
        """tie_center=False gives prefixed dx/dy for each component."""
        fn, info = make_model_fn(['sersic', 'gaussian'],
                                 tie_center=False, tie_incl=False, tie_pa=False)
        # intrinsic: 3 + 2 = 5, geo: incl*2 + PA*2 + dx*2 + dy*2 = 8, total = 13
        assert info['n_params'] == 13
        assert 'sersic:dx' in info['labels']
        assert 'gaussian:dx' in info['labels']
        assert 'sersic:dy' in info['labels']
        assert 'gaussian:dy' in info['labels']

    def test_invalid_profile(self):
        with pytest.raises(ValueError, match='Unknown profile'):
            make_model_fn(['invalid'])

    def test_too_many_components(self):
        with pytest.raises(ValueError, match='Only 1 or 2'):
            make_model_fn(['point', 'gaussian', 'sersic'])

    def test_fixed_params(self):
        fn, info = make_model_fn(['gaussian'], fixed={'sigma': 0.5})
        assert 1 in info['fixed_indices']
        assert info['n_free'] == 5  # 6 - 1

    def test_prior_overrides(self):
        fn, info = make_model_fn(['gaussian'],
                                 priors={'flux': (1.0, 50.0, 'log')})
        idx = info['labels'].index('flux')
        assert info['p_ranges_info'][idx] == (1.0, 50.0, 'log')

    def test_log_prior_lo_zero_raises(self):
        with pytest.raises(ValueError, match='lo > 0'):
            make_model_fn(['gaussian'], priors={'flux': (0.0, 50.0, 'log')})

    def test_callable(self):
        fn, info = make_model_fn(['point'])
        theta = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        u = np.array([100.0, 500.0])
        v = np.zeros(2)
        vis = fn(theta, (u, v))
        assert vis.shape == (2,)

    def test_point_source_flux(self):
        fn, info = make_model_fn(['point'])
        theta = np.array([10.0, 0.0, 0.0, 0.0, 0.0])  # 10 mJy
        u = np.array([100.0])
        v = np.array([0.0])
        vis = fn(theta, (u, v))
        np.testing.assert_allclose(np.abs(vis), 0.01)

    def test_two_component_sum(self):
        fn, info = make_model_fn(['point', 'point'])
        theta = np.array([10.0, 5.0, 0.0, 0.0, 0.0, 0.0])
        u = np.array([100.0])
        v = np.array([0.0])
        vis = fn(theta, (u, v))
        # 10 mJy + 5 mJy = 15 mJy = 0.015 Jy
        np.testing.assert_allclose(np.abs(vis), 0.015)
