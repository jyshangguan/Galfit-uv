"""Tests for galfit_uv.measure module."""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')

from galfit_uv.measure import calculate_w50, plot_1d_spectrum


# ============================================================
# Pure tests
# ============================================================

class TestCalculateW50:
    def test_gaussian(self):
        """W50 = 2.355*sigma for a Gaussian."""
        sigma = 10.0
        x = np.linspace(-50, 50, 10001)
        y = np.exp(-0.5 * (x / sigma) ** 2)
        w50 = calculate_w50(x, y)
        expected = 2.3548200450309493 * sigma  # 2*sqrt(2*ln(2))*sigma
        np.testing.assert_allclose(w50, expected, rtol=0.01)

    def test_below_half_max(self):
        """Returns finite W50 for a valid profile (non-NaN)."""
        # For a single-point peak above half-max, W50 should be 0
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 0.0])
        w50 = calculate_w50(x, y)
        assert w50 == 0.0  # only one point above half-max

    def test_single_peak(self):
        """Narrow spike -> small W50."""
        x = np.linspace(0, 100, 1001)
        y = np.zeros(1001)
        y[500] = 1.0  # single point spike
        w50 = calculate_w50(x, y)
        # Should be very small (width of single bin)
        assert w50 >= 0
        assert w50 < 1.0  # less than one bin width

    def test_array_input(self):
        """Works with numpy arrays (not just lists)."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 1.0, 0.5, 1.0, 0.0])
        w50 = calculate_w50(x, y)
        assert w50 > 0
        assert np.isfinite(w50)


class TestPlot1DSpectrum:
    def test_basic(self):
        x = np.linspace(0, 10, 50)
        y = np.exp(-0.5 * (x - 5) ** 2)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = plot_1d_spectrum(x, y, ax=ax)
        # Should return a StepPatch
        assert result is not None

    def test_with_ax(self):
        x = np.arange(10, dtype=float)
        y = np.ones(10)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = plot_1d_spectrum(x, y, ax=ax)
        assert ax is not None

    def test_dx_provided(self):
        x = np.arange(10, dtype=float)
        y = np.ones(10)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = plot_1d_spectrum(x, y, ax=ax, dx=1.0)
        assert result is not None

    def test_nan_dx_raises(self):
        x = np.array([1.0, 1.0])  # uniform spacing -> dx=0 -> edges will have NaN issues
        # Actually need dx to be NaN
        x = np.array([1.0, 1.0, 1.0])
        y = np.ones(3)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match='NaN'):
            plot_1d_spectrum(x, y, ax=ax, dx=float('nan'))


# ============================================================
# dynesty tests
# ============================================================

@pytest.mark.needs_dynesty
class TestFitDynesty:
    def test_gaussian_recovery(self):
        from galfit_uv.measure import fit_dynesty
        rng = np.random.default_rng(42)
        x = np.linspace(-50, 50, 100)
        true_a, true_b, true_c = 10.0, 0.0, 5.0
        y = true_a * np.exp(-0.5 * ((x - true_b) / true_c) ** 2)
        y += rng.normal(0, 0.1, len(x))

        result = fit_dynesty(
            x, y, yerr=0.1, model_type='gaussian',
            nlive=200, dlogz=0.5, plot=False, progress=False,
            rstate=np.random.default_rng(42),
        )
        assert np.abs(result['params']['a'] - true_a) / true_a < 0.2

    def test_double_peak_recovery(self):
        from galfit_uv.measure import fit_dynesty
        from galfit_uv.lineprofiles import Gaussian_DoublePeak
        rng = np.random.default_rng(42)
        x = np.linspace(-30, 30, 200)
        true_params = (5.0, 1.0, 0.0, 2.0, 3.0)
        y = Gaussian_DoublePeak(x, *true_params)
        y += rng.normal(0, 0.05, len(x))

        result = fit_dynesty(
            x, y, yerr=0.05, model_type='double_peak',
            nlive=200, dlogz=0.5, plot=False, progress=False,
            rstate=np.random.default_rng(42),
        )
        assert np.abs(result['params']['v0'] - 0.0) < 5.0

    def test_invalid_model(self):
        from galfit_uv.measure import fit_dynesty
        with pytest.raises(ValueError, match='model_type'):
            fit_dynesty([1, 2], [1, 2], 0.1, model_type='invalid')

    def test_result_keys(self):
        from galfit_uv.measure import fit_dynesty
        x = np.linspace(-10, 10, 50)
        y = np.exp(-0.5 * x ** 2)
        result = fit_dynesty(
            x, y, yerr=0.1, model_type='gaussian',
            nlive=100, dlogz=1.0, plot=False, progress=False,
        )
        for key in ['params', 'params_err', 'samples', 'logz', 'logzerr',
                     'model_type', 'derived']:
            assert key in result


# ============================================================
# Skipped tests (require spectral-cube / CASA)
# ============================================================

@pytest.mark.needs_spectral_cube
def test_source_mask_skipped():
    """Placeholder — source_mask requires spectral-cube and a real cube."""
    pass


@pytest.mark.needs_spectral_cube
def test_detect_source_skipped():
    """Placeholder — detect_source requires spectral-cube and a real cube."""
    pass
