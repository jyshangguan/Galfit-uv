"""Tests for galfit_uv.fit module."""

import os
import numpy as np
import pytest
from galfit_uv.fit import MCMCResult, compute_fit_stats, LogProbability, fit_mcmc
from galfit_uv.models import make_model_fn
from galfit_uv.export import Visibility


# ============================================================
# MCMCResult
# ============================================================

class TestMCMCResult:
    def test_creation(self):
        r = MCMCResult(
            bestfit=np.array([1.0, 2.0]),
            samples=np.zeros((10, 2)),
            all_samples=np.zeros((5, 4, 2)),
            labels=['a', 'b'],
            param_info={'n_params': 2},
        )
        assert r.bestfit is not None
        assert r.labels == ['a', 'b']

    def test_fields(self):
        r = MCMCResult(
            bestfit=np.array([1.0]),
            samples=np.zeros((10, 1)),
            all_samples=np.zeros((5, 4, 1)),
            labels=['x'],
            param_info={'n_params': 1},
            acceptance_fraction=np.array([0.3, 0.4]),
            burnin=50,
            outpath='/tmp',
            free_labels=['x'],
            fixed={'y': 1.0},
            fit_stats={'chi2': 1.0},
        )
        assert r.burnin == 50
        assert r.outpath == '/tmp'
        assert r.free_labels == ['x']
        assert r.fixed == {'y': 1.0}
        assert r.fit_stats == {'chi2': 1.0}


# ============================================================
# compute_fit_stats
# ============================================================

class TestComputeFitStats:
    def test_perfect_fit(self, simple_uv):
        u, v = simple_uv
        flux = 1e-3
        vis = np.full(len(u), flux, dtype=np.complex128)
        wgt = np.ones(len(u))
        dvis = Visibility(u, v, vis, wgt)

        fn, info = make_model_fn(['point'])
        theta = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        stats = compute_fit_stats(dvis, fn, theta, n_free_params=5)

        assert stats['chi2'] < 1e-10

    def test_expected_keys(self, simple_uv):
        u, v = simple_uv
        dvis = Visibility(u, v, np.ones(len(u), dtype=complex), np.ones(len(u)))
        fn, info = make_model_fn(['point'])
        stats = compute_fit_stats(dvis, fn, np.array([1.0, 0.0, 0.0, 0.0, 0.0]), 5)
        for key in ['chi2', 'dof', 'redchi2', 'bic', 'ndata', 'n_free']:
            assert key in stats

    def test_dof(self, simple_uv):
        u, v = simple_uv
        N = len(u)
        dvis = Visibility(u, v, np.ones(N, dtype=complex), np.ones(N))
        fn, _ = make_model_fn(['point'])
        stats = compute_fit_stats(dvis, fn, np.array([1.0, 0.0, 0.0, 0.0, 0.0]), 5)
        assert stats['dof'] == 2 * N - 5
        assert stats['ndata'] == 2 * N

    def test_bic(self, simple_uv):
        u, v = simple_uv
        N = len(u)
        dvis = Visibility(u, v, np.ones(N, dtype=complex), np.ones(N))
        fn, _ = make_model_fn(['point'])
        stats = compute_fit_stats(dvis, fn, np.array([1.0, 0.0, 0.0, 0.0, 0.0]), 5)
        expected_bic = stats['chi2'] + 5 * np.log(2 * N)
        np.testing.assert_allclose(stats['bic'], expected_bic)


# ============================================================
# LogProbability
# ============================================================

class TestLogProbability:
    def _make_lp(self, n_free=5):
        fn, info = make_model_fn(['point'])
        labels = info['labels']
        p_ranges_info = info['p_ranges_info']
        lp = LogProbability(
            fn, np.array([100.0]), np.array([0.0]),
            np.array([0.01+0j]), np.array([1.0]),
            p_ranges_info, labels,
        )
        return lp, info

    def test_prior_bounds_linear(self):
        """Outside bounds -> -inf for linear prior."""
        lp, info = self._make_lp()
        # flux default prior is (0.1, 100) log-scale, incl is (0, 90) linear
        # Set flux to something in bounds, incl to something out of bounds
        theta = np.array([1.0, 1.0, 0.0, 0.0, 0.0])  # flux=1, Re is unused but slot
        # For point source: labels = [flux, incl, PA, dx, dy]
        # Put incl at 91 (out of bounds 0-90)
        theta_oob = np.array([1.0, 91.0, 0.0, 0.0, 0.0])
        assert lp(theta) > -np.inf  # in bounds
        assert lp(theta_oob) == -np.inf

    def test_incl_prior_at_zero_and_90(self):
        """At incl=0 or 90 -> -inf due to sin(incl) prior."""
        lp, info = self._make_lp()
        # labels = [flux, incl, PA, dx, dy]
        theta_zero = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        theta_90 = np.array([1.0, 90.0, 0.0, 0.0, 0.0])
        theta_good = np.array([1.0, 45.0, 0.0, 0.0, 0.0])
        assert lp(theta_zero) == -np.inf
        assert lp(theta_90) == -np.inf
        assert lp(theta_good) > -np.inf

    def test_likelihood_better_model(self):
        """Better model (closer to data) gives higher log-prob."""
        lp, info = self._make_lp()
        # Point source, data has flux 0.01 Jy (10 mJy)
        # Good model: 10 mJy, incl=45 (in bounds for sin prior)
        theta_good = np.array([10.0, 45.0, 0.0, 0.0, 0.0])
        # Bad model: 100 mJy (way off)
        theta_bad = np.array([100.0, 45.0, 0.0, 0.0, 0.0])
        assert lp(theta_good) > lp(theta_bad)

    def test_fixed_params(self):
        """_reconstruct_full inserts fixed values correctly."""
        fn, info = make_model_fn(['point'], fixed={'incl': 45.0})
        labels = info['labels']
        p_ranges_info = info['p_ranges_info']
        lp = LogProbability(
            fn, np.array([100.0]), np.array([0.0]),
            np.array([0.01+0j]), np.array([1.0]),
            p_ranges_info, labels,
            fixed_indices=info['fixed_indices'],
            fixed_values=info['fixed_values'],
            full_ndim=info['n_params'],
        )
        # Free params: [flux, PA, dx, dy] (incl fixed at 45)
        theta_free = np.array([10.0, 0.0, 0.0, 0.0])
        full = lp._reconstruct_full(theta_free)
        assert full[1] == 45.0  # incl was fixed
        assert full[0] == 10.0  # flux from free


# ============================================================
# fit_mcmc integration
# ============================================================

class TestFitMCMC:
    def test_point_source_recovery(self, tmp_path, simple_uv):
        u, v = simple_uv
        flux_jy = 5e-3  # 5 mJy
        vis = np.full(len(u), flux_jy, dtype=np.complex128)
        wgt = np.ones(len(u))
        dvis = Visibility(u, v, vis, wgt)

        fn, info = make_model_fn(['point'])
        result = fit_mcmc(
            dvis, fn, info,
            p_init=[5.0, 0.0, 0.0, 0.0, 0.0],
            max_steps=50, burnin=10, n_workers=1,
            outpath=str(tmp_path), seed=42,
        )
        # Best-fit flux should be near 5 mJy
        flux_idx = info['labels'].index('flux')
        assert np.abs(result.bestfit[flux_idx] - 5.0) < 2.0

    def test_gaussian_recovery(self, tmp_path, simple_uv):
        u, v = simple_uv
        from galfit_uv.models import vis_gaussian
        true_sigma = 0.5
        mvis = vis_gaussian([5.0, true_sigma], (u, v), has_geometry=False)
        wgt = np.ones(len(u))
        dvis = Visibility(u, v, mvis, wgt)

        fn, info = make_model_fn(['gaussian'])
        result = fit_mcmc(
            dvis, fn, info,
            p_init=[5.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            max_steps=50, burnin=10, n_workers=1,
            outpath=str(tmp_path), seed=42,
        )
        sigma_idx = info['labels'].index('sigma')
        assert np.abs(result.bestfit[sigma_idx] - true_sigma) < 0.3

    def test_fixed_param(self, tmp_path, simple_uv):
        u, v = simple_uv
        flux_jy = 5e-3
        vis = np.full(len(u), flux_jy, dtype=np.complex128)
        wgt = np.ones(len(u))
        dvis = Visibility(u, v, vis, wgt)

        fn, info = make_model_fn(['point'], fixed={'incl': 30.0})
        result = fit_mcmc(
            dvis, fn, info,
            p_init=[5.0, 0.0, 0.0, 0.0],
            max_steps=50, burnin=10, n_workers=1,
            outpath=str(tmp_path), seed=42,
        )
        incl_idx = info['labels'].index('incl')
        assert result.bestfit[incl_idx] == 30.0

    def test_seed_reproducibility(self, tmp_path, simple_uv):
        u, v = simple_uv
        flux_jy = 5e-3
        vis = np.full(len(u), flux_jy, dtype=np.complex128)
        wgt = np.ones(len(u))
        dvis = Visibility(u, v, vis, wgt)

        fn, info = make_model_fn(['point'])

        p1 = tmp_path / 'run1'
        p2 = tmp_path / 'run2'
        r1 = fit_mcmc(dvis, fn, info,
                       p_init=[5.0, 0.0, 0.0, 0.0, 0.0],
                       max_steps=50, burnin=10, n_workers=1,
                       outpath=str(p1), seed=123)
        r2 = fit_mcmc(dvis, fn, info,
                       p_init=[5.0, 0.0, 0.0, 0.0, 0.0],
                       max_steps=50, burnin=10, n_workers=1,
                       outpath=str(p2), seed=123)
        np.testing.assert_allclose(r1.bestfit, r2.bestfit)

    def test_output_files(self, tmp_path, simple_uv):
        u, v = simple_uv
        flux_jy = 5e-3
        vis = np.full(len(u), flux_jy, dtype=np.complex128)
        wgt = np.ones(len(u))
        dvis = Visibility(u, v, vis, wgt)

        fn, info = make_model_fn(['point'])
        fit_mcmc(
            dvis, fn, info,
            p_init=[5.0, 0.0, 0.0, 0.0, 0.0],
            max_steps=50, burnin=10, n_workers=1,
            outpath=str(tmp_path), seed=42,
        )
        assert os.path.exists(str(tmp_path / 'fit_results.fits'))
        assert os.path.exists(str(tmp_path / 'chains.png'))
