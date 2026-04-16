"""Tests for galfit_uv.export module (Visibility and save_uvtable)."""

import os
import numpy as np
import pytest
from galfit_uv.export import Visibility, save_uvtable


class TestVisibility:
    def test_init_arrays(self):
        u = [1.0, 2.0, 3.0]
        v = [4.0, 5.0, 6.0]
        vis = [1+2j, 3+4j, 5+6j]
        wgt = [0.5, 1.0, 1.5]
        d = Visibility(u, v, vis, wgt)
        assert d.u.dtype == np.float64
        assert d.v.dtype == np.float64
        assert d.vis.dtype == np.complex128
        assert d.wgt.dtype == np.float64

    def test_uvdist(self):
        u = np.array([3.0, 0.0, 0.0])
        v = np.array([0.0, 4.0, 0.0])
        d = Visibility(u, v, np.zeros(3, dtype=complex), np.ones(3))
        np.testing.assert_allclose(d.uvdist, np.array([3.0, 4.0, 0.0]))

    def test_re_im(self):
        vis = np.array([1.0+2.0j, 3.0+4.0j])
        d = Visibility([0, 0], [0, 0], vis, [1, 1])
        np.testing.assert_allclose(d.re, vis.real)
        np.testing.assert_allclose(d.im, vis.imag)

    def test_repr(self):
        d = Visibility([1, 2], [3, 4], [1j, 2j], [1, 1])
        r = repr(d)
        assert 'Visibility' in r
        assert 'n=2' in r

    def test_list_input(self):
        """Accepts Python lists, converts to numpy arrays."""
        d = Visibility([1, 2], [3, 4], [1j, 2j], [1, 1])
        assert isinstance(d.u, np.ndarray)
        assert isinstance(d.vis, np.ndarray)


class TestSaveUvtable:
    def test_lambda(self, tmp_path):
        u = np.array([100.0, 500.0])
        v = np.array([0.0, 0.0])
        vis = np.array([0.01+0.0j, 0.005+0.0j])
        wgt = np.array([1.0, 1.0])
        fname = str(tmp_path / 'test.uvtable')
        save_uvtable(u, v, vis, wgt, fname, wle=0.003, uv_units='lambda')

        with open(fname) as f:
            lines = f.readlines()
        assert 'wavelength[m]' in lines[1]
        assert 'u v Re Im weights' in lines[2]
        assert len(lines) == 5  # 3 header + 2 data

    def test_meters(self, tmp_path):
        u = np.array([100.0])
        v = np.array([0.0])
        vis = np.array([0.01j])
        wgt = np.array([1.0])
        fname = str(tmp_path / 'test_m.uvtable')
        save_uvtable(u, v, vis, wgt, fname, uv_units='m')
        with open(fname) as f:
            text = ''.join(f.readlines())
        assert 'u[m]' in text
        assert 'v[m]' in text

    def test_no_wle(self, tmp_path):
        u = np.array([100.0])
        v = np.array([0.0])
        vis = np.array([0.01+0.0j])
        wgt = np.array([1.0])
        fname = str(tmp_path / 'test_nowle.uvtable')
        save_uvtable(u, v, vis, wgt, fname, wle=None)
        with open(fname) as f:
            text = ''.join(f.readlines())
        assert 'wavelength' not in text

    def test_roundtrip(self, tmp_path):
        u = np.array([100.0, 500.0, 1000.0])
        v = np.array([50.0, 200.0, 800.0])
        vis = np.array([0.01+0.005j, 0.005+0.001j, 0.001+0.0005j])
        wgt = np.array([1.0, 2.0, 0.5])
        fname = str(tmp_path / 'roundtrip.uvtable')
        save_uvtable(u, v, vis, wgt, fname, wle=0.001)

        # Read back
        data = np.loadtxt(fname, comments='#')
        np.testing.assert_allclose(data[:, 0], u)
        np.testing.assert_allclose(data[:, 1], v)
        np.testing.assert_allclose(data[:, 2], vis.real)
        np.testing.assert_allclose(data[:, 3], vis.imag)
        np.testing.assert_allclose(data[:, 4], wgt)

    def test_complex_values(self, tmp_path):
        """Re and Im columns correct for complex visibilities."""
        u = np.array([100.0, 200.0])
        v = np.array([0.0, 50.0])
        vis = np.array([3.14+2.72j, 1.0+1.0j])
        wgt = np.array([1.0, 2.0])
        fname = str(tmp_path / 'complex.uvtable')
        save_uvtable(u, v, vis, wgt, fname)

        data = np.loadtxt(fname, comments='#')
        assert data.shape[0] == 2
        assert np.isclose(data[0, 2], 3.14)
        assert np.isclose(data[0, 3], 2.72)
        assert np.isclose(data[1, 2], 1.0)
        assert np.isclose(data[1, 3], 1.0)


@pytest.mark.skip(reason="Requires CASA (casatools)")
def test_export_vis_skipped():
    """Placeholder — export_vis requires CASA and a real measurement set."""
    pass
