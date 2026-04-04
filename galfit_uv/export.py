"""
Export visibility data from CASA measurement sets.

Handles variable-shape columns (via tb.getvarcol) that arise when SPWs
have different numbers of channels.  Outputs uvplot-compatible text tables.
"""

import numpy as np


class Visibility:
    """Container for visibility data."""

    def __init__(self, u, v, vis, wgt):
        self.u = np.asarray(u, dtype=np.float64)
        self.v = np.asarray(v, dtype=np.float64)
        self.vis = np.asarray(vis, dtype=np.complex128)
        self.wgt = np.asarray(wgt, dtype=np.float64)

    def __repr__(self):
        return (f"Visibility(n={len(self.u)}, "
                f"uv=[{self.u.min():.0f}, {np.hypot(self.u, self.v).max():.0f}] lambda)")

    @property
    def uvdist(self):
        return np.hypot(self.u, self.v)

    @property
    def re(self):
        return self.vis.real

    @property
    def im(self):
        return self.vis.imag


def _getvarcol_safe(tb, colname):
    """Read a possibly variable column from a CASA table.

    Returns an array with shape ``(n_rows,)`` for scalar columns or
    ``(n_pol, n_rows)`` for array columns, regardless of whether the
    column is variable-shaped.

    For variable columns the rows are grouped by DATA_DESC_ID; each group
    may have a different number of channels.  We collapse over channels
    (sum for DATA, any for FLAG) so that every row contributes one value
    per polarization.
    """
    if not tb.isvarcol(colname):
        arr = np.squeeze(tb.getcol(colname))
        return arr

    data_dict = tb.getvarcol(colname)

    # Sort keys to preserve row order: 'r0', 'r1', ...
    keys = sorted(data_dict.keys(), key=lambda k: int(k[1:]))

    # Determine what we're dealing with by inspecting the first entry.
    v0 = data_dict[keys[0]]
    v0 = np.asarray(v0).squeeze()

    # Figure out ndim and whether the column has a channel axis.
    # After squeeze, typical shapes:
    #   DATA:  (n_pol, n_chan, 1) -> (n_pol, n_chan)  or  (n_pol, n_chan)
    #   WEIGHT: (n_pol, 1) -> (n_pol,)
    #   FLAG:   (n_pol, n_chan, 1) -> (n_pol, n_chan)
    #   UVW:    not variable, handled above
    col_is_scalar = (v0.ndim == 0)  # e.g. DATA_DESC_ID per row

    if col_is_scalar:
        # Each entry is a scalar.
        return np.array([np.squeeze(data_dict[k]) for k in keys])

    if v0.ndim == 1:
        # No channel axis, e.g. WEIGHT: (n_pol,) per row
        # All entries should have same n_pol.
        parts = [np.asarray(data_dict[k]).squeeze() for k in keys]
        # Stack along last axis to get (n_pol, n_rows)
        return np.column_stack(parts)

    if v0.ndim >= 2:
        # Has channel axis: (n_pol, n_chan, ...) per row
        # Sum / reduce over channels (axis 1) to get (n_pol,) per row.
        parts = []
        for k in keys:
            arr = np.asarray(data_dict[k]).squeeze()
            # Sum over all axes except the polarization axis (axis 0)
            reduced = arr.sum(axis=tuple(range(1, arr.ndim)))
            parts.append(reduced)
        return np.column_stack(parts)

    # Fallback
    return np.column_stack([np.squeeze(data_dict[k]) for k in keys])


def _getvarcol_flag(tb, colname):
    """Read a flag column (boolean), collapsing over channels with any().

    Returns shape ``(n_pol, n_rows)``.
    """
    if not tb.isvarcol(colname):
        arr = np.squeeze(tb.getcol(colname))
        if arr.ndim >= 2:
            # Collapse over channels: flag if any channel is flagged
            return np.any(arr, axis=tuple(range(1, arr.ndim)))
        return arr

    data_dict = tb.getvarcol(colname)
    keys = sorted(data_dict.keys(), key=lambda k: int(k[1:]))

    v0 = np.asarray(data_dict[keys[0]]).squeeze()
    if v0.ndim == 0:
        return np.array([bool(np.squeeze(data_dict[k])) for k in keys])

    parts = []
    for k in keys:
        arr = np.asarray(data_dict[k]).squeeze()
        # Flag row if any channel/pol is flagged
        reduced = np.any(arr, axis=tuple(range(1, arr.ndim)))
        parts.append(reduced)
    return np.column_stack(parts)


def export_vis(msfile, datacolumn='DATA', timebin=10.0, verbose=False):
    """Extract averaged visibility data from a measurement set.

    Handles variable-shape columns via ``tb.getvarcol`` so it works
    even when spectral windows have different numbers of channels.

    Parameters
    ----------
    msfile : str
        Path to the measurement set (must end in ``.ms``).
    datacolumn : str
        Which data column to read (``'DATA'``, ``'CORRECTED_DATA'``, etc.).
    timebin : float or None
        Time-bin width in seconds.  Rows within each bin are averaged
        per baseline (ANTENNA1, ANTENNA2), consistent with CASA
        ``split(timebin='10s')``.  Set to ``None`` or 0 to disable.
    verbose : bool
        Print column shapes for debugging.

    Returns
    -------
    galfit_uv.Visibility
        Object with ``.u``, ``.v``, ``.vis``, ``.wgt`` arrays (u,v in
        wavelengths, vis in Jy, wgt summed over polarizations).
    float
        Average wavelength in meters.
    """
    if not msfile.endswith('.ms'):
        raise ValueError("MS name must end in '.ms'")

    tb_tool = __import__('casatools', fromlist=['table']).table()

    # --- Read main table columns ---
    tb_tool.open(msfile)

    if verbose:
        for cn in [datacolumn, 'WEIGHT', 'FLAG', 'UVW', 'DATA_DESC_ID', 'TIME']:
            is_var = tb_tool.isvarcol(cn)
            print(f"  {cn}: isvarcol={is_var}")

    data = _getvarcol_safe(tb_tool, datacolumn)
    flag = _getvarcol_flag(tb_tool, 'FLAG')
    uvw = _getvarcol_safe(tb_tool, 'UVW')
    weight = _getvarcol_safe(tb_tool, 'WEIGHT')
    spwid = _getvarcol_safe(tb_tool, 'DATA_DESC_ID')

    # Read time and antenna columns (needed for time binning)
    _do_timebin = (timebin is not None) and (timebin > 0)
    if _do_timebin:
        time_col = np.squeeze(tb_tool.getcol('TIME'))
        ant1 = np.squeeze(tb_tool.getcol('ANTENNA1')).astype(np.int64)
        ant2 = np.squeeze(tb_tool.getcol('ANTENNA2')).astype(np.int64)

    tb_tool.close()

    if verbose:
        print(f"  data  shape: {data.shape}")
        print(f"  flag  shape: {flag.shape}")
        print(f"  uvw   shape: {uvw.shape}")
        print(f"  weight shape: {weight.shape}")
        print(f"  spwid shape: {spwid.shape}")
        if _do_timebin:
            print(f"  timebin: {timebin}s, time range [{time_col.min():.1f}, {time_col.max():.1f}]")

    # --- Read CHAN_FREQ from SPECTRAL_WINDOW subtable ---
    tb_tool.open(msfile + '/SPECTRAL_WINDOW')
    if tb_tool.isvarcol('CHAN_FREQ'):
        if verbose:
            print("  CHAN_FREQ is variable in SPECTRAL_WINDOW")
        cf_dict = tb_tool.getvarcol('CHAN_FREQ')
        keys = sorted(cf_dict.keys(), key=lambda k: int(k[1:]))
        freqlist = []
        for k in keys:
            arr = np.asarray(cf_dict[k]).squeeze()
            # Take mean frequency for each SPW
            freqlist.append(np.mean(arr))
        freqlist = np.array(freqlist)
    else:
        chan_freq_full = np.squeeze(tb_tool.getcol('CHAN_FREQ'))
        if chan_freq_full.ndim == 1:
            freqlist = chan_freq_full
        elif chan_freq_full.ndim == 2:
            freqlist = np.mean(chan_freq_full, axis=1)
        else:
            freqlist = np.array([np.mean(chan_freq_full)])
    tb_tool.close()

    # --- Ensure consistent row count ---
    # data: (n_pol, n_rows), flag: (n_pol, n_rows)
    # uvw: (3, n_rows), weight: (n_pol, n_rows), spwid: (n_rows,)
    n_rows_data = data.shape[-1]
    n_rows_uvw = uvw.shape[-1]
    n_rows = min(n_rows_data, n_rows_uvw)
    if verbose:
        print(f"  n_rows: data={n_rows_data}, uvw={n_rows_uvw}, using {n_rows}")

    data = data[..., :n_rows]
    flag = flag[..., :n_rows]
    uvw = uvw[..., :n_rows]
    weight = weight[..., :n_rows]
    spwid = spwid[:n_rows]

    # Handle scalar spwid (broadcast)
    spwid = np.atleast_1d(np.squeeze(spwid))
    if len(spwid) == 1 and n_rows > 1:
        spwid = np.broadcast_to(spwid, (n_rows,))

    # --- Identify unflagged data ---
    # flag may be (n_pol, n_rows) or (n_rows,)
    if flag.ndim == 2:
        good = ~np.any(flag, axis=0)
    else:
        good = ~flag
    good = good[:n_rows]

    data = data[:, good]
    weight = weight[:, good]
    uvw = uvw[:, good]
    spwid = spwid[good]
    if _do_timebin:
        time_col = time_col[good]
        ant1 = ant1[good]
        ant2 = ant2[good]

    # --- Average polarizations using weights ---
    w_sum = np.sum(weight, axis=0)
    w_sum = np.where(w_sum == 0, 1.0, w_sum)  # avoid division by zero
    re = np.sum(data.real * weight, axis=0) / w_sum
    im = np.sum(data.imag * weight, axis=0) / w_sum
    vis = re + 1j * im
    wgt = w_sum

    # --- Time binning (average per baseline per time bin) ---
    if _do_timebin:
        t0 = time_col.min()
        time_bin_id = np.floor((time_col - t0) / timebin).astype(np.int64)

        # Build composite key: (time_bin, ant1, ant2) encoded as integers.
        # Scale ant1/ant2 to avoid collisions with time_bin_id.
        n_ant = max(ant1.max(), ant2.max()) + 1
        composite = (time_bin_id * n_ant * n_ant + ant1 * n_ant + ant2)

        uniq_keys, inverse = np.unique(composite, return_inverse=True)
        n_binned = len(uniq_keys)

        vis_bin = np.zeros(n_binned, dtype=np.complex128)
        wgt_bin = np.zeros(n_binned, dtype=np.float64)
        um_bin = np.zeros(n_binned, dtype=np.float64)
        vm_bin = np.zeros(n_binned, dtype=np.float64)
        spwid_bin = np.empty(n_binned, dtype=spwid.dtype)

        # Use bincount for efficient aggregation
        wgt_sum_per_group = np.bincount(inverse, weights=wgt, minlength=n_binned)
        vis_re_sum = np.bincount(inverse, weights=vis.real * wgt, minlength=n_binned)
        vis_im_sum = np.bincount(inverse, weights=vis.imag * wgt, minlength=n_binned)
        um_sum = np.bincount(inverse, weights=uvw[0, :], minlength=n_binned)
        vm_sum = np.bincount(inverse, weights=uvw[1, :], minlength=n_binned)
        count = np.bincount(inverse, minlength=n_binned)

        # For spwid, take the first row in each group
        first_idx = np.zeros(n_binned, dtype=np.int64)
        np.minimum.at(first_idx, inverse, np.arange(len(composite)))

        safe_wgt = np.where(wgt_sum_per_group == 0, 1.0, wgt_sum_per_group)
        vis_bin = (vis_re_sum / safe_wgt) + 1j * (vis_im_sum / safe_wgt)
        wgt_bin = wgt_sum_per_group
        um_bin = um_sum / np.where(count == 0, 1, count)
        vm_bin = vm_sum / np.where(count == 0, 1, count)
        spwid_bin = spwid[first_idx]

        vis = vis_bin
        wgt = wgt_bin
        uvw = np.vstack([um_bin, vm_bin, np.zeros(n_binned)])
        spwid = spwid_bin

        if verbose:
            print(f"  After time binning ({timebin}s): {n_binned} points "
                  f"(from {len(time_bin_id)})")

    n_rows = len(vis)

    # --- Get frequency for each row ---
    spwid_int = spwid.astype(int)
    if spwid_int.max() < len(freqlist):
        freqs = freqlist[spwid_int]
    else:
        # Fallback: use mean frequency
        freqs = np.full(n_rows, freqlist.mean())

    # --- Convert u,v from meters to wavelengths ---
    c = 2.99792458e8  # m/s
    um = uvw[0, :]
    vm = uvw[1, :]
    u_lam = um * freqs / c
    v_lam = vm * freqs / c

    wle = c / np.mean(freqs)

    if verbose:
        print(f"  After averaging: {len(u_lam)} points")
        print(f"  u range: [{u_lam.min():.0f}, {u_lam.max():.0f}] lambda")
        print(f"  wavelength: {wle*1e3:.4f} mm")

    return Visibility(u_lam, v_lam, vis, wgt), wle


def save_uvtable(u, v, vis, wgt, filename, wle=None,
                 uv_units='lambda'):
    """Save visibility data as a uvplot-compatible ASCII table.

    Parameters
    ----------
    u, v : array-like
        u, v coordinates (in wavelengths or meters, see ``uv_units``).
    vis : array-like, complex
        Visibility values in Jy.
    wgt : array-like
        Visibility weights.
    filename : str
        Output file path.
    wle : float, optional
        Wavelength in meters.  Written into the header comment.
    uv_units : {'lambda', 'm'}
        Units of u, v.  uvplot expects wavelengths by default.
    """
    u = np.asarray(u)
    v = np.asarray(v)
    vis = np.asarray(vis)
    wgt = np.asarray(wgt)

    with open(filename, 'w') as f:
        if wle is not None:
            f.write(f"# Extracted from measurement set.\n")
            f.write(f"# wavelength[m] = {wle}\n")
        if uv_units == 'lambda':
            f.write("# Columns      u v Re Im weights\n")
        else:
            f.write("# Columns      u[m]    v[m]    Re(V)[Jy]       Im(V)[Jy]       weight\n")

        for i in range(len(u)):
            f.write(f"{u[i]:.15e}\t{v[i]:.15e}\t"
                    f"{vis[i].real:.15e}\t{vis[i].imag:.15e}\t"
                    f"{wgt[i]:.15e}\n")

    print(f"Saved uvtable to {filename} ({len(u)} points)")
