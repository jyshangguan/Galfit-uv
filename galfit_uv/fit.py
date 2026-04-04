"""
MCMC fitting of visibility data using emcee.

Provides a configurable fitter that works with any combination of
profile models from ``galfit_uv.models``.  Includes convergence diagnostics,
chain plots, and corner plots.
"""

import os
import numpy as np
import emcee
import dill
from dataclasses import dataclass
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

import corner

from galfit_uv.models import _bare_param_name


class _DillPool:
    """A multiprocessing.Pool wrapper that uses dill for pickling.

    This allows unpicklable objects (e.g. closures) to be distributed
    to worker processes.  Implements only the ``map`` / ``close`` /
    ``join`` interface that emcee needs.
    """

    def __init__(self, processes=None):
        self._pool = Pool(processes=processes)

    def map(self, func, iterable):
        return self._pool.map(_DillCallable(func), iterable)

    def close(self):
        self._pool.close()

    def join(self):
        self._pool.join()


class _DillCallable:
    """A picklable wrapper that uses dill to (de)serialize a callable."""

    def __init__(self, func):
        self._blob = dill.dumps(func)

    def __call__(self, *args, **kwargs):
        func = dill.loads(self._blob)
        return func(*args, **kwargs)


@dataclass
class MCMCResult:
    """Container for MCMC fit results."""
    bestfit: np.ndarray
    samples: np.ndarray          # (n_samples, n_params), post burn-in, flattened
    all_samples: np.ndarray      # (n_steps, n_walkers, n_params), full chain
    labels: list
    param_info: dict
    acceptance_fraction: np.ndarray = None
    burnin: int = 0
    outpath: str = '.'
    free_labels: list = None     # labels of free params only
    fixed: dict = None           # {label: value} of fixed params
    fit_stats: dict = None       # {chi2, dof, redchi2, bic, ndata, n_free}


class LogProbability:
    """Picklable log-probability function for emcee.

    Using a class instead of a closure so that the function can be
    serialized by multiprocessing.Pool.

    When fixed parameters are set, the __call__ method accepts a
    *free-only* theta vector and reconstructs the full parameter
    vector before evaluating the model.

    Supports three prior types:
    - ``incl`` parameters: ``sin(incl)`` prior + bounds check
    - ``log`` scale: Jeffreys prior ``ln_prior += -ln(theta[i])``,
      bounds ``lo < theta <= hi``
    - ``linear`` scale: uniform prior (bounds only)
    """

    def __init__(self, model_fn, u, v, vis, wgt, p_ranges_info, labels,
                 fixed_indices=None, fixed_values=None, full_ndim=None):
        self.model_fn = model_fn
        self.u = u
        self.v = v
        self.vis = vis
        self.wgt = wgt
        self.p_ranges_info = p_ranges_info  # list of (lo, hi, scale)
        self.labels = labels
        self._n = len(labels)
        # Fixed-parameter support
        self._fixed_set = set(fixed_indices or [])
        self._fixed_values = dict(zip(fixed_indices or [], fixed_values or []))
        self._full_ndim = full_ndim if full_ndim is not None else self._n

    def _reconstruct_full(self, theta_free):
        """Expand a free-only theta vector to full parameter vector."""
        theta = np.empty(self._full_ndim)
        free_idx = 0
        for i in range(self._full_ndim):
            if i in self._fixed_set:
                theta[i] = self._fixed_values[i]
            else:
                theta[i] = theta_free[free_idx]
                free_idx += 1
        return theta

    def __call__(self, theta_free):
        """Evaluate log-probability.

        Parameters
        ----------
        theta_free : ndarray
            If no params are fixed, this is the full parameter vector.
            If params are fixed, this is the free-only parameter vector
            and will be expanded to full length internally.
        """
        theta = self._reconstruct_full(theta_free)

        # --- Priors ---
        ln_prior = 0.0
        for i in range(self._full_ndim):
            if i in self._fixed_set:
                continue  # fixed params always in bounds
            lo, hi, scale = self.p_ranges_info[i]
            bare = _bare_param_name(self.labels[i])[0]

            if bare == 'incl':
                # sin(incl) prior
                if not (lo < theta[i] < hi):
                    return -np.inf
                ln_prior += np.log(np.sin(np.radians(theta[i])))
            elif scale == 'log':
                # Jeffreys prior: p(theta) ~ 1/theta
                if theta[i] <= lo or theta[i] > hi:
                    return -np.inf
                ln_prior += -np.log(theta[i])
            else:
                # Uniform (linear) prior
                if theta[i] < lo or theta[i] > hi:
                    return -np.inf

        # --- Likelihood ---
        mvis = self.model_fn(theta, (self.u, self.v))
        loglike = -0.5 * np.sum(self.wgt * np.abs(self.vis - mvis) ** 2)

        if np.isnan(loglike):
            return -np.inf

        return loglike + ln_prior


def compute_fit_stats(dvis, model_fn, theta_full, n_free_params):
    """Compute goodness-of-fit statistics for visibility data.

    Each complex visibility point contributes 2 independent real DOF
    (real and imaginary parts).

    Parameters
    ----------
    dvis : Visibility
        Data visibility object (must have ``vis``, ``wgt``, ``u``, ``v``).
    model_fn : callable
        ``model_fn(theta, uv) -> complex ndarray``
    theta_full : ndarray
        Full parameter vector (all params, including any fixed).
    n_free_params : int
        Number of free (non-fixed) parameters.

    Returns
    -------
    dict
        Keys: ``chi2``, ``dof``, ``redchi2``, ``bic``, ``ndata``, ``n_free``.
    """
    mvis = model_fn(theta_full, (dvis.u, dvis.v))
    chi2 = float(np.sum(dvis.wgt * np.abs(dvis.vis - mvis) ** 2))
    ndata = 2 * len(dvis.vis)
    dof = ndata - n_free_params
    redchi2 = chi2 / dof
    bic = chi2 + n_free_params * np.log(ndata)
    return {
        'chi2': chi2,
        'dof': dof,
        'redchi2': redchi2,
        'bic': bic,
        'ndata': ndata,
        'n_free': n_free_params,
    }


def fit_mcmc(dvis, model_fn, param_info, p_init=None,
             max_steps=5000, burnin=1000, nwalk_factor=5,
             outpath='.', seed=None, n_workers=None):
    """Run MCMC fitting using emcee.

    Parameters
    ----------
    dvis : Visibility
        Data visibility object.
    model_fn : callable
        ``model_fn(theta, uv) -> complex ndarray``
    param_info : dict
        From ``make_model_fn``, contains labels, priors, fixed info, etc.
    p_init : array-like, optional
        Initial parameter values (free parameters only).  Each value must
        be within its prior bounds.  Walkers are initialized with 0.1%
        perturbation around p_init.
    max_steps : int
        Maximum MCMC steps.
    burnin : int
        Number of burn-in steps to discard.
    nwalk_factor : int
        Number of walkers = ``nwalk_factor * n_free_params``.
    outpath : str
        Directory for output files.
    seed : int, optional
        Random seed for reproducibility.
    n_workers : int, optional
        Number of parallel workers.  None (default) uses a
        ``multiprocessing.Pool``; set to 1 to disable parallelism
        (useful for debugging or when closures can't be pickled).

    Returns
    -------
    MCMCResult
    """
    if seed is not None:
        np.random.seed(seed)

    # --- Extract config from param_info ---
    labels = param_info['labels']
    n_params = param_info['n_params']
    p_ranges_info = param_info['p_ranges_info']
    p_ranges = param_info['p_ranges']

    fixed_indices = param_info['fixed_indices']
    fixed_values = param_info['fixed_values']
    fixed_set = set(fixed_indices)
    free_labels = param_info['free_labels']
    free_indices = param_info['free_indices']
    n_free = param_info['n_free']
    n_fixed = len(fixed_indices)

    free_p_ranges = param_info['free_p_ranges']
    free_p_scales = param_info['free_p_scales']
    fixed_dict = param_info.get('fixed', {})

    # --- Print prior configuration ---
    print("Prior configuration:")
    max_lbl_len = max(len(lbl) for lbl in labels)
    for i, lbl in enumerate(labels):
        lo, hi, scale = p_ranges_info[i]
        if i in fixed_set:
            val = dict(zip(fixed_indices, fixed_values))[i]
            print(f"  {lbl:>{max_lbl_len + 2}s}: fixed at {val}")
        else:
            print(f"  {lbl:>{max_lbl_len + 2}s}: [{lo}, {hi}]  ({scale})")
    if n_fixed > 0:
        print(f"Fixed parameters ({n_fixed}): "
              + ", ".join(f"{labels[i]} = {fixed_values[j]:.4f}"
                          for j, i in enumerate(fixed_indices)))
        print(f"Free parameters ({n_free}): {', '.join(free_labels)}")

    ndim = n_free
    nwalk = nwalk_factor * ndim

    os.makedirs(outpath, exist_ok=True)

    # --- Walker initialization ---
    if p_init is not None:
        p_init = np.asarray(p_init, dtype=np.float64)
        if len(p_init) != n_free:
            raise ValueError(
                f"p_init length ({len(p_init)}) != n_free ({n_free})")
        free_lo = np.array([r[0] for r in free_p_ranges])
        free_hi = np.array([r[1] for r in free_p_ranges])
        for i in range(n_free):
            lo, hi = free_lo[i], free_hi[i]
            if not (lo <= p_init[i] <= hi):
                raise ValueError(
                    f"p_init[{i}]={p_init[i]} for '{free_labels[i]}' "
                    f"outside prior bounds [{lo}, {hi}]")
        p0 = []
        for _ in range(nwalk):
            walker = np.empty(ndim)
            for i in range(ndim):
                lo, hi = free_lo[i], free_hi[i]
                scale = free_p_scales[i]
                if scale == 'log':
                    walker[i] = p_init[i] * np.exp(np.random.normal(0, 0.001))
                    walker[i] = np.clip(walker[i], lo, hi)
                else:
                    walker[i] = p_init[i] + np.random.normal(0, 0.001) * (hi - lo)
                    walker[i] = np.clip(walker[i], lo, hi)
            p0.append(walker)
    else:
        # Random from prior
        p0 = []
        free_lo = np.array([r[0] for r in free_p_ranges])
        free_hi = np.array([r[1] for r in free_p_ranges])
        for _ in range(nwalk):
            walker = np.empty(ndim)
            for i in range(ndim):
                lo, hi = free_lo[i], free_hi[i]
                scale = free_p_scales[i]
                if scale == 'log':
                    walker[i] = np.exp(np.random.uniform(np.log(lo), np.log(hi)))
                else:
                    walker[i] = np.random.uniform(lo, hi)
            p0.append(walker)

    lnprob = LogProbability(
        model_fn, dvis.u, dvis.v, dvis.vis, dvis.wgt, p_ranges_info, labels,
        fixed_indices=fixed_indices,
        fixed_values=fixed_values,
        full_ndim=n_params,
    )

    # --- Run sampler ---
    if n_workers == 1:
        pool = None
    else:
        # Use dill for pool serialisation so that LogProbability
        # (which wraps a model_fn closure) can be pickled.
        pool = _DillPool(processes=n_workers)

    try:
        sampler = emcee.EnsembleSampler(nwalk, ndim, lnprob, pool=pool)

        index = 0
        autocorr = np.empty(max_steps)
        old_tau = np.inf

        for sample in sampler.sample(p0, iterations=max_steps, progress=True):
            if sampler.iteration % 100:
                continue
            try:
                tau = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1
                converged = np.all(tau * 50 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    print(f"Converged at step {sampler.iteration}")
                    break
                old_tau = tau
            except (emcee.autocorr.AutocorrError, RuntimeError):
                pass
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # --- Parse results ---
    all_samples = sampler.get_chain(discard=0, flat=False)  # free-only
    samples = sampler.get_chain(discard=burnin, flat=False)

    acceptance_fraction = sampler.acceptance_fraction
    accepted = acceptance_fraction > 0.2
    good_chain = sampler.chain[accepted, :, :]  # free-only
    nwalkers_good, nsteps_good, ndim_check = good_chain.shape
    re_samples_free = good_chain[:, burnin:, :].reshape(-1, ndim_check)

    # --- Reconstruct full parameter vectors ---
    fixed_vals_map = dict(zip(fixed_indices, fixed_values))

    def _to_full(free_vec):
        full = np.empty(n_params)
        for i in range(n_params):
            if i in fixed_set:
                full[i] = fixed_vals_map[i]
            else:
                full[i] = free_vec[free_indices.index(i)]
        return full

    bestfit_free = np.array([np.percentile(re_samples_free[:, i], 50)
                             for i in range(ndim_check)])
    bestfit = _to_full(bestfit_free)

    print(f"Best-fit values: {dict(zip(labels, bestfit))}")

    # --- Compute model visibility and expand samples ---
    n_samples = re_samples_free.shape[0]
    re_samples = np.column_stack([
        re_samples_free[:, free_indices.index(i)] if i not in fixed_set
        else np.full(n_samples, fixed_vals_map[i])
        for i in range(n_params)
    ])

    # --- Compute fit statistics ---
    fit_stats = compute_fit_stats(dvis, model_fn, bestfit, n_free)
    print(f"Fit statistics: chi2={fit_stats['chi2']:.1f}, "
          f"DOF={fit_stats['dof']}, "
          f"redchi2={fit_stats['redchi2']:.3f}, "
          f"BIC={fit_stats['bic']:.1f}")

    result = MCMCResult(
        bestfit=bestfit,
        samples=re_samples,
        all_samples=all_samples,
        labels=labels,
        param_info=param_info,
        acceptance_fraction=acceptance_fraction,
        burnin=burnin,
        outpath=outpath,
        free_labels=free_labels,
        fixed=fixed_dict if n_fixed > 0 else None,
        fit_stats=fit_stats,
    )

    # --- Save FITS ---
    uv = (dvis.u, dvis.v)
    mvis = model_fn(bestfit, uv)
    _save_fits(result, dvis, mvis, p_ranges, max_steps, nwalk_factor, seed)

    # --- Generate diagnostic plots ---
    _plot_chains(result)
    _plot_corner(result)

    return result


# ---- Save FITS results ----

def _save_fits(result, dvis, mvis, p_ranges, max_steps, nwalk_factor, seed):
    """Save MCMC results to a multi-extension FITS file.

    Extensions:
        0 (primary)  – header with model setup, MCMC config, best-fit summary
        1  DATA       – input visibility (u, v, Re, Im, weight)
        2  MODEL      – best-fit model visibility (Re, Im)
        3  BESTFIT    – best-fit parameters + 1-sigma uncertainties (one row)
        4  SAMPLES    – post-burn-in MCMC samples (N_samples × N_params)
    """
    from astropy.io import fits

    labels = result.labels
    ndim = len(labels)

    # --- Primary HDU: header ---
    hdr = fits.Header()
    hdr['PROFILES'] = ','.join(result.param_info['profiles'])
    hdr['TIECNTR'] = result.param_info['tie_center']
    hdr['NPARAMS'] = ndim
    for i, lbl in enumerate(labels):
        hdr[f'PLABEL{i}'] = lbl
        hdr[f'PRLO{i}'] = p_ranges[i][0]
        hdr[f'PRHI{i}'] = p_ranges[i][1]
    hdr['MAXSTEP'] = max_steps
    hdr['BURNIN'] = result.burnin
    hdr['NWFCTR'] = nwalk_factor
    hdr['NWALK'] = nwalk_factor * ndim
    hdr['SEED'] = (seed, '')
    # fit statistics
    if result.fit_stats is not None:
        hdr['NFREEPARS'] = result.fit_stats['n_free']
        hdr['NFIXEDPARS'] = len(result.fixed) if result.fixed else 0
        hdr['NDATA'] = result.fit_stats['ndata']
        hdr['DOF'] = result.fit_stats['dof']
        hdr['CHI2'] = result.fit_stats['chi2']
        hdr['REDCHI2'] = result.fit_stats['redchi2']
        hdr['BIC'] = result.fit_stats['bic']
    # fixed parameters
    if result.fixed:
        for j, (lbl, val) in enumerate(result.fixed.items()):
            hdr[f'FIXED{j}'] = f'{lbl}={val}'
    # best-fit + uncertainties
    samples = result.samples
    for i, lbl in enumerate(labels):
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
        hdr[f'BF{i}'] = (q50, f'best-fit {lbl}')
        hdr[f'BSIGP{i}'] = (q84 - q50, f'+1sigma {lbl}')
        hdr[f'BSIGM{i}'] = (q50 - q16, f'-1sigma {lbl}')

    primary_hdu = fits.PrimaryHDU(header=hdr)

    # --- Extension 1: DATA (input visibility) ---
    data_cols = [
        fits.Column(name='U', format='D', array=dvis.u),
        fits.Column(name='V', format='D', array=dvis.v),
        fits.Column(name='RE', format='D', array=dvis.vis.real),
        fits.Column(name='IM', format='D', array=dvis.vis.imag),
        fits.Column(name='WEIGHT', format='D', array=dvis.wgt),
    ]
    data_hdu = fits.BinTableHDU.from_columns(data_cols, name='DATA')

    # --- Extension 2: MODEL (best-fit model visibility) ---
    model_cols = [
        fits.Column(name='RE', format='D', array=mvis.real),
        fits.Column(name='IM', format='D', array=mvis.imag),
    ]
    model_hdu = fits.BinTableHDU.from_columns(model_cols, name='MODEL')

    # --- Extension 3: BESTFIT (single row) ---
    bestfit_cols = [
        fits.Column(name=f'P{i}', format='D',
                    array=np.array([result.bestfit[i]]))
        for i in range(ndim)
    ]
    for i in range(ndim):
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
        bestfit_cols.append(
            fits.Column(name=f'P{i}_P1SIG', format='D',
                        array=np.array([q84 - q50])))
        bestfit_cols.append(
            fits.Column(name=f'P{i}_M1SIG', format='D',
                        array=np.array([q50 - q16])))
    bestfit_hdu = fits.BinTableHDU.from_columns(bestfit_cols, name='BESTFIT')

    # --- Extension 4: SAMPLES (post-burn-in MCMC) ---
    sample_cols = [
        fits.Column(name=f'P{i}', format='D', array=samples[:, i])
        for i in range(ndim)
    ]
    samples_hdu = fits.BinTableHDU.from_columns(sample_cols, name='SAMPLES')

    # --- Write ---
    hdul = fits.HDUList([primary_hdu, data_hdu, model_hdu,
                         bestfit_hdu, samples_hdu])
    fits_path = os.path.join(result.outpath, 'fit_results.fits')
    hdul.writeto(fits_path, overwrite=True)
    print(f"FITS results saved to {fits_path}")


# ---- Diagnostic plots ----

def _setup_matplotlib():
    """Set publication-quality matplotlib defaults."""
    mpl.rc("font", family="serif", size=15)
    mpl.rc("xtick.major", size=7, width=1.5)
    mpl.rc("ytick.major", size=7, width=1.5)
    mpl.rc("xtick.minor", size=4, width=1.0)
    mpl.rc("ytick.minor", size=4, width=1.0)
    mpl.rc("axes", linewidth=2.0)
    mpl.rc("xtick", direction="in", top=True)
    mpl.rc("ytick", direction="in", right=True)


def _plot_chains(result):
    """Plot walker chains for all parameters in a single multi-panel figure.

    Annotates the integrated autocorrelation time tau for each parameter
    in the top-right corner of its subplot.
    """
    _setup_matplotlib()
    outpath = result.outpath
    # Chain arrays contain free-only dimensions
    chain = result.all_samples  # (n_steps, n_walkers, n_free_params)
    chain_labels = result.free_labels if result.free_labels else result.labels

    nsteps, nwalkers, ndim = chain.shape

    # Compute integrated autocorrelation time per parameter
    try:
        tau = emcee.autocorr.integrated_time(chain, tol=0)
    except (emcee.autocorr.AutocorrError, RuntimeError):
        tau = np.full(ndim, np.nan)

    fig, axes = plt.subplots(ndim, 1, figsize=(8, 2.5 * ndim),
                             sharex=True)
    if ndim == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(chain[:, :, i], color="DarkBlue", alpha=0.4,
                rasterized=True)
        ax.axvline(x=result.burnin, color='red', ls='--', lw=1,
                   label=f'burnin={result.burnin}')
        ax.set_ylabel(chain_labels[i], size=20)

        # Annotate tau
        if np.isfinite(tau[i]):
            ax.text(0.97, 0.92, f'$\\tau$ = {tau[i]:.0f}',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=13, bbox=dict(facecolor='white', alpha=0.8,
                                              edgecolor='none'))
        ax.legend(frameon=True, loc='upper left')

    axes[-1].set_xlabel('step number', size=20)
    fig.tight_layout()
    fig.savefig(os.path.join(outpath, 'chains.png'), dpi=150)
    plt.close(fig)

    print("Chain plots saved.")


def _plot_corner(result):
    """Plot corner plot of posterior distributions.

    When fixed parameters are present, only free parameters are plotted
    (fixed parameters have zero dynamic range and would cause errors).
    """
    _setup_matplotlib()
    outpath = result.outpath

    if result.fixed:
        free_labels = result.free_labels
        free_idx = [result.labels.index(l) for l in free_labels]
        samples = result.samples[:, free_idx]
        labels = free_labels
    else:
        samples = result.samples
        labels = result.labels

    fig = corner.corner(
        samples,
        plot_datapoints=False,
        show_titles=True,
        title_fmt='.3f',
        title_args={"fontsize": 13},
        label_kwargs={"fontsize": 16},
        color='DarkBlue',
        quantiles=[0.16, 0.5, 0.84],
        labels=labels,
    )
    fig.savefig(os.path.join(outpath, 'corner_plot.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    print("Corner plot saved.")
