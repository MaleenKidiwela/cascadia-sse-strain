"""Loading a Gualandi near-real-time Cascadia solution and rebuilding slip(x, t).

What the download contains
--------------------------
Each dated directory under https://near-real-time-sse.esc.cam.ac.uk/cascadia/
holds the *ingredients* of a slip model rather than the slip model itself.
Following ``driver_minimal_noplots.jl``:

``options.mat``       -> ``options``          inversion / smoothing / rate settings
``X.mat``             -> ``Xd``               input position time series, station llh, timeline
``ICA.mat``           -> ``ICA_essential``    U, S, V, var_U, var_V from the vbICA
``ind_comps.mat``     -> ``ind_comps``        indices of components retained for inversion
``misfit_comps.mat``  -> ``misfit_comps``     cross-validation misfit curves

The Green's functions are deliberately not stored (Gualandi, 2025, section 5:
regenerating them on the fly saves a large amount of disk).  So we rebuild
them, redo the linear inversion, and reconstruct

    slip_strike = L_strike @ S @ V.T          (n_patch, n_time)
    slip_dip    = L_dip    @ S @ V.T

The important structural point for everything downstream: slip is *low rank*.
With K retained components (typically 15-30), the whole 2007-present history is
a sum of K static patterns modulated by K scalar time functions.  Strain and
stress are linear in slip, so we only ever have to run K forward elastic
calculations, not one per epoch.  See :mod:`sse_strain.green`.

Sign convention on slip
-----------------------
``create_model`` in the Julia code applies a rake flip to the scalar slip
*magnitude* used for plotting, based on ``options.inversion.rake_pos``.  That
operation is nonlinear and is deliberately **not** applied here: we carry the
signed strike and dip components, which are linear in the component
amplitudes.  The magnitude with the flip applied is available from
:meth:`Solution.slip_magnitude` if you want to reproduce his figures.

Slip is relative to the long-term trend
---------------------------------------
Gualandi (2025) removes a secular linear trend before the vbICA.  A patch with
negative rate is loading faster than its long-term rate; a positive rate is
unloading.  The interseismic loading term is therefore *absent* from this
product and has to be supplied separately if you want total strain.  See
``docs/THEORY.md``, section "Adding the loading term".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .fault import FaultMesh
from .matio import load_mat

__all__ = ["Solution", "load_solution"]

_FILES = {
    "options": ("options.mat", "options"),
    "X": ("X.mat", "Xd"),
    "ICA": ("ICA.mat", "ICA_essential"),
    "ind_comps": ("ind_comps.mat", "ind_comps"),
    "misfit_comps": ("misfit_comps.mat", "misfit_comps"),
}


def _get(d, *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


@dataclass
class Solution:
    """One daily Gualandi solution, plus the fault it is inverted onto."""

    directory: Path
    options: dict
    X: dict
    ICA: dict
    fault: FaultMesh
    ind_comps: np.ndarray
    misfit_comps: object = None
    L: np.ndarray | None = field(default=None, repr=False)   # (2*n_patch, K)
    Cm: list | None = field(default=None, repr=False)
    sigma0_source: str | None = field(default=None, repr=False)

    # ---------------------------------------------------------------- basics
    @property
    def nu(self) -> float:
        """Poisson's ratio of the elastic half-space used for the inversion."""
        v = _get(self.options, "fault", "nu", default=0.25)
        return float(np.asarray(v).ravel()[0])

    @property
    def timeline(self) -> np.ndarray:
        """Decimal-year epochs, shape (n_time,)."""
        return np.asarray(self.X["timeline"], dtype=float).ravel()

    @property
    def dates(self):
        """Timeline as numpy datetime64[D]."""
        return decimal_year_to_datetime64(self.timeline)

    @property
    def station_llh(self) -> np.ndarray:
        """(n_station, 3) lon/lat/height of the stations used, one row each.

        ``X["llh"]`` repeats each station once per component (3x for 3-D data),
        so it is decimated here using the same rule as ``_get_llh_stations``.
        """
        llh = np.asarray(self.X["llh"], dtype=float)
        if llh.shape[0] < llh.shape[1]:
            llh = llh.T
        step = self._data_type_step()
        return llh[::step, :]

    def _data_type_step(self) -> int:
        names = self.X.get("name")
        if names is None:
            return 3
        names = [str(n) for n in np.asarray(names, dtype=object).ravel()]
        if not names:
            return 3
        if names[0].endswith("u"):
            return 1
        if len(names) > 2 and names[2].endswith("e"):
            return 2
        return 3

    @property
    def S(self) -> np.ndarray:
        """Component weights, as a 1-D array of length n_components."""
        S = np.asarray(self.ICA["S"], dtype=float)
        return np.diag(S) if S.ndim == 2 else S.ravel()

    @property
    def V(self) -> np.ndarray:
        """(n_time, n_components) temporal functions."""
        V = np.asarray(self.ICA["V"], dtype=float)
        if V.shape[0] != self.timeline.size and V.shape[1] == self.timeline.size:
            V = V.T
        return V

    @property
    def U(self) -> np.ndarray:
        """(n_data, n_components) spatial mixing matrix."""
        return np.asarray(self.ICA["U"], dtype=float)

    @property
    def var_U(self) -> np.ndarray:
        return np.asarray(self.ICA["var_U"], dtype=float)

    @property
    def component_index(self) -> np.ndarray:
        """0-based indices of the components retained for inversion."""
        idx = np.asarray(self.ind_comps, dtype=int).ravel()
        return idx - 1 if idx.min() >= 1 else idx

    # ----------------------------------------------------------- forward model
    def displacement_gf(self, max_pairs: int = 200_000) -> np.ndarray:
        """Station displacement Green's functions, shape (3*n_station, 2*n_patch).

        Thin wrapper over :func:`sse_strain.green.displacement_gf` using this
        solution's station list and Poisson's ratio.
        """
        from .green import displacement_gf as _gf

        return _gf(self.fault, self.station_llh[:, :2], self.nu,
                   max_pairs=max_pairs)

    def invert(self, G: np.ndarray | None = None, sigma0: float | None = None,
               verbose: bool = True) -> np.ndarray:
        """Redo the per-component linear inversion; returns L, shape (2*n_patch, K).

        Reproduces ``invert_comp`` / ``create_priors``: a Tarantola-Valette
        least-squares solution with an exponential (or Gaussian) prior model
        covariance whose correlation length is ``options.inversion.lambda_*``.

        The smoothing hyperparameter is per component in the published
        workflow (chosen by leave-one-out cross validation, ``ind_sigma0_comps``).
        If ``ind_sigma0_comps`` is not present in the download, pass ``sigma0``
        explicitly; a single scalar is then used for every component and the
        result will differ from the published slip model.  This is exactly the
        kind of thing to check before drawing conclusions -- see
        :meth:`misfit_check`.
        """
        opt = _get(self.options, "inversion", default={})
        lam_s = float(np.ravel(opt.get("lambda_strike", 21.0))[0])
        lam_d = float(np.ravel(opt.get("lambda_dip", 21.0))[0])
        lam0 = float(np.ravel(opt.get("lambda0", 21.0))[0])
        which = str(np.ravel(opt.get("which_smoothing", "exponential"))[0])
        m0_val = float(np.ravel(opt.get("m0", 0.0))[0])

        sigmas = np.ravel(np.asarray(opt.get("sigma", [1.0]), dtype=float))
        ind_sigma = _get(self.options, "inversion", "ind_sigma0_comps")
        if ind_sigma is None:
            ind_sigma = _get(self.options, "ind_sigma0_comps")

        if G is None:
            G = self.displacement_gf()

        xyz = self.fault.centroid_xyz_km
        d_patch = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1))
        n_p = self.fault.n_patch

        idx = self.component_index
        K = idx.size

        # ind_sigma0_comps is absent from the public download, but
        # misfit_comps.mat holds the cross-validation misfit over the
        # (n_sigma, K) grid the published workflow minimised. Recover the
        # per-component choice from the column minima. Only trust the argmin
        # when every column actually varies -- degenerate (e.g. all-zero)
        # arrays fall through to the middle-of-grid fallback.
        if sigma0 is not None:
            self.sigma0_source = "scalar sigma0 argument"
        elif ind_sigma is not None:
            self.sigma0_source = "ind_sigma0_comps from options"
        else:
            self.sigma0_source = "middle-of-grid fallback"
            if self.misfit_comps is not None:
                mis = np.asarray(self.misfit_comps, dtype=float)
                if (mis.ndim == 2 and mis.shape == (sigmas.size, K)
                        and np.all(np.nanmax(mis, 0) > np.nanmin(mis, 0))):
                    ind_sigma = np.nanargmin(mis, axis=0) + 1  # 1-based
                    self.sigma0_source = "per-component argmin of misfit_comps CV curves"

        L = np.zeros((2 * n_p, K))
        Cm = []
        for k in range(K):
            if sigma0 is not None:
                etm0 = float(sigma0)
            elif ind_sigma is not None:
                j = int(np.ravel(ind_sigma)[k])
                j = j - 1 if j >= 1 else j
                etm0 = float(sigmas[j])
            else:
                etm0 = float(sigmas[len(sigmas) // 2])

            Em_s = etm0 * lam0 / lam_s
            Em_d = etm0 * lam0 / lam_d
            if which.lower().startswith("gauss"):
                Cs = Em_s**2 * np.exp(-0.5 * d_patch**2 / lam_s)
                Cd_ = Em_d**2 * np.exp(-0.5 * d_patch**2 / lam_d)
            else:
                Cs = Em_s**2 * np.exp(-d_patch / lam_s)
                Cd_ = Em_d**2 * np.exp(-d_patch / lam_d)
            Cm0 = np.zeros((2 * n_p, 2 * n_p))
            Cm0[:n_p, :n_p] = Cs
            Cm0[n_p:, n_p:] = Cd_

            d = self.U[:, idx[k]]
            Cd = np.diag(self.var_U[:, idx[k]])
            m0 = np.full(2 * n_p, m0_val)

            GC = G @ Cm0
            A = GC @ G.T + Cd
            resid = d - G @ m0
            L[:, k] = m0 + GC.T @ np.linalg.solve(A, resid)
            Cm.append(Cm0 - GC.T @ np.linalg.solve(A, GC))
            if verbose:
                print(f"  component {idx[k] + 1:3d}  sigma0={etm0:.4g}")

        self.L, self.Cm = L, Cm
        return L

    def misfit_check(self, G: np.ndarray | None = None) -> np.ndarray:
        """Fractional data misfit per inverted component.

        Compare these numbers with the minima of ``misfit_comps.mat``.  If they
        agree to a few percent, the Green's functions, the local projection and
        the prior have all been reproduced correctly.  If they do not, stop and
        debug before computing any strain -- a sign error in the strike-slip
        convention shows up here and nowhere else.
        """
        if self.L is None:
            self.invert(G=G, verbose=False)
        if G is None:
            G = self.displacement_gf()
        idx = self.component_index
        out = np.empty(idx.size)
        for k in range(idx.size):
            d = self.U[:, idx[k]]
            out[k] = np.linalg.norm(G @ self.L[:, k] - d) / np.linalg.norm(d)
        return out

    # ------------------------------------------------------------------ slip
    @property
    def data_unit_to_m(self) -> float:
        """Factor converting the GNSS data unit (``X['unit']``) to metres.

        The Green's functions are dimensionless (displacement per unit slip),
        so ``L`` from the inversion carries the *data* unit. The published
        files store positions in mm; the synthetic fixture is in metres.
        """
        if "unit" not in self.X:
            return 1.0  # synthetic fixtures are built in metres
        u = str(self.X["unit"]).strip().lower()
        try:
            return {"mm": 1e-3, "cm": 1e-2, "m": 1.0}[u]
        except KeyError:
            raise ValueError(
                f"unrecognised data unit {self.X['unit']!r} in X['unit']; "
                "refusing to guess a slip unit conversion (a wrong factor "
                "is silently 1000x off)") from None

    def component_slip_patterns(self) -> tuple[np.ndarray, np.ndarray]:
        """Static slip pattern of each retained component.

        Returns ``(L_strike, L_dip)``, each ``(n_patch, K)`` in metres per unit
        amplitude of the corresponding time function.
        """
        if self.L is None:
            raise RuntimeError("call Solution.invert() first")
        n_p = self.fault.n_patch
        f = self.data_unit_to_m
        return f * self.L[:n_p, :], f * self.L[n_p:, :]

    def component_amplitudes(self) -> np.ndarray:
        """``(n_time, K)`` amplitude time functions, i.e. ``V[:, idx] * S[idx]``.

        Slip on patch *p* at epoch *t* is
        ``L_strike[p, k] * amplitude[t, k]`` summed over *k*.
        """
        idx = self.component_index
        return self.V[:, idx] * self.S[idx][None, :]

    def slip(self) -> tuple[np.ndarray, np.ndarray]:
        """Full ``(n_patch, n_time)`` strike and dip slip, in metres.

        Materialising this costs ``n_patch * n_time * 8`` bytes per component
        direction, which for a few thousand patches and 7000 epochs is tens of
        gigabytes.  Prefer the component form for anything large.
        """
        Ls, Ld = self.component_slip_patterns()
        A = self.component_amplitudes()
        return Ls @ A.T, Ld @ A.T

    def slip_magnitude(self, apply_rake_flip: bool = True) -> np.ndarray:
        """Scalar slip ``sqrt(ss^2 + ds^2)`` with the published rake flip."""
        ss, ds = self.slip()
        mag = np.hypot(ss, ds)
        if apply_rake_flip:
            rake_pos = float(np.ravel(
                _get(self.options, "inversion", "rake_pos", default=90.0))[0])
            rake = np.degrees(np.arctan2(ds, ss))
            flip = (rake < rake_pos - 90) | (rake > rake_pos + 90)
            mag[flip] *= -1.0
        return mag

    # ------------------------------------------------------------- rate model
    def smoothed_amplitudes(self, cutoff_per_day: float = 1.0 / 7.0,
                            n_taps: int = 365, causal: bool = False):
        """Low-pass the component time functions, following ``create_xsmooth``.

        Gualandi (2025) uses a FIR filter designed with a Hamming window of 365
        samples and a cutoff of 1/7 per day, applied non-causally with the last
        value removed and added back.

        FLAGGED: the Julia code calls ``Lowpass(Wn)`` without passing ``fs``,
        so ``Wn`` is interpreted as a fraction of Nyquist rather than as a
        frequency in per-day units.  Here ``cutoff_per_day`` is a genuine
        frequency and is converted internally (Nyquist = 0.5/day for daily
        sampling).  If you need bit-level agreement with his published rate,
        check which interpretation his ``options.smooth.filter_cutofffreq``
        actually carries.
        """
        from scipy.signal import filtfilt, firwin

        A = self.component_amplitudes()
        wn = cutoff_per_day / 0.5
        b = firwin(n_taps, wn, window="hamming")
        out = np.empty_like(A)
        for k in range(A.shape[1]):
            x = A[:, k]
            if causal:
                out[:, k] = np.convolve(x, b, mode="same")
            else:
                x_tr = x[-1]
                out[:, k] = x_tr + filtfilt(b, [1.0], x - x_tr)
        return out

    def amplitude_rates(self, window: int = 7, **kw):
        """Amplitude time derivative by rolling linear fit (``calc_derivative``).

        Returns ``(rates, timeline_centred)`` with ``rates`` in units of
        amplitude per year (the timeline is in decimal years).
        """
        A = self.smoothed_amplitudes(**kw)
        t = self.timeline
        n_t, K = A.shape
        n_out = n_t - window + 1
        rates = np.empty((n_out, K))
        for i in range(n_out):
            tt = t[i:i + window] - t[i]
            Tm = np.column_stack([np.ones(window), tt])
            coef, *_ = np.linalg.lstsq(Tm, A[i:i + window, :], rcond=None)
            rates[i, :] = coef[1, :]
        half = (window - 1) // 2
        return rates, t[half:half + n_out]


# ------------------------------------------------------------------ helpers
def decimal_year_to_datetime64(t):
    """Decimal year -> ``datetime64[D]`` (leap years handled exactly)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    year = np.floor(t).astype(int)
    frac = t - year
    starts = np.array([f"{y:04d}-01-01" for y in year], dtype="datetime64[D]")
    ends = np.array([f"{y + 1:04d}-01-01" for y in year], dtype="datetime64[D]")
    ndays = (ends - starts).astype(int)
    return starts + np.round(frac * ndays).astype("timedelta64[D]")


def load_solution(directory, fault: FaultMesh, strict: bool = False) -> Solution:
    """Read one dated solution directory.

    ``strict=True`` raises if any expected file is missing; otherwise missing
    optional files (``ind_comps``, ``misfit_comps``) are tolerated and the
    component selection falls back to every component in the decomposition.
    """
    directory = Path(directory)
    got = {}
    for key, (fname, varname) in _FILES.items():
        p = directory / fname
        if not p.exists():
            if strict or key in ("options", "X", "ICA"):
                raise FileNotFoundError(
                    f"{p} not found. Expected files: "
                    + ", ".join(f for f, _ in _FILES.values())
                )
            got[key] = None
            continue
        raw = load_mat(p)
        got[key] = raw.get(varname, raw)

    ICA = got["ICA"]
    X = got["X"]
    for fld in ("llh", "timeline", "name"):
        if fld in X and fld not in ICA:
            ICA[fld] = X[fld]

    ind = got["ind_comps"]
    if ind is None:
        n_comp = np.asarray(ICA["U"]).shape[1]
        ind = np.arange(1, n_comp + 1)

    return Solution(
        directory=directory,
        options=got["options"],
        X=X,
        ICA=ICA,
        fault=fault,
        ind_comps=np.asarray(ind),
        misfit_comps=got["misfit_comps"],
    )
