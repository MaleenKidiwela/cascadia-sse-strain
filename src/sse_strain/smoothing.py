r"""Matching the strain model to the dv/v measurement operator.

The coda-wave dv/v series is sampled daily but each sample is an average over a
window of about a month centred on that day.  That window is part of the
measurement, not a nuisance to be removed, so the strain model has to be put
through the same operator before the two are compared.  Comparing an unsmoothed
strain field against a monthly-averaged dv/v attributes the filter's effect to
the physics.

Why this is cheap
-----------------
Strain is linear in the component amplitudes (:mod:`sse_strain.solution`):

.. math::
   \varepsilon_{ij}(\mathbf{x}, t) = \sum_k E^{(k)}_{ij}(\mathbf{x})\, a_k(t)

A linear time-invariant operator :math:`W` acting on :math:`t` therefore
commutes with the spatial sum:

.. math::
   W[\varepsilon_{ij}](\mathbf{x}, t) = \sum_k E^{(k)}_{ij}(\mathbf{x})\, W[a_k](t)

so smoothing the K scalar amplitude functions gives exactly the smoothed strain
field, with no need to touch the Green's functions.  Pass the result to
``dataset.evaluate(..., amplitudes=...)``.

Cumulative or rate?
-------------------
dv/v is a *state* variable: a fractional velocity change relative to a
reference stack.  Its natural counterpart is cumulative strain relative to the
same reference epoch, not strain rate.  That pairing also happens to be the
robust one under a boxcar:

* A step in cumulative strain (an SSE) becomes a 30-day ramp.  The amplitude of
  the step is preserved exactly; only its timing is smeared.
* A pulse in strain *rate* of duration :math:`T` is attenuated by roughly
  :math:`T/W` once :math:`T < W`.  For a three-week episode under a 30-day
  window that is a ~30 % amplitude loss before any physics enters, and the loss
  varies with episode duration, so a moment-duration analysis done on smoothed
  rates is biased.

Use :func:`reference_to_epoch` to put the strain on the same reference as the
dv/v, then smooth.

Degrees of freedom
------------------
Neighbouring daily samples of a W-day moving average share :math:`W-1` of their
W inputs.  For white input the autocorrelation of the output is the triangular
kernel :math:`1 - |k|/W`, and the variance of a correlation coefficient
computed from N such samples is inflated by a factor of about
:math:`(2W^2+1)/(3W)`, so the effective sample size is roughly :math:`3N/(2W)`.
Fifteen years of daily samples under a 30-day window is of order 270
independent observations, not 5500.  :func:`effective_sample_size` computes
this properly from the empirical autocorrelations of the two series being
correlated (Bartlett's formula), which is what should go into any significance
statement.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "centered_moving_average", "smooth_amplitudes", "boxcar_response",
    "attenuation_of_pulse", "effective_sample_size", "reference_to_epoch",
    "reference_from_end", "saw_tooth_reference_bias",
]


def centered_moving_average(x, window: int, axis: int = 0,
                            min_periods: int | None = None):
    """Centred moving average with an explicit validity mask.

    Parameters
    ----------
    x : ndarray
        Input, smoothed along ``axis``.
    window : int
        Window length in samples.  An even window cannot be exactly centred on
        a sample; the result is then shifted by half a sample and a warning-free
        note is left in the returned ``info``.  Prefer an odd window (31 rather
        than 30) if the half-day matters, which for slow slip it does not.
    min_periods : int, optional
        Minimum number of valid input samples required to produce an output.
        Defaults to ``window`` (strict), which blanks ``(window-1)//2`` samples
        at each end.  Set it lower to taper into the edges, at the cost of a
        changing effective window there.

    Returns
    -------
    smoothed : ndarray, same shape as ``x``, NaN where undefined
    valid : bool ndarray, same shape
    info : dict with the realised window, edge loss and half-sample shift
    """
    x = np.asarray(x, dtype=float)
    if window < 1:
        raise ValueError("window must be >= 1")
    if min_periods is None:
        min_periods = window

    xm = np.moveaxis(x, axis, 0)
    n = xm.shape[0]
    if window > n:
        raise ValueError(f"window {window} exceeds series length {n}")

    finite = np.isfinite(xm)
    filled = np.where(finite, xm, 0.0)

    csum = np.concatenate([np.zeros((1,) + xm.shape[1:]), filled.cumsum(0)], 0)
    ccnt = np.concatenate([np.zeros((1,) + xm.shape[1:]),
                           finite.cumsum(0).astype(float)], 0)

    half_lo = (window - 1) // 2
    half_hi = window - 1 - half_lo

    out = np.full_like(xm, np.nan)
    cnt = np.zeros_like(xm, dtype=float)
    for i in range(n):
        a = max(0, i - half_lo)
        b = min(n, i + half_hi + 1)
        cnt[i] = ccnt[b] - ccnt[a]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i] = (csum[b] - csum[a]) / np.where(cnt[i] > 0, cnt[i], np.nan)

    valid = cnt >= min_periods
    out = np.where(valid, out, np.nan)

    info = {
        "window": window,
        "half_sample_shift": bool(window % 2 == 0),
        "edge_samples_lost_each_side": int(np.ceil((min_periods - 1) / 2)),
        "min_periods": min_periods,
    }
    return (np.moveaxis(out, 0, axis), np.moveaxis(valid, 0, axis), info)


def smooth_amplitudes(solution, window: int = 31, min_periods: int | None = None,
                      cumulative: bool = True, reference=None):
    """Amplitude functions put through the dv/v measurement operator.

    Parameters
    ----------
    solution : Solution
    window : int
        Averaging window in days, matching the dv/v processing.
    cumulative : bool
        Leave True to compare against dv/v as a state variable.  Set False only
        if your dv/v product is itself a rate.
    reference : str, np.datetime64, or (start, end), optional
        Reference epoch or interval for the dv/v stack.  The amplitudes are
        offset so that the mean over that interval is zero, which puts the
        strain on the same baseline as dv/v.  Without this the two series
        differ by an arbitrary constant and any regression through the origin
        is meaningless.

    Returns
    -------
    amplitudes, times, info
    """
    A = solution.component_amplitudes()
    t = solution.dates

    if reference is not None:
        A = reference_to_epoch(A, t, reference)

    if not cumulative:
        A = np.gradient(A, axis=0)

    As, valid, info = centered_moving_average(A, window,
                                              min_periods=min_periods)
    keep = valid.all(axis=1)
    info = dict(info, kept=int(keep.sum()), total=int(keep.size),
                cumulative=cumulative,
                effective_sample_size=float(3 * keep.sum() / (2 * window)))
    return As[keep], t[keep], info


def reference_to_epoch(A, times, reference):
    """Subtract the mean over a reference epoch or interval.

    ``reference`` may be a single date (nearest sample) or a ``(start, end)``
    pair.  Matching the dv/v reference stack here is what makes the zero of the
    strain series meaningful.
    """
    times = np.asarray(times)
    if isinstance(reference, (tuple, list)) and len(reference) == 2:
        t0, t1 = np.datetime64(reference[0]), np.datetime64(reference[1])
        sel = (times >= t0) & (times <= t1)
        if not sel.any():
            raise ValueError(f"no samples in reference interval {reference}")
        base = np.asarray(A)[sel].mean(axis=0)
    else:
        i = int(np.argmin(np.abs(times - np.datetime64(reference))))
        base = np.asarray(A)[i]
    return np.asarray(A) - base


def boxcar_response(window: int, freq_per_day=None):
    """Amplitude transfer function of the centred moving average.

    ``|H(f)| = |sin(pi f W) / (W sin(pi f))|`` for unit sampling.  Zeros sit at
    ``f = m/W``; between them the sidelobes alternate in sign, so a signal whose
    period falls in a sidelobe comes through inverted.  For a 30-day window the
    first zero is at a 30-day period and the first sidelobe peaks near a 20-day
    period at about 21 % amplitude, with negative sign.  A slow-slip episode of
    that duration is therefore not merely attenuated in a rate comparison; it
    can change sign.

    Returns ``(freq_per_day, H)`` with ``H`` signed.
    """
    if freq_per_day is None:
        freq_per_day = np.linspace(1e-4, 0.5, 2000)
    f = np.asarray(freq_per_day, dtype=float)
    num = np.sin(np.pi * f * window)
    den = window * np.sin(np.pi * f)
    H = np.where(np.abs(den) < 1e-15, 1.0, num / den)
    return f, H


def attenuation_of_pulse(duration_days: float, window: int) -> float:
    """Peak amplitude retained by a rectangular rate pulse under the average.

    A pulse of duration T through a W-day boxcar keeps a peak fraction
    ``min(T, W) / W``.  Use this to decide whether a rate-space comparison is
    viable for your episode durations, or whether to work in cumulative space.
    """
    return float(min(duration_days, window)) / float(window)


def effective_sample_size(x, y=None, max_lag: int | None = None) -> float:
    """Effective number of independent samples, via Bartlett's formula.

    For correlating two autocorrelated series,

    .. math::
       N_{eff} \\approx \\frac{N}{1 + 2\\sum_k \\rho_x(k)\\,\\rho_y(k)}

    Pass both series.  With one series the same expression is used with
    ``rho_y = rho_x``, giving the effective size for estimating its mean.

    Feed ``N_eff - 2`` to any t-test on a correlation coefficient.  Using the
    raw daily count instead will make almost any correlation look significant.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if max_lag is None:
        max_lag = max(1, n // 4)

    def acf(v):
        v = v - v.mean()
        denom = np.dot(v, v)
        if denom == 0:
            return np.zeros(max_lag + 1)
        return np.array([np.dot(v[:n - k], v[k:]) / denom
                         for k in range(max_lag + 1)])

    rx = acf(x)
    ry = rx if y is None else acf(np.asarray(y, dtype=float).ravel()[:n])
    s = 1.0 + 2.0 * np.sum(rx[1:] * ry[1:])
    return float(n / max(s, 1.0))


def reference_from_end(times, years: float = 1.0, edge_loss_days: int = 0):
    """The last ``years`` of a series, as a ``(start, end)`` interval.

    Parameters
    ----------
    times : array of datetime64
    years : float
    edge_loss_days : int
        Samples the centred average blanks at the end.  Pass
        ``config.DVV_EDGE_LOSS_DAYS`` so the reference interval sits inside the
        *smoothed* series rather than running off its end; otherwise the last
        ``edge_loss_days`` of the nominal interval contain NaN and the realised
        reference is quietly shorter than requested.
    """
    times = np.asarray(times)
    end = times.max() - np.timedelta64(int(edge_loss_days), "D")
    start = end - np.timedelta64(int(round(years * 365.25)), "D")
    if start < times.min():
        raise ValueError(
            f"series spans {times.min()} to {times.max()}, too short for a "
            f"{years}-year reference ending {end}"
        )
    return (start, end)


def saw_tooth_reference_bias(recurrence_days, reference_days: float = 365.25):
    """Residual baseline left by a finite reference window over a cyclic signal.

    Detrended cumulative slip on a Cascadia patch is close to a zero-mean
    saw-tooth: slow interseismic loading, then a rapid slip step.  Averaging
    over exactly one recurrence interval removes it; averaging over any other
    length leaves a phase-dependent offset that is absorbed into the baseline
    and shows up as a spurious constant difference between the strain and dv/v
    series.

    Returns the worst-case offset as a fraction of the peak-to-peak amplitude.
    For a one-year reference this is zero at a 12-month recurrence and grows to
    roughly 0.22 at 22 months, which is the northern-Cascadia end of the
    observed range (Bartlow, 2020; Michel et al., 2019a report 8-22 months
    along strike).

    This is a systematic offset in the *baseline*, so it biases any regression
    forced through the origin.  Fitting an intercept absorbs it; that is the
    recommended handling, with this function used to check the offset is small
    relative to the transients being interpreted.
    """
    T = np.atleast_1d(np.asarray(recurrence_days, dtype=float))
    R = float(reference_days)
    out = np.empty(T.shape)
    for i, Ti in enumerate(T.ravel()):
        dt = Ti / 4000.0
        t = np.arange(0.0, Ti, dt)
        saw = t / Ti
        saw -= saw.mean()
        n = max(1, int(round(R / dt)))
        reps = int(np.ceil((t.size + n) / t.size)) + 1
        k = np.ones(n) / n
        ma = np.convolve(np.tile(saw, reps), k, mode="valid")[:t.size]
        out.ravel()[i] = np.abs(ma).max()
    return out if out.size > 1 else float(out.ravel()[0])
