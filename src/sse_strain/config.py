"""Project-level defaults.

Values here are properties of *this* comparison, not of the method, so they
live in one place and get written into output metadata rather than being
retyped at each call site.
"""

from __future__ import annotations

#: Averaging window of the coda-wave dv/v product, in days.
#: 31 is odd, so the average is exactly centred on its sample and carries no
#: half-day shift.  Every strain series compared against dv/v must go through
#: the same window (see :mod:`sse_strain.smoothing`).
DVV_WINDOW_DAYS = 31

#: Samples blanked at each end by the centred average with strict validity.
DVV_EDGE_LOSS_DAYS = (DVV_WINDOW_DAYS - 1) // 2  # 15

#: Reference epoch of the dv/v stack: the last year of the stack, chosen
#: because station coverage is densest there.
#: Resolve it against an actual series with
#: ``smoothing.reference_from_end(times, DVV_REFERENCE_YEARS,
#: DVV_EDGE_LOSS_DAYS)``.
DVV_REFERENCE_YEARS = 1.0

#: FLAGGED: a one-year reference does not average over a whole ETS cycle
#: everywhere. Cascadia recurrence runs about 8-22 months along strike, so the
#: residual saw-tooth offset ranges from zero near a 12-month recurrence to
#: roughly 22 % of peak-to-peak at 22 months. Fit an intercept rather than
#: forcing a regression through the origin; see
#: ``smoothing.saw_tooth_reference_bias``.
DVV_REFERENCE_IS_FULL_CYCLE = False

#: Poisson's ratio of the half-space used by Gualandi (2025). Overridden at
#: run time by ``Solution.nu`` read from ``options.mat``.
DEFAULT_NU = 0.25
