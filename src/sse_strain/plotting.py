"""Figures.

Deliberately plain matplotlib with no basemap dependency, so the scripts run
anywhere.  The one figure worth getting right is the latitude-time map, since
it is the standard way Cascadia slow slip is displayed (Gualandi 2025, figure
7; Michel et al. 2019a) and it is what makes a strain field visually
comparable to a dv/v section.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

__all__ = ["latitude_time_map", "map_snapshot", "depth_profile",
           "component_gallery", "along_strike_profile"]


def _sym_norm(a, q=99.0):
    v = np.nanpercentile(np.abs(a), q)
    v = v if v > 0 else 1.0
    return TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)


def latitude_time_map(times, lat, field, ax=None, cmap="RdBu_r",
                      label="", title="", tremor=None, q=99.0):
    """Latitude vs time image of a quantity already reduced over longitude.

    ``field`` has shape ``(n_time, n_lat)``.  Reduce with the *signed* extremum
    rather than the mean if you want Gualandi-style panels: taking the value of
    largest magnitude at each latitude preserves the along-strike migration
    that averaging smears out.

    ``tremor`` may be an ``(n, 2)`` array of (decimal-year or datetime64,
    latitude) to overplot, for instance the PNSN catalog.
    """
    ax = ax or plt.subplots(figsize=(11, 4.5))[1]
    im = ax.pcolormesh(times, lat, field.T, cmap=cmap,
                       norm=_sym_norm(field, q), shading="auto")
    if tremor is not None and len(tremor):
        ax.plot(tremor[:, 0], tremor[:, 1], ".", ms=0.6, color="k", alpha=0.35)
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(title)
    cb = plt.colorbar(im, ax=ax, pad=0.01)
    cb.set_label(label)
    return ax


def signed_extremum(da, dim):
    """Value of largest absolute magnitude along ``dim``, keeping its sign.

    Averaging along strike smears the along-strike migration that makes these
    panels readable, so the extremum is taken instead.  Slices that are
    entirely NaN (outside the resolved interface) return NaN rather than
    raising.
    """
    a = np.asarray(da.values, dtype=float)
    ax = da.dims.index(dim)
    absa = np.abs(a)
    allnan = np.all(~np.isfinite(absa), axis=ax)
    filled = np.where(np.isfinite(absa), absa, -np.inf)
    i = np.argmax(filled, axis=ax)
    out = np.take_along_axis(a, np.expand_dims(i, ax), axis=ax).squeeze(ax)
    return np.where(allnan, np.nan, out)


def map_snapshot(ds, var, time, depth=None, ax=None, cmap="RdBu_r",
                 fault=None, q=99.0):
    """Map view of one variable at one epoch and depth."""
    ax = ax or plt.subplots(figsize=(5.5, 8))[1]
    sl = ds[var].sel(time=time, method="nearest")
    if "depth" in sl.dims:
        sl = sl.isel(depth=0) if depth is None else sl.sel(depth=depth,
                                                           method="nearest")
    im = ax.pcolormesh(ds.lon, ds.lat, sl.values, cmap=cmap,
                       norm=_sym_norm(sl.values, q), shading="auto")
    if fault is not None:
        ax.plot(fault.centroid_llh[:, 0], fault.centroid_llh[:, 1], ".",
                ms=0.8, color="0.35", alpha=0.5)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_aspect(1.0 / np.cos(np.deg2rad(float(ds.lat.mean()))))
    ttl = str(np.datetime_as_string(np.datetime64(sl.time.values), unit="D"))
    ax.set_title(f"{var}  {ttl}")
    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(ds[var].attrs.get("units", ""))
    return ax


def depth_profile(ds, var, lon, lat, ax=None):
    """Time-depth section at one location: the view that matters for dv/v.

    Coda-wave sensitivity spans a depth range rather than a surface, so the
    question of which depth to compare against is answered by looking at how
    fast the field decays over the sampled interval.
    """
    ax = ax or plt.subplots(figsize=(10, 3.5))[1]
    sl = ds[var].sel(lon=lon, lat=lat, method="nearest")
    im = ax.pcolormesh(ds.time, ds.depth, sl.T.values, cmap="RdBu_r",
                       norm=_sym_norm(sl.values), shading="auto")
    ax.invert_yaxis()
    ax.set_ylabel("Depth (km)")
    ax.set_title(f"{var} at {float(sl.lon):.2f}°E, {float(sl.lat):.2f}°N")
    plt.colorbar(im, ax=ax, pad=0.01).set_label(ds[var].attrs.get("units", ""))
    return ax


def component_gallery(solution, ncol=4, figsize=(13, 8)):
    """Amplitude time function of every retained component.

    Worth looking at before trusting anything: components that are seasonal
    leakage or unmodelled postseismic relaxation are obvious here, and if one
    of them is carrying a large share of the strain the result is not slow slip.
    """
    A = solution.component_amplitudes()
    t = solution.dates
    idx = solution.component_index + 1
    K = A.shape[1]
    nrow = int(np.ceil(K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharex=True)
    for k, ax in enumerate(np.atleast_1d(axes).ravel()):
        if k >= K:
            ax.axis("off")
            continue
        ax.plot(t, A[:, k], lw=0.7)
        ax.set_title(f"IC {idx[k]}", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def along_strike_profile(ds, var, target_depth_km=35.0, depth_index=0):
    """Latitude-time section following a fixed interface depth contour.

    A signed extremum over longitude is the wrong reduction for a *tensor*
    component.  Components such as eps_EN have positive and negative lobes on
    either side of the source, so the extremum jumps between lobes as an
    episode migrates and the panel fills with vertical stripes that are an
    artefact of the reduction.  It works for a scalar of one sign, such as slip
    magnitude, which is why it is the standard choice for slip panels.

    Following a fixed interface depth instead traces the along-strike band
    where slow slip occurs, keeps the down-dip position constant, and preserves
    the sign.

    Selection is by linear interpolation along longitude to the exact depth
    contour.  Taking the nearest grid cell instead makes the chosen longitude
    jump by one pixel between adjacent latitudes, which stripes the panel in
    the latitude direction for the components with strong across-strike
    gradients.

    Returns ``(field, lat, lon_used)`` with ``field`` of shape
    ``(n_time, n_lat)``.
    """
    iface = ds["interface_depth"].values
    n_lat, n_lon = iface.shape
    da = ds[var]
    if "depth" in da.dims:
        da = da.isel(depth=depth_index)
    a = da.values                                    # (time, lat, lon)

    out = np.full((a.shape[0], n_lat), np.nan)
    lon = ds.lon.values
    lon_used = np.full(n_lat, np.nan)

    for i in range(n_lat):
        row = iface[i]
        ok = np.isfinite(row) & np.isfinite(a[0, i])
        if ok.sum() < 2:
            continue
        idx = np.flatnonzero(ok)
        d = row[idx]
        order = np.argsort(d)
        d_s, idx_s = d[order], idx[order]
        if not (d_s[0] <= target_depth_km <= d_s[-1]):
            continue
        j = int(np.searchsorted(d_s, target_depth_km))
        j = min(max(j, 1), d_s.size - 1)
        j0, j1 = idx_s[j - 1], idx_s[j]
        d0, d1 = d_s[j - 1], d_s[j]
        w = 0.0 if d1 == d0 else (target_depth_km - d0) / (d1 - d0)
        out[:, i] = (1 - w) * a[:, i, j0] + w * a[:, i, j1]
        lon_used[i] = (1 - w) * lon[j0] + w * lon[j1]
    return out, ds.lat.values, lon_used
