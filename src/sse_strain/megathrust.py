r"""Observation surfaces that follow the megathrust.

Evaluating strain *on* the slipping interface needs care.  The Nikkhoo & Walter
solution is exact but the strain field of a dislocation diverges as :math:`1/r`
approaching the fault plane and as :math:`1/r` along each triangle edge, so the
value returned exactly on a slipping patch is meaningless: it is dominated by
the artificial edges of the discretisation, not by the physics.  A patch mesh
with a 20 km characteristic dimension cannot represent strain closer than a few
kilometres to itself.

The approach here is to evaluate on a surface offset from the interface by a
distance comparable to the patch scale, and to make the offset an explicit
parameter with a convergence diagnostic
(:func:`offset_sensitivity`) rather than a hidden constant.  Offsetting into the
hanging wall is the choice that matters physically: that is where the forearc
crust, the seismicity and the seismic stations are.

Two surfaces are useful and the same code builds both:

* ``make_slab_grid(..., offset_km=3.0)`` -- a regular lon/lat grid draped a
  fixed distance above the interface, following the slab down-dip.  This is the
  megathrust view.
* ``grid.make_grid(depths_km=(1., 3., 5.))`` -- flat constant-depth levels in
  the upper crust.  This is the view to compare against coda-wave dv/v, whose
  sensitivity kernels sit at a few kilometres depth regardless of what the slab
  is doing beneath.

The offset is applied vertically rather than along the local fault normal.  For
a plane dipping at :math:`\delta`, a vertical offset :math:`h` corresponds to a
perpendicular distance :math:`h\cos\delta`; Cascadia dips are 10-20 deg in the
slow-slip band, so the two differ by under 6 % and the vertical form has the
advantage that the resulting surface is single-valued on a lon/lat grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geodesy import llh2localxy
from .grid import ObsGrid

__all__ = ["slab_depth_interpolator", "make_slab_grid",
           "resolution_test", "resolution_filter"]


def slab_depth_interpolator(fault):
    """Interpolator returning interface depth (km, positive down) at lon/lat.

    Built from the patch centroids by linear interpolation on their Delaunay
    triangulation.  Returns NaN outside the convex hull of the mesh, which is
    the honest answer: there is no interface there to speak of.
    """
    from scipy.interpolate import LinearNDInterpolator

    pts = fault.centroid_llh[:, :2]
    depth_km = -fault.centroid_llh[:, 2]
    return LinearNDInterpolator(pts, depth_km)


def make_slab_grid(fault, lon_range=None, lat_range=None, spacing_deg=0.15,
                   offset_km=3.0, depth_limits_km=(20.0, 55.0), origin=None):
    """A regular lon/lat grid draped a fixed height above the megathrust.

    Parameters
    ----------
    fault : FaultMesh
    lon_range, lat_range : (min, max), optional
        Default to the bounding box of the mesh.
    spacing_deg : float
    offset_km : float
        Vertical height above the interface at which strain is evaluated.
        Must be comparable to or larger than the patch dimension; below that
        the result is controlled by the discretisation.  Check with
        :func:`offset_sensitivity`.
    depth_limits_km : (min, max)
        Restrict to the depth band where slow slip occurs.  Points where the
        interpolated interface falls outside this band are masked, which keeps
        the near-trench and deep-continuation parts of the mesh, where the
        inversion has no resolution, out of the figures.
    origin : (lon0, lat0), optional
        Defaults to ``fault.origin``, which is what you want.

    Returns
    -------
    ObsGrid with ``shape = (1, n_lat, n_lon)``, plus attributes
    ``interface_depth_km`` and ``valid`` on the same 2-D grid.
    """
    if origin is None:
        origin = fault.origin
    if lon_range is None:
        lon_range = (float(fault.lon.min()), float(fault.lon.max()))
    if lat_range is None:
        lat_range = (float(fault.lat.min()), float(fault.lat.max()))

    lon = np.arange(lon_range[0], lon_range[1] + 1e-9, spacing_deg)
    lat = np.arange(lat_range[0], lat_range[1] + 1e-9, spacing_deg)
    LO, LA = np.meshgrid(lon, lat, indexing="xy")

    interp = slab_depth_interpolator(fault)
    d_iface = interp(LO, LA)                       # km, positive down

    valid = np.isfinite(d_iface)
    if depth_limits_km is not None:
        valid &= (d_iface >= depth_limits_km[0]) & (d_iface <= depth_limits_km[1])

    d_obs = d_iface - offset_km                    # above the interface
    d_obs = np.where(valid, d_obs, np.nan)

    x_km, y_km = llh2localxy(LA.ravel(), LO.ravel(), origin)
    z = -np.nan_to_num(d_obs.ravel(), nan=0.0) * 1e3
    z = np.minimum(z, 0.0)
    pts = np.column_stack([x_km * 1e3, y_km * 1e3, z])

    g = ObsGrid(lon=lon, lat=lat,
                depth_m=np.array([np.nanmean(d_obs) * 1e3]),
                points_m=pts, shape=(1, lat.size, lon.size),
                origin=tuple(origin))
    g.interface_depth_km = d_iface
    g.observation_depth_km = d_obs
    g.valid = valid
    g.offset_km = offset_km
    return g


def resolution_filter(field, grid, length_km, lat=None, lon=None):
    """Spatial low-pass at the resolution length of the inversion.

    The slip model is piecewise constant on ~20 km triangles, and the prior
    correlation length (``options.inversion.lambda_*``, 21 km for Cascadia) says
    that nothing finer than that is resolved.  Evaluated a few kilometres above
    the interface, the strain field nonetheless shows structure at the patch
    scale, because every patch boundary carries a slip discontinuity of a few
    percent of peak.  That structure is a property of the parameterisation.

    This applies a Gaussian low-pass with ``sigma = length_km / 2``, ignoring
    NaN cells by normalised convolution so the mask edge does not bleed inward.

    ``field`` may be ``(..., n_lat, n_lon)``; the last two axes are filtered.
    """
    from scipy.ndimage import gaussian_filter

    lat = grid.lat if lat is None else lat
    lon = grid.lon if lon is None else lon
    dlat_km = 111.19 * float(np.mean(np.diff(lat)))
    dlon_km = 111.19 * float(np.mean(np.diff(lon))) * \
        float(np.cos(np.deg2rad(np.mean(lat))))
    sigma = (0.5 * length_km / abs(dlat_km), 0.5 * length_km / abs(dlon_km))

    a = np.asarray(field, dtype=float)
    shp = a.shape
    a = a.reshape((-1,) + shp[-2:])
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        w = np.isfinite(a[i]).astype(float)
        v = np.where(w > 0, a[i], 0.0)
        num = gaussian_filter(v, sigma, mode="constant", cval=0.0)
        den = gaussian_filter(w, sigma, mode="constant", cval=0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i] = np.where(den > 1e-6, num / den, np.nan)
        out[i] = np.where(w > 0, out[i], np.nan)
    return out.reshape(shp)


def resolution_test(fault, slip_patterns, nu, offsets_km=(3., 5., 10., 15., 20., 30.),
                    spacing_deg=0.2, depth_limits_km=(25.0, 50.0), component=0):
    """How far above the interface the field stops being parameterisation noise.

    At each offset the dilatation field is computed and its roughness measured
    as the RMS discrete Laplacian normalised by the RMS field.  A field set by
    the source varies on the scale of the source; one contaminated by the slip
    discontinuities between adjacent patches varies on the grid scale, giving a
    large ratio.  The ratio falls and then plateaus once the offset reaches
    roughly the patch dimension.

    Refining the triangles while holding the slip distribution fixed does *not*
    diagnose this, because subdivision leaves the patch-to-patch slip jumps
    exactly where they were.  Distance from the fault, or a spatial low-pass at
    the inversion's correlation length, is what removes them.

    Returns ``(offsets, roughness, rms)``.
    """
    from .green import strain_from_components
    from .strain import dilatation

    sp = np.asarray(slip_patterns)[:, :, component:component + 1]
    tris = fault.vertices_m()
    offsets = np.atleast_1d(np.asarray(offsets_km, dtype=float))
    rough, rms = [], []
    for h in offsets:
        g = make_slab_grid(fault, spacing_deg=spacing_deg, offset_km=float(h),
                           depth_limits_km=depth_limits_km)
        keep = g.valid.ravel()
        pts = np.ascontiguousarray(g.points_m[keep])
        E = strain_from_components(pts, tris, sp, nu, progress=False)
        a = np.full(g.valid.size, np.nan)
        a[keep] = dilatation(E[:, :, 0])
        a = a.reshape(g.valid.shape)
        lap = (4 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1]
               - a[1:-1, :-2] - a[1:-1, 2:])
        rms.append(float(np.nanstd(a)))
        rough.append(float(np.nanstd(lap) / np.nanstd(a)))
    return offsets, np.array(rough), np.array(rms)
