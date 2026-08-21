"""Observation grids.

A note on resolution.  The strain field at depth *z* produced by slip at depth
*h* has a characteristic horizontal wavelength of order *h*.  For Cascadia
slow slip at 30-45 km, that means the field varies on a 30-50 km scale, and a
grid much finer than ~10 km buys resolution the physics does not contain while
costing quadratically in Green's function evaluations.  ``spacing_deg=0.1``
(roughly 8-11 km here) is already generous; 0.05 is close to wasted effort.

The same argument sets the scale on which coda-wave measurements should be
binned before the comparison: station pairs whose sensitivity kernels fall
inside one strain wavelength are not independent samples of the strain field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geodesy import llh2localxy

__all__ = ["ObsGrid", "make_grid"]


@dataclass
class ObsGrid:
    """A regular lon/lat/depth grid with its local Cartesian counterpart."""

    lon: np.ndarray          # (n_lon,)
    lat: np.ndarray          # (n_lat,)
    depth_m: np.ndarray      # (n_depth,) positive downward
    points_m: np.ndarray     # (n_obs, 3) East-North-Up metres, z <= 0
    shape: tuple             # (n_depth, n_lat, n_lon)
    origin: tuple

    @property
    def n_obs(self) -> int:
        return self.points_m.shape[0]

    def reshape(self, arr, trailing=()):
        """``(n_obs, ...)`` -> ``(n_depth, n_lat, n_lon, ...)``."""
        return np.asarray(arr).reshape(self.shape + tuple(trailing))

    def depth_of_each_point(self) -> np.ndarray:
        """(n_obs,) depth in metres, positive down, matching ``points_m``."""
        return -self.points_m[:, 2]


def make_grid(lon_range=(-127.0, -120.5), lat_range=(39.5, 50.5),
              spacing_deg=0.1, depths_km=(0.5, 2.0, 5.0), origin=None):
    """Build an :class:`ObsGrid`.

    Parameters
    ----------
    lon_range, lat_range : (min, max) in degrees
    spacing_deg : float
        Grid spacing, applied to both longitude and latitude.  The cells are
        therefore not square in km; that is fine for the fields here, which are
        smooth on scales far larger than the anisotropy this introduces.
    depths_km : sequence
        Depths, positive downward.  Pick these to bracket the depth range your
        coda-wave sensitivity kernels actually sample.
    origin : (lon0, lat0) or None
        Local frame origin.  **Pass the fault mesh origin** (``fault.origin``)
        so that the grid and the fault share a frame.  Leaving it None centres
        the frame on the grid, which silently shifts everything.
    """
    lon = np.arange(lon_range[0], lon_range[1] + 1e-9, spacing_deg)
    lat = np.arange(lat_range[0], lat_range[1] + 1e-9, spacing_deg)
    depth_m = np.atleast_1d(np.asarray(depths_km, dtype=float)) * 1e3

    if origin is None:
        origin = (float(lon.mean()), float(lat.mean()))

    LO, LA = np.meshgrid(lon, lat, indexing="xy")
    x_km, y_km = llh2localxy(LA.ravel(), LO.ravel(), origin)

    n_h = x_km.size
    n_d = depth_m.size
    pts = np.empty((n_d * n_h, 3))
    for i, d in enumerate(depth_m):
        sl = slice(i * n_h, (i + 1) * n_h)
        pts[sl, 0] = x_km * 1e3
        pts[sl, 1] = y_km * 1e3
        pts[sl, 2] = -d
    return ObsGrid(lon=lon, lat=lat, depth_m=depth_m, points_m=pts,
                   shape=(n_d, lat.size, lon.size), origin=tuple(origin))
