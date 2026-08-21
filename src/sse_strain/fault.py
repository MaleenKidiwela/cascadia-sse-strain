"""Triangular fault mesh handling.

The Gualandi Cascadia mesh is an ASCII file with one row per triangular patch
and nine columns::

    lon1 lat1 h1  lon2 lat2 h2  lon3 lat3 h3

with longitude and latitude in degrees and height in km (negative below sea
level).  The geometry is that of Hayes et al. (2017); the file itself is *not*
part of the daily ``.mat`` download and has to be obtained separately (see
``docs/DATA_SOURCES.md``).

Strike, dip and area are recomputed here rather than read from anywhere,
following ``load_fault`` in ``src_load.jl``, so that the rake convention used
when we rotate strike/dip slip into the East-North-Up frame is the same one the
inversion used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geodesy import llh2local_xyz

__all__ = ["FaultMesh", "load_fault_ascii", "refine_mesh"]


def _plane_from_3points(A, B, C):
    """Plane coefficients a,b,c,d with a*x+b*y+c*z+d = 0, matching plane_3points.m."""
    AB = B - A
    AC = C - A
    n = np.cross(AB, AC)
    a, b, c = n
    d = -np.dot(n, A)
    return a, b, c, d


@dataclass
class FaultMesh:
    """A triangulated fault surface in both geographic and local coordinates.

    Attributes
    ----------
    lon, lat, height : (n_patch, 3) arrays
        Vertex coordinates, degrees and km.
    xE, yN, zV : (n_patch, 3) arrays
        Vertex coordinates in the local frame, km (zV negative downward).
    centroid_llh : (n_patch, 3)
        Patch centroids, (lon, lat, height_km).
    strike, dip : (n_patch,) arrays, degrees
    area : (n_patch,) array, km^2
    origin : (lon0, lat0) of the local frame
    """

    lon: np.ndarray
    lat: np.ndarray
    height: np.ndarray
    xE: np.ndarray
    yN: np.ndarray
    zV: np.ndarray
    centroid_llh: np.ndarray
    strike: np.ndarray
    dip: np.ndarray
    area: np.ndarray
    origin: tuple

    @property
    def n_patch(self) -> int:
        return self.lon.shape[0]

    @property
    def centroid_xyz_km(self) -> np.ndarray:
        return np.column_stack([self.xE.mean(1), self.yN.mean(1), self.zV.mean(1)])

    def vertices_m(self) -> np.ndarray:
        """(n_patch, 3, 3) vertex array in metres, East-North-Up.

        This is the layout ``cutde`` expects for its triangle argument.  The
        vertex ordering is left exactly as it appears in the mesh file; cutde,
        like Nikkhoo & Walter's own code, handles either orientation, but the
        *sign* of the strike-slip and dip-slip components is tied to that
        ordering, so see :meth:`slip_enu` for the rotation we apply instead.
        """
        return np.ascontiguousarray(
            np.stack([self.xE * 1e3, self.yN * 1e3, self.zV * 1e3], axis=-1)
        )

    def slip_enu(self, slip_strike: np.ndarray, slip_dip: np.ndarray,
                 slip_tensile: np.ndarray | None = None) -> np.ndarray:
        """Rotate (strike-slip, dip-slip, tensile) into East-North-Up.

        Parameters
        ----------
        slip_strike, slip_dip : (n_patch, ...) arrays
            Slip components along strike and dip, in metres.  Trailing
            dimensions (time, component index) are carried through.

        Returns
        -------
        (n_patch, 3, ...) array of East, North, Up slip in metres.

        Notes
        -----
        This reproduces ``fault2local.m`` / ``fault2local`` in the Julia code:
        with strike ``phi`` measured clockwise from north and dip ``delta``,
        the along-strike unit vector is ``(sin phi, cos phi, 0)`` and the
        along-dip (down-dip) unit vector is
        ``(cos phi cos delta, -sin phi cos delta, -sin delta)``.
        """
        phi = np.deg2rad(self.strike)
        delta = np.deg2rad(self.dip)
        # broadcast (n_patch,) against the trailing dims of the slip arrays
        extra = slip_strike.ndim - 1
        shp = (-1,) + (1,) * extra
        sp, cp = np.sin(phi).reshape(shp), np.cos(phi).reshape(shp)
        sd, cd = np.sin(delta).reshape(shp), np.cos(delta).reshape(shp)

        e = slip_strike * sp + slip_dip * cp * cd
        n = slip_strike * cp - slip_dip * sp * cd
        u = -slip_dip * sd
        if slip_tensile is not None:
            # tensile opening along the fault normal
            ne = -cp * sd
            nn = sp * sd
            nu = cd
            e = e + slip_tensile * ne
            n = n + slip_tensile * nn
            u = u + slip_tensile * nu
        return np.stack([e, n, u], axis=1)


def load_fault_ascii(path, origin=None, strict_ellipsoid: bool = False) -> FaultMesh:
    """Load the 9-column triangular mesh used by the Gualandi solutions."""
    path = Path(path)
    m = np.loadtxt(path)
    if m.ndim != 2 or m.shape[1] < 9:
        raise ValueError(
            f"{path} has shape {m.shape}; expected (n_patch, 9) with columns "
            "lon1 lat1 h1 lon2 lat2 h2 lon3 lat3 h3"
        )
    lon = m[:, [0, 3, 6]]
    lat = m[:, [1, 4, 7]]
    height = m[:, [2, 5, 8]]

    if origin is None:
        origin = (float(lon.mean()), float(lat.mean()))

    n = lon.shape[0]
    xE = np.empty_like(lon)
    yN = np.empty_like(lon)
    zV = np.empty_like(lon)
    for i in range(3):
        xE[:, i], yN[:, i], zV[:, i] = llh2local_xyz(
            lat[:, i], lon[:, i], height[:, i], origin,
            strict_ellipsoid=strict_ellipsoid,
        )

    strike = np.empty(n)
    dip = np.empty(n)
    area = np.empty(n)
    for i in range(n):
        A = np.array([xE[i, 0], yN[i, 0], zV[i, 0]])
        B = np.array([xE[i, 1], yN[i, 1], zV[i, 1]])
        C = np.array([xE[i, 2], yN[i, 2], zV[i, 2]])
        a, b, c, _ = _plane_from_3points(A, B, C)

        s = 90.0 - np.degrees(np.arctan2(-a, b))
        if -a / c < 0:
            if np.degrees(np.arctan2(-a, b)) < 0:
                s += 180.0
        else:
            if np.degrees(np.arctan2(-a, b)) > 0:
                s -= 180.0
        strike[i] = s % 360.0

        d = 90.0 - np.degrees(np.arctan2(c, np.hypot(a, b)))
        dip[i] = 180.0 - d if d > 90.0 else d

        AB = np.linalg.norm(B - A)
        AC = np.linalg.norm(C - A)
        cosang = np.dot(B - A, C - A) / (AB * AC)
        theta = np.arccos(np.clip(cosang, -1.0, 1.0))
        area[i] = 0.5 * AB * AC * np.sin(theta)

    centroid_llh = np.column_stack([lon.mean(1), lat.mean(1), height.mean(1)])
    return FaultMesh(lon, lat, height, xE, yN, zV, centroid_llh,
                     strike, dip, area, origin)


def refine_mesh(fault, slip_patterns=None):
    """Subdivide every triangle into four by edge midpoints.

    Returns ``(fine_mesh, fine_slip)``.  Each child inherits its parent's slip,
    so the slip *distribution* is unchanged and only the discretisation is
    refined.  Comparing fields computed on the two meshes therefore isolates
    the effect of the discretisation from the effect of the source.

    This is the diagnostic that decides how close to the interface a strain
    value can be believed.  Reducing the evaluation offset until the coarse and
    fine meshes disagree tells you where the artificial triangle edges start to
    dominate; correlation between fields at *different* offsets cannot, because
    the strain genuinely varies with distance from the fault.
    """
    lon, lat, h = fault.lon, fault.lat, fault.height
    n = lon.shape[0]

    def mid(i, j):
        return (0.5 * (lon[:, i] + lon[:, j]),
                0.5 * (lat[:, i] + lat[:, j]),
                0.5 * (h[:, i] + h[:, j]))

    m01, m12, m20 = mid(0, 1), mid(1, 2), mid(2, 0)
    rows = []
    for a, b, c in (((lon[:, 0], lat[:, 0], h[:, 0]), m01, m20),
                    (m01, (lon[:, 1], lat[:, 1], h[:, 1]), m12),
                    (m20, m12, (lon[:, 2], lat[:, 2], h[:, 2])),
                    (m01, m12, m20)):
        rows.append(np.column_stack([a[0], a[1], a[2],
                                     b[0], b[1], b[2],
                                     c[0], c[1], c[2]]))
    m = np.concatenate(rows, axis=0)

    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        np.savetxt(fh.name, m, fmt="%.8f")
        fine = load_fault_ascii(fh.name, origin=fault.origin)

    fine_slip = None
    if slip_patterns is not None:
        fine_slip = np.concatenate([slip_patterns] * 4, axis=0)
    return fine, fine_slip
