"""Local Cartesian projection used by the Gualandi (2025) Cascadia solutions.

The projection is a polyconic one.  It is ported line-for-line from
``src_coordinates.jl`` in https://github.com/Geolandi/sse_postprocessing
(and the equivalent ``llh2localxy.m``), because the fault mesh, the station
coordinates and the Green's functions of the published inversion are all
expressed in *this* frame.  Substituting a different projection (pyproj UTM,
for instance) would introduce a systematic offset between our observation grid
and his fault patches, of order a few hundred metres over the width of
Cascadia.  Fidelity to the original beats geodetic elegance here.

FLAGGED FOR VERIFICATION
------------------------
In the original ``polyconic`` routine the eccentricity term reads

    a = sqrt(1.0 - (esq * (2.0 * sinp2))) / (la * arcone)

The standard polyconic formulation has ``esq * sinp2**2`` in that position,
not ``esq * 2 * sinp2``.  This may be a long-standing transcription slip that
propagated from the original Matlab through to the Julia port.  Its numerical
effect is small (the term is a ~0.3 % correction on the scale factor), but it
is not zero.  ``polyconic()`` below reproduces the published expression so that
our coordinates match his; ``polyconic(..., strict_ellipsoid=True)`` uses the
textbook form.  Do not change the default without checking against a patch
centroid computed by his own code.
"""

from __future__ import annotations

import numpy as np

__all__ = ["polyconic", "llh2localxy", "llh2local_xyz"]

_ARCONE = 4.8481368e-6  # arcseconds -> radians
_ESQ = 6.7686580e-3     # first eccentricity squared (Clarke 1866)
_LA = 6378206.4         # semi-major axis, m (Clarke 1866)
_A0 = 6367399.7
_A2 = 32433.888
_A4 = 34.4187
_A6 = 0.0454
_A8 = 6.0e-5


def polyconic(lat_sec, dlon_sec, lat_orig_sec, strict_ellipsoid: bool = False):
    """Polyconic projection.

    Parameters
    ----------
    lat_sec, dlon_sec : array_like
        Latitude and longitude difference from the central meridian, in
        decimal *arcseconds*.
    lat_orig_sec : float
        Latitude of the projection origin, in decimal arcseconds.
    strict_ellipsoid : bool
        If True use ``esq * sin^2(phi)``; if False (default) reproduce the
        expression in the published code.  See module docstring.

    Returns
    -------
    x, y : ndarray
        Metres east of the central meridian and north of the origin.
    """
    p1 = float(lat_orig_sec)
    p2 = np.asarray(lat_sec, dtype=float)
    il = np.asarray(dlon_sec, dtype=float)

    ip = p2 - p1
    sinp2 = np.sin(p2 * _ARCONE)
    cosp2 = np.cos(p2 * _ARCONE)
    theta = il * sinp2

    if strict_ellipsoid:
        a = np.sqrt(1.0 - _ESQ * sinp2**2) / (_LA * _ARCONE)
    else:
        a = np.sqrt(1.0 - _ESQ * (2.0 * sinp2)) / (_LA * _ARCONE)

    cot = cosp2 / sinp2
    x = (cot * np.sin(theta * _ARCONE)) / (a * _ARCONE)

    ipr = ip * _ARCONE
    pr = ((p2 + p1) / 2.0) * _ARCONE
    y = (
        _A0 * ipr
        - (_A2 * np.cos(2.0 * pr)) * np.sin(ipr)
        + (_A4 * np.cos(4.0 * pr)) * np.sin(2.0 * ipr)
        - (_A6 * np.cos(6.0 * pr)) * np.sin(3.0 * ipr)
        + (_A8 * np.cos(8.0 * pr)) * np.sin(4.0 * ipr)
    )
    return x, y


def llh2localxy(lat, lon, origin_lonlat, strict_ellipsoid: bool = False):
    """Latitude/longitude (degrees) to local east/north in **km**.

    ``origin_lonlat`` is ``(lon0, lat0)`` in degrees, matching the convention
    of ``fault["origin"]`` in the Julia code.  The x axis is flipped so that
    positive x is east, as in ``llh2localxy.m``.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lon0, lat0 = float(origin_lonlat[0]), float(origin_lonlat[1])

    lat_sec = 3600.0 * lat
    lon_sec = 3600.0 * lon
    lat_orig_sec = 3600.0 * lat0
    dlon_sec = 3600.0 * lon0 - lon_sec

    x, y = polyconic(lat_sec, dlon_sec, lat_orig_sec,
                     strict_ellipsoid=strict_ellipsoid)
    # metres -> km, and flip the x axis (see llh2localxy.m)
    return -x / 1000.0, y / 1000.0


def llh2local_xyz(lat, lon, height_km, origin_lonlat, **kw):
    """As :func:`llh2localxy` but carrying the vertical coordinate through.

    ``height_km`` is altitude in km (negative below sea level), returned
    unchanged as ``zV``.
    """
    xE, yN = llh2localxy(lat, lon, origin_lonlat, **kw)
    return xE, yN, np.asarray(height_km, dtype=float)
