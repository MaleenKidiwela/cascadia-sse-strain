"""Strain Green's functions for the observation grid.

The expensive object here is the pairwise evaluation of the Nikkhoo & Walter
(2015) half-space triangular dislocation kernel over
``n_obs x n_patch`` pairs.  For a 60 x 60 x 3 grid and a few thousand patches
that is order 10^7-10^8 kernel calls, so the design matters:

* We never build the full ``(n_obs, 6, n_patch, 3)`` Green's matrix, which for
  realistic sizes is terabytes.  Instead we contract with the K component slip
  patterns inside each chunk, so peak memory is set by the chunk size and the
  result is only ``(n_obs, 6, K)``.
* Because slip is a sum of K static patterns times scalar time functions
  (see :mod:`sse_strain.solution`), K forward calculations give the entire
  time series.  Doing it epoch by epoch would be a factor of several hundred
  more work for identical output.

The output strain components follow cutde's ordering, which is Nikkhoo &
Walter's: ``[exx, eyy, ezz, exy, exz, eyz]`` in an East-North-Up frame, with
extension positive.
"""

from __future__ import annotations

import os
import numpy as np

__all__ = ["strain_from_components", "displacement_gf", "STRAIN_COMPONENTS"]

STRAIN_COMPONENTS = ("exx", "eyy", "ezz", "exy", "exz", "eyz")


def _chunk_strain(obs_block, tris, slip_patterns, nu):
    """Strain at ``obs_block`` from all triangles, for every component.

    Parameters
    ----------
    obs_block : (nb, 3) array, metres, East-North-Up (z <= 0)
    tris : (n_p, 3, 3) array, metres
    slip_patterns : (n_p, 3, K) array, metres of E/N/U slip per unit amplitude
    nu : float

    Returns
    -------
    (nb, 6, K)
    """
    import cutde.halfspace as HS

    nb = obs_block.shape[0]
    n_p = tris.shape[0]
    K = slip_patterns.shape[2]

    # (nb, 6, n_p, 3)
    M = HS.strain_matrix(obs_block, tris, nu)
    # contract over patch and slip direction
    return np.einsum("obpd,pdk->obk", M, slip_patterns, optimize=True)


def strain_from_components(obs_pts_m, tris_m, slip_patterns, nu,
                           max_pairs=2_000_000, n_jobs=1, progress=True):
    """Strain at observation points from each slip component.

    Parameters
    ----------
    obs_pts_m : (n_obs, 3)
        Observation points in the local frame, metres, East-North-Up.  z must
        be <= 0 (the half-space occupies z <= 0).
    tris_m : (n_patch, 3, 3)
        Triangle vertices, metres.
    slip_patterns : (n_patch, 3, K)
        East-North-Up slip of each component, metres per unit amplitude.
    nu : float
        Poisson's ratio of the half-space.  Use the same value as the
        inversion (``Solution.nu``) unless you intend to test sensitivity.
    max_pairs : int
        Target number of (obs, patch) pairs per chunk; controls peak memory.
        Roughly ``max_pairs * 18 * 8`` bytes are allocated per chunk.
    n_jobs : int
        Processes to spread chunks over.  ``-1`` uses all cores.

    Returns
    -------
    (n_obs, 6, K) array of strain per unit component amplitude.
    """
    obs = np.ascontiguousarray(np.asarray(obs_pts_m, dtype=float))
    tris = np.ascontiguousarray(np.asarray(tris_m, dtype=float))
    sp = np.ascontiguousarray(np.asarray(slip_patterns, dtype=float))

    if obs.ndim != 2 or obs.shape[1] != 3:
        raise ValueError(f"obs_pts_m must be (n_obs, 3), got {obs.shape}")
    if np.any(obs[:, 2] > 0):
        raise ValueError(
            "observation points must have z <= 0 (East-North-Up, metres). "
            "Depths are positive downward elsewhere in this package; convert "
            "with z = -depth."
        )
    if sp.shape[0] != tris.shape[0]:
        raise ValueError(
            f"slip_patterns has {sp.shape[0]} patches, tris has {tris.shape[0]}"
        )

    n_obs, n_p, K = obs.shape[0], tris.shape[0], sp.shape[2]
    per = max(1, int(max_pairs // max(n_p, 1)))
    starts = list(range(0, n_obs, per))

    if n_jobs in (0, 1):
        out = np.empty((n_obs, 6, K))
        for i, i0 in enumerate(starts):
            i1 = min(i0 + per, n_obs)
            out[i0:i1] = _chunk_strain(obs[i0:i1], tris, sp, nu)
            if progress:
                print(f"  strain chunk {i + 1}/{len(starts)} "
                      f"({i1}/{n_obs} points)", flush=True)
        return out

    from concurrent.futures import ProcessPoolExecutor

    if n_jobs < 0:
        n_jobs = os.cpu_count() or 1
    out = np.empty((n_obs, 6, K))
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = {}
        for i0 in starts:
            i1 = min(i0 + per, n_obs)
            futs[ex.submit(_chunk_strain, obs[i0:i1], tris, sp, nu)] = (i0, i1)
        done = 0
        for fut in futs:
            pass
        for fut, (i0, i1) in futs.items():
            out[i0:i1] = fut.result()
            done += 1
            if progress:
                print(f"  strain chunk {done}/{len(starts)}", flush=True)
    return out


def displacement_gf(fault, station_lonlat, nu, max_pairs=2_000_000):
    """Surface displacement Green's functions for a set of stations.

    Parameters
    ----------
    fault : FaultMesh
    station_lonlat : (n_station, 2) longitude, latitude in degrees
    nu : float

    Returns
    -------
    (3 * n_station, 2 * n_patch) array.  Rows cycle (East, North, Up) per
    station; the first ``n_patch`` columns are unit strike-slip and the second
    ``n_patch`` unit dip-slip, matching ``create_greens_function`` in
    Gualandi's ``src_load.jl``.

    Stations sit at z = 0 exactly, as in the original code, rather than at
    their ellipsoidal heights.  That choice is irrelevant for strain at depth
    and matters only for reproducing his inversion, where it must be kept.

    Note on cutde semantics
    -----------------------
    ``cutde.halfspace.disp_free(obs, tris, slips, nu)`` returns, for each
    observation point, the displacement summed over **all** triangles -- it is
    not a pairwise obs[i]/tris[i] evaluation, despite the shape requirement
    that the two arrays have the same length.  Building per-patch Green's
    function columns therefore requires ``disp_matrix``, which returns the
    unsummed ``(n_obs, 3, n_tri, 3)`` kernel.  Getting this backwards produces
    a G whose every column is the whole-fault response; the inversion still
    converges and still fits the data, so the error is invisible downstream.
    ``tests/test_pipeline.py::test_cutde_free_is_summed_not_pairwise`` pins the
    convention down.
    """
    import cutde.halfspace as HS

    from .geodesy import llh2localxy

    st = np.asarray(station_lonlat, dtype=float)
    sx, sy = llh2localxy(st[:, 1], st[:, 0], fault.origin)
    obs = np.ascontiguousarray(
        np.column_stack([sx * 1e3, sy * 1e3, np.zeros(sx.size)]))
    tris = np.ascontiguousarray(fault.vertices_m())

    n_s, n_p = obs.shape[0], tris.shape[0]
    G = np.empty((3 * n_s, 2 * n_p))
    per = max(1, int(max_pairs // max(n_p, 1)))
    for i0 in range(0, n_s, per):
        i1 = min(i0 + per, n_s)
        M = HS.disp_matrix(obs[i0:i1], tris, nu)   # (nb, 3, n_p, 3)
        rows = slice(3 * i0, 3 * i1)
        # slip direction 0 = strike, 1 = dip
        G[rows, :n_p] = M[:, :, :, 0].reshape(3 * (i1 - i0), n_p)
        G[rows, n_p:] = M[:, :, :, 1].reshape(3 * (i1 - i0), n_p)
    return G
