"""A Cascadia-shaped synthetic solution.

Geometry follows the broad shape of the Cascadia interface: a trench near
126 W curving to 124 W, dipping east, reaching 60 km beneath Puget Sound.  The
slip history is three along-strike segments with different recurrence
intervals, in the observed 8-22 month range, which is what makes the reference
window and degrees-of-freedom questions bite the same way they will on the real
data.

This is a fixture with the right shape and scale, not a model of Cascadia.
Every figure produced from it is labelled SYNTHETIC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

__all__ = ["cascadia_mesh", "cascadia_solution"]


def cascadia_mesh(path, n_strike=34, n_dip=9, lat_range=(40.0, 49.5),
                  depth_range_km=(8.0, 60.0)):
    """Write a Cascadia-like triangular interface mesh (9-column ASCII)."""
    lat = np.linspace(lat_range[0], lat_range[1], n_strike + 1)
    # trench longitude migrates east going north; interface widens with depth
    lon_trench = -126.2 + 0.16 * (lat - lat.min()) / np.ptp(lat) * 6.0
    width_deg = 2.9 + 0.7 * np.sin(np.pi * (lat - lat.min()) / np.ptp(lat))
    depth = np.linspace(depth_range_km[0], depth_range_km[1], n_dip + 1)
    # concave-up slab: shallow dip near the trench, steeper down-dip
    frac = (depth - depth[0]) / (depth[-1] - depth[0])
    across = frac ** 0.72

    rows = []
    for i in range(n_dip):
        for j in range(n_strike):
            def node(ii, jj):
                return (lon_trench[jj] + across[ii] * width_deg[jj],
                        lat[jj], -depth[ii])
            p00, p10 = node(i, j), node(i + 1, j)
            p01, p11 = node(i, j + 1), node(i + 1, j + 1)
            rows.append([*p00, *p10, *p11])
            rows.append([*p00, *p11, *p01])
    m = np.array(rows)
    np.savetxt(path, m, fmt="%.6f")
    return m


def cascadia_solution(outdir, mesh_path, start_year=2010.0, n_time=5600,
                      n_station=80, seed=3, nu=0.25, peak_slip_m=0.03,
                      sse_depth_km=(30.0, 45.0)):
    """Write a full synthetic solution directory with ETS-like time functions."""
    from .fault import load_fault_ascii
    from .green import displacement_gf

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    fault = load_fault_ascii(mesh_path)
    n_patch = fault.n_patch
    lat_c = fault.centroid_llh[:, 1]
    lon_c = fault.centroid_llh[:, 0]
    dep_c = -fault.centroid_llh[:, 2]

    # GNSS-like network onshore of the mesh
    st_lat = rng.uniform(lat_c.min(), lat_c.max(), n_station)
    st_lon = rng.uniform(lon_c.min() + 1.2, lon_c.max() + 1.0, n_station)
    llh = np.column_stack([st_lon, st_lat, np.zeros(n_station)])
    llh3 = np.repeat(llh, 3, axis=0)
    names = []
    for k in range(n_station):
        names += [f"C{k:03d}_e", f"C{k:03d}_n", f"C{k:03d}_u"]

    G = displacement_gf(fault, llh[:, :2], nu)

    # three along-strike segments, recurrence 14 / 11 / 8 months N to S
    seg_lat = [46.8, 44.6, 42.4, 48.4]
    seg_rec_d = [425.0, 335.0, 250.0, 500.0]
    n_comp = len(seg_lat)

    # Taper the down-dip edges of the slow-slip band rather than cutting them
    # square.  A hard slip edge radiates strong near-field structure that has
    # nothing to do with the method being demonstrated, and the real inverted
    # slip is smoothed by an exponential prior over the whole fault.
    d0, d1 = sse_depth_km
    taper = 4.0
    in_band = (np.clip((dep_c - (d0 - taper)) / taper, 0.0, 1.0)
               * np.clip(((d1 + taper) - dep_c) / taper, 0.0, 1.0))
    in_band = in_band ** 2 * (3 - 2 * in_band)      # smoothstep

    m_true = np.zeros((2 * n_patch, n_comp))
    for k in range(n_comp):
        w = np.exp(-0.5 * ((lat_c - seg_lat[k]) / 0.75) ** 2)
        w = w * in_band
        m_true[:n_patch, k] = 0.12 * w
        m_true[n_patch:, k] = 1.00 * w

    timeline = start_year + np.arange(n_time) / 365.25
    days = (timeline - timeline[0]) * 365.25

    V = np.empty((n_time, n_comp))
    for k in range(n_comp):
        T = seg_rec_d[k]
        phase = np.mod(days / T + 0.13 * k, 1.0)
        rise = 20.0 / T          # ~20 day slip episode
        # continuous saw-tooth: slow loading down to -(1-rise) over the
        # interseismic part, then recovery to zero during a ~20 day episode
        load = -phase
        u = np.clip((phase - (1.0 - rise)) / rise, 0.0, 1.0)
        slip = -(1.0 - rise) * (1.0 - u)
        V[:, k] = np.where(phase < 1.0 - rise, load, slip)
        V[:, k] -= V[:, k].mean()
    V /= np.abs(V).max(axis=0, keepdims=True)

    U = G @ m_true
    scale = np.linalg.norm(U, axis=0, keepdims=True)
    U = U / scale
    m_norm = m_true / scale

    S = np.diag(np.linspace(1.0, 0.7, n_comp)
                * peak_slip_m / float(np.abs(m_norm).max()))
    noise = 0.01 * np.abs(U).max()
    U = U + rng.normal(0.0, noise, U.shape)
    var_U = np.full_like(U, noise ** 2)
    var_V = np.full_like(V, 1e-6)

    options = {
        "fault": {"fault_file": str(Path(mesh_path).name), "nu": nu},
        "inversion": {
            "lambda_strike": 21.0, "lambda_dip": 21.0, "lambda0": 21.0,
            "which_smoothing": "exponential", "m0": 0.0,
            "sigma": float(np.abs(m_norm).max())
                     * np.array([0.1, 0.3, 1.0, 3.0, 10.0]),
            "rake_pos": 90.0,
        },
        "smooth": {"filter_cutofffreq": 2.0 / 7.0, "filter_window": 365},
        "slip_rate": {"windowsize": 7},
    }
    sio.savemat(outdir / "options.mat", {"options": options})
    sio.savemat(outdir / "X.mat", {"Xd": {
        "llh": llh3, "timeline": timeline[None, :],
        "name": np.array(names, dtype=object), "decmode": "3d"}})
    sio.savemat(outdir / "ICA.mat", {"ICA_essential": {
        "U": U, "S": S, "V": V, "var_U": var_U, "var_V": var_V}})
    sio.savemat(outdir / "ind_comps.mat",
                {"ind_comps": np.arange(1, n_comp + 1)})
    sio.savemat(outdir / "misfit_comps.mat",
                {"misfit_comps": np.zeros((5, n_comp))})
    np.save(outdir / "m_true.npy", m_norm)
    np.save(outdir / "segment_recurrence_days.npy", np.array(seg_rec_d))
    return outdir
