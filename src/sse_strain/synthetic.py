"""A synthetic Cascadia-like solution, for testing and for the demo.

This writes a fault mesh file and the five ``.mat`` files with the same names,
variable names and shapes as a real Gualandi solution directory, so that
``load_solution`` and everything downstream can be exercised before any real
data is in hand.  The slip history is a caricature: a few Gaussian patches on
the interface switched on and off by saw-tooth time functions.

It is a test fixture, not a model of Cascadia.  Nothing produced from it should
appear in a figure that is not labelled synthetic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

__all__ = ["make_synthetic_mesh", "write_synthetic_solution"]


def make_synthetic_mesh(path, lon_range=(-124.6, -123.2), lat_range=(45.0, 49.0),
                        n_strike=18, n_dip=6, depth_top_km=25.0,
                        depth_bot_km=50.0):
    """Write a 9-column triangular mesh dipping east, roughly Cascadia-like."""
    lon = np.linspace(lon_range[0], lon_range[1], n_dip + 1)
    lat = np.linspace(lat_range[0], lat_range[1], n_strike + 1)
    dep = np.linspace(depth_top_km, depth_bot_km, n_dip + 1)

    rows = []
    for i in range(n_dip):
        for j in range(n_strike):
            p00 = (lon[i], lat[j], -dep[i])
            p10 = (lon[i + 1], lat[j], -dep[i + 1])
            p01 = (lon[i], lat[j + 1], -dep[i])
            p11 = (lon[i + 1], lat[j + 1], -dep[i + 1])
            rows.append([*p00, *p10, *p11])
            rows.append([*p00, *p11, *p01])
    m = np.array(rows)
    np.savetxt(path, m, fmt="%.6f")
    return m


def write_synthetic_solution(outdir, mesh_path, n_time=1200, n_comp=6,
                             n_station=40, seed=0, nu=0.25,
                             peak_slip_m=0.025):
    """Write options/X/ICA/ind_comps/misfit_comps ``.mat`` files."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    mesh = np.loadtxt(mesh_path)
    n_patch = mesh.shape[0]
    lon_c = mesh[:, [0, 3, 6]].mean(1)
    lat_c = mesh[:, [1, 4, 7]].mean(1)

    # stations scattered onshore of the mesh
    st_lon = rng.uniform(lon_c.min() - 1.5, lon_c.max() + 0.5, n_station)
    st_lat = rng.uniform(lat_c.min() - 0.5, lat_c.max() + 0.5, n_station)
    llh = np.column_stack([st_lon, st_lat, np.zeros(n_station)])
    llh3 = np.repeat(llh, 3, axis=0)
    names = []
    for k in range(n_station):
        names += [f"S{k:02d}_e", f"S{k:02d}_n", f"S{k:02d}_u"]

    t0 = 2015.0
    timeline = t0 + np.arange(n_time) / 365.25

    # saw-tooth amplitude functions with different recurrence intervals
    V = np.empty((n_time, n_comp))
    for k in range(n_comp):
        period = 1.1 + 0.35 * k
        phase = 0.17 * k
        V[:, k] = np.mod((timeline - t0) / period + phase, 1.0)
        V[:, k] -= V[:, k].mean()
    V /= np.abs(V).max(axis=0, keepdims=True)
    # S is set after m_norm is known, so that peak reconstructed slip lands
    # near peak_slip_m rather than at whatever the normalisation happens to give

    # Forward-consistent mixing matrix.  Each component gets a Gaussian slip
    # patch on the interface; U is then the surface response predicted by the
    # same Green's functions the inversion will use, plus noise.  A round trip
    # through Solution.invert() must therefore recover the input patches, and
    # Solution.misfit_check() must return small numbers.  That is what makes
    # this a test rather than a demo.
    from .fault import load_fault_ascii
    from .green import displacement_gf

    fault = load_fault_ascii(mesh_path)
    G = displacement_gf(fault, llh[:, :2], nu)

    lat_patch = fault.centroid_llh[:, 1]
    lon_patch = fault.centroid_llh[:, 0]
    m_true = np.zeros((2 * n_patch, n_comp))
    for k in range(n_comp):
        c_lat = lat_patch.min() + (k + 0.5) / n_comp * np.ptp(lat_patch)
        c_lon = lon_patch.mean()
        w = np.exp(-0.5 * (((lat_patch - c_lat) / 0.45) ** 2
                           + ((lon_patch - c_lon) / 0.35) ** 2))
        m_true[:n_patch, k] = 0.15 * w          # a little strike-slip
        m_true[n_patch:, k] = 1.00 * w          # mostly dip-slip

    U = G @ m_true
    # vbICA returns unit-norm spatial vectors, with the physical amplitude
    # carried by S; normalise here the same way, and keep the model that
    # actually generates the normalised data for the round-trip test.
    scale = np.linalg.norm(U, axis=0, keepdims=True)
    U = U / scale
    m_norm = m_true / scale
    S = np.diag(np.linspace(1.0, 0.4, n_comp)
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
            # bracket the true model amplitude so the middle entry is
            # a sensible prior standard deviation for this fixture
            "sigma": float(np.abs(m_norm).max())
                     * np.array([0.1, 0.3, 1.0, 3.0, 10.0]),
            "rake_pos": 90.0, "verbose": False, "calc_slip_cov": False,
        },
        "smooth": {"name": "lowpass", "filter_cutofffreq": 2.0 / 7.0,
                   "filter_window": 365, "filter_type": "hamming",
                   "causal": False, "fs": 1.0},
        "slip_rate": {"windowsize": 7},
        "scen": {"unit_output": "m"},
        "flags": {"flag_parallel": 1},
    }

    sio.savemat(outdir / "options.mat", {"options": options})
    sio.savemat(outdir / "X.mat", {"Xd": {
        "llh": llh3, "timeline": timeline[None, :],
        "name": np.array(names, dtype=object), "decmode": "3d",
        "type": "position",
    }})
    sio.savemat(outdir / "ICA.mat", {"ICA_essential": {
        "U": U, "S": S, "V": V, "var_U": var_U, "var_V": var_V,
    }})
    sio.savemat(outdir / "ind_comps.mat",
                {"ind_comps": np.arange(1, n_comp + 1)})
    sio.savemat(outdir / "misfit_comps.mat",
                {"misfit_comps": np.zeros((5, n_comp))})
    np.save(outdir / "m_true.npy", m_norm)
    return outdir
