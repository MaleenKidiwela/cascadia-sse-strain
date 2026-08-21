"""Tests.

The physics tests (free surface, numerical gradient) are the ones that matter.
They check that the strain we compute is genuinely the gradient of the
displacement field of a half-space dislocation, independently of anything in
this package.  The round-trip test checks the bookkeeping: reshaping, the
strike/dip rotation, the component contraction, and the regularised inversion.
"""

from __future__ import annotations

import numpy as np
import pytest

from sse_strain import (ElasticModel, load_fault_ascii, load_solution,
                        make_grid)
from sse_strain import strain as st
from sse_strain.geodesy import llh2localxy
from sse_strain.green import displacement_gf, strain_from_components
from sse_strain.synthetic import make_synthetic_mesh, write_synthetic_solution

NU = 0.25
MU = 30e9


@pytest.fixture(scope="module")
def solution(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("syn")
    mesh = tmp / "mesh.txt"
    make_synthetic_mesh(mesh, n_strike=10, n_dip=4)
    d = write_synthetic_solution(tmp / "2025-01-26", mesh, n_time=300,
                                 n_comp=4, n_station=30)
    fault = load_fault_ascii(mesh)
    sol = load_solution(d, fault)
    return sol, fault, d


# --------------------------------------------------------------------- physics
def test_free_surface_traction_vanishes():
    """A half-space solution must carry zero traction on z = 0."""
    import cutde.halfspace as HS

    tri = np.array([[[-2000.0, 0.0, -8000.0],
                     [3000.0, -1500.0, -9000.0],
                     [500.0, 2500.0, -11000.0]]])
    pts = np.array([[1000.0, 500.0, 0.0],
                    [-4000.0, 2000.0, 0.0],
                    [7000.0, -3000.0, 0.0]])
    for slip in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        e = HS.strain_free(pts, np.repeat(tri, 3, 0),
                           np.array([slip] * 3, float), NU)
        s = HS.strain_to_stress(e, MU, NU)
        scale = np.abs(s).max()
        # szz, sxz, syz are components 2, 4, 5
        assert np.abs(s[:, [2, 4, 5]]).max() < 1e-9 * max(scale, 1.0)


def test_strain_equals_numerical_displacement_gradient():
    """eps_ij must equal 1/2 (du_i/dx_j + du_j/dx_i) computed by differencing."""
    import cutde.halfspace as HS

    tri = np.array([[[-4000.0, -2000.0, -20000.0],
                     [5000.0, -1000.0, -22000.0],
                     [0.0, 6000.0, -26000.0]]])
    slip = np.array([[0.3, 1.0, 0.0]])
    p0 = np.array([2500.0, 1200.0, -4000.0])
    h = 20.0

    def u(pt):
        return HS.disp_free(pt[None, :], tri, slip, NU)[0]

    grad = np.empty((3, 3))
    for j in range(3):
        dp = np.zeros(3)
        dp[j] = h
        grad[:, j] = (u(p0 + dp) - u(p0 - dp)) / (2 * h)
    eps_num = 0.5 * (grad + grad.T)

    e = HS.strain_free(p0[None, :], tri, slip, NU)[0]
    eps_ana = st.voigt_to_tensor(e)
    assert np.allclose(eps_num, eps_ana, rtol=2e-4,
                       atol=1e-3 * np.abs(eps_ana).max())


def test_strain_decays_with_distance():
    """Far from a finite source, strain should fall off faster than 1/r^2."""
    import cutde.halfspace as HS

    tri = np.array([[[-3000.0, 0.0, -30000.0],
                     [3000.0, 0.0, -30000.0],
                     [0.0, 4000.0, -34000.0]]])
    r = np.array([5e4, 1e5, 2e5])
    pts = np.column_stack([r, np.zeros(3), np.full(3, -2000.0)])
    e = HS.strain_free(pts, np.repeat(tri, 3, 0),
                       np.tile([0.0, 1.0, 0.0], (3, 1)), NU)
    amp = np.abs(e).max(axis=1)
    slope = np.polyfit(np.log(r), np.log(amp), 1)[0]
    assert slope < -2.0


# ------------------------------------------------------------------- elasticity
def test_hooke_round_trip_isotropic():
    """Uniaxial strain must give the textbook lambda + 2mu response."""
    lam, mu = 30e9, 30e9
    eps = np.array([[1e-6, 0, 0, 0, 0, 0]])
    sig = st.hooke(eps, np.array([mu]), np.array([lam]))
    assert np.isclose(sig[0, 0], (lam + 2 * mu) * 1e-6)
    assert np.isclose(sig[0, 1], lam * 1e-6)
    assert np.isclose(sig[0, 3], 0.0)


def test_invariants_are_rotation_invariant():
    rng = np.random.default_rng(3)
    e = rng.normal(size=(1, 6))
    t = st.voigt_to_tensor(e)[0]
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    tr = q @ t @ q.T
    e_rot = np.array([[tr[0, 0], tr[1, 1], tr[2, 2],
                       tr[0, 1], tr[0, 2], tr[1, 2]]])
    assert np.isclose(st.dilatation(e)[0], st.dilatation(e_rot)[0])
    assert np.isclose(st.j2_strain(e)[0], st.j2_strain(e_rot)[0])
    assert np.isclose(st.max_shear_strain(e)[0], st.max_shear_strain(e_rot)[0])


def test_elastic_model_consistency():
    em = ElasticModel.homogeneous(vs_m_s=3500.0, nu=0.25, rho_kg_m3=2800.0)
    assert np.isclose(em.mu(5000.0), 2800.0 * 3500.0**2)
    assert np.isclose(em.poisson(5000.0), 0.25, atol=1e-6)
    ph = ElasticModel.placeholder_cascadia()
    assert ph.is_placeholder
    assert 0.15 < float(ph.poisson(10000.0)) < 0.35
    assert 1e10 < float(ph.mu(10000.0)) < 8e10


# ---------------------------------------------------------------- bookkeeping
def test_geodesy_origin_maps_to_zero():
    x, y = llh2localxy([47.0], [-123.0], (-123.0, 47.0))
    assert abs(float(x[0])) < 1e-6 and abs(float(y[0])) < 1e-6


def test_fault_geometry_sane(solution):
    _, fault, _ = solution
    assert fault.n_patch > 0
    assert np.all(fault.area > 0)
    assert np.all((fault.dip >= 0) & (fault.dip <= 90))
    assert np.all(fault.centroid_llh[:, 2] < 0)  # below sea level


def test_slip_enu_is_unit_preserving(solution):
    _, fault, _ = solution
    ss = np.ones((fault.n_patch, 1))
    ds = np.zeros((fault.n_patch, 1))
    v = fault.slip_enu(ss, ds)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    v2 = fault.slip_enu(ds, ss)
    assert np.allclose(np.linalg.norm(v2, axis=1), 1.0)
    # strike and dip directions must be orthogonal
    assert np.allclose(np.einsum("pi...,pi...->p...", v, v2), 0.0, atol=1e-12)


def test_inversion_round_trip(solution):
    """Recover the synthetic slip patches and fit the synthetic data."""
    sol, fault, d = solution
    G = sol.displacement_gf()
    sol.invert(G=G, verbose=False)
    misfit = sol.misfit_check(G=G)
    assert misfit.max() < 0.15, f"poor fit: {misfit}"

    # Where the slip is, rather than its detailed shape.  A regularised
    # inversion of 30 surface stations cannot recover a 50 km patch at 25-50 km
    # depth point by point, and asserting that it does would be testing the
    # fixture rather than the code.  What must hold is that the recovered
    # dip-slip maximum sits at the latitude of the patch that generated the
    # data: that exercises the projection, the Green's functions, the
    # strike/dip rotation and the column ordering all at once.
    m_true = np.load(d / "m_true.npy")
    n_p = fault.n_patch
    lat = fault.centroid_llh[:, 1]
    for k in range(m_true.shape[1]):
        lat_true = lat[np.argmax(np.abs(m_true[n_p:, k]))]
        lat_rec = lat[np.argmax(np.abs(sol.L[n_p:, k]))]
        assert abs(lat_rec - lat_true) < 0.8, (
            f"component {k}: recovered slip at {lat_rec:.2f} N, "
            f"input patch at {lat_true:.2f} N")


def test_slip_amplitudes_are_physical(solution):
    sol, _, _ = solution
    if sol.L is None:
        sol.invert(verbose=False)
    ss, ds = sol.slip()
    peak = np.abs(np.hypot(ss, ds)).max()
    assert 1e-4 < peak < 0.5, f"peak slip {peak} m is not slow-slip-like"


def test_cutde_free_is_summed_not_pairwise():
    """Pin down the cutde convention that displacement_gf depends on.

    ``disp_free(obs, tris, slips, nu)`` returns the response of the *whole* set
    of triangles at each observation point, not the obs[i]/tris[i] pair.  If a
    future cutde release changes this, our Green's functions would silently
    become the whole-fault response in every column, and every fit downstream
    would still look fine.
    """
    import cutde.halfspace as HS

    tris = np.array([
        [[-3000.0, 0.0, -20000.0], [3000.0, -1000.0, -21000.0],
         [0.0, 4000.0, -24000.0]],
        [[20000.0, 10000.0, -22000.0], [26000.0, 12000.0, -23000.0],
         [23000.0, 15000.0, -26000.0]]])
    slip = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    obs = np.array([[0.0, 0.0, 0.0], [20000.0, 12000.0, 0.0]])

    free = HS.disp_free(obs, tris, slip, NU)
    matrix = np.einsum("oipd,pd->oi", HS.disp_matrix(obs, tris, NU), slip)
    assert np.allclose(free, matrix)

    single = HS.disp_matrix(obs, tris[:1], NU)[:, :, 0, 1]
    assert not np.allclose(free, single)


def test_displacement_gf_columns_are_single_patch(solution):
    """Each column of G must be one patch's response, not the fault's."""
    import cutde.halfspace as HS

    from sse_strain.geodesy import llh2localxy

    sol, fault, _ = solution
    G = sol.displacement_gf()
    st = sol.station_llh[:, :2]
    sx, sy = llh2localxy(st[:, 1], st[:, 0], fault.origin)
    obs = np.ascontiguousarray(
        np.column_stack([sx * 1e3, sy * 1e3, np.zeros(sx.size)]))
    tris = fault.vertices_m()
    n_p = fault.n_patch

    for j in (0, n_p // 3, n_p - 1):
        ss = HS.disp_matrix(obs, tris[j:j + 1], sol.nu)[:, :, 0, 0].ravel()
        ds = HS.disp_matrix(obs, tris[j:j + 1], sol.nu)[:, :, 0, 1].ravel()
        assert np.allclose(G[:, j], ss)
        assert np.allclose(G[:, n_p + j], ds)


def test_component_contraction_matches_direct(solution):
    """Summing component strains must equal a direct forward model of slip(t)."""
    import cutde.halfspace as HS

    sol, fault, _ = solution
    if sol.L is None:
        sol.invert(verbose=False)
    Ls, Ld = sol.component_slip_patterns()
    slip_k = fault.slip_enu(Ls, Ld)

    grid = make_grid(lon_range=(-124.0, -123.0), lat_range=(46.0, 47.0),
                     spacing_deg=0.5, depths_km=(2.0,), origin=fault.origin)
    E = strain_from_components(grid.points_m, fault.vertices_m(), slip_k,
                               sol.nu, progress=False)

    ti = 137
    A = sol.component_amplitudes()[ti]
    eps_comp = np.einsum("obk,k->ob", E, A)

    slip_t = fault.slip_enu(Ls @ A, Ld @ A)          # (n_patch, 3)
    tris = fault.vertices_m()
    eps_dir = HS.strain_free(np.ascontiguousarray(grid.points_m),
                             np.repeat(tris, 1, 0), slip_t, sol.nu) \
        if grid.n_obs == tris.shape[0] else None
    if eps_dir is None:
        # disp/strain_free requires equal lengths; sum single-patch calls
        eps_dir = np.zeros_like(eps_comp)
        for j in range(tris.shape[0]):
            eps_dir += HS.strain_matrix(
                np.ascontiguousarray(grid.points_m), tris[j:j + 1], sol.nu
            )[:, :, 0, :] @ slip_t[j]

    scale = max(np.abs(eps_dir).max(), 1e-30)
    assert np.allclose(eps_comp, eps_dir, rtol=1e-7, atol=1e-9 * scale)


def test_dataset_shapes_and_metadata(solution):
    from sse_strain import dataset

    sol, fault, _ = solution
    if sol.L is None:
        sol.invert(verbose=False)
    Ls, Ld = sol.component_slip_patterns()
    slip_k = fault.slip_enu(Ls, Ld)
    grid = make_grid(lon_range=(-124.5, -122.5), lat_range=(45.5, 48.5),
                     spacing_deg=1.0, depths_km=(1.0, 5.0), origin=fault.origin)
    E = strain_from_components(grid.points_m, fault.vertices_m(), slip_k,
                               sol.nu, progress=False)
    em = ElasticModel.placeholder_cascadia(nu_greens=sol.nu)

    ds = dataset.evaluate(grid, sol, em, E, stride=25,
                          receiver=(0.0, 15.0, 90.0))
    assert ds.dilatation.dims == ("time", "depth", "lat", "lon")
    assert ds.sizes["depth"] == 2
    assert "coulomb" in ds
    assert ds.attrs["elastic_is_placeholder"] == 1
    assert "Gualandi" in ds.attrs["source_catalog"]
    assert np.isfinite(ds.pressure.values).all()

    cds = dataset.component_dataset(grid, sol, em, E)
    assert cds.strain_component.sizes["tensor"] == 6
    assert cds.strain_component.sizes["component"] == E.shape[2]


def test_evaluate_refuses_oversized_request(solution):
    from sse_strain import dataset

    sol, fault, _ = solution
    if sol.L is None:
        sol.invert(verbose=False)
    Ls, Ld = sol.component_slip_patterns()
    slip_k = fault.slip_enu(Ls, Ld)
    grid = make_grid(lon_range=(-124.5, -122.5), lat_range=(45.5, 48.5),
                     spacing_deg=1.0, depths_km=(1.0,), origin=fault.origin)
    E = strain_from_components(grid.points_m, fault.vertices_m(), slip_k,
                               sol.nu, progress=False)
    em = ElasticModel.homogeneous()
    n = 5000
    huge = np.zeros((n, E.shape[2]))
    times = np.datetime64("2000-01-01") + np.arange(n).astype("timedelta64[D]")
    with pytest.raises(MemoryError, match="GB"):
        dataset.evaluate(grid, sol, em, E, amplitudes=huge, times=times,
                         max_gb=1e-6)


def test_grid_rejects_positive_z(solution):
    sol, fault, _ = solution
    bad = np.array([[0.0, 0.0, 100.0]])
    with pytest.raises(ValueError, match="z <= 0"):
        strain_from_components(bad, fault.vertices_m(),
                               np.zeros((fault.n_patch, 3, 1)), sol.nu,
                               progress=False)


# ------------------------------------------------------- measurement operator
def test_moving_average_preserves_step_amplitude():
    """A step survives a boxcar with its amplitude intact, only smeared."""
    from sse_strain.smoothing import centered_moving_average

    n, w = 400, 31
    x = np.zeros(n)
    x[200:] = 1.0
    s, valid, info = centered_moving_average(x, w)
    assert np.isclose(np.nanmax(s[valid]), 1.0)
    assert np.isclose(np.nanmin(s[valid]), 0.0)
    # the transition occupies about one window
    trans = np.flatnonzero((s > 0.02) & (s < 0.98))
    assert w - 2 <= trans.size <= w + 2
    assert info["edge_samples_lost_each_side"] == (w - 1) // 2


def test_moving_average_attenuates_short_rate_pulse():
    """A pulse shorter than the window loses peak amplitude as T/W."""
    from sse_strain.smoothing import (attenuation_of_pulse,
                                      centered_moving_average)

    n, w, T = 400, 30, 21
    x = np.zeros(n)
    x[200:200 + T] = 1.0
    s, _, _ = centered_moving_average(x, w)
    assert np.isclose(np.nanmax(s), attenuation_of_pulse(T, w), rtol=0.05)
    assert np.nanmax(s) < 0.75


def test_boxcar_sidelobe_changes_sign():
    """Signals in the first sidelobe come through inverted, not just weakened."""
    from sse_strain.smoothing import boxcar_response

    w = 30
    f, H = boxcar_response(w, np.linspace(1e-4, 0.12, 5000))
    assert np.isclose(np.interp(1.0 / w, f, H), 0.0, atol=1e-3)
    sidelobe = H[(f > 1.0 / w) & (f < 2.0 / w)]
    assert sidelobe.min() < -0.15


def test_smoothing_commutes_with_spatial_sum(solution):
    """Smoothing amplitudes must equal smoothing the assembled strain field."""
    from sse_strain.smoothing import centered_moving_average

    sol, fault, _ = solution
    if sol.L is None:
        sol.invert(verbose=False)
    Ls, Ld = sol.component_slip_patterns()
    slip_k = fault.slip_enu(Ls, Ld)
    grid = make_grid(lon_range=(-124.0, -123.0), lat_range=(46.0, 47.0),
                     spacing_deg=0.5, depths_km=(2.0,), origin=fault.origin)
    E = strain_from_components(grid.points_m, fault.vertices_m(), slip_k,
                               sol.nu, progress=False)

    A = sol.component_amplitudes()
    w = 31
    A_s, valid_a, _ = centered_moving_average(A, w)

    eps_full = np.einsum("obk,tk->tob", E, A)
    eps_then_smooth, valid_e, _ = centered_moving_average(eps_full, w)
    eps_smooth_first = np.einsum("obk,tk->tob", E, np.nan_to_num(A_s))

    m = valid_a.all(axis=1)
    assert np.allclose(eps_then_smooth[m], eps_smooth_first[m], rtol=1e-10,
                       atol=1e-16 * max(np.abs(eps_full).max(), 1e-30))


def test_effective_sample_size_collapses_under_smoothing():
    """Daily samples of a monthly average are not daily degrees of freedom."""
    from sse_strain.smoothing import (centered_moving_average,
                                      effective_sample_size)

    rng = np.random.default_rng(11)
    n, w = 3000, 31
    x = rng.normal(size=n)
    xs, valid, _ = centered_moving_average(x, w)
    xs = xs[valid]

    n_raw = effective_sample_size(x)
    n_smooth = effective_sample_size(xs)
    assert n_raw > 0.5 * n
    assert n_smooth < 0.15 * xs.size
    # order-of-magnitude agreement with the 3N/(2W) rule of thumb
    assert 0.3 < n_smooth / (3 * xs.size / (2 * w)) < 3.0


def test_reference_to_epoch_zeroes_the_baseline(solution):
    from sse_strain.smoothing import reference_to_epoch

    sol, _, _ = solution
    A = sol.component_amplitudes()
    t = sol.dates
    ref = (t[50], t[100])
    A0 = reference_to_epoch(A, t, ref)
    sel = (t >= ref[0]) & (t <= ref[1])
    assert np.allclose(A0[sel].mean(axis=0), 0.0, atol=1e-12)


def test_smooth_amplitudes_reports_effective_size(solution):
    from sse_strain.smoothing import smooth_amplitudes

    sol, _, _ = solution
    A, t, info = smooth_amplitudes(sol, window=31)
    assert A.shape[0] == t.size == info["kept"]
    assert info["kept"] == info["total"] - 2 * ((31 - 1) // 2)
    assert info["effective_sample_size"] < 0.1 * info["kept"]


def test_reference_from_end_respects_edge_loss():
    from sse_strain.config import DVV_EDGE_LOSS_DAYS, DVV_REFERENCE_YEARS
    from sse_strain.smoothing import reference_from_end

    t = (np.datetime64("2010-01-01")
         + np.arange(5000).astype("timedelta64[D]"))
    s, e = reference_from_end(t, DVV_REFERENCE_YEARS, DVV_EDGE_LOSS_DAYS)
    assert e == t.max() - np.timedelta64(DVV_EDGE_LOSS_DAYS, "D")
    assert (e - s).astype(int) == 365
    with pytest.raises(ValueError, match="too short"):
        reference_from_end(t[:100], 1.0)


def test_saw_tooth_reference_bias_vanishes_at_matched_period():
    from sse_strain.smoothing import saw_tooth_reference_bias

    # The offset vanishes when the reference spans a whole number of cycles,
    # i.e. when the recurrence divides the reference length -- not when the
    # reference divides the recurrence.
    assert saw_tooth_reference_bias(365.25) < 1e-3     # one full cycle
    assert saw_tooth_reference_bias(365.25 / 2) < 1e-3  # two full cycles
    assert saw_tooth_reference_bias(730.5) > 0.2       # only half a cycle
    assert 0.15 < saw_tooth_reference_bias(660.0) < 0.30
    assert saw_tooth_reference_bias(480.0) > saw_tooth_reference_bias(420.0)


def test_reference_field_zeroes_the_interval(solution):
    from sse_strain import dataset
    from sse_strain.smoothing import reference_from_end

    sol, fault, _ = solution
    if sol.L is None:
        sol.invert(verbose=False)
    Ls, Ld = sol.component_slip_patterns()
    slip_k = fault.slip_enu(Ls, Ld)
    grid = make_grid(lon_range=(-124.5, -123.0), lat_range=(46.0, 48.0),
                     spacing_deg=1.0, depths_km=(2.0,), origin=fault.origin)
    E = strain_from_components(grid.points_m, fault.vertices_m(), slip_k,
                               sol.nu, progress=False)
    ds = dataset.evaluate(grid, sol, ElasticModel.homogeneous(), E, stride=5)

    ref = reference_from_end(ds.time.values, years=0.3, edge_loss_days=15)
    out = dataset.reference_field(ds, ref)
    sel = (out.time >= np.datetime64(ref[0])) & (out.time <= np.datetime64(ref[1]))
    resid = out.dilatation.isel(time=sel.values).mean("time").values
    scale = np.abs(ds.dilatation.values).max()
    assert np.abs(resid).max() < 1e-5 * scale   # float32 storage
    assert "dvv_reference_interval" in out.attrs
