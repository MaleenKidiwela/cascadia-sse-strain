"""Assembling results into xarray Datasets.

Two products, and the distinction matters for whether this fits on disk.

**The component Dataset** (:func:`component_dataset`) stores the K static
strain fields and the K amplitude time functions separately.  It is complete
and lossless: any epoch can be reconstructed exactly, and it is small, of
order ``n_obs * 6 * K`` numbers.  This is the thing to archive.

**The evaluated Dataset** (:func:`evaluate`) materialises strain and stress on
a (time, depth, lat, lon) grid for a chosen time window.  Materialising all
6600 daily epochs on a full grid would run to terabytes, so ``evaluate`` takes
a time selection and a list of variables and refuses to be clever about it.
Ask for the derived scalars over fifteen years at weekly sampling, or all
twelve tensor components over one ETS episode at daily sampling, and both fit
comfortably.

Both carry provenance in ``attrs``: solution date, Poisson's ratio, elastic
model source and whether it is a placeholder, and the citation for the input
catalog.  That metadata should survive into any figure caption.

The files are plain netCDF4/CF and open in Julia through NCDatasets.jl or
YAXArrays.jl without conversion.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import xarray as xr

from . import strain as st
from .green import STRAIN_COMPONENTS

__all__ = ["component_dataset", "evaluate", "reference_field",
           "DERIVED_VARIABLES"]

DERIVED_VARIABLES = (
    "dilatation", "eps_eq", "max_shear_strain",
    "pressure", "mean_stress", "von_mises_stress", "max_shear_stress",
)

_ATTRS = {
    "dilatation": dict(long_name="volumetric strain change (trace of strain)",
                       units="1", positive_meaning="extension"),
    "eps_eq": dict(long_name="equivalent (von Mises) deviatoric shear strain",
                   units="1"),
    "max_shear_strain": dict(long_name="maximum shear strain, eps_1 - eps_3",
                             units="1"),
    "pressure": dict(long_name="pressure change, -mean stress",
                     units="Pa", positive_meaning="compression"),
    "mean_stress": dict(long_name="mean stress change, trace/3",
                        units="Pa", positive_meaning="tension"),
    "von_mises_stress": dict(long_name="von Mises equivalent stress change",
                             units="Pa"),
    "max_shear_stress": dict(long_name="maximum shear stress change",
                             units="Pa"),
    "coulomb": dict(long_name="Coulomb failure stress change on receiver plane",
                    units="Pa"),
}


def _base_attrs(solution, elastic, extra=None):
    a = {
        "title": "Cascadia slow-slip strain and stress change",
        "source_catalog": (
            "Gualandi, A. (2025). Near real-time Cascadia slow slip events. "
            "Geophysical Journal International, 242(2), ggaf198. "
            "https://doi.org/10.1093/gji/ggaf198"
        ),
        "source_solution_directory": str(getattr(solution, "directory", "")),
        "greens_functions": (
            "Nikkhoo & Walter (2015) triangular dislocation, half-space, "
            "evaluated with cutde"
        ),
        "slip_reference_frame": (
            "Slip is relative to the long-term secular trend, which the vbICA "
            "pre-processing removes. Interseismic loading is NOT included; add "
            "a locking model separately if total strain is required."
        ),
        "sign_convention": "East-North-Up; extension and tension positive",
        "sigma0_selection": getattr(solution, "sigma0_source", None) or "unknown",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "software": "sse-strain 0.1.0",
    }
    a.update(elastic.metadata())
    if extra:
        a.update(extra)
    return a


def component_dataset(grid, solution, elastic, strain_components,
                      amplitudes=None, times=None):
    """Archive-format Dataset: static component fields + amplitude functions.

    Parameters
    ----------
    grid : ObsGrid
    solution : Solution
    elastic : ElasticModel
    strain_components : (n_obs, 6, K)
    amplitudes : (n_time, K), optional
        Defaults to ``solution.component_amplitudes()``.
    """
    if amplitudes is None:
        amplitudes = solution.component_amplitudes()
    if times is None:
        times = solution.dates

    K = strain_components.shape[2]
    arr = grid.reshape(strain_components, trailing=(6, K))

    ds = xr.Dataset(
        data_vars={
            "strain_component": (
                ("depth", "lat", "lon", "tensor", "component"),
                arr.astype("float32"),
                dict(long_name="strain per unit component amplitude", units="1"),
            ),
            "amplitude": (
                ("time", "component"),
                np.asarray(amplitudes, dtype="float64"),
                dict(long_name="vbICA component amplitude (S * V)",
                     units="1",
                     note="dimensionless; slip [m] = L @ amplitude with L "
                          "already converted from the data unit"),
            ),
        },
        coords={
            "lon": ("lon", grid.lon, dict(units="degrees_east")),
            "lat": ("lat", grid.lat, dict(units="degrees_north")),
            "depth": ("depth", grid.depth_m / 1e3,
                      dict(units="km", positive="down")),
            "tensor": ("tensor", list(STRAIN_COMPONENTS)),
            "component": ("component", solution.component_index + 1),
            "time": ("time", np.asarray(times)),
        },
        attrs=_base_attrs(solution, elastic, {
            "product": "component form (reconstruct with einsum over 'component')",
            "poisson_ratio_used": solution.nu,
        }),
    )
    return ds


def evaluate(grid, solution, elastic, strain_components, amplitudes=None,
             times=None, time_slice=None, stride=1,
             variables=DERIVED_VARIABLES, receiver=None, friction=0.4,
             tensor_output=False, float32=True, max_gb=8.0):
    """Materialise strain and stress on the (time, depth, lat, lon) grid.

    Parameters
    ----------
    time_slice : (start, end) as anything numpy can compare to datetime64, or None
    stride : int
        Temporal decimation applied after the slice.  Weekly is ``stride=7``.
    variables : sequence of str
        Subset of :data:`DERIVED_VARIABLES`.
    receiver : (strike, dip, rake) or None
        If given, a ``coulomb`` variable is added for that receiver geometry.
    tensor_output : bool
        Also write the six strain and six stress components.  Multiplies the
        file size by roughly four, so off by default.
    max_gb : float
        Guard on the size of the intermediate strain array.  Raises
        ``MemoryError`` rather than letting the machine swap.

    Notes
    -----
    Memory is the binding constraint.  The array built here is
    ``n_time * n_depth * n_lat * n_lon`` per variable; check that product
    before asking for a decade of daily output.
    """
    if amplitudes is None:
        amplitudes = solution.component_amplitudes()
    if times is None:
        times = solution.dates
    times = np.asarray(times)
    amplitudes = np.asarray(amplitudes, dtype=float)

    sel = np.ones(times.size, dtype=bool)
    if time_slice is not None:
        t0, t1 = np.datetime64(time_slice[0]), np.datetime64(time_slice[1])
        sel &= (times >= t0) & (times <= t1)
    idx = np.flatnonzero(sel)[::stride]
    times_out = times[idx]
    A = amplitudes[idx, :]

    n_t = idx.size
    n_obs = grid.n_obs
    est_gb = n_t * n_obs * 6 * 8 / 1e9
    if est_gb > max_gb:
        raise MemoryError(
            f"this request would allocate about {est_gb:.1f} GB for the strain "
            f"tensor alone ({n_t} epochs x {n_obs} points). Narrow time_slice, "
            "raise stride, or coarsen the grid."
        )

    # (n_time, n_obs, 6)
    eps = np.einsum("obk,tk->tob", strain_components, A, optimize=True)

    depth_per_pt = grid.depth_of_each_point()
    mu = elastic.mu(depth_per_pt)
    lam = elastic.lam(depth_per_pt)
    sig = st.hooke(eps, np.broadcast_to(mu, (n_t, n_obs)),
                   np.broadcast_to(lam, (n_t, n_obs)))

    dtype = "float32" if float32 else "float64"
    dims = ("time", "depth", "lat", "lon")

    def pack(a):
        return (dims, a.reshape((n_t,) + grid.shape).astype(dtype))

    inv = st.stress_invariants(sig)
    pool = {
        "dilatation": st.dilatation(eps),
        "eps_eq": st.equivalent_shear_strain(eps),
        "max_shear_strain": st.max_shear_strain(eps),
        "mean_stress": inv["mean_stress"],
        "pressure": inv["pressure"],
        "von_mises_stress": inv["von_mises_stress"],
        "max_shear_stress": inv["max_shear_stress"],
    }

    data = {}
    for name in variables:
        if name not in pool:
            raise KeyError(f"unknown variable {name!r}; "
                           f"choose from {sorted(pool)}")
        data[name] = (*pack(pool[name]), _ATTRS.get(name, {}))

    if receiver is not None:
        dcfs, shear, normal = st.coulomb_stress(sig, *receiver, friction=friction)
        data["coulomb"] = (*pack(dcfs), dict(
            _ATTRS["coulomb"],
            receiver_strike=receiver[0], receiver_dip=receiver[1],
            receiver_rake=receiver[2], apparent_friction=friction))
        data["shear_stress"] = (*pack(shear),
                                dict(long_name="shear stress change on receiver",
                                     units="Pa"))
        data["normal_stress"] = (*pack(normal),
                                 dict(long_name="normal stress change on receiver",
                                      units="Pa",
                                      positive_meaning="tension (unclamping)"))

    if tensor_output:
        for i, c in enumerate(STRAIN_COMPONENTS):
            data[f"strain_{c}"] = (*pack(eps[..., i]),
                                   dict(long_name=f"strain {c}", units="1"))
            data[f"stress_s{c[1:]}"] = (*pack(sig[..., i]),
                                        dict(long_name=f"stress s{c[1:]}",
                                             units="Pa"))

    ds = xr.Dataset(
        data_vars={k: (v[0], v[1], v[2]) for k, v in data.items()},
        coords={
            "time": ("time", times_out),
            "depth": ("depth", grid.depth_m / 1e3,
                      dict(units="km", positive="down")),
            "lat": ("lat", grid.lat, dict(units="degrees_north")),
            "lon": ("lon", grid.lon, dict(units="degrees_east")),
        },
        attrs=_base_attrs(solution, elastic, {
            "product": "evaluated fields",
            "poisson_ratio_used": solution.nu,
            "temporal_stride_days": stride,
        }),
    )
    return ds


def write(ds, path, complevel=4):
    """Write netCDF4 with compression on every float variable."""
    enc = {
        v: dict(zlib=True, complevel=complevel)
        for v in ds.data_vars
        if np.issubdtype(ds[v].dtype, np.floating)
    }
    ds.to_netcdf(path, engine="netcdf4", encoding=enc)
    return path


def reference_field(ds, reference, per_location=None, variables=None):
    """Zero each field over the dv/v reference interval.

    Parameters
    ----------
    ds : Dataset from :func:`evaluate`
    reference : (start, end)
        Global reference interval, e.g. from
        ``smoothing.reference_from_end(ds.time, 1.0, config.DVV_EDGE_LOSS_DAYS)``.
    per_location : DataArray, optional
        Two variables named ``ref_start`` and ``ref_end`` on the ``(lat, lon)``
        grid, for the case where each dv/v pair has its own usable span.  When
        given, ``reference`` is used only where these are missing.
    variables : sequence of str, optional
        Defaults to every floating-point data variable.

    Why a per-location option exists
    --------------------------------
    A global reference can be folded into the component amplitudes, which is
    free (see :mod:`sse_strain.smoothing`).  A reference that differs by
    location cannot: the subtracted constant becomes a field, so it has to be
    applied after evaluation.  Station coverage in Cascadia grew unevenly, so
    a pair that only came online part way through has a different usable
    reference window from one that spans the whole record, and forcing them
    onto a common baseline reintroduces exactly the offset the referencing was
    meant to remove.
    """
    if variables is None:
        variables = [v for v in ds.data_vars
                     if np.issubdtype(ds[v].dtype, np.floating)
                     and "time" in ds[v].dims]

    out = ds.copy()
    t0, t1 = np.datetime64(reference[0]), np.datetime64(reference[1])
    sel = (ds.time >= t0) & (ds.time <= t1)
    if not bool(sel.any()):
        raise ValueError(f"no epochs inside reference interval {reference}")

    for v in variables:
        base = ds[v].isel(time=sel.values).mean("time")
        if per_location is not None:
            base = _per_location_baseline(ds[v], per_location, fallback=base)
        out[v] = ds[v] - base
        out[v].attrs = dict(ds[v].attrs)
        out[v].attrs["reference_interval"] = (
            f"{np.datetime_as_string(t0, unit='D')}/"
            f"{np.datetime_as_string(t1, unit='D')}"
        )
    out.attrs["dvv_reference_interval"] = out[variables[0]].attrs[
        "reference_interval"]
    out.attrs["dvv_reference_note"] = (
        "Fields are zeroed over the dv/v reference interval. A one-year "
        "reference does not span a whole ETS cycle everywhere in Cascadia, so "
        "a residual saw-tooth offset remains; fit an intercept rather than "
        "forcing a regression through the origin."
    )
    return out


def _per_location_baseline(da, per_location, fallback):
    starts = per_location["ref_start"].values
    ends = per_location["ref_end"].values
    base = fallback.copy()
    it = np.ndindex(starts.shape)
    for ij in it:
        s, e = starts[ij], ends[ij]
        if np.isnat(s) or np.isnat(e):
            continue
        sel = (da.time.values >= s) & (da.time.values <= e)
        if not sel.any():
            continue
        idx = {d: ij[k] for k, d in enumerate(per_location["ref_start"].dims)}
        base.loc[idx] = da.isel(time=sel).mean("time").isel(idx).values
    return base
