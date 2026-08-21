#!/usr/bin/env python
"""Strain and stress on the Cascadia megathrust from slow slip -> xarray + figures.

    # real data, once the download and the mesh are in hand
    python scripts/04_megathrust_strain.py \
        --solution data/2025-01-26 --mesh data/cascadia_mesh.txt --out out/

    # synthetic Cascadia-shaped fixture, no download needed
    python scripts/04_megathrust_strain.py --synthetic --out synthetic/

Everything is evaluated on a surface offset a fixed height above the interface.
Strain exactly on a slipping patch is singular and is controlled by the mesh
discretisation rather than by the physics, so the offset is an explicit
parameter and its effect is measured (figure 1) rather than assumed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sse_strain import (ElasticModel, config, dataset, load_fault_ascii,
                        load_solution, smoothing)
from sse_strain.green import strain_from_components
from sse_strain.megathrust import (make_slab_grid, resolution_filter,
                                   resolution_test)
from sse_strain.plotting import along_strike_profile

PROFILE_DEPTH = [35.0]

STRAIN_LABELS = {
    "exx": r"$\varepsilon_{EE}$", "eyy": r"$\varepsilon_{NN}$",
    "ezz": r"$\varepsilon_{UU}$", "exy": r"$\varepsilon_{EN}$",
    "exz": r"$\varepsilon_{EU}$", "eyz": r"$\varepsilon_{NU}$",
}


def sym_limits(a, q=99.0):
    v = np.nanpercentile(np.abs(a), q)
    return (-v, v) if v > 0 else (-1.0, 1.0)


def build(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        from sse_strain.synthetic_cascadia import cascadia_mesh, cascadia_solution
        mesh = out / "synthetic_mesh.txt"
        cascadia_mesh(mesh)
        soldir = cascadia_solution(out / "synthetic_solution", mesh,
                                   n_time=args.n_time)
        tag = "SYNTHETIC"
    else:
        mesh, soldir, tag = Path(args.mesh), Path(args.solution), ""

    fault = load_fault_ascii(mesh)
    sol = load_solution(soldir, fault)
    print(f"fault: {fault.n_patch} patches, dip {fault.dip.min():.1f}-"
          f"{fault.dip.max():.1f} deg, depth "
          f"{-fault.centroid_llh[:, 2].max():.0f}-"
          f"{-fault.centroid_llh[:, 2].min():.0f} km")
    print(f"solution: {sol.timeline.size} epochs, "
          f"{sol.component_index.size} components, nu={sol.nu}")

    print("inverting components ...")
    sol.invert(verbose=False)
    print("  fractional misfit:", np.round(sol.misfit_check(), 4))

    Ls, Ld = sol.component_slip_patterns()
    slip_k = fault.slip_enu(Ls, Ld)

    # ---- figure 1: where does the field stop being parameterisation noise?
    print("resolution test ...")
    offs, rough, rms = resolution_test(fault, slip_k, sol.nu)
    patch_km = float(np.sqrt(fault.area.mean()))
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax[0].plot(offs, rough, "o-")
    ax[0].set_xlabel("offset above interface (km)")
    ax[0].set_ylabel("RMS Laplacian / RMS field")
    ax[1].plot(offs, rms, "o-")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("offset above interface (km)")
    ax[1].set_ylabel("RMS dilatation (per unit amp.)")
    for a in ax:
        a.axvline(patch_km, color="C3", ls="--", lw=1)
        a.text(patch_km, a.get_ylim()[0], " patch dimension", color="C3",
               fontsize=8, va="bottom")
    fig.suptitle("Grid-scale roughness from patch-to-patch slip jumps   "
                 f"{tag}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig01_resolution_test.png", dpi=150)
    plt.close(fig)
    for h, r, m in zip(offs, rough, rms):
        print(f"   offset {h:5.1f} km   roughness {r:.2f}   RMS {m:.3e}")
    print(f"  mean patch dimension {patch_km:.1f} km")

    # ---- the megathrust surface -------------------------------------------
    grid = make_slab_grid(fault, spacing_deg=args.spacing,
                          offset_km=args.offset,
                          depth_limits_km=(args.dmin, args.dmax))
    n_valid = int(grid.valid.sum())
    print(f"grid: {grid.lon.size} x {grid.lat.size}, {n_valid} valid points, "
          f"offset {args.offset} km above interface")

    print(f"strain Green's functions ({n_valid} pts x {fault.n_patch} patches) ...")
    E = strain_from_components(grid.points_m, fault.vertices_m(), slip_k,
                               sol.nu, progress=True)

    # ---- measurement operator ---------------------------------------------
    ref = smoothing.reference_from_end(sol.dates, config.DVV_REFERENCE_YEARS,
                                       config.DVV_EDGE_LOSS_DAYS)
    A, times, info = smoothing.smooth_amplitudes(
        sol, window=config.DVV_WINDOW_DAYS, reference=ref)
    print(f"31-day centred average: kept {info['kept']}/{info['total']} epochs, "
          f"N_eff ~ {info['effective_sample_size']:.0f}")
    print(f"reference interval: {ref[0]} to {ref[1]}")

    # ---- elastic properties at megathrust depth ---------------------------
    if args.elastic_csv:
        em = ElasticModel.from_csv(args.elastic_csv, nu_greens=sol.nu,
                                   source=args.elastic_source)
    else:
        em = ElasticModel.placeholder_cascadia(nu_greens=sol.nu)
    d_mid = np.nanmean(grid.observation_depth_km) * 1e3
    print(f"elastic at {d_mid / 1e3:.0f} km: mu={em.mu(d_mid) / 1e9:.1f} GPa, "
          f"lambda={em.lam(d_mid) / 1e9:.1f} GPa, nu={float(em.poisson(d_mid)):.3f}")
    if em.is_placeholder:
        print("  *** PLACEHOLDER elastic model -- replace before quantitative use")

    # depth varies across the grid, so give evaluate the per-point depths
    grid.depth_m = np.array([d_mid])
    ds = dataset.evaluate(grid, sol, em, E, amplitudes=A, times=times,
                          stride=args.stride, tensor_output=True,
                          receiver=(args.rstrike, args.rdip, args.rrake))

    # mask outside the resolved interface
    mask = grid.valid[None, None, :, :]
    for v in ds.data_vars:
        if {"lat", "lon"} <= set(ds[v].dims):
            ds[v] = ds[v].where(mask)

    # spatial low-pass at the inversion's own resolution length
    lam = float(np.ravel(sol.options.get("inversion", {}).get(
        "lambda_dip", 21.0))[0])
    if args.filter_km is not None:
        lam = args.filter_km
    if lam > 0:
        print(f"low-passing at the inversion correlation length, {lam:.0f} km")
        for v in list(ds.data_vars):
            if {"lat", "lon"} <= set(ds[v].dims):
                a = ds[v].values
                sh = a.shape
                ds[v] = (ds[v].dims,
                         resolution_filter(a.reshape((-1,) + sh[-2:]), grid,
                                           lam).reshape(sh).astype("float32"),
                         ds[v].attrs)

    ds = ds.assign_coords(
        interface_depth=(("lat", "lon"), grid.interface_depth_km),
        observation_depth=(("lat", "lon"), grid.observation_depth_km))
    ds.attrs.update(
        surface="megathrust-following, fixed vertical offset above interface",
        offset_above_interface_km=args.offset,
        dvv_window_days=config.DVV_WINDOW_DAYS,
        dvv_reference_interval=f"{ref[0]}/{ref[1]}",
        effective_sample_size=float(info["effective_sample_size"]),
        synthetic=int(bool(args.synthetic)),
        spatial_lowpass_km=lam,
        spatial_lowpass_note=(
            "Gaussian low-pass at the inversion prior correlation length. "
            "The slip model is piecewise constant on ~20 km patches, so "
            "structure finer than this reflects the parameterisation."),
    )
    if args.synthetic:
        ds.attrs["WARNING"] = (
            "SYNTHETIC fixture with Cascadia-like geometry. Not Cascadia.")

    nc = out / ("megathrust_strain_SYNTHETIC.nc" if args.synthetic
                else "megathrust_strain.nc")
    dataset.write(ds, nc)
    print(f"wrote {nc}  ({nc.stat().st_size / 1e6:.1f} MB)")
    return ds, sol, fault, grid, out, tag


def figures(ds, sol, fault, grid, out, tag, profile_depth=35.0):
    PROFILE_DEPTH[0] = profile_depth
    lonc, latc = ds.lon.values, ds.lat.values
    aspect = 1.0 / np.cos(np.deg2rad(float(latc.mean())))

    # pick the epoch of largest deviatoric strain
    ti = int(np.nanargmax(np.abs(ds.eps_eq).max(("lat", "lon", "depth")).values))
    tstr = str(ds.time.values[ti])[:10]

    # ---- figure 2: all six strain components, map view --------------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 11), sharex=True, sharey=True)
    allv = np.concatenate([ds[f"strain_{c}"].isel(time=ti).values.ravel()
                           for c in STRAIN_LABELS])
    vmin, vmax = sym_limits(allv)
    for ax, c in zip(axes.ravel(), STRAIN_LABELS):
        a = ds[f"strain_{c}"].isel(time=ti, depth=0).values
        im = ax.pcolormesh(lonc, latc, a, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                           shading="auto")
        ax.contour(lonc, latc, ds.interface_depth.values,
                   levels=[30, 35, 40, 45], colors="0.35", linewidths=0.5)
        ax.set_title(STRAIN_LABELS[c], fontsize=13)
        ax.set_aspect(aspect)
    for ax in axes[-1]:
        ax.set_xlabel("Longitude")
    for ax in axes[:, 0]:
        ax.set_ylabel("Latitude")
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("strain (dimensionless)")
    fig.suptitle(f"Strain tensor on the megathrust, {tstr}   {tag}", fontsize=13)
    fig.savefig(out / "fig02_strain_components_map.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    # ---- figure 3: all six components, latitude-time ----------------------
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True, sharey=True)
    for ax, c in zip(axes.ravel(), STRAIN_LABELS):
        f, _, _ = along_strike_profile(ds, f"strain_{c}", PROFILE_DEPTH[0])
        vmin, vmax = sym_limits(f)
        im = ax.pcolormesh(ds.time.values, latc, f.T, cmap="RdBu_r",
                           vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(STRAIN_LABELS[c], fontsize=12)
        plt.colorbar(im, ax=ax, pad=0.01)
    for ax in axes[-1]:
        ax.set_xlabel("Time")
    for ax in axes[:, 0]:
        ax.set_ylabel("Latitude")
    fig.suptitle(f"Strain tensor along the {PROFILE_DEPTH[0]:.0f} km interface "
                 f"depth contour   {tag}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "fig03_strain_components_lat_time.png", dpi=140)
    plt.close(fig)

    # ---- figure 4: invariants ---------------------------------------------
    inv = [("dilatation", r"$\varepsilon_{kk}$ (volumetric)", "1"),
           ("eps_eq", r"$\varepsilon_{eq}$ (equivalent shear)", "1"),
           ("max_shear_strain", r"$\varepsilon_1-\varepsilon_3$", "1")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    for j, (v, lab, unit) in enumerate(inv):
        a = ds[v].isel(time=ti, depth=0).values
        vmin, vmax = sym_limits(a)
        if v != "dilatation":
            vmin = 0.0
        im = axes[0, j].pcolormesh(lonc, latc, a,
                                   cmap="RdBu_r" if v == "dilatation" else "magma",
                                   vmin=vmin, vmax=vmax, shading="auto")
        axes[0, j].set_aspect(aspect)
        axes[0, j].set_title(f"{lab}\n{tstr}", fontsize=11)
        plt.colorbar(im, ax=axes[0, j], pad=0.02)

        f, _, _ = along_strike_profile(ds, v, PROFILE_DEPTH[0])
        vmin, vmax = sym_limits(f)
        # the shear invariants are positive definite; a diverging map would
        # imply a sign they cannot have
        seq = v != "dilatation"
        im = axes[1, j].pcolormesh(
            ds.time.values, latc, f.T,
            cmap="magma" if seq else "RdBu_r",
            vmin=0.0 if seq else vmin, vmax=vmax, shading="auto")
        axes[1, j].set_ylabel("Latitude")
        plt.colorbar(im, ax=axes[1, j], pad=0.02)
    fig.suptitle(f"Strain invariants   {tag}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "fig04_invariants.png", dpi=140)
    plt.close(fig)

    # ---- figure 5: stress and Coulomb -------------------------------------
    sv = [("pressure", "pressure change (Pa, + compression)"),
          ("von_mises_stress", "von Mises stress (Pa)"),
          ("coulomb", "Coulomb stress change (Pa)"),
          ("normal_stress", "normal stress (Pa, + unclamping)")]
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    for j, (v, lab) in enumerate(sv):
        a = ds[v].isel(time=ti, depth=0).values
        vmin, vmax = sym_limits(a)
        im = axes[0, j].pcolormesh(lonc, latc, a, cmap="RdBu_r",
                                   vmin=vmin, vmax=vmax, shading="auto")
        axes[0, j].set_aspect(aspect)
        axes[0, j].set_title(f"{lab}\n{tstr}", fontsize=10)
        plt.colorbar(im, ax=axes[0, j], pad=0.02)
        f, _, _ = along_strike_profile(ds, v, PROFILE_DEPTH[0])
        vmin, vmax = sym_limits(f)
        im = axes[1, j].pcolormesh(ds.time.values, latc, f.T, cmap="RdBu_r",
                                   vmin=vmin, vmax=vmax, shading="auto")
        axes[1, j].set_ylabel("Latitude")
        plt.colorbar(im, ax=axes[1, j], pad=0.02)
    fig.suptitle(f"Stress change on the megathrust   {tag}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "fig05_stress.png", dpi=140)
    plt.close(fig)

    # ---- figure 6: internal consistency checks ----------------------------
    eps = np.stack([ds[f"strain_{c}"].isel(depth=0).values
                    for c in STRAIN_LABELS], -1)
    tr = eps[..., 0] + eps[..., 1] + eps[..., 2]
    dil = ds.dilatation.isel(depth=0).values
    p = ds.pressure.isel(depth=0).values

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.plot(tr.ravel(), dil.ravel(), ".", ms=1, alpha=0.3)
    ax.set_xlabel(r"$\varepsilon_{EE}+\varepsilon_{NN}+\varepsilon_{UU}$")
    ax.set_ylabel("dilatation variable")
    ax.set_title(f"trace consistency, max |diff| = "
                 f"{np.nanmax(np.abs(tr - dil)):.2e}", fontsize=9)

    ax = axes[0, 1]
    lat_i = ds.sizes["lat"] // 2
    lon_i = ds.sizes["lon"] // 2
    for c in STRAIN_LABELS:
        ax.plot(ds.time.values,
                ds[f"strain_{c}"].isel(depth=0, lat=lat_i, lon=lon_i).values,
                lw=0.8, label=STRAIN_LABELS[c])
    ax.legend(fontsize=7, ncol=3)
    ax.set_title(f"all components at {float(ds.lat[lat_i]):.1f}N, "
                 f"{float(ds.lon[lon_i]):.1f}E", fontsize=9)

    ax = axes[1, 0]
    A = sol.component_amplitudes()
    for k in range(A.shape[1]):
        ax.plot(sol.dates, A[:, k], lw=0.7,
                label=f"IC {sol.component_index[k] + 1}")
    ax.legend(fontsize=7)
    ax.set_title("component amplitude functions (unsmoothed)", fontsize=9)

    ax = axes[1, 1]
    ax.plot(ds.time.values,
            np.nanmax(np.abs(ds.eps_eq.isel(depth=0).values), axis=(1, 2)),
            lw=0.9)
    ax.set_yscale("log")
    ax.set_title(r"peak $\varepsilon_{eq}$ over the grid vs time", fontsize=9)
    fig.suptitle(f"Consistency checks   {tag}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "fig06_consistency.png", dpi=140)
    plt.close(fig)

    print("\nMagnitudes (peak over grid and time):")
    for v in ("dilatation", "eps_eq", "max_shear_strain"):
        print(f"  {v:<18} {np.nanmax(np.abs(ds[v].values)):.3e}")
    for v in ("pressure", "von_mises_stress", "coulomb"):
        print(f"  {v:<18} {np.nanmax(np.abs(ds[v].values)):.3e} Pa "
              f"({np.nanmax(np.abs(ds[v].values)) / 1e3:.2f} kPa)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solution", type=Path)
    ap.add_argument("--mesh", type=Path)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory; defaults to out/ for real runs "
                         "and synthetic/ for --synthetic, so a synthetic run "
                         "never overwrites real results")
    ap.add_argument("--spacing", type=float, default=0.15)
    ap.add_argument("--offset", type=float, default=3.0,
                    help="km above the interface")
    ap.add_argument("--dmin", type=float, default=25.0)
    ap.add_argument("--dmax", type=float, default=50.0)
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--profile-depth", type=float, default=35.0,
                    help="interface depth contour for latitude-time panels")
    ap.add_argument("--filter-km", type=float, default=None,
                    help="spatial low-pass length; default = inversion lambda")
    ap.add_argument("--n-time", type=int, default=5600)
    ap.add_argument("--elastic-csv", type=Path, default=None,
                    help="depth profile CSV (depth_km, vp_km_s[, vs_km_s, "
                         "rho_kg_m3]) used for the strain-to-stress moduli; "
                         "default is the flagged placeholder")
    ap.add_argument("--elastic-source", type=str, default=None,
                    help="provenance string recorded in the netCDF for "
                         "--elastic-csv")
    ap.add_argument("--rstrike", type=float, default=0.0)
    ap.add_argument("--rdip", type=float, default=12.0)
    ap.add_argument("--rrake", type=float, default=90.0)
    args = ap.parse_args()
    if not args.synthetic and not (args.solution and args.mesh):
        ap.error("give --solution and --mesh, or --synthetic")
    if args.out is None:
        args.out = Path("synthetic" if args.synthetic else "out")
    figures(*build(args), profile_depth=args.profile_depth)


if __name__ == "__main__":
    main()
