#!/usr/bin/env python
"""Slip distribution and slip rate from a Gualandi solution.

    python scripts/03_slip.py --solution data/2026-02-24 --mesh data/cascadia_mesh.txt --out out/
    python scripts/03_slip.py --synthetic --out synthetic/

Produces the panels that are conventionally used to read Cascadia slow slip:
a latitude-time map of slip rate (the migration plot), cumulative slip, map
views of the slip distribution at chosen epochs, and the equivalent moment
rate.  Look at these before the strain: if the slip does not look like slow
slip, nothing downstream will.

Slip is plotted on the triangular mesh with ``tripcolor`` rather than
interpolated to a regular grid, so what you see is the model as parameterised,
including the patch-to-patch jumps that the strain calculation has to contend
with.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from sse_strain import config, load_fault_ascii, load_solution, smoothing


def build_triangulation(fault):
    """A matplotlib Triangulation over the mesh vertices in lon/lat."""
    verts = np.column_stack([fault.lon.ravel(), fault.lat.ravel()])
    uniq, inv = np.unique(np.round(verts, 6), axis=0, return_inverse=True)
    tri = inv.reshape(fault.n_patch, 3)
    return mtri.Triangulation(uniq[:, 0], uniq[:, 1], tri)


def latitude_bins(fault, n_bins=None):
    """Bin patch centroids by latitude.

    The number of bins is set from the mesh rather than chosen: a triangular
    mesh has a finite number of distinct along-strike rows, and asking for more
    bins than that leaves empty rows that appear as white stripes in the
    latitude-time panels. Half the distinct centroid latitudes is a safe
    default, giving roughly two rows per bin.
    """
    lat = fault.centroid_llh[:, 1]
    if n_bins is None:
        n_bins = max(10, int(np.unique(np.round(lat, 4)).size // 2))
    edges = np.linspace(lat.min(), lat.max(), n_bins + 1)
    idx = np.clip(np.digitize(lat, edges) - 1, 0, n_bins - 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return idx, centres, n_bins


def fill_empty_bins(a):
    """Nearest-neighbour fill along the latitude axis of an (n_time, n_bin) array.

    Empty bins are a property of the mesh, not of the slip, so leaving them as
    gaps misrepresents the model as having no slip there.
    """
    a = np.asarray(a, dtype=float).copy()
    good = np.flatnonzero(np.isfinite(a).any(axis=0))
    if good.size == 0:
        return a
    allb = np.arange(a.shape[1])
    nearest = good[np.argmin(np.abs(allb[:, None] - good[None, :]), axis=1)]
    return a[:, nearest]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solution", type=Path)
    ap.add_argument("--mesh", type=Path)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory; defaults to out/ for real runs "
                         "and synthetic/ for --synthetic, so a synthetic run "
                         "never overwrites real results")
    ap.add_argument("--depth-band", type=float, nargs=2, default=(25.0, 50.0),
                    help="depth range to include in the latitude-time panels")
    ap.add_argument("--n-snapshots", type=int, default=6)
    args = ap.parse_args()
    if not args.synthetic and not (args.solution and args.mesh):
        ap.error("give --solution and --mesh, or --synthetic")

    out = args.out or Path("synthetic" if args.synthetic else "out")
    out.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        from sse_strain.synthetic_cascadia import cascadia_mesh, cascadia_solution
        mesh = out / "synthetic_mesh.txt"
        if not mesh.exists():
            cascadia_mesh(mesh)
        soldir = out / "synthetic_solution"
        if not (soldir / "ICA.mat").exists():
            cascadia_solution(soldir, mesh)
        tag = "SYNTHETIC"
    else:
        mesh, soldir, tag = args.mesh, args.solution, ""

    fault = load_fault_ascii(mesh)
    sol = load_solution(soldir, fault)
    print(f"fault {fault.n_patch} patches; solution {sol.timeline.size} epochs, "
          f"{sol.component_index.size} components")
    sol.invert(verbose=False)
    print("  fractional misfit:", np.round(sol.misfit_check(), 4))

    Ls, Ld = sol.component_slip_patterns()
    A = sol.component_amplitudes()
    times = sol.dates
    dep = -fault.centroid_llh[:, 2]
    in_band = (dep >= args.depth_band[0]) & (dep <= args.depth_band[1])

    # cumulative slip magnitude on every patch, with the published rake flip
    slip_mag = sol.slip_magnitude(apply_rake_flip=True)          # (n_patch, n_time)
    print(f"  peak |slip| = {np.abs(slip_mag).max() * 100:.1f} cm")

    # slip rate via the smoothed amplitudes, so the rate is not differentiated noise
    rate_amp, t_rate = sol.amplitude_rates(window=7)             # per year
    ss_rate = Ls @ rate_amp.T
    ds_rate = Ld @ rate_amp.T
    rate_mag = np.hypot(ss_rate, ds_rate)                        # m/yr, unsigned
    from sse_strain.solution import decimal_year_to_datetime64
    t_rate = decimal_year_to_datetime64(t_rate)
    print(f"  peak slip rate = {rate_mag.max() * 100:.1f} cm/yr")

    idx, lat_centres, n_bins = latitude_bins(fault)

    def bin_by_latitude(field, weights=None, reducer="max"):
        """(n_patch, n_time) -> (n_time, n_bins)."""
        w = fault.area if weights is None else weights
        out_ = np.full((field.shape[1], n_bins), np.nan)
        for b in range(n_bins):
            m = (idx == b) & in_band
            if not m.any():
                continue
            if reducer == "max":
                out_[:, b] = np.abs(field[m]).max(axis=0) * \
                    np.sign(field[m][np.abs(field[m]).argmax(axis=0),
                                     np.arange(field.shape[1])])
            else:
                out_[:, b] = np.average(field[m], axis=0, weights=w[m])
        return out_

    slip_lat = fill_empty_bins(bin_by_latitude(slip_mag, reducer="mean"))
    rate_lat = fill_empty_bins(bin_by_latitude(rate_mag, reducer="mean"))

    # ---- figure 7: latitude-time slip rate and cumulative slip -------------
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    v = np.nanpercentile(rate_lat, 99.0)
    im = axes[0].pcolormesh(t_rate, lat_centres, rate_lat.T, cmap="magma",
                            vmin=0, vmax=v, shading="auto")
    axes[0].set_ylabel("Latitude (deg N)")
    axes[0].set_title(f"Slip rate, area-weighted mean over "
                      f"{args.depth_band[0]:.0f}-{args.depth_band[1]:.0f} km depth")
    plt.colorbar(im, ax=axes[0], pad=0.01).set_label("slip rate (m/yr)")

    v = np.nanpercentile(np.abs(slip_lat), 99.0)
    im = axes[1].pcolormesh(times, lat_centres, slip_lat.T, cmap="RdBu_r",
                            vmin=-v, vmax=v, shading="auto")
    axes[1].set_ylabel("Latitude (deg N)")
    axes[1].set_xlabel("Time")
    axes[1].set_title("Cumulative slip relative to the long-term trend "
                      "(negative = loading faster than secular)")
    plt.colorbar(im, ax=axes[1], pad=0.01).set_label("slip (m)")
    fig.suptitle(f"Cascadia slow slip, latitude-time   {tag}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "fig07_slip_latitude_time.png", dpi=140)
    plt.close(fig)

    # ---- figure 8: map-view slip snapshots at the largest rate epochs ------
    peak = np.nanmax(rate_lat, axis=1)
    order = np.argsort(peak)[::-1]
    picks, used = [], []
    for i in order:
        if all(abs(int(i) - int(j)) > 120 for j in used):
            picks.append(int(i))
            used.append(int(i))
        if len(picks) == args.n_snapshots:
            break
    picks = sorted(picks)

    trg = build_triangulation(fault)
    ncol = 3
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 6.2 * nrow),
                             sharex=True, sharey=True)
    vmax = np.nanpercentile(rate_mag, 99.5)
    for ax, i in zip(np.atleast_1d(axes).ravel(), picks):
        val = rate_mag[:, i]
        tp = ax.tripcolor(trg, facecolors=val, cmap="magma", vmin=0, vmax=vmax)
        ax.set_aspect(1.0 / np.cos(np.deg2rad(float(fault.lat.mean()))))
        ax.set_title(str(np.datetime_as_string(t_rate[i], unit="D")), fontsize=10)
    for ax in np.atleast_1d(axes).ravel()[len(picks):]:
        ax.axis("off")
    cb = fig.colorbar(tp, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("slip rate (m/yr)")
    fig.suptitle(f"Slip rate on the interface at the six largest episodes   {tag}",
                 fontsize=13)
    fig.savefig(out / "fig08_slip_snapshots.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- figure 9: moment rate and the measurement operator ---------------
    mu_fault = 40e9   # FLAGGED: nominal rigidity at 30-45 km; replace with the
                      # value from your elastic model at the interface depth
    area_m2 = fault.area * 1e6
    pot_rate = (rate_mag * area_m2[:, None]).sum(axis=0)       # m^3/yr
    moment_rate = mu_fault * pot_rate

    A31, t31, info = smoothing.smooth_amplitudes(
        sol, window=config.DVV_WINDOW_DAYS,
        reference=smoothing.reference_from_end(times, config.DVV_REFERENCE_YEARS,
                                               config.DVV_EDGE_LOSS_DAYS))
    slip31 = np.hypot(Ls @ A31.T, Ld @ A31.T)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(t_rate, moment_rate, lw=0.8)
    axes[0].set_ylabel("moment rate (N m / yr)")
    axes[0].set_title(f"Equivalent moment rate (mu = {mu_fault / 1e9:.0f} GPa, "
                      "nominal)")

    for k in range(A.shape[1]):
        axes[1].plot(times, A[:, k], lw=0.6,
                     label=f"IC {sol.component_index[k] + 1}")
    axes[1].legend(fontsize=7, ncol=4)
    axes[1].set_ylabel("amplitude (dimensionless)")
    axes[1].set_title("Component amplitude functions, unsmoothed")

    axes[2].plot(times, np.abs(slip_mag[in_band]).max(axis=0), lw=0.8,
                 label="raw daily")
    axes[2].plot(t31, np.abs(slip31[in_band]).max(axis=0), lw=1.4,
                 label=f"{config.DVV_WINDOW_DAYS}-day centred average")
    axes[2].legend(fontsize=8)
    axes[2].set_ylabel("peak |slip| (m)")
    axes[2].set_xlabel("Time")
    axes[2].set_title("Effect of the dv/v measurement operator on peak slip")
    fig.suptitle(f"Moment, components, and smoothing   {tag}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "fig09_moment_and_smoothing.png", dpi=140)
    plt.close(fig)

    print(f"  peak moment rate = {moment_rate.max():.3e} N m/yr "
          f"(Mw {(2 / 3) * (np.log10(moment_rate.max()) - 9.1):.2f} per year "
          "equivalent)")
    print(f"  wrote fig07-fig09 to {out}")


if __name__ == "__main__":
    main()
