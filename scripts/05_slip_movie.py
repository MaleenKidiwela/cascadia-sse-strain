#!/usr/bin/env python
"""Animate slip rate on the interface through time.

    python scripts/05_slip_movie.py --solution data/2026-08-18 \
        --mesh data/cascadia_mesh.txt --out out/

Writes ``slip_rate_movie.mp4`` (or ``.gif`` without an ffmpeg backend): one
map frame per ``--stride`` days of the slip-rate magnitude used in fig08,
with the Natural Earth 10m coastline for geographic reference. The rate
comes from the low-passed amplitudes (`Solution.amplitude_rates`), so frames
are not differentiated noise.

ffmpeg is found through ``imageio-ffmpeg`` if installed (pip install
imageio-ffmpeg) or on PATH; otherwise the script falls back to an animated
GIF, which is larger for the same content.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib import animation

from sse_strain import load_fault_ascii, load_solution
from sse_strain.solution import decimal_year_to_datetime64


def build_triangulation(fault):
    """A matplotlib Triangulation over the mesh vertices in lon/lat."""
    verts = np.column_stack([fault.lon.ravel(), fault.lat.ravel()])
    uniq, inv = np.unique(np.round(verts, 6), axis=0, return_inverse=True)
    tri = inv.reshape(fault.n_patch, 3)
    return mtri.Triangulation(uniq[:, 0], uniq[:, 1], tri)


def _writer(fps):
    """FFMpeg if reachable (PATH or imageio-ffmpeg), else Pillow GIF."""
    try:
        import imageio_ffmpeg

        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    if animation.FFMpegWriter.isAvailable():
        return animation.FFMpegWriter(fps=fps, bitrate=1800), ".mp4"
    return animation.PillowWriter(fps=fps), ".gif"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solution", type=Path)
    ap.add_argument("--mesh", type=Path)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory; defaults to out/ for real runs "
                         "and synthetic/ for --synthetic")
    ap.add_argument("--stride", type=int, default=7,
                    help="days between frames")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--coastline", type=Path,
                    default=Path("data/cascadia_coastline_ne10m.json"),
                    help="clipped Natural Earth 10m segments; skipped if absent")
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
        fault = load_fault_ascii(mesh)
        sol = load_solution(soldir, fault)
        tag = "SYNTHETIC"
    else:
        fault = load_fault_ascii(args.mesh)
        sol = load_solution(args.solution, fault)
        tag = ""

    sol.invert(verbose=False)
    Ls, Ld = sol.component_slip_patterns()
    rate_amp, t_rate = sol.amplitude_rates(window=7)              # per year
    rate_mag = np.hypot(Ls @ rate_amp.T, Ld @ rate_amp.T)         # (n_patch, n_t)
    dates = decimal_year_to_datetime64(t_rate)
    frames = np.arange(0, rate_mag.shape[1], args.stride)
    vmax = float(np.nanpercentile(rate_mag, 99.5))
    print(f"{len(frames)} frames, {dates[0]} to {dates[-1]}, "
          f"vmax {vmax * 100:.1f} cm/yr")

    trg = build_triangulation(fault)
    fig, ax = plt.subplots(figsize=(4.6, 7.2))
    tp = ax.tripcolor(trg, facecolors=rate_mag[:, frames[0]], cmap="magma",
                      vmin=0.0, vmax=vmax)
    if args.coastline.exists():
        from matplotlib import patheffects
        halo = [patheffects.withStroke(linewidth=2.4, foreground="white")]
        for seg in json.load(open(args.coastline)):
            seg = np.asarray(seg)
            ax.plot(seg[:, 0], seg[:, 1], color="0.15", lw=0.9, zorder=3,
                    path_effects=halo)
    else:
        print(f"  (no coastline file at {args.coastline}, skipping)")
    pad = 0.7
    ax.set_xlim(fault.lon.min() - pad, fault.lon.max() + 2.2)     # room for coast
    ax.set_ylim(fault.lat.min() - pad, fault.lat.max() + pad)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(float(fault.lat.mean()))))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(tp, ax=ax, shrink=0.7, pad=0.03).set_label("slip rate (m/yr)")
    title = ax.set_title("", fontsize=11)
    fig.tight_layout()

    def update(k):
        i = frames[k]
        tp.set_array(rate_mag[:, i])
        title.set_text(f"Slip rate  {np.datetime_as_string(dates[i], unit='D')}"
                       f"   {tag}")
        return tp, title

    writer, ext = _writer(args.fps)
    path = out / f"slip_rate_movie{ext}"
    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save(path, writer=writer, dpi=110)
    plt.close(fig)
    dur = len(frames) / args.fps
    print(f"wrote {path}  ({len(frames)} frames, {dur:.0f} s at {args.fps} fps)")


if __name__ == "__main__":
    main()
