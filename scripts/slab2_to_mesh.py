#!/usr/bin/env python
"""Build a triangular interface mesh from a Slab2 depth grid.

    python scripts/slab2_to_mesh.py cas_slab2_dep_02.24.18.grd \
        --out data/cascadia_mesh_slab2.txt --spacing 0.25 --depth 10 60

This is the **fallback** geometry, for use when Gualandi's own mesh file is not
available. It will not reproduce his inversion: different patches mean
different Green's functions, and the slip model you recover will differ by an
amount this script cannot bound. See docs/RUN_ON_REAL_DATA.md.

Provenance, flagged
-------------------
Gualandi (2025) cites Hayes et al. (2017) for the Cascadia geometry. Slab2 is
Hayes et al. (2018, Science 362, 58-61), and McCrory et al. (2012, JGR) is the
other interface model in common use for Cascadia. Which of these the "Hayes et
al. (2017)" citation refers to I have not established. Record whichever you use
in the output header and do not assume it matches his.

Slab2 grids are distributed by the USGS (ScienceBase, "Slab2 - A Comprehensive
Subduction Zone Geometry Model"); the Cascadia depth grid is the ``cas_slab2_dep``
file. Depths in Slab2 are negative below sea level, which is the same sign
convention this mesh format uses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_grid(path):
    """Read a Slab2 depth grid (netCDF/GMT .grd) as lon, lat, depth arrays."""
    import xarray as xr

    ds = xr.open_dataset(path)
    name = None
    for cand in ("z", "depth", "Band1"):
        if cand in ds:
            name = cand
            break
    if name is None:
        name = list(ds.data_vars)[0]
    da = ds[name]

    dims = {d.lower(): d for d in da.dims}
    lat_d = dims.get("lat") or dims.get("y")
    lon_d = dims.get("lon") or dims.get("x")
    if lat_d is None or lon_d is None:
        raise ValueError(f"could not identify lon/lat dims in {da.dims}")

    lon = np.asarray(ds[lon_d].values, dtype=float)
    lat = np.asarray(ds[lat_d].values, dtype=float)
    z = np.asarray(da.transpose(lat_d, lon_d).values, dtype=float)
    if lon.max() > 180.0:
        lon = np.where(lon > 180.0, lon - 360.0, lon)
        order = np.argsort(lon)
        lon, z = lon[order], z[:, order]
    return lon, lat, z


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("grid", type=Path, help="Slab2 depth grid")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--spacing", type=float, default=0.25,
                    help="node spacing in degrees; 0.25 gives ~25 km patches, "
                         "comparable to the 21 km correlation length of the "
                         "published inversion")
    ap.add_argument("--depth", type=float, nargs=2, default=(10.0, 60.0),
                    help="depth range in km, positive down")
    ap.add_argument("--lon", type=float, nargs=2, default=None)
    ap.add_argument("--lat", type=float, nargs=2, default=None)
    ap.add_argument("--source", type=str, default="Slab2 (Hayes et al. 2018)")
    args = ap.parse_args()

    from scipy.interpolate import RegularGridInterpolator

    lon, lat, z = read_grid(args.grid)
    print(f"grid {z.shape}, lon {lon.min():.2f}..{lon.max():.2f}, "
          f"lat {lat.min():.2f}..{lat.max():.2f}")

    interp = RegularGridInterpolator((lat, lon), z, bounds_error=False,
                                     fill_value=np.nan)

    lo_r = args.lon or (float(lon.min()), float(lon.max()))
    la_r = args.lat or (float(lat.min()), float(lat.max()))
    LON = np.arange(lo_r[0], lo_r[1] + 1e-9, args.spacing)
    LAT = np.arange(la_r[0], la_r[1] + 1e-9, args.spacing)
    GLO, GLA = np.meshgrid(LON, LAT, indexing="xy")
    H = interp(np.column_stack([GLA.ravel(), GLO.ravel()])).reshape(GLA.shape)

    # Slab2 depths are negative below sea level, matching the mesh convention
    depth_km = -H
    ok = np.isfinite(depth_km) & (depth_km >= args.depth[0]) \
        & (depth_km <= args.depth[1])
    print(f"{ok.sum()} of {ok.size} nodes inside the depth band")

    rows = []
    n_lat, n_lon = GLA.shape
    for i in range(n_lat - 1):
        for j in range(n_lon - 1):
            quad = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            if not all(ok[a, b] for a, b in quad):
                continue
            p = [(GLO[a, b], GLA[a, b], H[a, b]) for a, b in quad]
            rows.append([*p[0], *p[1], *p[2]])
            rows.append([*p[0], *p[2], *p[3]])

    if not rows:
        raise SystemExit("no complete cells inside the depth band; widen "
                         "--depth or coarsen --spacing")

    m = np.array(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = (f"triangular interface mesh from {args.grid.name}\n"
              f"source: {args.source}\n"
              f"spacing {args.spacing} deg, depth band {args.depth} km\n"
              "FALLBACK GEOMETRY: does not reproduce Gualandi's inversion\n"
              "columns: lon1 lat1 h1 lon2 lat2 h2 lon3 lat3 h3 "
              "(deg, deg, km with h negative below sea level)")
    np.savetxt(args.out, m, fmt="%.6f", header=header)
    print(f"wrote {len(rows)} triangles to {args.out}")

    from sse_strain import load_fault_ascii
    f = load_fault_ascii(args.out)
    print(f"  dip {f.dip.min():.1f}-{f.dip.max():.1f} deg, "
          f"depth {(-f.centroid_llh[:, 2]).min():.0f}-"
          f"{(-f.centroid_llh[:, 2]).max():.0f} km, "
          f"mean patch dimension {np.sqrt(f.area.mean()):.1f} km")


if __name__ == "__main__":
    main()
