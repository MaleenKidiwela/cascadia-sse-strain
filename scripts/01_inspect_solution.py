#!/usr/bin/env python
"""Print the structure of a downloaded Gualandi solution directory.

Run this first, before anything else.  The file and variable names this package
expects were reconstructed from Gualandi's own post-processing code
(https://github.com/Geolandi/sse_postprocessing), not from the files, so the
first thing to establish is whether the download matches.

    python scripts/01_inspect_solution.py data/2025-01-26

If a name differs, edit ``_FILES`` in ``sse_strain/solution.py`` rather than
renaming files, so the mismatch stays documented.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sse_strain.matio import describe, load_mat
from sse_strain.solution import _FILES


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("--depth", type=int, default=3,
                    help="max nesting depth to print")
    args = ap.parse_args()

    d = args.directory
    print(f"Directory: {d.resolve()}\n")

    present = sorted(p.name for p in d.glob("*"))
    print("Files found:")
    for name in present:
        size = (d / name).stat().st_size / 1e6
        print(f"  {name:<24} {size:8.1f} MB")

    expected = {f for f, _ in _FILES.values()}
    missing = expected - set(present)
    extra = set(present) - expected
    if missing:
        print(f"\n  MISSING (expected by sse_strain): {sorted(missing)}")
    if extra:
        print(f"  Not used by sse_strain: {sorted(extra)}")

    for key, (fname, varname) in _FILES.items():
        p = d / fname
        if not p.exists():
            continue
        print(f"\n{'=' * 70}\n{fname}   (expecting top-level variable "
              f"'{varname}')\n{'=' * 70}")
        raw = load_mat(p)
        print(f"top-level keys: {list(raw)}")
        obj = raw.get(varname, raw)
        describe(obj, varname, max_depth=args.depth)

    # the two things most likely to be wrong
    print(f"\n{'=' * 70}\nCritical values to confirm\n{'=' * 70}")
    try:
        opt = load_mat(d / "options.mat")
        opt = opt.get("options", opt)
        nu = opt.get("fault", {}).get("nu")
        ff = opt.get("fault", {}).get("fault_file")
        print(f"  Poisson's ratio (options.fault.nu): {np.ravel(nu)}")
        print(f"  Fault mesh file name:               {ff}")
        print("  -> the mesh file is NOT in this download. See "
              "docs/DATA_SOURCES.md.")
    except Exception as exc:  # noqa: BLE001
        print(f"  could not read options.mat: {exc}")


if __name__ == "__main__":
    main()
