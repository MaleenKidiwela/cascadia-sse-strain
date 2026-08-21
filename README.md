# sse-strain

Space-time strain and stress fields on the Cascadia megathrust from published
slow-slip catalogs, built for comparison against coda-wave interferometry dv/v.

The input is a daily solution from Gualandi (2025), *Near real-time Cascadia
slow slip events*, GJI 242(2), ggaf198, https://doi.org/10.1093/gji/ggaf198.
The output is a CF-compliant netCDF holding the full strain and stress tensors
on a lon/lat grid that follows the subduction interface, sampled through time,
already put through the same 31-day averaging operator as the dv/v measurement.

**`out/` holds REAL results** (the 2026-08-18 Gualandi solution on a Slab2
fallback mesh, casc1.6 elastic model — see `docs/RUN_ON_REAL_DATA.md` and the
netCDF attrs for provenance). Synthetic verification output goes to
`synthetic/` instead, comes from a Cascadia-*shaped* fixture, and no figure
produced from it should appear anywhere without the SYNTHETIC label it
carries. The scripts default `--out` accordingly, so a `--synthetic` run can
never overwrite real results.

---

## Install

```bash
git clone <this>  &&  cd cascadia-sse-strain
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                       # 32 tests, ~2 s
```

Full instructions, including conda and GPU backends for `cutde`, are in
[docs/INSTALL.md](docs/INSTALL.md).

## Run it on nothing (no download required)

```bash
python scripts/03_slip.py            --synthetic --out synthetic/
python scripts/04_megathrust_strain.py --synthetic --out synthetic/ --spacing 0.12 --offset 10
```

This produces every figure and the netCDF, from the real code, on a synthetic
fixture. Use it to confirm the install works and to see the output schema
before committing to a download.

## Run it on the real catalog

```bash
./scripts/00_fetch_solution.sh 2026-02-24 data/     # ~96 MB
python scripts/01_inspect_solution.py data/2026-02-24
# obtain the fault mesh -- see docs/RUN_ON_REAL_DATA.md, this is the one obstacle
python scripts/03_slip.py --solution data/2026-02-24 --mesh data/cascadia_mesh.txt --out out/
python scripts/04_megathrust_strain.py --solution data/2026-02-24 \
    --mesh data/cascadia_mesh.txt --out out/ --spacing 0.1 --offset 10 --stride 7 \
    --elastic-csv data/cascadia_casc16_profile.csv \
    --elastic-source "casc1.6 forearc median profile (Stephenson et al., 2017)"
```

Read the slip figures before the strain figures. If the slip does not look like
Cascadia slow slip, nothing downstream is worth reading.

---

## What it does

Slip is reconstructed from the vbICA decomposition as a sum of K static
patterns modulated by K scalar time functions. Strain is linear in slip, so K
forward elastic calculations give the entire 2007-present history; computing
epoch by epoch would be several hundred times the work for identical output.
That structure runs through the whole package — the 31-day measurement operator
is applied to the K amplitude functions, which is mathematically identical to
smoothing the assembled field and costs microseconds.

Green's functions are Nikkhoo & Walter (2015) triangular dislocations in a
homogeneous half-space, evaluated with `cutde`. Strain is converted to stress
with depth-dependent moduli. Both parameterisations, and the approximation
involved in mixing them, are documented in [docs/THEORY.md](docs/THEORY.md).

## Project layout

```
src/sse_strain/
  geodesy.py       polyconic projection, ported from Gualandi's own code
  matio.py         MAT v7 / v7.3 readers, struct flattening
  fault.py         triangular mesh, strike/dip/area, slip rotation, refinement
  solution.py      load a daily solution, rebuild the inversion, slip(x, t)
  green.py         displacement and strain Green's functions via cutde
  elastic.py       Brocher relations, layered models, mu(z) and lambda(z)
  strain.py        invariants, Hooke's law, Coulomb stress
  grid.py          flat constant-depth observation grids
  megathrust.py    interface-following grids, resolution test, spatial low-pass
  smoothing.py     the dv/v measurement operator and its diagnostics
  dataset.py       xarray assembly, referencing, netCDF output
  plotting.py      latitude-time profiles, map views, galleries
  config.py        project constants (31-day window, reference, nu)
  synthetic*.py    test fixtures

scripts/
  00_fetch_solution.sh      download one daily solution
  01_inspect_solution.py    print the .mat tree; run this first
  03_slip.py                slip distribution, rate, moment (fig07-fig09)
  04_megathrust_strain.py   strain and stress tensors (fig01-fig06) + netCDF
  slab2_to_mesh.py          fallback geometry from a Slab2 grid

docs/
  INSTALL.md            environment, backends, troubleshooting
  RUN_ON_REAL_DATA.md   the real-data path, and the mesh problem
  THEORY.md             the mathematics, conventions, and approximations
  DATA_SOURCES.md       provenance of every input and constant
```

## Output

A netCDF with 22 variables on `(time, depth, lat, lon)`: six strain components,
six stress components, the invariants (`dilatation`, `eps_eq`,
`max_shear_strain`), stress measures (`pressure`, `mean_stress`,
`von_mises_stress`, `max_shear_stress`), and Coulomb with its shear and normal
parts. `interface_depth` and `observation_depth` ride along as 2-D coordinates.
Plain CF netCDF4, so `NCDatasets.jl` and `YAXArrays.jl` open it directly.

Provenance travels in `attrs`: solution directory, Poisson's ratio, the
averaging window, the reference interval, the spatial low-pass length, the
elastic model source and whether it is a placeholder, and the effective sample
size. Those belong in figure captions.

---

## Three things established while building this

**cutde's `disp_free`/`strain_free` sum over all triangles.** They are not
pairwise `obs[i]`/`tris[i]` evaluations, despite requiring equal-length inputs.
Building per-patch Green's function columns requires `disp_matrix`. Getting
this backwards produces a `G` whose every column is the whole-fault response;
the inversion still converges and still fits the data, so the error is
invisible downstream. `test_cutde_free_is_summed_not_pairwise` pins it.

**Strain within a few km of the interface is parameterisation noise.** The slip
model is piecewise constant on ~20 km triangles, with 5-8% of peak jumping
across every patch boundary. Field roughness falls from 3.14 at a 3 km
evaluation offset to 1.44 at 30 km, plateauing near the patch dimension.
Refining the triangles while holding slip fixed does *not* diagnose this, since
subdivision leaves the slip jumps where they were. The defaults evaluate 10 km
above the interface and low-pass at the inversion's own correlation length.

**A signed extremum over longitude is the wrong reduction for a tensor.** It is
standard for slip panels because slip magnitude has one sign; components like
eps_EN have lobes of both signs, so the extremum jumps between them as an
episode migrates and stripes the panel with pure artefact. The latitude-time
panels follow a fixed interface depth contour instead.

## Standing flags

- The mesh is rebuilt from Slab2, not Gualandi's own fault file, so the
  inversion is not an exact reproduction of his. A matched-resolution
  McCrory (2012) mesh (`data/cascadia_mesh_mccrory.txt`) agrees to ~1% in
  misfit and ~0.01 in episode Mw, so the geometry choice is second-order —
  but slip near the mesh edges is an artefact, and his mesh (one email)
  remains the clean fix.
- The default elastic model is still the flagged placeholder; the committed
  run in `out/` instead used `--elastic-csv data/cascadia_casc16_profile.csv`,
  a median forearc profile from casc1.6 (Stephenson et al., 2017, via the
  CRESCENT CVM), and carries `elastic_is_placeholder = 0`.
- The Brocher (2005) coefficients were verified against two independent
  secondary sources (see `elastic.py`); the original BSSA paper itself has
  not been checked.
- `geodesy.polyconic` reproduces an eccentricity term that looks like a
  transcription slip in the original code. It is kept for fidelity, with the
  textbook form available behind `strict_ellipsoid=True`.
- Slip is relative to the long-term trend; interseismic loading is absent by
  construction. `fitresult.mat` holds per-station rate terms
  (`mu_east/north/vertical`) but the full trajectory fits are MATLAB opaque
  objects, unreadable outside MATLAB. See DATA_SOURCES.md.

## The committed run

`out/` holds figures from the 2026-08-18 solution: slip reconstructions
whose five largest northern episodes come out at Mw 6.6-6.8 with 1-2 cm of
slip over 2-5 weeks, matching documented ETS events (Aug 2011, Sep 2012,
Sep 2013, ...), and strain/stress fields peaking at ~1.6e-7 max shear and
kPa-scale stress (pressure ~7 kPa, von Mises ~19 kPa, Coulomb ~6 kPa). The
netCDF (117 MB) is gitignored; regenerate it with the commands above after
`./scripts/00_fetch_solution.sh`.

## Citing

Cite Gualandi (2025) for the catalog and Nikkhoo & Walter (2015) for the
Green's functions. `cutde` is Ben Thompson's implementation of the latter.

## License

MIT.
