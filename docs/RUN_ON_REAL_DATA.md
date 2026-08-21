# Running on the real Gualandi catalog

Real output lives in `out/`; synthetic verification output (a Cascadia-shaped
fixture, not Cascadia) goes to `synthetic/` and always carries a SYNTHETIC
label. This document is the path that produced the real output — it was walked
end to end on 2026-08-20 with the 2026-08-18 solution (see the session notes
in the parent directory's `notes/` folder).

---

## 1. Download a solution

```bash
chmod +x scripts/00_fetch_solution.sh
./scripts/00_fetch_solution.sh 2026-02-24 data/
```

The server is `https://near-real-time-sse.esc.cam.ac.uk/cascadia/`, with dated
directories from 2024-07-08 onward (a few days are missing). The listing for
2026-02-24 was confirmed to contain:

| file | size | contents |
|---|---|---|
| `ICA.mat` | 25 MB | vbICA decomposition: `U`, `S`, `V`, `var_U`, `var_V` |
| `X.mat` | 38 MB | input position time series, station `llh`, `timeline` |
| `fitresult.mat` | 31 MB | trajectory-model fit |
| `options.mat` | 1.8 MB | inversion, smoothing and rate settings; `fault.nu` |
| `ind_comps.mat` | 2 KB | indices of retained components |
| `misfit_comps.mat` | 3 KB | cross-validation misfit curves |

**One dated directory is enough.** Each daily solution reconstructs the entire
spatiotemporal history from the whole GNSS record, so a 2026 date covers 2007
to 2026. Downloading many dates only helps if you want to study how the
near-real-time solution itself evolves as data arrives.

**`fitresult.mat` deserves a look.** It is commented out in Gualandi's own
`driver.jl` (line 53) and holds the trajectory-model fit of the GNSS series.
That model is where the secular trend removed before the ICA lives. Since the
slip in the ICA product is defined *relative to the long-term trend*, the
interseismic loading term you need for total strain is plausibly recoverable
from this file. I have not opened it, so treat that as a hypothesis to check
with `scripts/01_inspect_solution.py`, not an established fact.

---

## 2. Get the fault mesh — the one genuine obstacle

`options.fault.fault_file` names a 9-column ASCII file
(`lon1 lat1 h1 lon2 lat2 h2 lon3 lat3 h3`, degrees and km with height negative
below sea level). Gualandi's `src_load.jl` strips the directory from that name
and looks in `dir_data/Faults/<case>/`, so **the mesh is not in the download**.
Three options, in order of preference:

1. **Ask Gualandi for the mesh file.** This is the only route that guarantees
   your patches match his inversion exactly. Anything else changes the Green's
   functions, and therefore the slip model you recover, by an amount you cannot
   easily bound. One email.

2. **Check `options.mat` first.** It is 1.8 MB, which is large for a settings
   struct, so it may embed geometry. `scripts/01_inspect_solution.py` prints
   the whole tree — look for anything patch-shaped before assuming you need
   route 1.

3. **Rebuild from a slab model** with `scripts/slab2_to_mesh.py`. The paper
   cites Hayes et al. (2017) for the geometry. Note that Slab2 is Hayes et al.
   (2018, *Science*), and McCrory et al. (2012) is the other commonly used
   Cascadia interface; which of these corresponds to "Hayes et al. (2017)" I
   have not established, so a rebuilt mesh is an approximation of unknown
   fidelity. Use it to get the pipeline moving, not for published numbers.

---

## 3. Inspect before trusting

```bash
python scripts/01_inspect_solution.py data/2026-02-24
```

The file and variable names this package expects were reconstructed from
Gualandi's post-processing source, not from the files themselves. Reconcile
what this prints against `_FILES` in `sse_strain/solution.py` before going
further. Confirm in particular:

- `options.fault.nu` (expected 0.25) — it enters the strain field directly.
- `options.inversion.lambda_dip` (expected 21 km) — sets the spatial low-pass.
- whether `ind_sigma0_comps` is present. Without it the per-component
  smoothing hyperparameter falls back to a single value and **your slip model
  will differ from the published one.**

---

## 4. Validate the inversion before computing any strain

```python
from sse_strain import load_fault_ascii, load_solution
fault = load_fault_ascii("data/cascadia_mesh.txt")
sol   = load_solution("data/2026-02-24", fault)
G     = sol.displacement_gf()
sol.invert(G=G)
print(sol.misfit_check(G=G))          # compare against misfit_comps.mat minima
```

If those fractional misfits sit near the minima of `misfit_comps.mat`, the
projection, the Green's functions and the prior have all been reproduced. If
they do not, **stop and debug there.** A sign error in the strike-slip
convention shows up in this number and nowhere else downstream — the strain
fields will look entirely plausible and be wrong.

---

## 5. Run

```bash
python scripts/03_slip.py \
    --solution data/2026-02-24 --mesh data/cascadia_mesh.txt --out out/

python scripts/04_megathrust_strain.py \
    --solution data/2026-02-24 --mesh data/cascadia_mesh.txt --out out/ \
    --spacing 0.1 --offset 10 --stride 7 \
    --elastic-csv data/cascadia_casc16_profile.csv \
    --elastic-source "casc1.6 forearc median profile (Stephenson et al., 2017)"
```

Look at `fig07`–`fig09` (slip) before `fig02`–`fig06` (strain). If the slip
does not look like Cascadia slow slip — episodes of a few centimetres, two to
four weeks long, recurring every 8–22 months and migrating along strike —
nothing downstream is worth reading.

**Cost.** `cutde` ran at roughly 44k obs-patch pairs per second single-threaded
here. The real mesh is likely a few thousand patches; a 0.1° grid over Cascadia
is a few thousand points. Expect tens of minutes on several cores. `cutde`
picks up a CUDA or OpenCL backend automatically if one is available, which is
worth an order of magnitude.

---

## 6. Before anything quantitative

Two flags are live in the code and will follow you into the netCDF metadata:

- The elastic model is `ElasticModel.placeholder_cascadia()`, and the output
  carries `elastic_is_placeholder = 1`. Substitute Stephenson (2007) or the
  CRESCENT community velocity model.
- The Brocher (2005) coefficients in `elastic.py` were transcribed from the
  standard published forms without being checked against the paper.

And one methodological point that the synthetic run established: at an
evaluation offset of 3 km above the interface, the strain field is dominated by
the slip discontinuities between adjacent patches rather than by the source.
Roughness plateaus only once the offset approaches the patch dimension. The
defaults evaluate at 10 km and low-pass at the inversion's own correlation
length; `resolution_test()` re-measures this for the real mesh, and you should
run it, since the real patch dimension may differ from the fixture's 21.6 km.
