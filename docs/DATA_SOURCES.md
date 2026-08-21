# Data sources and provenance

Every input and constant, with an explicit statement of how it was established.
The distinction between **verified**, **reconstructed** and **flagged** is the
point of this document: reconstructed items were inferred from source code
rather than from the data, and flagged items are unchecked.

---

## The catalog

**Gualandi, A. (2025). Near real-time Cascadia slow slip events.**
*Geophysical Journal International* **242**(2), ggaf198.
https://doi.org/10.1093/gji/ggaf198

Solutions: `https://near-real-time-sse.esc.cam.ac.uk/cascadia/`

**VERIFIED** by fetching the directory listing. Dated directories run daily
from 2024-07-08 through at least late February 2026, with occasional gaps. The
listing for `2026-02-24` contains exactly six files:

| file | size | contents | how established |
|---|---|---|---|
| `ICA.mat` | 25 MB | `ICA_essential`: `U`, `S`, `V`, `var_U`, `var_V` | reconstructed from `src_load.jl` |
| `X.mat` | 38 MB | `Xd`: `llh`, `timeline`, `name`, `decmode` | reconstructed from `src_load.jl` |
| `fitresult.mat` | 31 MB | `fitresult`, the trajectory-model fit | reconstructed from `driver.jl` line 53, where it is commented out |
| `options.mat` | 1.8 MB | `options`: inversion, smoothing, rate settings; `fault.nu` | reconstructed from `src_options.jl` |
| `ind_comps.mat` | 2 KB | `ind_comps`, retained component indices | reconstructed |
| `misfit_comps.mat` | 3 KB | `misfit_comps`, cross-validation misfit | reconstructed |

The file *names and sizes* are verified; the *variable names inside them* are
reconstructed from Gualandi's post-processing source, not from the files.
Run `scripts/01_inspect_solution.py` and reconcile against `_FILES` in
`solution.py` before trusting anything downstream.

**Post-processing source** (the basis for every reconstruction above):
- https://github.com/Geolandi/sse_postprocessing (Julia)
- https://github.com/Geolandi/sse_postprocessing_matlab

**Coverage.** Each daily solution reconstructs the entire spatiotemporal
history from the whole GNSS record, so one recent directory covers 2007 to that
date. This is stated in the paper and is why the package reads a single
directory.

**Slip reference frame.** Slip is relative to the long-term trend, which the
vbICA pre-processing removes; there is no positivity constraint. Both stated in
the paper. Interseismic loading is therefore absent by construction.

**`fitresult.mat` — FLAGGED HYPOTHESIS.** The trajectory model is where the
removed secular trend lives, so the loading term is plausibly recoverable from
this file. I have not opened it. Treat as a lead, not a fact.

---

## The fault mesh — NOT IN THE DOWNLOAD

`options.fault.fault_file` names a 9-column ASCII file
(`lon1 lat1 h1 lon2 lat2 h2 lon3 lat3 h3`, degrees and km, height negative
below sea level). `src_load.jl` strips the directory from that name and reads
from `dir_data/Faults/<case>/`, which is Gualandi's own tree.

**VERIFIED** by grep of `src_load.jl` and `load_geom.m`, and by the directory
listing containing no mesh file.

**Geometry provenance — FLAGGED.** The paper cites Hayes et al. (2017). Slab2
is Hayes et al. (2018, *Science* **362**, 58–61); McCrory et al. (2012, *JGR*)
is the other Cascadia interface model in common use. Which of these the 2017
citation refers to has not been established. `scripts/slab2_to_mesh.py` builds
a fallback from a Slab2 grid and labels it as an approximation of unknown
fidelity.

**Recommended action:** request the mesh file from Gualandi. It is the only
route that guarantees matching Green's functions, and it is one email.

---

## Green's functions

**Nikkhoo, M., & Walter, T. R. (2015).** Triangular dislocation: an analytical,
artefact-free solution. *GJI* **201**(2), 1119–1141.
https://doi.org/10.1093/gji/ggv035

Implemented by `cutde` (Ben Thompson), https://github.com/tbenthompson/cutde.
The same kernel family Gualandi used, so the two are consistent by
construction.

**VERIFIED** in this package by two independent physical tests: free-surface
tractions vanish to $10^{-9}$ of the stress scale, and analytic strain matches
a numerically differenced displacement gradient to $2\times10^{-4}$.

---

## Elastic properties

**Brocher, T. M. (2005).** Empirical relations between elastic wavespeeds and
density in the Earth's crust. *BSSA* **95**(6), 2081–2092.

Nafe–Drake density and the $V_p$–$V_s$ regression, his equations 1 and 9.

**FLAGGED.** The polynomial coefficients in `elastic.py` were entered from the
standard published forms of these relations. They have **not** been checked
against the paper. Verify before publication.

**The 1-D profile is a PLACEHOLDER.** `ElasticModel.placeholder_cascadia()`
ships a coarse forearc-crust profile that exists so the pipeline runs end to
end and the output shape is correct. It is not a Cascadia velocity model and is
not cited as one. The Dataset carries `elastic_is_placeholder = 1`.

Candidate replacements, neither verified against the region of interest:
- Stephenson, W. J. (2007). USGS Open-File Report — Cascadia velocity model.
- The CRESCENT community velocity model.
- A local $V_p$/$V_s$ profile from your own tomography.

Poisson's ratio for the half-space is read at run time from
`options.fault.nu`; 0.25 in the published solutions.

---

## Projection

`geodesy.polyconic` is a line-for-line port of `src_coordinates.jl`, kept
deliberately faithful because the fault mesh, station coordinates and Green's
functions of the published inversion are all in that frame. A different
projection would offset the observation grid from the fault patches by a few
hundred metres across Cascadia.

**FLAGGED.** The original eccentricity term reads
`sqrt(1 - esq * (2 * sinp2))` where the textbook polyconic has
`esq * sin^2(phi)`. This looks like a transcription slip that propagated from
the original Matlab through to the Julia port. Its numerical effect is a ~0.3%
correction on the scale factor. The default reproduces the published
expression so coordinates match; `strict_ellipsoid=True` gives the textbook
form. Do not change the default without checking against a patch centroid
computed by his own code.

---

## dv/v measurement parameters

Supplied by the user, recorded in `config.py`:

- `DVV_WINDOW_DAYS = 31`. Odd, so exactly centred, no half-sample shift.
- `DVV_REFERENCE_YEARS = 1.0`, the last year of the stack, chosen because
  station coverage is densest there.
- `DVV_EDGE_LOSS_DAYS = 15`, a consequence of the centred average under strict
  validity.

**FLAGGED:** a one-year reference does not span a whole ETS cycle everywhere
(`DVV_REFERENCE_IS_FULL_CYCLE = False`). See THEORY.md §6.

Still needed from the dv/v side, and not assumed anywhere in the code:
frequency band and lapse-time window (these set the sampled depth range),
per-pair time spans, per-epoch uncertainties, station coordinates, and whether
any detrending or seasonal/hydrological correction was applied.

---

## Other catalogs considered

Retained here because the choice of Gualandi (2025) was not obvious.

**Michel, Gualandi & Avouac (2019b),** *Nature* **574**, 522–526,
doi:10.1038/s41586-019-1673-6. Cascadia SSEs 2007–2017 from surface
deformation. The CaltechAUTHORS record gives the slip model under
`ftp.gps.caltech.edu/pub/avouac/Cascadia_SSE_Nature/`. **UNVERIFIED** — FTP
endpoints of that era often no longer resolve. The companion PAGEOPH paper is
the more useful one here, since it supplies the locking distribution.

**Bartlow (2020),** *GRL*, doi:10.1029/2019GL085303. Mendeley Data
`mc49zmg7n7` holds `results_for_paper.mat` with a long-term **average** ETS
slip rate on a triangular mesh — a time-averaged field, not a space-time
catalog. Her time-dependent NIF catalog for 2006–2019 appears in AGU abstract
form (2021AGUFM.T25C0193B); no public release found. Worth an email rather
than a search.

**Costantino et al. (2026),** *GRL*, doi:10.1029/2025GL117446. GNSS denoising
at 200 Cascadia stations 2007–2023, inverted day by day without temporal
smoothing, giving 1-day slip-rate resolution and four catalogs at different
thresholds. Attractive if you want transients on the timescale of individual
dv/v measurements. The data availability statement could not be reached
(publisher blocks automated fetches), so the repository location is
**UNVERIFIED**.

**PNSN tremor catalog** (Wech). Independent cross-check on episode timing and
along-strike migration; not slip, but useful for overplotting on the
latitude-time panels — `plotting.latitude_time_map` takes a `tremor` argument.

---

## Numbers quoted in this repository that came from the synthetic fixture

To avoid these being mistaken for Cascadia results: the peak slip (3.7 cm),
peak slip rate (1.07 m/yr), moment rate ($2.3\times10^{21}$ N m/yr), strain
magnitudes ($\sim10^{-7}$), stress magnitudes (pressure 12 kPa, von Mises
39 kPa, Coulomb 8 kPa) and the roughness-versus-offset table all come from
`synthetic_cascadia.py`. They have the right order of magnitude — the stress
values sit in the published range for SSE stress drops, which is the sanity
check they were used for — but they are properties of a fixture.

The `cutde` throughput figure (~44,000 obs-patch pairs per second) is real,
measured on a single-core container.
