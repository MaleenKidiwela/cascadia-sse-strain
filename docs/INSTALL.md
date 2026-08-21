# Installation

## Requirements

Python 3.10 or newer. The package was developed and tested on:

| package | tested version | why |
|---|---|---|
| Python | 3.12.3 | |
| numpy | 2.4.4 | |
| scipy | 1.17.1 | FIR filtering, interpolation, Gaussian smoothing |
| xarray | 2026.7.0 | output container |
| netCDF4 | 1.7.4 | netCDF writing with compression |
| h5py | 3.16.0 | MAT v7.3 files, which `scipy.io.loadmat` cannot read |
| matplotlib | 3.10.8 | figures |
| cutde | 26.3.6 | Nikkhoo & Walter (2015) triangular dislocation kernels |

Lower versions will probably work; these are what the 32 tests were run
against. `pyproject.toml` pins only lower bounds.

---

## venv

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"
```

## conda

```bash
conda create -n sse-strain -c conda-forge python=3.12 \
    numpy scipy xarray netcdf4 h5py matplotlib pytest
conda activate sse-strain
pip install cutde                  # not on conda-forge; pip is fine here
pip install -e .
```

Installing the compiled scientific stack through conda-forge and only `cutde`
through pip avoids the usual netCDF/HDF5 library conflicts on macOS and on
HPC systems with a module-provided HDF5.

---

## Verify

```bash
pytest -q
```

Expect `27 passed` in about two seconds. Then run the pipeline on the synthetic
fixture, which needs no download and exercises every code path:

```bash
python scripts/03_slip.py --synthetic --out out/
python scripts/04_megathrust_strain.py --synthetic --out out/ --spacing 0.2 --offset 10
```

That writes nine figures and a netCDF to `out/`. If those appear and the
printed misfits are of order 0.01-0.02, the install is good.

---

## cutde backends

`cutde` picks a backend at import: CUDA, then OpenCL, then a C++ CPU backend
compiled on first use. The first kernel call therefore takes several seconds
while it compiles; that is not a hang.

Throughput matters here. On the single-core container used for development the
CPU backend ran at roughly **44,000 observation-patch pairs per second**. The
cost of a strain field is `n_obs x n_patch` kernel evaluations:

| grid | patches | pairs | CPU, 1 core | CPU, 8 cores |
|---|---|---|---|---|
| 0.2 deg, 1 level | 600 | 6 x 10^5 | ~15 s | ~2 s |
| 0.1 deg, 1 level | 3000 | 1 x 10^7 | ~4 min | ~30 s |
| 0.1 deg, 3 levels | 3000 | 3 x 10^7 | ~12 min | ~90 s |

A GPU is worth roughly an order of magnitude. To check what you got:

```python
import cutde
print(cutde.backend)
```

For CUDA install `pycuda`; for OpenCL install `pyopencl` plus an ICD for your
device. Neither is required — the CPU path is the tested one.

Parallelism inside this package is per-chunk, controlled by `n_jobs` on
`green.strain_from_components`; `-1` uses every core. Memory per chunk is about
`max_pairs * 18 * 8` bytes, so the 2,000,000 default is roughly 300 MB.

---

## Memory

The binding constraint is `dataset.evaluate`, which allocates
`n_time x n_obs x 6` float64 for the strain tensor before deriving anything.
There is a guard: requests above `max_gb` (default 8) raise `MemoryError` with
the size rather than sending the machine into swap. If you hit it, narrow
`time_slice`, raise `stride`, or coarsen the grid.

For the whole record at full grid resolution, do not materialise it. Use
`dataset.component_dataset`, which stores the K static strain fields and the K
amplitude functions separately — complete, lossless, and small.

---

## Troubleshooting

**`ModuleNotFoundError: cutde`** — `pip install cutde`. It is not on
conda-forge.

**First strain call takes 10+ seconds** — backend compilation. Subsequent calls
are fast.

**`UserWarning: The tris input array is not C-contiguous`** — a `cutde`
performance warning. `FaultMesh.vertices_m()` returns contiguous arrays, so this
only appears if you slice or transpose them yourself; wrap in
`np.ascontiguousarray`.

**`ValueError: could not read <path>` on a `.mat`** — the file is MAT v7.3.
`matio.load_mat` sniffs the HDF5 header and dispatches to `h5py`, so check that
`h5py` imported cleanly.

**Fault mesh loads but dips look wrong** — check the column order. The expected
format is `lon1 lat1 h1 lon2 lat2 h2 lon3 lat3 h3` with height in km, negative
below sea level. `load_fault_ascii` recomputes strike, dip and area from the
vertices rather than reading them, so a column mix-up shows up as implausible
dip values rather than as an error.

**Misfits from `sol.misfit_check()` are large (>0.1)** — see step 4 of
[RUN_ON_REAL_DATA.md](RUN_ON_REAL_DATA.md). Do not proceed to strain; the
likely causes are a mismatched mesh, a missing `ind_sigma0_comps`, or a
projection origin that differs from the one the inversion used.
