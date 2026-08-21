"""Reading the Gualandi ``.mat`` files from Python.

The published solutions are Matlab structs.  Depending on how they were saved
they are either MAT v7 (zlib, readable by ``scipy.io.loadmat``) or MAT v7.3
(HDF5, readable by ``h5py``).  ``load_mat`` sniffs the header and dispatches,
and both paths return the same thing: a plain nested ``dict`` of numpy arrays
and strings, with Matlab struct fields as dict keys.

One trap worth knowing about: MAT v7.3 stores arrays in Fortran order, so
everything read through ``h5py`` comes back transposed relative to the Matlab
view.  ``_h5_to_py`` transposes it back, which means downstream code can treat
the two paths identically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

__all__ = ["load_mat", "describe"]


def _is_hdf5(path: Path) -> bool:
    # MAT v7.3 files carry a 512-byte MATLAB user block, so the HDF5
    # signature sits at offset 512, not 0. h5py checks every offset the
    # HDF5 spec allows (0, 512, 1024, ...).
    import h5py

    return h5py.is_hdf5(str(path))


def _simplify(obj):
    """Turn scipy's mat_struct / object arrays into dicts, lists and arrays."""
    if isinstance(obj, sio.matlab.mat_struct):
        return {k: _simplify(getattr(obj, k)) for k in obj._fieldnames}
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            if obj.size == 1:
                return _simplify(obj.item())
            return [_simplify(o) for o in obj.ravel()]
        if obj.dtype.names:  # struct array read without squeeze_me
            return {n: _simplify(obj[n]) for n in obj.dtype.names}
        return obj
    return obj


def _h5_to_py(h5obj, fh):
    import h5py

    if isinstance(h5obj, h5py.Group):
        return {k: _h5_to_py(v, fh) for k, v in h5obj.items()
                if not k.startswith("#")}
    arr = h5obj[()]
    if isinstance(arr, np.ndarray) and arr.dtype == h5py.ref_dtype:
        out = [_h5_to_py(fh[r], fh) for r in arr.ravel()]
        return out[0] if len(out) == 1 else out
    if isinstance(arr, np.ndarray) and arr.dtype.kind == "u" and arr.ndim == 2 \
            and h5obj.attrs.get("MATLAB_class", b"") == b"char":
        return "".join(chr(c) for c in arr.ravel())
    if isinstance(arr, np.ndarray) and arr.ndim >= 2:
        return arr.T  # undo Fortran ordering
    return arr


def load_mat(path) -> dict:
    """Load a ``.mat`` file into nested plain-Python containers."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if _is_hdf5(path):
        import h5py

        with h5py.File(path, "r") as fh:
            return {k: _h5_to_py(v, fh) for k, v in fh.items()
                    if not k.startswith("#")}
    raw = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    return {k: _simplify(v) for k, v in raw.items() if not k.startswith("__")}


def describe(obj, name: str = "", indent: int = 0, max_depth: int = 6) -> None:
    """Print the tree structure of a loaded ``.mat``.

    Used by ``scripts/01_inspect_solution.py``.  The structure of the published
    files is documented in ``docs/DATA_SOURCES.md``, but that documentation was
    reconstructed from Gualandi's post-processing source rather than from the
    files themselves, so run this first and reconcile.
    """
    pad = "  " * indent
    if indent > max_depth:
        print(f"{pad}{name}: ...")
        return
    if isinstance(obj, dict):
        print(f"{pad}{name}/  ({len(obj)} fields)")
        for k, v in obj.items():
            describe(v, k, indent + 1, max_depth)
    elif isinstance(obj, list):
        print(f"{pad}{name}[]  (list, n={len(obj)})")
        if obj:
            describe(obj[0], f"{name}[0]", indent + 1, max_depth)
    elif isinstance(obj, np.ndarray):
        rng = ""
        if obj.size and obj.dtype.kind in "fiu":
            rng = f"  min={np.nanmin(obj):.4g} max={np.nanmax(obj):.4g}"
        print(f"{pad}{name}: ndarray{obj.shape} {obj.dtype}{rng}")
    else:
        s = str(obj)
        print(f"{pad}{name}: {type(obj).__name__} = {s[:70]}")
