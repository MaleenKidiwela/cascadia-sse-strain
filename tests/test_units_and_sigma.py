"""Guards added after the first real-data run.

Two failure modes surfaced there that the synthetic fixtures cannot see:
the published files store positions in mm while the fixtures are built in
metres (a silent 1000x slip error), and the fixtures ship an all-zero
misfit_comps that must not be mistaken for a cross-validation curve.
"""

from pathlib import Path

import numpy as np
import pytest

from sse_strain.solution import Solution


def _sol(X):
    return Solution(directory=Path("."), options={}, X=X, ICA={},
                    fault=None, ind_comps=np.array([1.0]))


def test_data_unit_factor_known_units():
    assert _sol({"unit": "mm"}).data_unit_to_m == 1e-3
    assert _sol({"unit": "cm"}).data_unit_to_m == 1e-2
    assert _sol({"unit": "m"}).data_unit_to_m == 1.0
    assert _sol({"unit": " MM "}).data_unit_to_m == 1e-3


def test_data_unit_factor_missing_defaults_to_metres():
    assert _sol({}).data_unit_to_m == 1.0


def test_data_unit_factor_unknown_raises():
    with pytest.raises(ValueError, match="unrecognised data unit"):
        _sol({"unit": "furlong"}).data_unit_to_m


def _load_fixture(tmp_path):
    from sse_strain.synthetic import make_synthetic_mesh, write_synthetic_solution
    from sse_strain import load_fault_ascii, load_solution

    mesh = tmp_path / "mesh.txt"
    make_synthetic_mesh(mesh, n_strike=10, n_dip=4)
    write_synthetic_solution(tmp_path / "sol", mesh, n_time=300, n_comp=3)
    fault = load_fault_ascii(mesh)
    return load_solution(tmp_path / "sol", fault)


def test_degenerate_misfit_comps_falls_back(tmp_path):
    """All-zero CV curves (the fixtures) must not drive sigma selection."""
    sol = _load_fixture(tmp_path)
    assert np.all(np.asarray(sol.misfit_comps) == 0.0)
    sol.invert(verbose=False)
    assert sol.sigma0_source == "middle-of-grid fallback"


def test_informative_misfit_comps_drives_selection(tmp_path):
    """U-shaped CV curves select the per-component argmin."""
    sol = _load_fixture(tmp_path)
    K = sol.component_index.size
    sigmas = np.ravel(np.asarray(sol.options["inversion"]["sigma"], float))
    curves = np.ones((sigmas.size, K))
    best = np.arange(K) % sigmas.size
    for k in range(K):
        curves[best[k], k] = 0.5  # interior minimum per component
    sol.misfit_comps = curves
    sol.invert(verbose=False)
    assert sol.sigma0_source == "per-component argmin of misfit_comps CV curves"
