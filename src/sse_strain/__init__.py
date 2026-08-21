"""sse-strain: space-time strain and stress from Cascadia slow-slip catalogs.

Typical use::

    from sse_strain import load_fault_ascii, load_solution, make_grid
    from sse_strain.elastic import ElasticModel
    from sse_strain.green import strain_from_components
    from sse_strain import dataset

    fault = load_fault_ascii("cascadia_mesh.txt")
    sol   = load_solution("2025-01-26", fault)
    sol.invert()
    grid  = make_grid(origin=fault.origin, depths_km=(1., 3., 6.))
    Ls, Ld = sol.component_slip_patterns()
    slip  = fault.slip_enu(Ls, Ld)                      # (n_patch, 3, K)
    E     = strain_from_components(grid.points_m, fault.vertices_m(), slip, sol.nu)
    # put the model through the same operator as the dv/v measurement
    A, t, info = smoothing.smooth_amplitudes(sol, window=31,
                                             reference=("2012-01-01", "2012-12-31"))
    ds    = dataset.evaluate(grid, sol, ElasticModel.placeholder_cascadia(), E,
                             amplitudes=A, times=t, stride=7)
"""

from .fault import FaultMesh, load_fault_ascii          # noqa: F401
from .solution import Solution, load_solution           # noqa: F401
from .grid import ObsGrid, make_grid                    # noqa: F401
from .elastic import ElasticModel                       # noqa: F401
from .green import strain_from_components, displacement_gf  # noqa: F401
from . import strain, dataset, plotting, matio, smoothing  # noqa: F401

__version__ = "0.1.0"
__all__ = [
    "FaultMesh", "load_fault_ascii", "Solution", "load_solution",
    "ObsGrid", "make_grid", "ElasticModel", "strain_from_components",
    "displacement_gf",
    "strain", "dataset", "plotting", "matio", "smoothing",
]
