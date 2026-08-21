"""In-situ elastic properties for Cascadia.

Two distinct elastic parameterisations enter this calculation and it is worth
being explicit about the difference, because conflating them is the easiest
way to get a plausible-looking but wrong stress field.

1. **The half-space used for the Green's functions.**  The Nikkhoo & Walter
   (2015) triangular dislocation solution is for a *homogeneous, isotropic*
   elastic half-space, and the only elastic parameter that enters the
   displacement and strain kernels is Poisson's ratio.  Gualandi (2025) used
   nu = 0.25.  Changing nu changes the strain field itself.

2. **The moduli used to convert strain to stress.**  Hooke's law needs mu and
   lambda at the *observation* point.  Nothing forces these to be the same
   constants as in (1), and using a depth-dependent mu(z) here is closer to the
   truth than a single crustal average.

Doing (2) with a depth-dependent model while (1) assumes homogeneity is
internally inconsistent: a layered medium would also modify the strain field,
not only the strain-to-stress map.  The inconsistency is second order for a
smoothly varying crust, and it is the standard compromise in the Coulomb
stress literature, but it is a real approximation and
:class:`ElasticModel` records it in the output metadata so the choice travels
with the data.  ``ElasticModel.homogeneous(nu=...)`` gives the fully consistent
alternative.

Empirical relations
-------------------
``brocher_density`` and ``brocher_vs`` implement the Nafe-Drake and Vp-Vs
regressions of Brocher (2005).

VERIFIED 2026-08-20 against two independent secondary sources: Stephenson
(2007, USGS OFR 2007-1348, p. 7) quotes the Nafe-Drake density polynomial
digit for digit, and Ranjbar & Kamani (2023, J. Oil Gas Petrochem. Tech.
10, 95-109, table 2) quotes the Vs regression coefficients (their leading
"0.07858" is a typesetting slip for 0.7858: with 0.7858 the relation gives
Vs = 3.55 km/s at Vp = 6 km/s, i.e. Vp/Vs = 1.69, as published).  The
original BSSA paper itself has not been checked; do that before quoting the
coefficients in a manuscript.

References
----------
Brocher, T. M. (2005). Empirical relations between elastic wavespeeds and
    density in the Earth's crust. *Bulletin of the Seismological Society of
    America*, *95*(6), 2081-2092.
Nikkhoo, M., & Walter, T. R. (2015). Triangular dislocation: an analytical,
    artefact-free solution. *Geophysical Journal International*, *201*(2),
    1119-1141. https://doi.org/10.1093/gji/ggv035
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["brocher_density", "brocher_vs", "ElasticModel", "PLACEHOLDER_NOTE"]

PLACEHOLDER_NOTE = (
    "PLACEHOLDER 1-D profile. Replace with a documented Cascadia velocity "
    "model (e.g. Stephenson, 2007 USGS OFR; the CRESCENT community velocity "
    "model; or a local Vp/Vs profile from your own tomography) before using "
    "these stresses quantitatively."
)


def brocher_density(vp_km_s):
    """Nafe-Drake density from Vp.  Vp in km/s, returns density in kg/m^3.

    Nominally valid for 1.5 <= Vp <= 8.5 km/s.  Outside that range the
    polynomial is extrapolated and the result should not be trusted.
    """
    v = np.asarray(vp_km_s, dtype=float)
    rho_g_cm3 = (
        1.6612 * v
        - 0.4721 * v**2
        + 0.0671 * v**3
        - 0.0043 * v**4
        + 0.000106 * v**5
    )
    return rho_g_cm3 * 1000.0


def brocher_vs(vp_km_s):
    """Vs from Vp (Brocher 2005 regression).  Both in km/s.

    Nominally valid for 1.5 <= Vp <= 8.0 km/s.
    """
    v = np.asarray(vp_km_s, dtype=float)
    return (
        0.7858
        - 1.2344 * v
        + 0.7949 * v**2
        - 0.1238 * v**3
        + 0.0064 * v**4
    )


@dataclass
class ElasticModel:
    """Depth-dependent isotropic elastic properties.

    Depths are positive downward in metres.  Moduli are returned in Pa.

    Parameters
    ----------
    depth_m : (n,) array
        Node depths, positive down, monotonically increasing.
    vp_m_s, vs_m_s, rho_kg_m3 : (n,) arrays
    nu_greens : float
        Poisson's ratio of the half-space used for the dislocation Green's
        functions.  Kept separate from ``poisson(z)`` on purpose (see module
        docstring).
    source : str
        Provenance string written into the output metadata.
    """

    depth_m: np.ndarray
    vp_m_s: np.ndarray
    vs_m_s: np.ndarray
    rho_kg_m3: np.ndarray
    nu_greens: float = 0.25
    source: str = "unspecified"
    is_placeholder: bool = False
    notes: list = field(default_factory=list)

    # ------------------------------------------------------------ constructors
    @classmethod
    def homogeneous(cls, vs_m_s=3500.0, nu=0.25, rho_kg_m3=2800.0,
                    max_depth_m=60000.0):
        """Uniform half-space, fully consistent with the Green's functions."""
        vp = vs_m_s * np.sqrt(2 * (1 - nu) / (1 - 2 * nu))
        d = np.array([0.0, max_depth_m])
        return cls(
            depth_m=d,
            vp_m_s=np.full(2, vp),
            vs_m_s=np.full(2, vs_m_s),
            rho_kg_m3=np.full(2, rho_kg_m3),
            nu_greens=nu,
            source=f"homogeneous half-space Vs={vs_m_s} m/s, nu={nu}",
            notes=["Strain and stress use the same elastic constants."],
        )

    @classmethod
    def from_csv(cls, path, nu_greens=0.25, source=None):
        """Load a profile with columns ``depth_km, vp_km_s[, vs_km_s, rho_kg_m3]``.

        Missing Vs and density are filled with the Brocher (2005) relations.
        """
        path = Path(path)
        arr = np.genfromtxt(path, delimiter=",", names=True)
        depth = np.atleast_1d(arr["depth_km"]) * 1e3
        vp = np.atleast_1d(arr["vp_km_s"])
        vs = np.atleast_1d(arr["vs_km_s"]) if "vs_km_s" in arr.dtype.names \
            else brocher_vs(vp)
        rho = np.atleast_1d(arr["rho_kg_m3"]) if "rho_kg_m3" in arr.dtype.names \
            else brocher_density(vp)
        return cls(depth, vp * 1e3, vs * 1e3, rho, nu_greens=nu_greens,
                   source=source or f"CSV {path.name}")

    @classmethod
    def placeholder_cascadia(cls, nu_greens=0.25):
        """A coarse forearc-crust profile, explicitly marked as a placeholder.

        The numbers are order-of-magnitude only.  They exist so that the
        pipeline runs end to end and so that the shape of the output is correct;
        they are not a Cascadia velocity model and are not cited as one.
        """
        depth_km = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 16.0, 24.0, 32.0, 45.0, 60.0])
        vp_km_s = np.array([2.5, 4.0, 5.3, 6.0, 6.3, 6.6, 6.8, 7.1, 7.8, 8.0])
        vs = brocher_vs(vp_km_s)
        rho = brocher_density(vp_km_s)
        return cls(
            depth_m=depth_km * 1e3,
            vp_m_s=vp_km_s * 1e3,
            vs_m_s=vs * 1e3,
            rho_kg_m3=rho,
            nu_greens=nu_greens,
            source="PLACEHOLDER generic forearc crust + Brocher (2005)",
            is_placeholder=True,
            notes=[PLACEHOLDER_NOTE],
        )

    # ------------------------------------------------------------- evaluation
    def _interp(self, arr, depth_m):
        d = np.asarray(depth_m, dtype=float)
        return np.interp(d, self.depth_m, arr)

    def mu(self, depth_m):
        """Shear modulus, Pa."""
        return self._interp(self.rho_kg_m3 * self.vs_m_s**2, depth_m)

    def lam(self, depth_m):
        """First Lame parameter, Pa."""
        rho = self._interp(self.rho_kg_m3, depth_m)
        vp = self._interp(self.vp_m_s, depth_m)
        vs = self._interp(self.vs_m_s, depth_m)
        return rho * (vp**2 - 2.0 * vs**2)

    def bulk_modulus(self, depth_m):
        return self.lam(depth_m) + 2.0 / 3.0 * self.mu(depth_m)

    def poisson(self, depth_m):
        """Local Poisson's ratio implied by the velocity profile."""
        lam = self.lam(depth_m)
        mu = self.mu(depth_m)
        return lam / (2.0 * (lam + mu))

    def young(self, depth_m):
        lam = self.lam(depth_m)
        mu = self.mu(depth_m)
        return mu * (3 * lam + 2 * mu) / (lam + mu)

    def metadata(self) -> dict:
        return {
            "elastic_source": self.source,
            "elastic_is_placeholder": int(self.is_placeholder),
            "nu_greens_functions": self.nu_greens,
            "elastic_notes": " | ".join(self.notes) if self.notes else "",
            "elastic_consistency": (
                "Green's functions assume a homogeneous half-space with "
                f"nu={self.nu_greens}; strain-to-stress conversion uses "
                "depth-dependent mu and lambda. See docs/THEORY.md."
                if len(np.unique(self.vs_m_s)) > 1 else
                "Homogeneous: Green's functions and Hooke's law use the same "
                "elastic constants."
            ),
        }
