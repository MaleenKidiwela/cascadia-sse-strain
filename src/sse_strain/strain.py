r"""Strain invariants, Hooke's law, and derived stress measures.

Conventions
-----------
Frame is East-North-Up.  Strain components arrive in the Voigt-like ordering
``[exx, eyy, ezz, exy, exz, eyz]`` where the off-diagonal entries are tensor
components (``exy = 1/2 (du_x/dy + du_y/dx)``), not engineering shear strains.
Extension is positive.  For stress this makes tension positive, so a
compressive change appears as a negative ``skk``.

Quantities computed
-------------------
Dilatation (volumetric strain)

.. math:: \varepsilon_{kk} = \varepsilon_{11}+\varepsilon_{22}+\varepsilon_{33}

Deviatoric strain

.. math:: e_{ij} = \varepsilon_{ij} - \tfrac{1}{3}\varepsilon_{kk}\delta_{ij}

Second invariant of the deviatoric strain and the equivalent (von Mises) shear
strain

.. math::
   J_2' = \tfrac{1}{2} e_{ij}e_{ij}, \qquad
   \varepsilon_{eq} = \sqrt{\tfrac{4}{3} J_2'}

Maximum shear strain from the principal values
:math:`\varepsilon_1 \ge \varepsilon_2 \ge \varepsilon_3`

.. math:: \gamma_{max} = \varepsilon_1 - \varepsilon_3

Hooke's law for an isotropic medium

.. math:: \sigma_{ij} = \lambda\,\varepsilon_{kk}\,\delta_{ij} + 2\mu\,\varepsilon_{ij}

Mean stress and pressure (pressure positive in compression)

.. math:: \sigma_m = \tfrac{1}{3}\sigma_{kk}, \qquad p = -\sigma_m

von Mises equivalent stress and maximum shear stress

.. math::
   \sigma_{eq} = \sqrt{3 J_2}, \qquad \tau_{max} = \tfrac{1}{2}(\sigma_1-\sigma_3)

Coulomb stress change on a receiver plane with unit normal :math:`n` and slip
direction :math:`s`, apparent friction :math:`\mu'`

.. math:: \Delta CFS = \Delta\tau_s + \mu' \Delta\sigma_n

with :math:`\Delta\sigma_n = n_i \sigma_{ij} n_j` positive in tension (so
unclamping raises :math:`\Delta CFS`) and
:math:`\Delta\tau_s = s_i \sigma_{ij} n_j`.

Which of these to compare against dv/v
--------------------------------------
There is no single right answer, and that is the point of computing several.
Laboratory and field work on stress sensitivity of seismic velocity generally
finds the strongest control from the confining or mean stress acting on crack
populations, so ``pressure`` and ``dilatation`` are the natural first targets.
Deviatoric measures matter if crack opening is orientation-selective.  Keeping
all of them in the same Dataset means the correlation analysis can be done
without recomputing anything.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "voigt_to_tensor", "dilatation", "deviatoric", "j2_strain",
    "equivalent_shear_strain", "principal_values", "max_shear_strain",
    "hooke", "stress_invariants", "coulomb_stress",
]


def voigt_to_tensor(v):
    """``(..., 6)`` -> ``(..., 3, 3)`` symmetric tensor.

    Input order ``[xx, yy, zz, xy, xz, yz]``.
    """
    v = np.asarray(v, dtype=float)
    if v.shape[-1] != 6:
        raise ValueError(f"last axis must be 6, got {v.shape}")
    t = np.empty(v.shape[:-1] + (3, 3))
    t[..., 0, 0] = v[..., 0]
    t[..., 1, 1] = v[..., 1]
    t[..., 2, 2] = v[..., 2]
    t[..., 0, 1] = t[..., 1, 0] = v[..., 3]
    t[..., 0, 2] = t[..., 2, 0] = v[..., 4]
    t[..., 1, 2] = t[..., 2, 1] = v[..., 5]
    return t


def dilatation(v):
    """Trace of the strain (or stress) tensor from Voigt-ordered input."""
    v = np.asarray(v, dtype=float)
    return v[..., 0] + v[..., 1] + v[..., 2]


def deviatoric(v):
    """Deviatoric part, same Voigt ordering."""
    v = np.asarray(v, dtype=float)
    out = v.copy()
    tr = dilatation(v) / 3.0
    out[..., 0] -= tr
    out[..., 1] -= tr
    out[..., 2] -= tr
    return out


def j2_strain(v):
    r"""Second invariant :math:`J_2' = \tfrac12 e_{ij}e_{ij}` of the deviator."""
    e = deviatoric(v)
    return 0.5 * (e[..., 0] ** 2 + e[..., 1] ** 2 + e[..., 2] ** 2) + (
        e[..., 3] ** 2 + e[..., 4] ** 2 + e[..., 5] ** 2
    )


def equivalent_shear_strain(v):
    r""":math:`\varepsilon_{eq} = \sqrt{4J_2'/3}`, the von Mises equivalent."""
    return np.sqrt(4.0 / 3.0 * j2_strain(v))


def principal_values(v):
    """Sorted descending principal values, shape ``(..., 3)``."""
    t = voigt_to_tensor(v)
    w = np.linalg.eigvalsh(t)
    return w[..., ::-1]


def max_shear_strain(v):
    """``eps_1 - eps_3``."""
    w = principal_values(v)
    return w[..., 0] - w[..., 2]


def hooke(strain_voigt, mu, lam):
    """Isotropic Hooke's law.  ``mu`` and ``lam`` broadcast against the leading axes.

    Returns stress in the same Voigt ordering, in Pa if ``mu``/``lam`` are Pa.
    """
    e = np.asarray(strain_voigt, dtype=float)
    mu = np.asarray(mu, dtype=float)[..., None]
    lam = np.asarray(lam, dtype=float)[..., None]
    tr = dilatation(e)[..., None]
    s = 2.0 * mu * e
    s[..., :3] += lam * tr
    return s


def stress_invariants(stress_voigt):
    """Dictionary of mean stress, pressure, von Mises stress and max shear."""
    s = np.asarray(stress_voigt, dtype=float)
    mean = dilatation(s) / 3.0
    j2 = j2_strain(s)  # same algebra applies to any symmetric tensor
    w = principal_values(s)
    return {
        "mean_stress": mean,
        "pressure": -mean,
        "von_mises_stress": np.sqrt(3.0 * j2),
        "max_shear_stress": 0.5 * (w[..., 0] - w[..., 2]),
        "sigma_1": w[..., 0],
        "sigma_3": w[..., 2],
    }


def plane_vectors(strike_deg, dip_deg, rake_deg):
    """Unit normal and slip vectors of a receiver plane, East-North-Up."""
    phi = np.deg2rad(strike_deg)
    delta = np.deg2rad(dip_deg)
    lam_ = np.deg2rad(rake_deg)
    # normal to a plane with strike phi (clockwise from N) and dip delta
    n = np.array([-np.cos(phi) * np.sin(delta),
                  np.sin(phi) * np.sin(delta),
                  np.cos(delta)])
    strike_vec = np.array([np.sin(phi), np.cos(phi), 0.0])
    dip_vec = np.array([np.cos(phi) * np.cos(delta),
                        -np.sin(phi) * np.cos(delta),
                        -np.sin(delta)])
    s = np.cos(lam_) * strike_vec + np.sin(lam_) * dip_vec
    return n, s


def coulomb_stress(stress_voigt, strike_deg, dip_deg, rake_deg,
                   friction=0.4):
    """Coulomb failure stress change on a receiver plane.

    Returns ``(dCFS, shear, normal)`` in Pa.  ``normal`` is positive in
    tension, so a positive value means the plane is being unclamped.

    ``friction`` is the *apparent* friction, absorbing pore pressure effects
    through the Skempton coefficient in the usual way.  For the Cascadia
    subduction interface, values in the range 0.0-0.4 are commonly used;
    the result scales linearly with the choice, so report it.
    """
    s = voigt_to_tensor(stress_voigt)
    n, d = plane_vectors(strike_deg, dip_deg, rake_deg)
    tvec = np.einsum("...ij,j->...i", s, n)
    normal = np.einsum("...i,i->...", tvec, n)
    shear = np.einsum("...i,i->...", tvec, d)
    return shear + friction * normal, shear, normal
