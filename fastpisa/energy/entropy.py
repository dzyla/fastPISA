# Copyright (c) 2026 Dawid Zyla. Part of fastPISA.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rigid-body entropy of assembly dissociation (``T*dS``), calibrated to PISA.

WHAT PISA REPORTS
-----------------
PISA's assembly page gives a dissociation free energy and an entropy term
related by

    dG_diss = -sum(stabilization_energy over the CUT interfaces) - T*dS

Recovered exactly (to +-0.01 kcal/mol) from the PDBe PISA 2.0 JSON of 20
biological assemblies; see ``tests/test_dissociation.py``. ``T*dS`` is the
entropy cost of releasing the dissociated components as independent bodies,
so it OPPOSES dissociation and is subtracted.

THE MODEL
---------
Dissociating one body of mass ``M`` into ``k`` bodies of masses ``m_i``
releases translational freedom. From the Sackur-Tetrode expression for the
translational entropy of an ideal rigid body,

    T*dS = RT * [ sum_i (3/2) ln m_i  -  (3/2) ln M  +  (k - 1) * C ]

with one constant ``C`` collecting the standard state, the thermal
wavelength prefactor and the 5/2 of Sackur-Tetrode. The mass dependence is
therefore NOT fitted -- it is the (3/2) power of the ideal-gas expression --
and ``C`` is the single calibrated number.

Rotational freedom is deliberately NOT added. The ideal classical-rotor term
would contribute a further ~+17 kcal/mol for a typical dimer, which
overshoots PISA by more than a factor of two; PISA's values track the
translational term alone. Keeping one fitted constant rather than bolting on
a rotational term with a fitted coefficient keeps the functional form
physical.

CALIBRATION
-----------
``C`` is fitted to PISA's own ``entropy`` over the 13 reference assemblies
whose dissociation cut can be identified from PISA's numbers alone (the
interface subset satisfying the identity above whose removal disconnects the
chain graph) -- so the partition is not chosen by this model, and the fit is
not circular. Weighted least squares on ``(k - 1)``:

    Pearson r = 0.983, median |error| = 0.91 kcal/mol, bias -0.10,
    max |error| = 2.81 kcal/mol   (n = 13)

The fitted ``C`` sits 2.95 RT below its ideal-gas 1 M value, i.e. an
effective free volume ~19x smaller (factor 0.052) than the 1 M standard
volume -- the usual cage/free-volume correction, and the one place this
model is empirical.

HONEST LIMITS
-------------
* n = 13 assemblies, all 2 to 4 released bodies; do not read it as validated
  for a 60-mer.
* It models translation of rigid bodies only: no side-chain immobilisation,
  no conformational entropy, no solvent reorganisation beyond what the
  solvation term already carries.
* A refit needs a larger set of PISA assemblies; see
  ``examples/calibrate_entropy.py``.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

#: Gas constant times temperature, kcal/mol, at 298.15 K.
R_KCAL_PER_MOL_K = 8.314462618 / 4184.0
TEMPERATURE_K = 298.15
RT_KCAL = R_KCAL_PER_MOL_K * TEMPERATURE_K

#: Mass exponent of the Sackur-Tetrode translational term. NOT fitted.
MASS_EXPONENT = 1.5

#: The one calibrated constant: ``C`` above, in units of RT, per released
#: body, for masses expressed in DALTONS. Fitted 2026-10-03 by weighted
#: least squares against PISA's own assembly entropies over the 13 reference
#: assemblies with an identifiable dissociation cut. Its ideal-gas value at a
#: 1 M standard state is 9.8819 in these units; the fit lands 2.95 RT lower
#: (free-volume factor 0.052), which is the correction discussed above.
#:
#: Refit with ``examples/calibrate_entropy.py``;
#: ``tests/test_dissociation.py::test_shipped_entropy_constant_matches_a_refit``
#: fails if this value drifts from what the reference data refits to.
STANDARD_STATE_LN_TERM = 6.9298

#: Atomic masses (Da) for the elements that occur in PDB/mmCIF models.
ATOMIC_MASSES = {
    "H": 1.008, "D": 2.014, "HE": 4.003, "LI": 6.94, "BE": 9.012,
    "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
    "NE": 20.180, "NA": 22.990, "MG": 24.305, "AL": 26.982, "SI": 28.085,
    "P": 30.974, "S": 32.06, "CL": 35.45, "AR": 39.948, "K": 39.098,
    "CA": 40.078, "SC": 44.956, "TI": 47.867, "V": 50.942, "CR": 51.996,
    "MN": 54.938, "FE": 55.845, "CO": 58.933, "NI": 58.693, "CU": 63.546,
    "ZN": 65.38, "GA": 69.723, "GE": 72.63, "AS": 74.922, "SE": 78.97,
    "BR": 79.904, "KR": 83.798, "RB": 85.468, "SR": 87.62, "Y": 88.906,
    "ZR": 91.224, "NB": 92.906, "MO": 95.95, "RU": 101.07, "RH": 102.906,
    "PD": 106.42, "AG": 107.868, "CD": 112.414, "IN": 114.818,
    "SN": 118.710, "SB": 121.760, "TE": 127.60, "I": 126.904,
    "XE": 131.293, "CS": 132.905, "BA": 137.327, "LA": 138.905,
    "CE": 140.116, "PR": 140.908, "ND": 144.242, "SM": 150.36,
    "EU": 151.964, "GD": 157.25, "TB": 158.925, "DY": 162.500,
    "HO": 164.930, "ER": 167.259, "TM": 168.934, "YB": 173.045,
    "LU": 174.967, "HF": 178.486, "TA": 180.948, "W": 183.84,
    "RE": 186.207, "OS": 190.23, "IR": 192.217, "PT": 195.084,
    "AU": 196.967, "HG": 200.592, "TL": 204.38, "PB": 207.2,
    "BI": 208.980, "TH": 232.038, "U": 238.029,
}


def element_mass(element: str) -> float:
    """Atomic mass in daltons. Raises ``KeyError`` for an unknown element.

    Strict on purpose: silently substituting carbon for an unrecognised
    element would bias a component mass, and component masses are what the
    entropy term is computed from.
    """
    return ATOMIC_MASSES[element.strip().upper()]


def atoms_mass(atoms: Iterable, default: float = 12.011) -> float:
    """Total mass (Da) of parsed atoms, falling back to ``default`` per atom.

    Hydrogens are included when the model has them. ``default`` keeps an
    exotic element from aborting an analysis; it is carbon-weighted and the
    error it can introduce is one atom's worth.
    """
    total = 0.0
    for atom in atoms:
        try:
            total += element_mass(atom.element)
        except KeyError:
            total += default
    return total


def dissociation_entropy(masses: Sequence[float]) -> float:
    """``T*dS`` (kcal/mol, positive) for releasing bodies of these masses.

    Parameters
    ----------
    masses : sequence of float
        Mass in daltons of each body the assembly dissociates INTO. Fewer
        than two bodies means nothing dissociates, so the term is zero.

    Notes
    -----
    The expression is additive over a splitting tree:
    ``dS(A, B, C) == dS(A+B, C) + dS(A, B)``. The minimum-cut search in
    :mod:`fastpisa.energy.dissociation` relies on that, so it is pinned by
    ``tests/test_dissociation.py``.
    """
    bodies = [float(m) for m in masses if m > 0.0]
    k = len(bodies)
    if k < 2:
        return 0.0
    total = math.fsum(bodies)
    term = math.fsum(MASS_EXPONENT * math.log(m) for m in bodies)
    term -= MASS_EXPONENT * math.log(total)
    term += (k - 1) * STANDARD_STATE_LN_TERM
    return RT_KCAL * term
