"""Atomic solvation parameters (ASP), calibrated against original PISA.

The interface solvation free-energy gain is

    dG_solv = sum_k  sigma(class_k) * BSA_k        [kcal/mol]

where BSA_k is the pair-specific buried area of atom k (isolated-monomer ASA
minus in-pair ASA, heavy atoms only) and sigma is the per-class parameter
below. Negative dG_solv = favourable, matching PISA's ``int_solv_en``.

CALIBRATION (2026-09-01, sampled benchmark). sigma is fitted by ordinary
least squares (no intercept -- PISA's solvation gain is a pure sum of
per-atom terms) to the original PISA engine's ``int_solv_en`` over **6881
matched identity interfaces from 674 PDB entries**.

The entries are not hand-picked. 400 of them are a seeded random draw from a
stated sampling frame (X-ray, resolution <= 3.0 A, >= 2 polymer chains,
<= 12000 atoms, released within the frozen EBI PISA CGI's coverage),
de-duplicated to one entry per 30% sequence-identity cluster so that
over-deposited proteins cannot dominate the fit; see
``fastpisa/reference/sampling.py``. The remaining 36 are the legacy
hand-picked benchmark, kept for continuity. Reproduce with
``examples/build_calibration_set.py`` then ``examples/calibrate.py``.

PERFORMANCE, by grouped 10-fold cross-validation (folds never split a PDB
entry, because interfaces within an entry share chains and chemistry and are
not independent observations):

* polymer-polymer interfaces -- the cryo-EM / predicted-model use case, and
  the regime this tool is for: Pearson r = 0.971, R^2 about the 1:1 line
  = 0.940, median |error| 0.74 kcal/mol, bias +0.15 kcal/mol (n = 2303).
* ligand-involving interfaces: substantially worse -- r = 0.81, R^2 = 0.65
  over everything, with a heavy tail (p99 |error| ~34 kcal/mol). Metal ions
  (K, Zn, Mg, Mn, Fe) and small crystallisation additives (acetate, iodide,
  MPD, BOG) dominate it, and the underlying geometry is already off: the
  interface *area* of ligand pairs has a 12% median relative error against
  PISA versus 1.8% for polymer pairs. Treat ligand-interface energies as
  indicative, not calibrated.

An earlier version of this file quoted r = 0.92 overall / 0.974 for polymer
pairs from a 36-entry hand-picked set. The polymer figure survives honest
out-of-sample testing (0.963 blind for the OLD constants on the 638 entries
they had never seen); the "overall" figure did not -- it was an artifact of
a benchmark whose ligand interfaces were unrepresentatively easy.

Two reporting notes, because correlation alone is misleading here:
interface sizes span two orders of magnitude, so Pearson r looks excellent
even when the scale is wrong; R^2 about the 1:1 line and the bias are quoted
alongside. And a regression-to-the-mean slope below 1 (~0.90 for polymer
pairs) is an expected property of a least-squares fit, not a defect.

The fitted values remain physically sensible and close to classic
Eisenberg-McLachlan scales: burying carbon/sulfur is favourable
(sigma_C = -14.8 cal/mol/A^2), burying charged nitrogen costs energy, and
inorganic anion oxygens / metal cations carry large desolvation penalties.
Every fitted class is determined at |z| > 3.7 with a design-matrix condition
number of 56; per-class evidence is printed by ``examples/calibrate.py``.

Reproduce / refresh with ``python examples/calibrate.py`` (offline, from
the committed feature table) or ``python examples/compare_vs_pisa.py``
(the legacy 36-entry head-to-head report).
"""

from typing import Optional

# Metal elements (ZN has its own fitted class; the rest share MET).
METAL_ELEMENTS = frozenset({
    "FE", "MG", "CA", "CU", "MN", "NI", "CO", "MO", "W", "CD", "HG",
    "NA", "K", "LI", "SR", "BA", "RB", "CS",
})

# Phosphate / acid ester oxygens (nucleic-acid backbone, ATP-like).
PHOSPHATE_OXYGENS = frozenset({
    "OP1", "OP2", "OP3", "O1P", "O2P", "O3P",
    "O1A", "O2A", "O3A", "O1B", "O2B", "O3B", "O1G", "O2G", "O3G",
})

# Small inorganic / organic acid ion ligands whose oxygens carry the large
# ion-desolvation penalty PISA assigns them (validated on SO4/PO4 cases).
ION_RESIDUES = frozenset({
    "SO4", "PO4", "PO3", "VO4", "WO4", "NO3", "CO3", "ACT", "FMT", "OXL",
    "CIT",
})

# sigma per atom class, kcal mol^-1 A^-2 (see calibration note above).
SIGMA = {
    "C":   -0.01482,   # carbon (hydrophobic burial favourable)
    "N":   -0.00423,   # neutral nitrogen
    "N+":   0.02033,   # charged side-chain N (Arg NE/NH*, Lys NZ, His ND1/NE2)
    "O":    0.00880,   # neutral oxygen
    "O-":   0.00890,   # carboxylate O (Asp/Glu)
    "OP":  -0.01591,   # phosphate / acid-ester O (DNA backbone, ATP)
    "OI":  -0.07590,   # inorganic ion O (SO4, PO4, ...)
    "S":   -0.03583,   # sulfur / selenium
    "P":    0.0,       # phosphorus: buried area is negligible (median 1.5 A^2
                       # when present -- the OP oxygens shield it), and the
                       # fitted value is not distinguishable from zero
                       # (z = 0.4). Pinned rather than fitted; see the note.
    "HAL": -0.05144,   # halogen / halide (F, Cl, Br, I)
    "MET": -0.11217,   # metal ions other than Zn
    "ZN":  -0.26378,   # zinc (strongly desolvating in PISA)
    "X":    0.0,       # anything else: no contribution
}

#: Halogen elements. As free halide ions (CL-, BR-, ...) and as covalently
#: bound halogen substituents these are chemically distinct from both carbon
#: and the metals, and lumping them into ``X`` gave them sigma = 0 -- i.e. a
#: buried chloride contributed nothing at all to the solvation gain, against
#: the reference engine's ~-12 kcal/mol for such an interface.
HALOGEN_ELEMENTS = frozenset({"F", "CL", "BR", "I"})


def atom_class(atom_name: str, element: str, res_name: str = "") -> str:
    """Map one heavy atom to its solvation class (a key of :data:`SIGMA`)."""
    from fastpisa.interface.bonds import SALT_CHARGES

    el = element.strip().upper()
    res = res_name.strip().upper()
    name = atom_name.strip().upper()
    if el == "ZN":
        return "ZN"
    if el in METAL_ELEMENTS:
        return "MET"
    if res in ION_RESIDUES and el == "O":
        return "OI"
    if el == "C":
        return "C"
    if el in ("S", "SE"):
        return "S"
    if el == "P":
        return "P"
    if el in HALOGEN_ELEMENTS:
        return "HAL"
    if el == "N":
        return "N+" if SALT_CHARGES.get((res, name), 0) > 0 else "N"
    if el == "O":
        if SALT_CHARGES.get((res, name), 0) < 0:
            return "O-"
        if name in PHOSPHATE_OXYGENS:
            return "OP"
        return "O"
    return "X"


def get_asp(atom_name: str, element: Optional[str] = None,
            res_name: str = "") -> float:
    """ASP (kcal mol^-1 A^-2) for an atom.

    ``res_name`` disambiguates charged side-chain atoms; without it the
    neutral class is assumed. Hydrogens return 0 (they carry no surface).
    """
    el = (element or "").strip().upper()
    if el in ("H", "D"):
        return 0.0
    if not el:
        # Derive a coarse element from the atom name's first letter.
        stripped = atom_name.strip().upper()
        el = stripped[0] if stripped else "C"
    return SIGMA[atom_class(atom_name, el, res_name)]
