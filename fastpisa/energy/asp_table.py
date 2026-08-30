"""Atomic solvation parameters (ASP), calibrated against original PISA.

The interface solvation free-energy gain is

    dG_solv = sum_k  sigma(class_k) * BSA_k        [kcal/mol]

where BSA_k is the pair-specific buried area of atom k (isolated-monomer ASA
minus in-pair ASA, heavy atoms only) and sigma is the per-class parameter
below. Negative dG_solv = favourable, matching PISA's ``int_solv_en``.

CALIBRATION (2026-08-29). sigma was fitted by least squares to the original
PISA engine's ``int_solv_en`` (EBI PDBe PISA service XML) over 117 matched
identity interfaces from 21 diverse PDB entries (protease-inhibitor,
antibody-antigen, hormone-receptor, hemoglobin+heme, protein-DNA, insulin
with inter-chain disulfides and Zn, ATP/ion/glycan ligand interfaces):
1ktz 1brs 1vfb 2ptc 1acb 3hhr 4ins 1a3n 1fin 1lmb 1dfj 1ay7 1gcq 1tro 3cro
1rva 9ant 2sni 1cho 1stf 1cbw.

Performance (leave-one-PDB-out cross-validation, i.e. on entries the fit
never saw): Pearson r = 0.94 overall, r = 0.97 for interfaces > 300 A^2,
median |error| 1.16 kcal/mol (14% relative) on those interfaces. Small
single-ion interfaces (a lone Zn/Ca/Ni) carry the largest relative errors:
PISA gives ions large, ion-specific desolvation penalties that a single MET
class only approximates.

The fitted values are physically sensible and close to classic
Eisenberg-McLachlan scales: burying carbon/sulfur is favourable
(sigma_C = -17 cal/mol/A^2), burying charged nitrogen/oxygen costs energy.

Reproduce / refresh with ``python examples/compare_vs_pisa.py --recalibrate``.
"""

from typing import Optional

# Metal elements treated as the MET desolvation class.
METAL_ELEMENTS = frozenset({
    "FE", "ZN", "MG", "CA", "CU", "MN", "NI", "CO", "MO", "W", "CD", "HG",
    "NA", "K", "LI", "SR", "BA", "RB", "CS",
})

# Phosphate / acid ester oxygens (nucleic-acid backbone, ATP-like, sulfate).
PHOSPHATE_OXYGENS = frozenset({
    "OP1", "OP2", "OP3", "O1P", "O2P", "O3P",
    "O1A", "O2A", "O3A", "O1B", "O2B", "O3B", "O1G", "O2G", "O3G",
})

# sigma per atom class, kcal mol^-1 A^-2 (see calibration note above).
SIGMA = {
    "C":   -0.01711,   # carbon (hydrophobic burial favourable)
    "N":   -0.01336,   # neutral nitrogen
    "N+":   0.02151,   # charged side-chain N (Arg NE/NH*, Lys NZ, His ND1/NE2)
    "O":    0.01206,   # neutral oxygen
    "O-":   0.02314,   # carboxylate O (Asp/Glu)
    "OP":  -0.01455,   # phosphate / acid-ester O (DNA backbone, ATP, SO4)
    "S":   -0.03326,   # sulfur / selenium
    "MET": -0.13820,   # metal ions (single-class approximation)
    "X":    0.0,       # anything else (P, halogens, ...): no contribution
}


def atom_class(atom_name: str, element: str, res_name: str = "") -> str:
    """Map one heavy atom to its solvation class (a key of :data:`SIGMA`)."""
    from fastpisa.interface.bonds import SALT_CHARGES

    el = element.strip().upper()
    res = res_name.strip().upper()
    name = atom_name.strip().upper()
    if el in METAL_ELEMENTS:
        return "MET"
    if el == "C":
        return "C"
    if el in ("S", "SE"):
        return "S"
    if el == "N":
        return "N+" if SALT_CHARGES.get((res, name), 0) > 0 else "N"
    if el == "O":
        if SALT_CHARGES.get((res, name), 0) < 0:
            return "O-"
        if name in PHOSPHATE_OXYGENS or (res == "SO4" and name in ("O1", "O2", "O3", "O4")):
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
