"""Atomic solvation parameters (ASP), calibrated against original PISA.

The interface solvation free-energy gain is

    dG_solv = sum_k  sigma(class_k) * BSA_k        [kcal/mol]

where BSA_k is the pair-specific buried area of atom k (isolated-monomer ASA
minus in-pair ASA, heavy atoms only) and sigma is the per-class parameter
below. Negative dG_solv = favourable, matching PISA's ``int_solv_en``.

CALIBRATION (2026-08-29, extended set). sigma was fitted by least squares to
the original PISA engine's ``int_solv_en`` (EBI PDBe PISA service XML) over
262 matched identity interfaces from 36 diverse PDB entries
(protease-inhibitor, antibody-antigen, hormone-receptor, hemoglobin+heme,
protein-DNA/RNA, insulin with inter-chain disulfides and Zn, ATP / inorganic
ion / glycan / cofactor ligand interfaces):
1ktz 1brs 1vfb 2ptc 1acb 3hhr 4ins 1a3n 1fin 1lmb 1dfj 1ay7 1gcq 1tro 3cro
1rva 9ant 2sni 1cho 1stf 1cbw 1fdl 3hfm 1nca 1cgi 1eaw 1r0r 1oph 1jck 1gpw
1tsr 1aay 1urn 1prc 1f34 1e6e.

Performance (leave-one-PDB-out cross-validation, i.e. on entries the fit
never saw): Pearson r = 0.92 overall; polymer-polymer interfaces (the
cryo-EM / predicted-model use case) r = 0.974, median |error| 1.35 kcal/mol;
ligand-involving interfaces r = 0.74. Before the 15-entry blind-test
extension the polymer-polymer figure was verified truly out-of-sample at
r = 0.977. Largest residual errors: exotic hydrophobic cofactors
(quinones/pheophytins), where PISA applies CCD-specific chemistry.

The fitted values are physically sensible and close to classic
Eisenberg-McLachlan scales: burying carbon/sulfur is favourable
(sigma_C = -14 cal/mol/A^2), burying charged nitrogen/oxygen costs energy,
and inorganic anion oxygens / metal cations carry large desolvation
penalties.

Reproduce / refresh with ``python examples/compare_vs_pisa.py``.
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
    "C":   -0.01360,   # carbon (hydrophobic burial favourable)
    "N":   -0.01062,   # neutral nitrogen
    "N+":   0.01432,   # charged side-chain N (Arg NE/NH*, Lys NZ, His ND1/NE2)
    "O":    0.00673,   # neutral oxygen
    "O-":   0.01744,   # carboxylate O (Asp/Glu)
    "OP":  -0.01017,   # phosphate / acid-ester O (DNA backbone, ATP)
    "OI":  -0.10438,   # inorganic ion O (SO4, PO4, ...)
    "S":   -0.04276,   # sulfur / selenium
    "MET": -0.09959,   # metal ions other than Zn
    "ZN":  -0.36134,   # zinc (strongly desolvating in PISA)
    "X":    0.0,       # anything else (P, halogens, ...): no contribution
}


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
