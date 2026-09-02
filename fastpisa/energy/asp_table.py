"""Atomic solvation parameters (ASP), calibrated against original PISA.

The interface solvation free-energy gain is

    dG_solv = sum_k  sigma(atom_k) * BSA_k        [kcal/mol]

where BSA_k is the pair-specific buried area of atom k (isolated-monomer ASA
minus in-pair ASA, heavy atoms only, NACCESS radii) and sigma is the
per-atom parameter below. Negative dG_solv = favourable, matching PISA's
``int_solv_en``.

MODEL (2026-09-01, hierarchical scheme). Every heavy atom gets a FINE type
(``"RES:ATOM"`` for the 20 amino acids, MSE and the canonical nucleotides;
``"het:<class>"`` otherwise) and

    sigma(atom) = SIGMA[class_of_fine(fine)] + DELTA.get(fine, 0)

i.e. a chemically motivated CLASS value (32 fitted classes: backbone vs
side-chain, sp3 / aromatic / heteroatom-bonded sp2 carbon,
amine / ring nitrogen and oxygen, Met vs Cys sulfur, nucleotide sugar / base
/ phosphate, plus the hetero classes) and a per-atom-type DEVIATION shrunk
toward zero by an L2 penalty (ridge 1000 on the deviations only, chosen by
grouped cross-validation -- results are flat from 100 to 10^4). Atom types
with a lot of buried area earn their own value; sparse ones fall back to
their class.

CALIBRATION DATA. Fitted at the RESIDUE level to PISA's own per-residue
solvation energies: 119,078 PISA-matched interface residues from 674 PDB
entries (400 a seeded random draw from a stated sampling frame,
de-duplicated at 30% sequence identity; 36 legacy hand-picked). Residue-level
fitting is what makes the fine types identifiable -- the same model fitted
to the 6,904 interface *sums* is markedly worse out of fold (median error
0.52 vs 0.32 kcal/mol), because a sum cannot separate the atoms it adds.
Reproduce with ``python examples/calibrate.py`` from the committed tables in
``tests/data/calibration/``.

PERFORMANCE, grouped 10-fold cross-validation at the INTERFACE level (folds
never split a PDB entry; each fold refitted at residue level and scored on
the held-out interfaces' full pipeline features), polymer-polymer
interfaces (n = 2314):

    Pearson r = 0.987, R^2 about the 1:1 line = 0.975,
    median |error| = 0.33 kcal/mol, bias -0.01, slope 0.98.

    (previous 11-class model under the same geometry: r 0.977, R^2 0.951,
    median 0.71; its published figure of 0.74 was under the older radii.)

Protein-nucleic-acid interfaces (n = 241): R^2 0.96, median 0.30. On the
legacy 36-entry head-to-head (in-sample) polymer pairs: r 0.997, median
0.24 kcal/mol.
Ligand-involving interfaces remain much weaker (R^2 ~0.69 overall) -- their
interface AREAS are still 9% off PISA at the median (polymer pairs 1.8%),
so no solvation constant can rescue them; treat those energies as indicative.

The class values are physically sensible: every carbon class is favourable
to bury (-0.014 to -0.017), sulfur -0.039, neutral N/O cost
+0.010 to +0.016, charged N +0.022 to +0.031, metals and inorganic-ion
oxygens carry large penalties. The nucleotide classes are inverted relative
to protein chemistry (positive C, negative O); that is not collinearity (the
nucleotide block has condition number 24 and every value is > 7 standard
errors from zero) -- it is what PISA's own parameter set does for nucleic
acids, and it is reproduced rather than "corrected".

The side-chain sp2 carbons bonded to heteroatoms (Asn CG, Gln CD, Asp CG,
Glu CD, Arg CZ) share one class, ``C_sp2_polar``: each is shielded by its
own O/N and buries too little area to be determined alone (amide carbon
z = 1.9, Arg CZ z = 2.8 as separate classes). Their fine-type deviations
carry whatever real difference the data support.

Two classes are PINNED at zero rather than fitted: hetero ``P`` and
nucleotide ``NA_P``. Phosphorus buries a median 1.5 A^2 (its oxygens shield
it) and both fitted values are within 2 standard errors of zero. Two hetero
classes (``N+``, ``O-``) have no data in the benchmark -- charged atoms only
occur in fine-typed standard residues -- and keep their 2026-09-01 values as
fallbacks for modified residues.
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

#: Halogen elements. As free halide ions and as covalently bound substituents
#: these are chemically distinct from carbon and from the metals; lumping
#: them into ``X`` (sigma 0) made a buried chloride contribute nothing.
HALOGEN_ELEMENTS = frozenset({"F", "CL", "BR", "I"})

# Residues whose atoms get a per-(residue, atom-name) FINE type.
FINE_TYPED_RESIDUES = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE",
    "A", "G", "C", "U", "DA", "DG", "DC", "DT", "DU",
})
_NUCLEOTIDES = frozenset({"A", "G", "C", "U", "DA", "DG", "DC", "DT", "DU"})
_AROMATIC_C = {
    ("PHE", a) for a in ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")
} | {
    ("TYR", a) for a in ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")
} | {
    ("TRP", a) for a in ("CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2")
} | {("HIS", "CG"), ("HIS", "CD2"), ("HIS", "CE1")}

# sigma per CLASS, kcal mol^-1 A^-2 (see the model note above).
SIGMA = {
    "C_CA":         -0.01506,   # backbone CA
    "C_bb":         -0.01389,   # backbone carbonyl C
    "C_ali":        -0.01596,   # sp3 side-chain C
    "C_aro":        -0.01687,   # aromatic ring C (Phe/Tyr/Trp/His)
    "C_sp2_polar":  -0.01844,   # Asn CG / Gln CD / Asp CG / Glu CD / Arg CZ
    "N_bb":          0.01148,   # backbone N
    "N_amide":       0.01107,   # Asn ND2 / Gln NE2
    "N_arg":         0.02281,   # Arg NE/NH1/NH2
    "N_lys":         0.03058,   # Lys NZ
    "N_his":         0.02221,   # His ND1/NE2
    "N_trp":         0.01009,   # Trp NE1
    "O_bb":          0.01031,   # backbone O / OXT
    "O_amide":       0.01630,   # Asn OD1 / Gln OE1
    "O_carbox":      0.01393,   # Asp OD1/OD2, Glu OE1/OE2
    "O_hyd":         0.01094,   # Ser OG / Thr OG1 / Tyr OH
    "S_met":        -0.03918,   # Met SD / Mse SE
    "S_cys":        -0.03924,   # Cys SG
    "NA_C_sugar":    0.03200,   # nucleotide sugar C
    "NA_C_base":     0.03342,   # nucleotide base C
    "NA_N_base":     0.01083,   # nucleotide base N
    "NA_O_base":    -0.05824,   # nucleotide base O
    "NA_O_sugar":   -0.04133,   # nucleotide sugar O
    "NA_OP":        -0.05202,   # nucleotide phosphate OP1/OP2
    "NA_P":          0.00000,   # nucleotide P -- pinned, see note
    "C":            -0.00115,   # hetero-group carbon
    "N":             0.01948,   # hetero neutral N
    "N+":            0.02033,   # hetero charged N (no data; fallback)
    "O":             0.00290,   # hetero neutral O
    "O-":            0.00890,   # hetero carboxylate O (no data; fallback)
    "OP":           -0.03808,   # hetero phosphate / acid-ester O (ATP-like)
    "OI":           -0.08578,   # inorganic ion O (SO4, PO4, ...)
    "S":            -0.03103,   # hetero sulfur / selenium
    "P":             0.00000,   # phosphorus (hetero) -- pinned, see note
    "HAL":          -0.05631,   # halogen / halide (F, Cl, Br, I)
    "MET":          -0.11008,   # metal ions other than Zn
    "ZN":           -0.26351,   # zinc
    "X":             0.00000,   # anything else: no contribution
}

# Per-fine-type deviation from the class value (kcal mol^-1 A^-2), shrunk
# toward zero (ridge 1000). Types absent here use their class value alone.
DELTA = {
    "A:N6": +0.00256,
    "ALA:C": +0.00004,
    "ALA:CA": -0.00045,
    "ALA:CB": +0.00029,
    "ALA:N": -0.00020,
    "ALA:O": +0.00070,
    "ARG:CA": -0.00468,
    "ARG:CB": +0.00159,
    "ARG:CD": +0.00064,
    "ARG:CG": +0.00399,
    "ARG:CZ": -0.01147,
    "ARG:N": -0.00316,
    "ARG:NE": +0.00012,
    "ARG:NH1": +0.00267,
    "ARG:NH2": -0.00279,
    "ARG:O": +0.00092,
    "ASN:CA": -0.00021,
    "ASN:CB": +0.00027,
    "ASN:CG": +0.00366,
    "ASN:N": -0.00002,
    "ASN:ND2": +0.00003,
    "ASN:O": +0.00092,
    "ASN:OD1": -0.00013,
    "ASP:CA": -0.00009,
    "ASP:CB": +0.00050,
    "ASP:CG": +0.00256,
    "ASP:N": -0.00051,
    "ASP:O": +0.00075,
    "ASP:OD1": +0.00027,
    "ASP:OD2": -0.00046,
    "CYS:CA": -0.00118,
    "CYS:CB": -0.00028,
    "CYS:O": +0.00025,
    "CYS:SG": +0.00000,
    "DA:C2": -0.00192,
    "DA:N1": -0.00357,
    "DA:N6": +0.00503,
    "DA:OP1": +0.00407,
    "DA:OP2": +0.00064,
    "DC:N3": -0.00117,
    "DC:N4": +0.00701,
    "DC:O2": -0.00105,
    "DC:OP1": +0.00464,
    "DC:OP2": +0.00223,
    "DG:N1": -0.00707,
    "DG:N2": +0.00845,
    "DG:O6": -0.00149,
    "DG:OP1": +0.00451,
    "DG:OP2": -0.00139,
    "DT:C7": -0.02988,
    "DT:N3": -0.00205,
    "DT:O2": +0.00133,
    "DT:O4": -0.00223,
    "DT:OP1": +0.00379,
    "DT:OP2": -0.00228,
    "GLN:CA": +0.00016,
    "GLN:CB": +0.00032,
    "GLN:CD": +0.00339,
    "GLN:CG": +0.00164,
    "GLN:NE2": -0.00003,
    "GLN:O": +0.00072,
    "GLN:OE1": +0.00013,
    "GLU:CA": -0.00094,
    "GLU:CB": +0.00016,
    "GLU:CD": +0.00186,
    "GLU:CG": +0.00181,
    "GLU:N": -0.00053,
    "GLU:O": +0.00074,
    "GLU:OE1": +0.00043,
    "GLU:OE2": -0.00024,
    "GLY:C": +0.00036,
    "GLY:CA": -0.00045,
    "GLY:N": -0.00068,
    "GLY:O": +0.00055,
    "HIS:CA": +0.01208,
    "HIS:CB": -0.00268,
    "HIS:CD2": +0.00144,
    "HIS:CE1": +0.00083,
    "HIS:CG": -0.01488,
    "HIS:ND1": +0.01468,
    "HIS:NE2": -0.01468,
    "HIS:O": +0.00256,
    "ILE:CA": -0.00264,
    "ILE:CB": +0.00046,
    "ILE:CD1": +0.00051,
    "ILE:CG1": +0.00062,
    "ILE:CG2": +0.00058,
    "ILE:N": +0.00002,
    "ILE:O": +0.00095,
    "LEU:C": -0.00204,
    "LEU:CA": -0.00019,
    "LEU:CB": +0.00042,
    "LEU:CD1": +0.00037,
    "LEU:CD2": +0.00036,
    "LEU:CG": +0.00016,
    "LEU:N": -0.00089,
    "LEU:O": +0.00052,
    "LYS:CA": +0.00034,
    "LYS:CB": -0.00013,
    "LYS:CD": -0.00040,
    "LYS:CE": -0.00191,
    "LYS:CG": -0.00006,
    "LYS:N": -0.00013,
    "LYS:NZ": +0.00000,
    "LYS:O": +0.00083,
    "MET:CA": -0.00322,
    "MET:CB": -0.00134,
    "MET:CE": +0.00087,
    "MET:CG": +0.00103,
    "MET:O": +0.00031,
    "MET:SD": -0.00044,
    "MSE:CE": +0.00151,
    "MSE:CG": -0.00321,
    "MSE:SE": +0.00044,
    "PHE:CA": +0.00026,
    "PHE:CB": +0.00044,
    "PHE:CD1": +0.00101,
    "PHE:CD2": +0.00097,
    "PHE:CE1": +0.00157,
    "PHE:CE2": +0.00120,
    "PHE:CG": -0.00094,
    "PHE:CZ": +0.00124,
    "PHE:N": -0.00096,
    "PHE:O": +0.00036,
    "PRO:CA": -0.00008,
    "PRO:CB": +0.00004,
    "PRO:CD": +0.00021,
    "PRO:CG": +0.00033,
    "PRO:O": +0.00084,
    "SER:C": -0.00181,
    "SER:CA": -0.00064,
    "SER:CB": +0.00051,
    "SER:N": -0.00033,
    "SER:O": +0.00064,
    "SER:OG": -0.00040,
    "THR:CA": -0.00104,
    "THR:CB": +0.00015,
    "THR:CG2": +0.00027,
    "THR:N": -0.00106,
    "THR:O": +0.00073,
    "THR:OG1": +0.00030,
    "TRP:CB": -0.00025,
    "TRP:CD1": +0.00180,
    "TRP:CE2": +0.00146,
    "TRP:CE3": +0.00090,
    "TRP:CH2": +0.00074,
    "TRP:CZ2": +0.00185,
    "TRP:CZ3": +0.00268,
    "TRP:NE1": -0.00000,
    "TRP:O": +0.00093,
    "TYR:CA": -0.00232,
    "TYR:CB": +0.00081,
    "TYR:CD1": +0.00103,
    "TYR:CD2": +0.00088,
    "TYR:CE1": +0.00287,
    "TYR:CE2": +0.00137,
    "TYR:CG": +0.00025,
    "TYR:CZ": +0.00102,
    "TYR:N": +0.00124,
    "TYR:O": +0.00072,
    "TYR:OH": +0.00010,
    "VAL:CA": -0.00094,
    "VAL:CB": -0.00067,
    "VAL:CG1": +0.00036,
    "VAL:CG2": +0.00040,
    "VAL:N": +0.00041,
    "VAL:O": +0.00088,
}


def _hetero_class(atom_name: str, el: str, res: str) -> str:
    """Solvation class of a non-standard-residue heavy atom."""
    from fastpisa.interface.bonds import SALT_CHARGES

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


def fine_atom_type(atom_name: str, element: str, res_name: str) -> str:
    """Finest atom type: ``"RES:ATOM"`` for standard residues, ``"het:<class>"``
    otherwise, ``"H"`` for hydrogens."""
    el = element.strip().upper()
    if el in ("H", "D"):
        return "H"
    res = res_name.strip().upper()
    if res in FINE_TYPED_RESIDUES:
        return f"{res}:{atom_name.strip().upper()}"
    return "het:" + _hetero_class(atom_name, el, res)


def class_of_fine(fine: str) -> str:
    """Map a fine type to its SIGMA class (the shipped scheme)."""
    if fine == "H":
        return "H"
    if fine.startswith("het:"):
        return fine[4:]
    res, atom = fine.split(":", 1)
    el = "SE" if (res == "MSE" and atom == "SE") else atom[0]
    if res in _NUCLEOTIDES:
        if el == "P":
            return "NA_P"
        if atom in ("OP1", "OP2", "OP3", "O1P", "O2P"):
            return "NA_OP"
        if atom.endswith("'"):
            return "NA_C_sugar" if el == "C" else "NA_O_sugar"
        return {"C": "NA_C_base", "N": "NA_N_base", "O": "NA_O_base"}.get(el, "X")
    if el == "C":
        if atom == "CA":
            return "C_CA"
        if atom == "C":
            return "C_bb"
        if (res, atom) in _AROMATIC_C:
            return "C_aro"
        if (res, atom) in (("ASN", "CG"), ("GLN", "CD"),
                           ("ASP", "CG"), ("GLU", "CD"), ("ARG", "CZ")):
            return "C_sp2_polar"
        return "C_ali"
    if el == "N":
        if atom == "N":
            return "N_bb"
        if res == "ARG":
            return "N_arg"
        if res == "LYS":
            return "N_lys"
        if res == "HIS":
            return "N_his"
        if res == "TRP":
            return "N_trp"
        return "N_amide"
    if el == "O":
        if atom in ("O", "OXT"):
            return "O_bb"
        if res in ("ASP", "GLU"):
            return "O_carbox"
        if res in ("ASN", "GLN"):
            return "O_amide"
        return "O_hyd"
    if el in ("S", "SE"):
        return "S_cys" if res == "CYS" else "S_met"
    return "X"


def atom_class(atom_name: str, element: str, res_name: str = "") -> str:
    """Solvation CLASS of one heavy atom (a key of :data:`SIGMA`)."""
    return class_of_fine(fine_atom_type(atom_name, element, res_name))


def is_apolar_class(cls: str) -> bool:
    """Carbon / sulfur solvation classes -- the hydrophobic part of dG_solv.

    PISA does not list hydrophobic contacts; its hydrophobic term IS the
    favourable burial of these atoms inside the solvation gain. Splitting the
    sum by this predicate exposes it without adding any model.
    """
    return cls.startswith(("C", "S", "NA_C")) and cls != "X"


def sigma_of_fine(fine: str, sigma=None, delta=None) -> float:
    """sigma for a fine type under a (class table, deviation table) pair."""
    if fine == "H":
        return 0.0
    sigma = SIGMA if sigma is None else sigma
    delta = DELTA if delta is None else delta
    return sigma.get(class_of_fine(fine), 0.0) + delta.get(fine, 0.0)


def get_asp(atom_name: str, element: Optional[str] = None,
            res_name: str = "") -> float:
    """ASP (kcal mol^-1 A^-2) for an atom. Hydrogens return 0."""
    el = (element or "").strip().upper()
    if el in ("H", "D"):
        return 0.0
    if not el:
        stripped = atom_name.strip().upper()
        el = stripped[0] if stripped else "C"
    return sigma_of_fine(fine_atom_type(atom_name, el, res_name))
