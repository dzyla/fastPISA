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
never split a PDB entry), polymer-polymer interfaces (n = 2314):

    Pearson r = 0.993, R^2 about the 1:1 line = 0.987,
    median |error| = 0.32 kcal/mol, bias +0.03.

    (previous 11-class model under the same geometry: r 0.977, R^2 0.951,
    median 0.71; its published figure of 0.74 was under the older radii.)

Protein-nucleic-acid interfaces (n = 241): R^2 0.961, median 0.30.
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
    "C_CA":         -0.01507,   # backbone CA
    "C_bb":         -0.01389,   # backbone carbonyl C
    "C_ali":        -0.01597,   # sp3 side-chain C
    "C_aro":        -0.01685,   # aromatic ring C (Phe/Tyr/Trp/His)
    "C_sp2_polar":  -0.01852,   # Asn CG / Gln CD / Asp CG / Glu CD / Arg CZ
    "N_bb":          0.01148,   # backbone N
    "N_amide":       0.01107,   # Asn ND2 / Gln NE2
    "N_arg":         0.02281,   # Arg NE/NH1/NH2
    "N_lys":         0.03060,   # Lys NZ
    "N_his":         0.02218,   # His ND1/NE2
    "N_trp":         0.01012,   # Trp NE1
    "O_bb":          0.01033,   # backbone O / OXT
    "O_amide":       0.01633,   # Asn OD1 / Gln OE1
    "O_carbox":      0.01399,   # Asp OD1/OD2, Glu OE1/OE2
    "O_hyd":         0.01094,   # Ser OG / Thr OG1 / Tyr OH
    "S_met":        -0.03919,   # Met SD / Mse SE
    "S_cys":        -0.03928,   # Cys SG
    "NA_C_sugar":    0.03201,   # nucleotide sugar C
    "NA_C_base":     0.03358,   # nucleotide base C
    "NA_N_base":     0.01071,   # nucleotide base N
    "NA_O_base":    -0.05815,   # nucleotide base O
    "NA_O_sugar":   -0.04136,   # nucleotide sugar O
    "NA_OP":        -0.05212,   # nucleotide phosphate OP1/OP2
    "NA_P":          0.00000,   # nucleotide P -- pinned, see note
    "C":            -0.00135,   # hetero-group carbon
    "N":             0.02071,   # hetero neutral N
    "N+":            0.02033,   # hetero charged N (no data; fallback)
    "O":             0.00315,   # hetero neutral O
    "O-":            0.00890,   # hetero carboxylate O (no data; fallback)
    "OP":           -0.03866,   # hetero phosphate / acid-ester O (ATP-like)
    "OI":           -0.08546,   # inorganic ion O (SO4, PO4, ...)
    "S":            -0.03986,   # hetero sulfur / selenium
    "P":             0.00000,   # phosphorus (hetero) -- pinned, see note
    "HAL":          -0.01570,   # halogen / halide; per-element DELTA below
    "MET":          -0.08818,   # metal ions other than Zn; per-element DELTA below
    "ZN":           -0.33657,   # zinc
    "X":             0.00000,   # anything else: no contribution
}

# Per-fine-type deviation from the class value (kcal mol^-1 A^-2), shrunk
# toward zero (ridge 1000). Types absent here use their class value alone.
DELTA = {
    "A:N6": +0.00275,
    "ALA:C": +0.00005,
    "ALA:CA": -0.00044,
    "ALA:CB": +0.00029,
    "ALA:N": -0.00019,
    "ALA:O": +0.00067,
    "ARG:CA": -0.00466,
    "ARG:CB": +0.00159,
    "ARG:CD": +0.00066,
    "ARG:CG": +0.00399,
    "ARG:CZ": -0.01146,
    "ARG:N": -0.00316,
    "ARG:NE": +0.00012,
    "ARG:NH1": +0.00268,
    "ARG:NH2": -0.00279,
    "ARG:O": +0.00090,
    "ASN:CA": -0.00023,
    "ASN:CB": +0.00028,
    "ASN:CG": +0.00368,
    "ASN:N": -0.00002,
    "ASN:ND2": +0.00003,
    "ASN:O": +0.00090,
    "ASN:OD1": -0.00012,
    "ASP:CA": -0.00012,
    "ASP:CB": +0.00046,
    "ASP:CG": +0.00245,
    "ASP:N": -0.00053,
    "ASP:O": +0.00073,
    "ASP:OD1": +0.00028,
    "ASP:OD2": -0.00043,
    "CYS:CA": -0.00113,
    "CYS:CB": -0.00025,
    "CYS:O": +0.00023,
    "CYS:SG": +0.00000,
    "DA:C2": -0.00203,
    "DA:N1": -0.00350,
    "DA:N6": +0.00515,
    "DA:OP1": +0.00416,
    "DA:OP2": +0.00073,
    "DC:N3": -0.00112,
    "DC:N4": +0.00710,
    "DC:O2": -0.00110,
    "DC:OP1": +0.00474,
    "DC:OP2": +0.00224,
    "DG:N1": -0.00707,
    "DG:N2": +0.00859,
    "DG:O6": -0.00155,
    "DG:OP1": +0.00461,
    "DG:OP2": -0.00130,
    "DT:C7": -0.03001,
    "DT:N3": -0.00204,
    "DT:O2": +0.00128,
    "DT:O4": -0.00231,
    "DT:OP1": +0.00392,
    "DT:OP2": -0.00228,
    "GLN:CA": +0.00017,
    "GLN:CB": +0.00031,
    "GLN:CD": +0.00349,
    "GLN:CG": +0.00164,
    "GLN:NE2": -0.00003,
    "GLN:O": +0.00069,
    "GLN:OE1": +0.00012,
    "GLU:CA": -0.00094,
    "GLU:CB": +0.00014,
    "GLU:CD": +0.00184,
    "GLU:CG": +0.00181,
    "GLU:N": -0.00053,
    "GLU:O": +0.00073,
    "GLU:OE1": +0.00042,
    "GLU:OE2": -0.00026,
    "GLY:C": +0.00037,
    "GLY:CA": -0.00045,
    "GLY:N": -0.00065,
    "GLY:O": +0.00053,
    "HIS:CA": +0.01211,
    "HIS:CB": -0.00270,
    "HIS:CD2": +0.00145,
    "HIS:CE1": +0.00085,
    "HIS:CG": -0.01490,
    "HIS:ND1": +0.01473,
    "HIS:NE2": -0.01473,
    "HIS:O": +0.00257,
    "ILE:CA": -0.00263,
    "ILE:CB": +0.00044,
    "ILE:CD1": +0.00051,
    "ILE:CG1": +0.00063,
    "ILE:CG2": +0.00058,
    "ILE:N": +0.00003,
    "ILE:O": +0.00093,
    "LEU:C": -0.00206,
    "LEU:CA": -0.00017,
    "LEU:CB": +0.00042,
    "LEU:CD1": +0.00038,
    "LEU:CD2": +0.00036,
    "LEU:CG": +0.00016,
    "LEU:N": -0.00090,
    "LEU:O": +0.00050,
    "LYS:CA": +0.00033,
    "LYS:CB": -0.00012,
    "LYS:CD": -0.00042,
    "LYS:CE": -0.00192,
    "LYS:CG": -0.00007,
    "LYS:N": -0.00010,
    "LYS:NZ": +0.00000,
    "LYS:O": +0.00082,
    "MET:CA": -0.00321,
    "MET:CB": -0.00133,
    "MET:CE": +0.00087,
    "MET:CG": +0.00103,
    "MET:O": +0.00028,
    "MET:SD": -0.00044,
    "MSE:CE": +0.00151,
    "MSE:CG": -0.00320,
    "MSE:SE": +0.00044,
    "PHE:CA": +0.00027,
    "PHE:CB": +0.00044,
    "PHE:CD1": +0.00099,
    "PHE:CD2": +0.00096,
    "PHE:CE1": +0.00155,
    "PHE:CE2": +0.00118,
    "PHE:CG": -0.00095,
    "PHE:CZ": +0.00122,
    "PHE:N": -0.00097,
    "PHE:O": +0.00034,
    "PRO:CA": -0.00007,
    "PRO:CB": +0.00005,
    "PRO:CD": +0.00021,
    "PRO:CG": +0.00033,
    "PRO:O": +0.00082,
    "SER:C": -0.00181,
    "SER:CA": -0.00063,
    "SER:CB": +0.00051,
    "SER:N": -0.00033,
    "SER:O": +0.00062,
    "SER:OG": -0.00040,
    "THR:CA": -0.00104,
    "THR:CB": +0.00012,
    "THR:CG2": +0.00028,
    "THR:N": -0.00104,
    "THR:O": +0.00073,
    "THR:OG1": +0.00030,
    "TRP:CB": -0.00027,
    "TRP:CD1": +0.00177,
    "TRP:CE2": +0.00149,
    "TRP:CE3": +0.00097,
    "TRP:CH2": +0.00079,
    "TRP:CZ2": +0.00177,
    "TRP:CZ3": +0.00259,
    "TRP:NE1": +0.00000,
    "TRP:O": +0.00091,
    "TYR:CA": -0.00231,
    "TYR:CB": +0.00080,
    "TYR:CD1": +0.00106,
    "TYR:CD2": +0.00085,
    "TYR:CE1": +0.00281,
    "TYR:CE2": +0.00135,
    "TYR:CG": +0.00017,
    "TYR:CZ": +0.00104,
    "TYR:N": +0.00126,
    "TYR:O": +0.00069,
    "TYR:OH": +0.00010,
    "VAL:CA": -0.00093,
    "VAL:CB": -0.00067,
    "VAL:CG1": +0.00037,
    "VAL:CG2": +0.00040,
    "VAL:N": +0.00046,
    "VAL:O": +0.00084,
    "het:HAL:CL": -0.07600,
    "het:HAL:I": +0.01755,
    "het:MET:CA": -0.05242,
    "het:MET:CD": +0.01752,
    "het:MET:CU": +0.01485,
    "het:MET:FE": +0.00840,
    "het:MET:HG": -0.04835,
    "het:MET:K": -0.35487,
    "het:MET:MG": +0.01092,
    "het:MET:MN": +0.05046,
    "het:MET:NA": -0.00257,
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
    cls = _hetero_class(atom_name, el, res)
    # metals / halogens: element-resolved, so each ion can earn its own
    # shrunk deviation from the class value (Mg is not K)
    if cls in ("MET", "HAL"):
        return f"het:{cls}:{el}"
    return "het:" + cls


def class_of_fine(fine: str) -> str:
    """Map a fine type to its SIGMA class (the shipped scheme)."""
    if fine == "H":
        return "H"
    if fine.startswith("het:"):
        return fine.split(":")[1]
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
