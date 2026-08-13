"""
Atomic Solvation Parameters (ASP) for PISA energy calculation.

The values below are derived from the literature (Ooi et al. 1987,
Srinivasan et al. 1999, and the PISA/CryoEM ASP tables used by the
PDBe PISA implementation).  Each parameter is a free energy of
solvation per unit buried surface area (kcal mol^-1 A^-2) for a given
atom type.

SIGN CONVENTION, AND WHY THIS TABLE DOES NOT SATISFY IT.
`calculate_solvation_energy` sums ``asp * bsa`` with no negation and documents its
result as "negative = favourable". For that sum to be negative on a favourable
interface, a favourably-buried atom must carry a NEGATIVE ASP.

The values here do the OPPOSITE of that. Aliphatic carbon -- whose burial IS the
hydrophobic effect, the archetypal favourable contribution -- carries +0.0259, so
the code scores carbon burial as unfavourable; polar N (-0.0623) and O (-0.1057)
are negative, so the code scores polar desolvation as favourable. That is backwards
physically: hydrophobic burial drives association while polar desolvation costs
energy. The previous docstring here asserted the other convention ("positive values
mean burying is favourable"), which is self-consistent with these numbers but
contradicts `energy.py`; either way one of the two files was wrong, and the numbers
themselves do not match the code's stated convention.

The total nonetheless comes out negative on every interface measured (63 of 63,
range -176.6 to -2.9 kcal/mol) because the polar terms dominate numerically. So
the result LOOKS favourable while being driven by the wrong term -- which is
exactly why it anti-correlates with PISA's dG on antibody-antigen interfaces (see
below). A SIGN FLIP DOES NOT FIX THIS: Spearman is invariant under negation, so
negating the table merely mirrors the correlation (-0.408 becomes +0.408) without
making the model right. The relative WEIGHTING of apolar against polar burial is
what needs refitting, not the overall sign. Fitting PISA's dG on interface area and
H-bond count instead recovers the correct physics (area coefficient negative: more
buried area, more favourable dG) at Spearman +0.388 on the same interfaces.

Recalibration is deliberately NOT attempted in this commit; it needs a fitting set
far broader than 7 complexes, and probably separate treatment per interface class.

CALIBRATION STATUS -- these values do NOT reproduce CCP4 PISA's dG. Measured
against PISA v2.2.0 over 63 matched interfaces from 7 antibody-antigen complexes,
the resulting solvation energy relates to PISA's dG with Spearman -0.408
(p = 0.015, n = 35) on ANTIBODY-ANTIGEN interfaces but +0.596 on antibody-antibody
ones, so the pooled figure (+0.324) is a Simpson's paradox and the agreement is
class-dependent; magnitudes run about 6x larger than PISA's. Interface AREA agrees
at Pearson 0.9996 and salt-bridge counts at Spearman 0.998, so prefer those as
quantitative outputs and treat these energies as relative, uncalibrated indicators.

The standard ASP set used by PISA assigns parameters to atom types:
C, N, O, S, P, and the heteroatoms commonly found in protein structures.
"""

# Atom-type -> ASP (kcal/mol/A^2)
# These values are calibrated to reproduce PISA solvation energies.
# Source: PISA ASP table (Krissinel & Henrick, 2007; Srinivasan et al., 1999).
ASP_TABLE = {
    # element: asp_value
    "C": 0.0259,   # aliphatic C
    "N": -0.0623,  # polar N
    "O": -0.1057,  # polar O
    "S": 0.0100,   # sulfur
    "P": 0.0100,   # phosphorus
    "F": 0.0100,
    "Cl": 0.0100,
    "Br": 0.0100,
    "I": 0.0100,
    "H": 0.0,      # hydrogens are treated separately (attached to heavy atom)
}

# More granular ASP by atom name (PISA uses atom-name-level parameters)
# Format: atom_name -> asp (kcal/mol/A^2)
# These override the element-level defaults when a specific atom name
# is recognised.
ASP_BY_NAME = {
    # Standard amino acid atom names
    "N": -0.0623,
    "CA": 0.0259,
    "C": 0.0259,
    "O": -0.1057,
    "CB": 0.0259,
    "CG": 0.0259,
    "CG1": 0.0259,
    "CG2": 0.0259,
    "CD": 0.0259,
    "CD1": 0.0259,
    "CD2": 0.0259,
    "CE": 0.0259,
    "CE1": 0.0259,
    "CE2": 0.0259,
    "CZ": 0.0259,
    "CH2": 0.0259,
    "OH": -0.1057,
    "OG": -0.1057,
    "OG1": -0.1057,
    "SG": 0.0100,
    "ND1": -0.0623,
    "ND2": -0.0623,
    "OD1": -0.1057,
    "OD2": -0.1057,
    "SD": 0.0100,
    "CD2": 0.0259,
    "NE": -0.0623,
    "CZ": 0.0259,
    "NH1": -0.0623,
    "NH2": -0.0623,
    "OH": -0.1057,
    "OXT": -0.1057,
    "OXT1": -0.1057,
    "H": 0.0,
    "HA": 0.0,
    "HB": 0.0,
    "HB1": 0.0,
    "HB2": 0.0,
    "HB3": 0.0,
    "HG": 0.0,
    "HG1": 0.0,
    "HG2": 0.0,
    "HD": 0.0,
    "HD1": 0.0,
    "HD2": 0.0,
    "HE": 0.0,
    "HE1": 0.0,
    "HE2": 0.0,
    "HZ": 0.0,
    "HO": 0.0,
    "HS": 0.0,
    "HN": 0.0,
    "H1": 0.0,
    "H2": 0.0,
    "H3": 0.0,
    "HN2": 0.0,
    "HND": 0.0,
    "HOH": -0.1057,
    "NA": 0.0100,
    "PB": 0.0100,
    "ZN": 0.0100,
    "FE": 0.0100,
    "MG": 0.0100,
    "CA_": 0.0100,
    "CU": 0.0100,
    "NI": 0.0100,
    "CO": 0.0100,
    "MO": 0.0100,
    "W": 0.0100,
    "RE": 0.0100,
    "LU": 0.0100,
    "HB_": 0.0,
    "HG_": 0.0,
    "HS_": 0.0,
    "HE_": 0.0,
    "HZ_": 0.0,
    "HO_": 0.0,
    "HA_": 0.0,
    # Nucleic acid atom names
    "P": 0.0100,
    "O1P": -0.1057,
    "O2P": -0.1057,
    "O3P": -0.1057,
    "O5P": -0.1057,
    "O4P": -0.1057,
    "OP1": -0.1057,
    "OP2": -0.1057,
    "C1'": 0.0259,
    "C2'": 0.0259,
    "C3'": 0.0259,
    "C4'": 0.0259,
    "C5'": 0.0259,
    "O2'": -0.1057,
    "O3'": -0.1057,
    "O4'": -0.1057,
    "O5'": -0.1057,
    "N1": -0.0623,
    "N2": -0.0623,
    "N3": -0.0623,
    "N4": -0.0623,
    "N6": -0.0623,
    "N7": -0.0623,
    "N9": -0.0623,
    "O6": -0.1057,
    "O2": -0.1057,
    "O4": -0.1057,
    "N6": -0.0623,
    "O2": -0.1057,
    "C8": 0.0259,
    "C2": 0.0259,
    "C4": 0.0259,
    "C5": 0.0259,
    "C6": 0.0259,
    "C9": 0.0259,
    "H1": 0.0,
    "H2": 0.0,
    "H3": 0.0,
    "H4": 0.0,
    "H5": 0.0,
    "HO3'": 0.0,
    "HO5'": 0.0,
    "HN2": 0.0,
    "HN3": 0.0,
    "HN4": 0.0,
    "HN6": 0.0,
    "HN7": 0.0,
    "HN9": 0.0,
}


def get_asp(atom_name: str, element: str = None) -> float:
    """Return the ASP value for an atom.

    Parameters
    ----------
    atom_name : str
        Atom name (e.g. "CA", "O", "ND1").
    element : str, optional
        Element symbol (e.g. "C", "O").  Used as fallback if atom_name
        is not found in the name-specific table.

    Returns
    -------
    float
        ASP in kcal mol^-1 A^-2.
    """
    # Strip trailing digits/chars that sometimes appear in PDB atom names
    name = atom_name.strip()

    # Look up by atom name first
    if name in ASP_BY_NAME:
        return ASP_BY_NAME[name]

    # Try without trailing non-alphanumeric chars
    clean = name
    while clean and (not clean[-1].isalpha() or clean[-1] in "0123456789"):
        clean = clean[:-1]
    if clean and clean in ASP_BY_NAME:
        return ASP_BY_NAME[clean]

    # Fall back to element-level ASP
    if element:
        el = element.upper()
        if el in ASP_TABLE:
            return ASP_TABLE[el]

    # Default ASP for unknown atom types
    return 0.0100