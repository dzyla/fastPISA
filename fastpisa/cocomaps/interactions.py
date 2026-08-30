"""
COCOMAPS atomic interaction classifier.

Implements the interaction-type classification described in COCOMAPS 2.0
(Chawla et al., Bioinformatics 2025, btaf606) for protein-protein and
protein-nucleic acid interfaces.

The following interaction types are classified (subset implementable without
an external H-add / HBPLUS backend; vdW + pi + electrostatic rules follow the
criteria described in the COCOMAPS 2.0 paper and the literature it cites):

  hydrogen_bond      - N/O-H ... N/O       (H-bond donor-acceptor)
  weak_hbond         - C-H ... O/N         (CH-O / CH-N weak H-bond)
  salt_bridge        - + ... - charged pair< 4.0 A
  disulfide          - Cys Sgamma-Sgamma   < 3.0 A
  halogen_bond       - halogen ... O/N
  pi_pi              - aromatic ring ... aromatic ring
  cation_pi          - + charged ... pi ring
  ch_pi              - C-H(apolar sp3) ... pi ring
  polar_vdw          - polar vdW contact, distance <= r1 + r2 + 0.5 A
  apolar_vdw         - apolar vdW contact, distance <= r1 + r2 + 0.5 A
  clash              - interatomic distance < sum of vdW radii (no other class)
  water_mediated     - contact bridged by a crystallographic water
  metal_mediated     - salt/coordination contact involving a metal ion
  proximal           - within the 5 A cutoff but beyond vdW contact range
                       (COCOMAPS 2.0 "Proximal contact")

Each interface residue pair gets a dominant interaction type.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from fastpisa.interface.contacts import HBOND_DISTANCE

# ---------------------------------------------------------------------------
# Amino-acid / nucleotide atom classification tables
# ---------------------------------------------------------------------------

# Charged side-chain atoms: (residue, atom) -> charge (+1 / -1 / None)
CHARGED_ATOMS: Dict[Tuple[str, str], int] = {
    # Positive
    ("ARG", "NH1"): 1, ("ARG", "NH2"): 1, ("ARG", "CZ"): 1,
    ("LYS", "NZ"): 1,
    ("HIS", "ND1"): 1, ("HIS", "NE2"): 1,
    # Negative
    ("ASP", "OD1"): -1, ("ASP", "OD2"): -1,
    ("GLU", "OE1"): -1, ("GLU", "OE2"): -1,
}

# COCOMAPS 2.0 salt-bridge convention (differs from PISA's, which fastPISA
# uses for the PISA-schema number_salt_bridges counts): cationic Lys NZ /
# Arg NH1/NH2 vs Asp/Glu carboxylate O, or vs nucleic-acid phosphate OP1/OP2,
# within 4.5 A.
COCOMAPS_SALT_CATIONS = {("LYS", "NZ"), ("ARG", "NH1"), ("ARG", "NH2")}
COCOMAPS_SALT_ANIONS = {("ASP", "OD1"), ("ASP", "OD2"),
                        ("GLU", "OE1"), ("GLU", "OE2")}
_NA_RES = {"A", "U", "G", "C", "T", "DA", "DG", "DC", "DT", "DU", "PSU"}
COCOMAPS_SALT_DIST = 4.5

# Carbons that carry NO hydrogen (carbonyl / carboxyl / guanidinium /
# aromatic junction carbons): they cannot donate a weak C-H...O/N bond.
_NO_H_CARBONS = {
    ("*", "C"),
    ("ASP", "CG"), ("GLU", "CD"), ("ASN", "CG"), ("GLN", "CD"),
    ("ARG", "CZ"), ("TYR", "CG"), ("TYR", "CZ"), ("PHE", "CG"),
    ("TRP", "CG"), ("TRP", "CD2"), ("TRP", "CE2"), ("HIS", "CG"),
}
_NA_NO_H_CARBONS = {
    "A": {"C4", "C5", "C6"}, "G": {"C2", "C4", "C5", "C6"},
    "C": {"C2", "C4"}, "T": {"C2", "C4", "C5"}, "U": {"C2", "C4"},
}
WEAK_HBOND_DIST = 3.6  # COCOMAPS 2.0 CH_ON_DIST


def _carbon_bears_h(res: str, name: str) -> bool:
    if ("*", name) in _NO_H_CARBONS or (res, name) in _NO_H_CARBONS:
        return False
    base = res[1:] if len(res) == 2 and res[0] in ("D", "R") else res
    if base in _NA_NO_H_CARBONS and name in _NA_NO_H_CARBONS[base]:
        return False
    return True


def _cocomaps_salt_bridge(res1: str, a1: str, res2: str, a2: str) -> bool:
    for (rc, ac), (ra, aa) in (((res1, a1), (res2, a2)),
                               ((res2, a2), (res1, a1))):
        if (rc, ac) in COCOMAPS_SALT_CATIONS:
            if (ra, aa) in COCOMAPS_SALT_ANIONS:
                return True
            if ra in _NA_RES and aa in ("OP1", "OP2", "O1P", "O2P"):
                return True
    return False


# Aromatic (pi) side-chain atoms for pi interactions
AROMATIC_RESIDUES = {"PHE", "TYR", "TRP", "HIS"}
AROMATIC_RING_ATOMS = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
    "TRP": {"CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2", "NE1"},
    "HIS": {"CG", "ND1", "CD2", "CE1", "NE2"},
}
# Aromatic nucleotides (base rings)
NUCLEOBASE_AROMATIC = {
    "A": {"N1", "C2", "N3", "C4", "C5", "C6"},
    "G": {"N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"},
    "C": {"N1", "C2", "N3", "C4", "C5", "C6"},
    "T": {"N1", "C2", "N3", "C4", "C5", "C6", "C7"},
    "U": {"N1", "C2", "N3", "C4", "C5", "C6"},
    "DA": {"N1", "C2", "N3", "C4", "C5", "C6"},
    "DG": {"N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"},
    "DC": {"N1", "C2", "N3", "C4", "C5", "C6"},
    "DT": {"N1", "C2", "N3", "C4", "C5", "C6", "C7"},
    "DU": {"N1", "C2", "N3", "C4", "C5", "C6"},
}

# H-bond donor / acceptor heavy atoms (N and O) by residue
# Keyed by residue+atom; value is set of {"donor", "acceptor"}
HBOND_ATOMS_AA = {
    ("ASN", "OD1"): {"acceptor"}, ("ASN", "ND2"): {"donor"},
    ("ASP", "OD1"): {"acceptor"}, ("ASP", "OD2"): {"acceptor"},
    ("GLN", "OE1"): {"acceptor"}, ("GLN", "NE2"): {"donor"},
    ("GLU", "OE1"): {"acceptor"}, ("GLU", "OE2"): {"acceptor"},
    ("HIS", "ND1"): {"donor", "acceptor"}, ("HIS", "NE2"): {"donor", "acceptor"},
    ("SER", "OG"): {"donor", "acceptor"},
    ("THR", "OG1"): {"donor", "acceptor"},
    ("TYR", "OH"): {"donor", "acceptor"},
    ("TRP", "NE1"): {"donor"},
    ("CYS", "SG"): {"donor", "acceptor"},
    ("MET", "SD"): {"acceptor"},
    # Backbone
    ("_BB", "N"): {"donor"}, ("_BB", "O"): {"acceptor"},
}

# Atoms considered polar (candidate H-bond / polar contact)
def is_polar_atom(atom_name: str, element: str) -> bool:
    """Heuristic: N, O, S, halogen atoms and phosphoryl groups are polar."""
    el = element.upper().strip()
    name = atom_name.strip().upper()
    if el in ("N", "O", "S"):
        return True
    if el in ("F", "CL", "BR", "I"):
        return True
    # Nucleotide phosphate oxygens
    if name in ("OP1", "OP2", "O1P", "O2P", "P"):
        return True
    return False


def is_apolar_atom(atom_name: str, element: str) -> bool:
    """Heuristic: carbon atoms are apolar unless in a polar environment."""
    el = element.upper().strip()
    name = atom_name.strip().upper()
    if el == "C":
        # Carbonyl carbon and carboxylate carbon are polar-ish
        if name in ("C", "CA", "CB", "CG", "CD", "CE", "CZ", "CH2", "CG2", "CD1", "CD2", "CE1", "CE2", "CZ2", "CZ3"):
            return True
    return False


def is_metal(element: str) -> bool:
    return element.upper().strip() in {
        "FE", "ZN", "MG", "CA", "CU", "MN", "NI", "CO", "MO", "W", "CD", "HG", "NA", "K",
    }


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

# COCOMAPS 2.0 interaction categories (subset)
INTERACTION_TYPES = [
    "hydrogen_bond", "weak_hbond", "salt_bridge", "disulfide",
    "halogen_bond", "pi_pi", "cation_pi", "ch_pi", "polar_vdw",
    "apolar_vdw", "water_mediated", "metal_mediated", "clash", "proximal",
]

# vdW-contact tolerance (A): COCOMAPS 2.0 counts a polar/apolar vdW contact
# only within r1 + r2 + tolerance; anything farther (but inside the 5 A
# cutoff) is a "proximal" contact.
VDW_CONTACT_TOLERANCE = 0.5

PROMISCUOUS_AA_SET = {nuc for nuc in NUCLEOBASE_AROMATIC}


def classify_atom_pair(
    res1: str,
    atom1: str,
    el1: str,
    res2: str,
    atom2: str,
    el2: str,
    dist: float,
    vdw_radius1: float,
    vdw_radius2: float,
    is_hbond: Optional[bool] = None,
    pi_verdicts: Optional[tuple] = None,
) -> str:
    """Classify the interaction type for a single atom-atom contact.

    Returns one of INTERACTION_TYPES.

    ``is_hbond``: pre-computed geometric H-bond verdict for this pair (from
    :mod:`fastpisa.interface.bonds`). When provided it replaces the
    table-only ``_hbond`` rule so all fastPISA outputs share one H-bond
    definition; ``None`` falls back to the legacy rule.

    ``pi_verdicts``: pre-computed ring-geometry verdicts
    ``(pi_pi_ok, cation_pi_ok, ch_pi_ok)`` from
    :class:`fastpisa.cocomaps.rings.RingContext`. When provided, pi classes
    require the geometric verdict (centroid distance and position above the
    ring plane) in addition to the chemical atom-type rules; ``None`` falls
    back to legacy proximity-only rules.
    """
    res1_u = res1.strip().upper()
    res2_u = res2.strip().upper()
    a1 = atom1.strip().upper()
    a2 = atom2.strip().upper()
    el1_u = el1.upper().strip()
    el2_u = el2.upper().strip()

    # 1. Disulfide (valid covalent S-S between Cys), < 3.0 A
    if (el1_u == "S" and el2_u == "S" and res1_u == "CYS"
            and res2_u == "CYS" and dist < 3.0):
        return "disulfide"

    # 2. Salt bridge, COCOMAPS 2.0 convention: Lys NZ / Arg NH* vs Asp/Glu
    #    carboxylate O or nucleic-acid phosphate OP1/OP2, <= 4.5 A. (The
    #    PISA-schema number_salt_bridges counts use PISA's own convention
    #    from fastpisa.interface.bonds instead.)
    if dist <= COCOMAPS_SALT_DIST and _cocomaps_salt_bridge(res1_u, a1, res2_u, a2):
        return "salt_bridge"

    # 3. Halogen bond: halogen ... O/N, < 3.6 A
    if dist < 3.6:
        halogens = {"F", "CL", "BR", "I"}
        if (el1_u in halogens and el2_u in ("O", "N")) or (
            el2_u in halogens and el1_u in ("O", "N")
        ):
            return "halogen_bond"

    # 4. Metal-mediated: a metal ion coordinates a polar atom, < 3.0 A
    if dist < 3.0 and (is_metal(el1_u) or is_metal(el2_u)):
        return "metal_mediated"

    # 5. Hydrogen bond / weak H-bond (before clash: H-bonds are short-range).
    #    When the caller supplies the geometric verdict (fastpisa.interface.
    #    bonds; distance + antecedent angles), use it -- single source of
    #    truth with the PISA-calibrated counts. Weak C-H...O/N keeps its own
    #    3.8 A band.
    if is_hbond if is_hbond is not None else (
            dist < HBOND_DISTANCE and _hbond(res1_u, a1, res2_u, a2, el1_u, el2_u)):
        return "hydrogen_bond"
    if dist <= WEAK_HBOND_DIST:
        # weak C-H ... O/N: the carbon must actually carry a hydrogen
        if (el1_u == "C" and el2_u in ("O", "N")
                and _carbon_bears_h(res1_u, a1)) or (
                el2_u == "C" and el1_u in ("O", "N")
                and _carbon_bears_h(res2_u, a2)):
            return "weak_hbond"

    # 7. Pi-pi / cation-pi / ch-pi (ring-ring or ring-charge/alkyl)
    ring1 = _is_ring_atom(res1_u, a1, el1_u)
    ring2 = _is_ring_atom(res2_u, a2, el2_u)
    if pi_verdicts is not None:
        pi_pi_ok, cation_pi_ok, ch_pi_ok = pi_verdicts
    else:
        pi_pi_ok, cation_pi_ok, ch_pi_ok = dist < 5.5, dist < 6.0, dist < 5.0
    if ring1 and ring2 and pi_pi_ok:
        return "pi_pi"
    if ring1 or ring2:
        # cation-pi: + charged atom vs ring
        if ring1 and cation_pi_ok:
            c2 = CHARGED_ATOMS.get((res2_u, a2))
            if c2 == 1:
                return "cation_pi"
        if ring2 and cation_pi_ok:
            c1 = CHARGED_ATOMS.get((res1_u, a1))
            if c1 == 1:
                return "cation_pi"
        # ch-pi: sp3 carbon (apolar) vs ring
        if ring1 and el2_u == "C" and not ring2 and ch_pi_ok:
            return "ch_pi"
        if ring2 and el1_u == "C" and not ring1 and ch_pi_ok:
            return "ch_pi"

    # 8. Clash: below the sum of vdW radii and no specific interaction
    #    (COCOMAPS 2.0 convention)
    if dist < (vdw_radius1 + vdw_radius2):
        return "clash"

    # 9. Polar / apolar vdW contacts require actual vdW-range proximity
    #    (r1 + r2 + tolerance), matching COCOMAPS 2.0.
    if dist <= (vdw_radius1 + vdw_radius2 + VDW_CONTACT_TOLERANCE):
        pol1 = is_polar_atom(a1, el1_u)
        pol2 = is_polar_atom(a2, el2_u)
        if pol1 or pol2:
            return "polar_vdw"
        return "apolar_vdw"

    # 10. Within the 5 A cutoff but beyond vdW contact range
    return "proximal"


def _is_ring_atom(res_name: str, atom_name: str, element: str) -> bool:
    """Whether the atom belongs to an aromatic ring (aa side chain or base)."""
    if res_name in AROMATIC_RESIDUES:
        if atom_name in AROMATIC_RING_ATOMS[res_name]:
            return True
    if res_name in NUCLEOBASE_AROMATIC:
        if atom_name in NUCLEOBASE_AROMATIC[res_name]:
            return True
    return False


def _hbond(res1: str, a1: str, res2: str, a2: str, el1: str, el2: str) -> bool:
    """Rule-based H-bond detection between two N/O heavy atoms.

    A contact is an H-bond when one side provides a donor (N-H or O-H) and
    the other an acceptor (N or O). If neither side has an explicit donor
    role, accept N-donor/O-acceptor generic roles (e.g. for backbone atoms).
    A contact between two "acceptor-only" oxygens is not an H-bond.
    """
    nuc = ("N", "O")
    if el1 not in nuc or el2 not in nuc:
        return False

    role1 = _atoms_roles(res1, a1)
    role2 = _atoms_roles(res2, a2)
    if "donor" in role1 and "acceptor" in role2:
        return True
    if "donor" in role2 and "acceptor" in role1:
        return True
    # Neither side is a donor: only accept if at least one atom is an N
    # (generic backbone amide N is a donor). Two acceptor-only O's: no.
    if "donor" in role1 or "donor" in role2:
        return True
    if el1 == "N" or el2 == "N":
        return True
    return False


def _atoms_roles(res_name: str, atom_name: str) -> set:
    key = (res_name, atom_name)
    if key in HBOND_ATOMS_AA:
        return HBOND_ATOMS_AA[key]
    # Backbone defaults
    if atom_name == "N":
        return {"donor"}
    if atom_name == "O":
        return {"acceptor"}
    # Generic N = donor, generic O = acceptor
    el = atom_name[:1]
    if el == "N":
        return {"donor"}
    if el == "O":
        return {"acceptor"}
    return set()