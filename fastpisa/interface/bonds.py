"""PISA-grade inter-molecular bond detection (single source of truth).

Original PISA's per-interface bond lists follow McDonald-Thornton-style
geometric criteria evaluated on heavy atoms (no explicit hydrogens):

  Hydrogen bond   donor(N/O/S with implicit H) ... acceptor(N/O/S), with
                  donor-acceptor distance <= 3.89 A AND every covalent
                  antecedent X of either end making an angle X-D-A (resp.
                  X-A-D) >= 90 deg. The angle test is what separates real
                  H-bonds from generic polar proximity (validated against
                  EBI PISA bond lists: >=90 deg keeps 98% of PISA's bonds
                  and rejects ~80% of the false positives a pure distance
                  cutoff admits).
  Salt bridge     charged side-chain N ... charged side-chain O <= 4.0 A
                  (Arg NE/NH1/NH2, Lys NZ, His ND1/NE2 vs Asp OD1/OD2,
                  Glu OE1/OE2). PISA does not count phosphate backbone
                  oxygens as salt-bridge partners.
  Disulfide       Cys SG ... Cys SG < 3.0 A.

IMPORTANT: the classes are NOT mutually exclusive. PISA lists a charged pair
that also satisfies H-bond geometry in BOTH its h-bond and salt-bridge lists
(~44% of its salt bridges are). Counting must therefore use independent
predicates -- see :func:`detect_bond_flags`.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from fastpisa.interface.contacts import AMINO_ACIDS, NUCLEIC_ACIDS, is_disulfide

HBOND_MAX_DIST = 3.89
HBOND_MIN_ANGLE = 90.0
SALT_BRIDGE_MAX_DIST = 4.0

# ---------------------------------------------------------------------------
# Hydrogen-bond donor / acceptor roles (protein + nucleic acid)
# ---------------------------------------------------------------------------
D = frozenset({"donor"})
A = frozenset({"acceptor"})
DA = frozenset({"donor", "acceptor"})

_PROTEIN_ROLES: Dict[Tuple[str, str], FrozenSet[str]] = {
    ("ARG", "NE"): D, ("ARG", "NH1"): D, ("ARG", "NH2"): D,
    ("LYS", "NZ"): D,
    ("HIS", "ND1"): DA, ("HIS", "NE2"): DA,
    ("ASN", "ND2"): D, ("ASN", "OD1"): A,
    ("GLN", "NE2"): D, ("GLN", "OE1"): A,
    ("ASP", "OD1"): A, ("ASP", "OD2"): A,
    ("GLU", "OE1"): A, ("GLU", "OE2"): A,
    ("SER", "OG"): DA, ("THR", "OG1"): DA, ("TYR", "OH"): DA,
    ("TRP", "NE1"): D,
    ("CYS", "SG"): DA, ("MET", "SD"): A, ("MSE", "SE"): A,
}

_NA_BASE_ROLES: Dict[Tuple[str, str], FrozenSet[str]] = {
    ("A", "N6"): D, ("A", "N1"): A, ("A", "N3"): A, ("A", "N7"): A,
    ("G", "N1"): D, ("G", "N2"): D, ("G", "O6"): A, ("G", "N3"): A, ("G", "N7"): A,
    ("C", "N4"): D, ("C", "O2"): A, ("C", "N3"): A,
    ("T", "N3"): D, ("T", "O2"): A, ("T", "O4"): A,
    ("U", "N3"): D, ("U", "O2"): A, ("U", "O4"): A,
}
# The same base chemistry applies to DNA (DA/DG/DC/DT/DU) and the R* aliases.
_NA_ROLES: Dict[Tuple[str, str], FrozenSet[str]] = {}
for (base, atom), roles in _NA_BASE_ROLES.items():
    for prefix in ("", "D", "R"):
        _NA_ROLES[(prefix + base, atom)] = roles

# Nucleic-acid backbone/sugar oxygens (any residue)
_NA_BACKBONE_ROLES: Dict[str, FrozenSet[str]] = {
    "OP1": A, "OP2": A, "OP3": A, "O1P": A, "O2P": A, "O3P": A,
    "O3'": A, "O5'": A, "O4'": A,
    "O2'": DA,  # ribose 2'-OH donates and accepts
}

HB_ROLES: Dict[Tuple[str, str], FrozenSet[str]] = {}
HB_ROLES.update(_PROTEIN_ROLES)
HB_ROLES.update(_NA_ROLES)

_NONE: FrozenSet[str] = frozenset()


def hb_roles(res_name: str, atom_name: str, element: str) -> FrozenSet[str]:
    """Donor/acceptor roles for one heavy atom."""
    res = res_name.strip().upper()
    name = atom_name.strip().upper()
    el = element.strip().upper()

    specific = HB_ROLES.get((res, name))
    if specific is not None:
        return specific

    is_std = res in AMINO_ACIDS or res in NUCLEIC_ACIDS
    if is_std:
        if res in NUCLEIC_ACIDS and name in _NA_BACKBONE_ROLES:
            return _NA_BACKBONE_ROLES[name]
        if name == "N":  # backbone amide (PRO has no H but PISA still lists it rarely)
            return _NONE if res == "PRO" else D
        if name in ("O", "OXT"):
            return A
        # Unlisted standard-residue N/O (e.g. modified residues): generic roles
        if el == "N":
            return D
        if el == "O":
            return A
        return _NONE

    # Non-standard residues / ligands: CCD chemistry is unknown. Generic
    # conservative roles (validated against EBI PISA bond lists: the
    # permissive donor+acceptor fallback flagged ligand carboxylate /
    # phosphate oxygens as donors and over-counted badly).
    if el == "N":
        return D
    if el == "O":
        return A
    return _NONE


# ---------------------------------------------------------------------------
# Salt-bridge charges (heavy, genuinely ionisable side-chain atoms only)
# ---------------------------------------------------------------------------
SALT_CHARGES: Dict[Tuple[str, str], int] = {
    ("ARG", "NE"): 1, ("ARG", "NH1"): 1, ("ARG", "NH2"): 1,
    ("LYS", "NZ"): 1,
    ("HIS", "ND1"): 1, ("HIS", "NE2"): 1,
    ("ASP", "OD1"): -1, ("ASP", "OD2"): -1,
    ("GLU", "OE1"): -1, ("GLU", "OE2"): -1,
}


def salt_charge(res_name: str, atom_name: str) -> Optional[int]:
    return SALT_CHARGES.get((res_name.strip().upper(), atom_name.strip().upper()))


# ---------------------------------------------------------------------------
# Geometric detection
# ---------------------------------------------------------------------------
def _covalent_neighbors(idx, atoms, coords, kd_tree):
    """Heavy-atom covalent antecedents of atom ``idx`` in the same chain."""
    el = atoms[idx].element.strip().upper()
    r = 2.1 if el in ("S", "P", "SE") else 1.8
    out = []
    for j in kd_tree.query_ball_point(coords[idx], r):
        if j == idx:
            continue
        aj = atoms[j]
        if aj.element.strip().upper() == "H":
            continue
        if aj.auth_asym_id != atoms[idx].auth_asym_id:
            continue
        out.append(j)
    return out


def _min_antecedent_angle(i, j, atoms, coords, kd_tree, cache) -> float:
    """Minimum angle X-i-j over covalent antecedents X of atom i (degrees)."""
    if i not in cache:
        cache[i] = _covalent_neighbors(i, atoms, coords, kd_tree)
    antecedents = cache[i]
    if not antecedents:
        return 180.0
    v_ij = coords[j] - coords[i]
    n_ij = np.linalg.norm(v_ij) + 1e-12
    best = 180.0
    for x in antecedents:
        v_ix = coords[x] - coords[i]
        cosang = float(np.dot(v_ix, v_ij) / ((np.linalg.norm(v_ix) + 1e-12) * n_ij))
        ang = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
        best = min(best, ang)
    return best


_METALS = {
    "FE", "ZN", "MG", "CA", "CU", "MN", "NI", "CO", "MO", "W", "CD", "HG",
    "NA", "K",
}
_METAL_COORD_DIST = 2.4


def _is_metal_coordinated(i, atoms, coords, kd_tree, metal_cache) -> bool:
    """True when atom ``i`` is directly coordinated to a metal ion.

    A metal-coordinated N/O (heme pyrrole N, His NE2 on Zn/Fe, ...) has its
    lone pair / proton engaged by the metal and does not hydrogen-bond;
    original PISA does not list such pairs.
    """
    if i not in metal_cache:
        found = False
        for j in kd_tree.query_ball_point(coords[i], _METAL_COORD_DIST):
            if j != i and atoms[j].element.strip().upper() in _METALS:
                found = True
                break
        metal_cache[i] = found
    return metal_cache[i]


# Number of implicit hydrogens a donor can donate / lone pairs an acceptor
# can accept. Limiting these (greedy, closest bond first) reproduces PISA's
# counts where a pure geometric test over-counts (e.g. one amide donating to
# three nearby carbonyls, DNA duplex cross-strand extras).
_DONOR_CAP: Dict[Tuple[str, str], int] = {
    ("ARG", "NE"): 1, ("ARG", "NH1"): 2, ("ARG", "NH2"): 2,
    ("LYS", "NZ"): 3,
    ("HIS", "ND1"): 1, ("HIS", "NE2"): 1,
    ("ASN", "ND2"): 2, ("GLN", "NE2"): 2,
    ("SER", "OG"): 1, ("THR", "OG1"): 1, ("TYR", "OH"): 1,
    ("TRP", "NE1"): 1, ("CYS", "SG"): 1,
}
for _base, _atom, _cap in (("A", "N6", 2), ("G", "N1", 1), ("G", "N2", 2),
                           ("C", "N4", 2), ("T", "N3", 1), ("U", "N3", 1)):
    for _p in ("", "D", "R"):
        _DONOR_CAP[(_p + _base, _atom)] = _cap


def _donor_capacity(atom) -> int:
    cap = _DONOR_CAP.get((atom.res_name.strip().upper(),
                          atom.atom_name.strip().upper()))
    if cap is None:
        cap = 1  # backbone amide N, hydroxyls, generic ligand donors
    # +1: PISA lists bifurcated H-bonds (one donor H shared by two
    # acceptors). Validated on the EBI benchmark: this setting minimises
    # the count error (mean |diff| 0.56/interface, 93% within +-1).
    return cap + 1


def _acceptor_capacity(atom) -> int:
    el = atom.element.strip().upper()
    if el == "O":
        return 2
    if el == "S":
        return 2
    return 1  # N acceptors


def _explicit_hydrogens(i, atoms, coords, kd_tree, h_cache):
    """Explicit H atoms covalently attached to atom ``i`` (same chain)."""
    if i not in h_cache:
        hs = []
        for j in kd_tree.query_ball_point(coords[i], 1.25):
            if j != i and atoms[j].element.strip().upper() == "H" \
                    and atoms[j].auth_asym_id == atoms[i].auth_asym_id:
                hs.append(j)
        h_cache[i] = hs
    return h_cache[i]


def _hbond_assignments(i, j, distance, atoms, coords, kd_tree,
                       cache, metal_cache, h_cache):
    """(donor, acceptor) assignments of pair (i, j) passing PISA geometry."""
    if distance > HBOND_MAX_DIST:
        return []
    a1, a2 = atoms[i], atoms[j]
    r1 = hb_roles(a1.res_name, a1.atom_name, a1.element)
    r2 = hb_roles(a2.res_name, a2.atom_name, a2.element)
    if not (("donor" in r1 and "acceptor" in r2)
            or ("donor" in r2 and "acceptor" in r1)):
        return []
    if (_is_metal_coordinated(i, atoms, coords, kd_tree, metal_cache)
            or _is_metal_coordinated(j, atoms, coords, kd_tree, metal_cache)):
        return []
    if _min_antecedent_angle(i, j, atoms, coords, kd_tree, cache) < HBOND_MIN_ANGLE:
        return []
    if _min_antecedent_angle(j, i, atoms, coords, kd_tree, cache) < HBOND_MIN_ANGLE:
        return []

    out = []
    for d, a, rd, ra in ((i, j, r1, r2), (j, i, r2, r1)):
        if "donor" not in rd or "acceptor" not in ra:
            continue
        # If the model carries explicit hydrogens on the donor, hold the
        # bond to real H geometry (HBPLUS-style): H...A <= 2.5 A and
        # D-H...A angle >= 90 deg. PISA does this on hydrogenated models.
        hs = _explicit_hydrogens(d, atoms, coords, kd_tree, h_cache)
        if hs:
            ok = False
            for h in hs:
                v_hd = coords[d] - coords[h]
                v_ha = coords[a] - coords[h]
                dist_ha = float(np.linalg.norm(v_ha))
                if dist_ha > 2.5:
                    continue
                cosang = float(np.dot(v_hd, v_ha) /
                               ((np.linalg.norm(v_hd) + 1e-12) * (dist_ha + 1e-12)))
                ang = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
                if ang >= HBOND_MIN_ANGLE:
                    ok = True
                    break
            if not ok:
                continue
        out.append((d, a))
    return out


def is_geometric_hbond(i, j, distance, atoms, coords, kd_tree, cache,
                       metal_cache=None) -> bool:
    """Full PISA-style H-bond test for the atom pair (i, j) (no capacity)."""
    if metal_cache is None:
        metal_cache = {}
    return bool(_hbond_assignments(i, j, distance, atoms, coords, kd_tree,
                                   cache, metal_cache, {}))


def detect_bond_flags(contacts, atoms, coords, kd_tree) -> List[Set[str]]:
    """Per-contact independent bond predicates.

    Returns one ``set`` per contact drawn from {"hbond", "salt_bridge",
    "disulfide"}; an empty set means plain van-der-Waals contact. A single
    pair can carry several flags (PISA lists charged H-bonded pairs in both
    its h-bond and salt-bridge tables).
    """
    cache: Dict[int, list] = {}
    metal_cache: Dict[int, bool] = {}
    h_cache: Dict[int, list] = {}
    flags: List[Set[str]] = []
    candidates = []  # (distance, position, assignments)
    for pos, c in enumerate(contacts):
        f: Set[str] = set()
        a1, a2 = atoms[c.atom1_idx], atoms[c.atom2_idx]
        if is_disulfide(a1.res_name, a2.res_name, a1.element, a2.element, c.distance):
            f.add("disulfide")
        q1 = salt_charge(a1.res_name, a1.atom_name)
        q2 = salt_charge(a2.res_name, a2.atom_name)
        if (q1 is not None and q2 is not None and q1 * q2 < 0
                and c.distance <= SALT_BRIDGE_MAX_DIST):
            f.add("salt_bridge")
        assigns = _hbond_assignments(c.atom1_idx, c.atom2_idx, c.distance,
                                     atoms, coords, kd_tree,
                                     cache, metal_cache, h_cache)
        if assigns:
            candidates.append((c.distance, pos, assigns))
        flags.append(f)

    # Greedy H-bond assignment, closest pair first, respecting per-atom
    # donor (implicit H count) and acceptor (lone pair) capacities.
    donor_used: Dict[int, int] = {}
    acceptor_used: Dict[int, int] = {}
    for distance, pos, assigns in sorted(candidates, key=lambda t: t[0]):
        for d, a in assigns:
            if donor_used.get(d, 0) >= _donor_capacity(atoms[d]):
                continue
            if acceptor_used.get(a, 0) >= _acceptor_capacity(atoms[a]):
                continue
            donor_used[d] = donor_used.get(d, 0) + 1
            acceptor_used[a] = acceptor_used.get(a, 0) + 1
            flags[pos].add("hbond")
            break
    return flags
