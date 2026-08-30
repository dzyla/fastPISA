"""
Residue-residue contact map for COCOMAPS mode.

COCOMAPS builds an intermolecular contact map showing which residue of
molecule 1 contacts which residue of molecule 2 across an interface, with
each residue-pair contact annotated by its dominant interaction type and
the minimum inter-residue distance.

Uses the same interface atom detection as the PISA pipeline (5 A cutoff)
so that both modes identify identical interfaces.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from fastpisa.cocomaps.interactions import classify_atom_pair
from fastpisa.interface.contacts import AtomContact
from fastpisa.surface.shrake_rupley import get_vdw_radius


class ResidueContact:
    """A contact between two residues across an interface."""

    __slots__ = (
        "atom1_idx", "atom2_idx", "distance", "atom1_name", "atom2_name",
        "res1", "res2", "chain1", "chain2", "interaction_type",
    )

    def __init__(self, atom1_idx, atom2_idx, distance, atom1_name, atom2_name,
                 res1, res2, chain1, chain2, interaction_type):
        self.atom1_idx = atom1_idx
        self.atom2_idx = atom2_idx
        self.distance = distance
        self.atom1_name = atom1_name
        self.atom2_name = atom2_name
        self.res1 = res1
        self.res2 = res2
        self.chain1 = chain1
        self.chain2 = chain2
        self.interaction_type = interaction_type


def build_residue_contact_map(
    atoms,
    mol1_mask,
    mol2_mask,
    mol1_atom_indices,
    mol2_atom_indices,
    interface_cutoff: float = 5.0,
    hbond_pairs: Optional[set] = None,
) -> List[ResidueContact]:
    """Build a residue-residue contact map between two molecules.

    Returns a list of ResidueContact objects (one per contacting atom pair),
    sorted by distance. Each is classified by interaction type.

    ``hbond_pairs`` (optional): set of ``(min_idx, max_idx)`` atom-index
    pairs pre-validated by the geometric H-bond detector
    (:mod:`fastpisa.interface.bonds`). When given, the hydrogen_bond class
    uses it instead of the table-only rule, so the contact map agrees with
    the PISA-calibrated bond counts.
    """
    contacts = []

    coords1 = np.array([[atoms[i].x, atoms[i].y, atoms[i].z] for i in mol1_atom_indices])
    coords2 = np.array([[atoms[i].x, atoms[i].y, atoms[i].z] for i in mol2_atom_indices])

    if len(coords1) == 0 or len(coords2) == 0:
        return contacts

    from fastpisa.cocomaps.rings import RingContext
    ring_ctx = RingContext(atoms, mol1_atom_indices, mol2_atom_indices)

    tree = cKDTree(coords2)
    dist_pairs = tree.query_ball_point(coords1, interface_cutoff)

    for i1, neighbors in enumerate(dist_pairs):
        g1 = mol1_atom_indices[i1]
        a1 = atoms[g1]
        for j2 in neighbors:
            g2 = mol2_atom_indices[j2]
            a2 = atoms[g2]
            d = float(np.linalg.norm(coords1[i1] - coords2[j2]))
            if d >= interface_cutoff:
                continue

            is_hb = None
            if hbond_pairs is not None:
                is_hb = (min(g1, g2), max(g1, g2)) in hbond_pairs
            pi_verdicts = (
                ring_ctx.is_pi_pi(g1, g2),
                ring_ctx.is_cation_pi(g1, g2) or ring_ctx.is_cation_pi(g2, g1),
                ring_ctx.is_ch_pi(g1, g2) or ring_ctx.is_ch_pi(g2, g1),
            )
            itype = classify_atom_pair(
                res1=a1.res_name,
                atom1=a1.atom_name,
                el1=a1.element,
                res2=a2.res_name,
                atom2=a2.atom_name,
                el2=a2.element,
                dist=d,
                vdw_radius1=get_vdw_radius(a1.element),
                vdw_radius2=get_vdw_radius(a2.element),
                is_hbond=is_hb,
                pi_verdicts=pi_verdicts,
            )
            contacts.append(ResidueContact(
                atom1_idx=g1,
                atom2_idx=g2,
                distance=d,
                atom1_name=a1.atom_name,
                atom2_name=a2.atom_name,
                res1=a1.res_name,
                res2=a2.res_name,
                chain1=a1.auth_asym_id,
                chain2=a2.auth_asym_id,
                interaction_type=itype,
            ))

    contacts.sort(key=lambda c: c.distance)
    return contacts


def aggregate_residue_pairs(
    contacts: List[ResidueContact], atoms
) -> List[dict]:
    """Aggregate atom-level contacts into residue-pair entries.

    Each residue pair is represented once with:
      - residue_1 / residue_2 identifiers
      - minimum distance
      - dominant interaction type
      - number of atom-atom contacts
      - per-type interaction counts

    Returns a list of dicts, one per residue pair.
    """
    pairs: Dict[Tuple, dict] = {}
    # Map residue key -> residue type (from first atom seen for that residue)
    res_type: Dict[Tuple, str] = {}

    for c in contacts:
        # residue key: (chain, seq, icode)
        a1 = atoms[c.atom1_idx]
        a2 = atoms[c.atom2_idx]
        rkey1 = (a1.auth_asym_id, a1.res_seq, (a1.icode or ""))
        rkey2 = (a2.auth_asym_id, a2.res_seq, (a2.icode or ""))
        res_type.setdefault(rkey1, a1.res_name)
        res_type.setdefault(rkey2, a2.res_name)
        # canonical pair order (avoid duplicates)
        pair = (rkey1, rkey2) if rkey1 <= rkey2 else (rkey2, rkey1)

        if pair not in pairs:
            pairs[pair] = {
                "residue_1_chain": pair[0][0],
                "residue_1_seq": pair[0][1],
                "residue_1_icode": pair[0][2] or None,
                "residue_1_type": res_type.get(pair[0], ""),
                "residue_2_chain": pair[1][0],
                "residue_2_seq": pair[1][1],
                "residue_2_icode": pair[1][2] or None,
                "residue_2_type": res_type.get(pair[1], ""),
                "min_distance": c.distance,
                "num_contacts": 0,
                "interaction_counts": defaultdict(int),
                "dominant_interaction": None,
            }
        entry = pairs[pair]
        entry["min_distance"] = min(entry["min_distance"], c.distance)
        entry["num_contacts"] += 1
        entry["interaction_counts"][c.interaction_type] += 1

    # Determine dominant type; keep the full per-class breakdown per pair
    # (COCOMAPS 2.0 lists every class a residue pair participates in).
    # "proximal" is only dominant when NO other class is present, and
    # specific interactions dominate generic vdW contacts.
    _GENERIC = ("proximal", "clash", "polar_vdw", "apolar_vdw", "weak_hbond")
    result = []
    for key, entry in sorted(pairs.items()):
        counts = dict(entry.pop("interaction_counts"))
        specific = {t: n for t, n in counts.items() if t not in _GENERIC}
        if specific:
            dominant = max(specific.items(), key=lambda kv: kv[1])[0]
        elif counts:
            dominant = next((t for t in ("weak_hbond", "apolar_vdw", "polar_vdw",
                                         "clash", "proximal") if t in counts),
                            "proximal")
        else:
            dominant = "proximal"
        entry["dominant_interaction"] = dominant
        entry["interaction_counts"] = counts
        result.append(entry)

    return result


def build_contact_matrix(atom_contacts: List[AtomContact], atoms) -> Tuple[np.ndarray, list, list]:
    """Build a 2D residue-residue contact matrix (contact map).

    Rows/columns are residues of reactant vs product chains. Value is
    1 if any atom-atom contact exists between the residue pair in the
    given contact list, else 0.

    Returns a tuple (matrix, row_labels, col_labels) where labels are
    (chain, seq, icode) tuples.
    """
    if not atom_contacts:
        return np.zeros((0, 0), dtype=int), [], []

    def rkey(idx):
        a = atoms[idx]
        return (a.auth_asym_id, a.res_seq, a.icode or "")

    rows = set()
    cols = set()
    for c in atom_contacts:
        rows.add(rkey(c.atom1_idx))
        cols.add(rkey(c.atom2_idx))

    rows = sorted(rows, key=lambda r: (r[0], r[1]))
    cols = sorted(cols, key=lambda r: (r[0], r[1]))

    n1, n2 = len(rows), len(cols)
    mat = np.zeros((n1, n2), dtype=int)
    row_idx = {r: i for i, r in enumerate(rows)}
    col_idx = {r: i for i, r in enumerate(cols)}

    for c in atom_contacts:
        i = row_idx[rkey(c.atom1_idx)]
        j = col_idx[rkey(c.atom2_idx)]
        mat[i, j] = 1
    return mat, rows, cols


def _reskey(atoms, idx):
    a = atoms[idx]
    return (a.auth_asym_id, a.res_seq, a.icode or "")