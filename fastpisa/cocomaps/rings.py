"""Aromatic ring geometry for pi-interaction classification.

COCOMAPS 2.0 validates pi interactions against ring centroids and normals
(computed after adding hydrogens); proximity-only rules over-count them
badly (e.g. every atom of a Phe within 5 A of any carbon became "ch_pi").
This module computes ring centroids/normals from heavy atoms and provides
the geometric verdicts used by the contact-map classifier.

Heavy-atom criteria (calibrated against COCOMAPS 2.0 standalone output):

  pi_pi      ring-centroid ... ring-centroid distance <= 5.5 A
  cation_pi  charged N (Arg NE/NH*, Lys NZ, His) ... centroid <= 5.0 A and
             the cation sits above the ring plane (angle from the normal
             <= 60 deg)
  ch_pi      carbon ... centroid <= 4.6 A, above the plane
             (angle from the normal <= 55 deg)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

PI_PI_CENTROID_DIST = 5.5
CATION_PI_DIST = 5.0
CATION_PI_MAX_ANGLE = 60.0
CH_PI_DIST = 4.6
CH_PI_MAX_ANGLE = 55.0

# Ring atom groups per residue (aromatic side chains; purines have two rings)
RING_GROUPS: Dict[str, List[Tuple[str, ...]]] = {
    "PHE": [("CG", "CD1", "CD2", "CE1", "CE2", "CZ")],
    "TYR": [("CG", "CD1", "CD2", "CE1", "CE2", "CZ")],
    "HIS": [("CG", "ND1", "CD2", "CE1", "NE2")],
    "TRP": [("CG", "CD1", "NE1", "CE2", "CD2"),
            ("CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3")],
}
_PYRIMIDINE = ("N1", "C2", "N3", "C4", "C5", "C6")
_IMIDAZOLE = ("C4", "C5", "N7", "C8", "N9")
for _base, _rings in (("A", [_PYRIMIDINE, _IMIDAZOLE]),
                      ("G", [_PYRIMIDINE, _IMIDAZOLE]),
                      ("C", [_PYRIMIDINE]),
                      ("T", [_PYRIMIDINE]),
                      ("U", [_PYRIMIDINE])):
    for _p in ("", "D", "R"):
        RING_GROUPS[_p + _base] = _rings


class Ring:
    __slots__ = ("res_key", "centroid", "normal", "atom_ids")

    def __init__(self, res_key, centroid, normal, atom_ids):
        self.res_key = res_key
        self.centroid = centroid
        self.normal = normal
        self.atom_ids = atom_ids


def build_rings(atoms, atom_ids) -> List[Ring]:
    """Rings (centroid + unit normal) among ``atom_ids``, grouped by residue."""
    by_res: Dict[tuple, Dict[str, int]] = {}
    for gi in atom_ids:
        a = atoms[gi]
        rn = a.res_name.strip().upper()
        if rn not in RING_GROUPS:
            continue
        key = (a.auth_asym_id, a.res_seq, (a.icode or "").strip(), rn)
        by_res.setdefault(key, {})[a.atom_name.strip().upper()] = gi

    rings: List[Ring] = []
    for key, name_map in by_res.items():
        for group in RING_GROUPS[key[3]]:
            ids = [name_map[n] for n in group if n in name_map]
            if len(ids) < len(group) - 1 or len(ids) < 4:
                continue  # too incomplete to define the plane
            coords = np.array([[atoms[i].x, atoms[i].y, atoms[i].z] for i in ids])
            centroid = coords.mean(axis=0)
            # plane normal = smallest singular vector of centered coords
            _, _, vt = np.linalg.svd(coords - centroid)
            rings.append(Ring(key[:3], centroid, vt[2], frozenset(ids)))
    return rings


def _angle_from_normal(ring: Ring, point: np.ndarray) -> float:
    v = point - ring.centroid
    nv = np.linalg.norm(v)
    if nv < 1e-9:
        return 0.0
    cosang = abs(float(np.dot(ring.normal, v) / nv))
    return float(np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0))))


class RingContext:
    """Pre-built ring geometry for one interface (both molecules)."""

    def __init__(self, atoms, mol1_atom_ids, mol2_atom_ids):
        self._atoms = atoms
        self.rings1 = build_rings(atoms, mol1_atom_ids)
        self.rings2 = build_rings(atoms, mol2_atom_ids)
        self._member1 = {i: r for r in self.rings1 for i in r.atom_ids}
        self._member2 = {i: r for r in self.rings2 for i in r.atom_ids}

    def _coord(self, gi) -> np.ndarray:
        a = self._atoms[gi]
        return np.array([a.x, a.y, a.z])

    def _rings_of(self, gi) -> List[Ring]:
        out = []
        if gi in self._member1:
            out.append(self._member1[gi])
        if gi in self._member2:
            out.append(self._member2[gi])
        return out

    def is_pi_pi(self, gi1: int, gi2: int) -> bool:
        for r1 in self._rings_of(gi1):
            for r2 in self._rings_of(gi2):
                if np.linalg.norm(r1.centroid - r2.centroid) <= PI_PI_CENTROID_DIST:
                    return True
        return False

    def _point_vs_ring(self, ring_atom: int, partner: int,
                       max_dist: float, max_angle: float) -> bool:
        p = self._coord(partner)
        for ring in self._rings_of(ring_atom):
            if (np.linalg.norm(p - ring.centroid) <= max_dist
                    and _angle_from_normal(ring, p) <= max_angle):
                return True
        return False

    def is_cation_pi(self, ring_atom: int, cation_atom: int) -> bool:
        return self._point_vs_ring(ring_atom, cation_atom,
                                   CATION_PI_DIST, CATION_PI_MAX_ANGLE)

    def is_ch_pi(self, ring_atom: int, carbon_atom: int) -> bool:
        return self._point_vs_ring(ring_atom, carbon_atom,
                                   CH_PI_DIST, CH_PI_MAX_ANGLE)
