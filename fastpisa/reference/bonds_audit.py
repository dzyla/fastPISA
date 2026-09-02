"""Atom-level audit of fastPISA's H-bonds / salt bridges against PISA's.

The EBI PISA XML lists every hydrogen bond and salt bridge as an atom pair.
Comparing *counts* per interface (what the accuracy benchmark does) hides
compensating errors -- one missed bond plus one spurious bond scores as a
perfect count. This module matches pair by pair and, for every pair PISA
lists that we do not, records *why* our detector rejected it (too far, no
donor/acceptor role, antecedent angle, metal-coordinated, or lost to the
capacity limit), so criteria can be adjusted from evidence rather than
guessed at.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fastpisa.reference.ebi_pisa import (
    cached_pdb_path, identity_interfaces, load_cached_reference,
)


def _key(chain_ids) -> frozenset:
    return frozenset(c.replace(" ", "") for c in chain_ids)


def _atom_key(chain: str, seqnum, icode: str, atname: str) -> tuple:
    return (chain.strip(), int(seqnum), (icode or "").strip(),
            atname.strip().upper())


def _ref_pairs(bonds: Sequence[dict]) -> Dict[frozenset, float]:
    out = {}
    for b in bonds:
        try:
            k1 = _atom_key(b["chain1"], b["seqnum1"], b.get("inscode1", ""), b["atname1"])
            k2 = _atom_key(b["chain2"], b["seqnum2"], b.get("inscode2", ""), b["atname2"])
        except ValueError:
            continue
        out[frozenset((k1, k2))] = b["dist"]
    return out


def _reject_reason(i: int, j: int, dist: float, atoms, coords, kd_tree) -> str:
    """Why the H-bond detector does not accept pair (i, j) on geometry."""
    from fastpisa.interface.bonds import (
        HBOND_MAX_DIST, HBOND_MIN_ANGLE, _hbond_assignments,
        _is_metal_coordinated, _min_antecedent_angle, hb_roles,
    )
    if dist > HBOND_MAX_DIST:
        return f"distance>{HBOND_MAX_DIST}"
    a1, a2 = atoms[i], atoms[j]
    r1 = hb_roles(a1.res_name, a1.atom_name, a1.element)
    r2 = hb_roles(a2.res_name, a2.atom_name, a2.element)
    if not (("donor" in r1 and "acceptor" in r2)
            or ("donor" in r2 and "acceptor" in r1)):
        return "no donor/acceptor role"
    mc: Dict[int, bool] = {}
    if (_is_metal_coordinated(i, atoms, coords, kd_tree, mc)
            or _is_metal_coordinated(j, atoms, coords, kd_tree, mc)):
        return "metal-coordinated"
    cache: Dict[int, list] = {}
    if (_min_antecedent_angle(i, j, atoms, coords, kd_tree, cache) < HBOND_MIN_ANGLE
            or _min_antecedent_angle(j, i, atoms, coords, kd_tree, cache) < HBOND_MIN_ANGLE):
        return "antecedent angle"
    if not _hbond_assignments(i, j, dist, atoms, coords, kd_tree, cache, mc, {}):
        return "explicit-H geometry"
    return "capacity"


def audit_entry(pdb_id: str) -> List[dict]:
    """Per matched identity interface: matched / missed / extra bond pairs."""
    from scipy.spatial import cKDTree
    from fastpisa.core import run_core

    ref_all = load_cached_reference(pdb_id)
    pdb = cached_pdb_path(pdb_id)
    if ref_all is None or pdb is None:
        return []
    refk = {_key(m["chain_id"] for m in i["molecules"]): i
            for i in identity_interfaces(ref_all)}
    state = run_core(pdb, mode="pisa")
    atoms = state.atoms
    coords = np.array([[a.x, a.y, a.z] for a in atoms])
    kd = cKDTree(coords)
    by_key: Dict[tuple, int] = {}
    h_idx: List[int] = []
    for idx, a in enumerate(atoms):
        if a.element.strip().upper() in ("H", "D"):
            h_idx.append(idx)
            continue
        by_key.setdefault(
            (a.auth_asym_id.strip(), a.res_seq, (a.icode or "").strip(),
             a.atom_name.strip().upper()), idx)
    # On hydrogenated models PISA names the donor by its H atom ("H",
    # "HH12", ...). Our contacts are heavy-atom pairs, so map every explicit
    # H onto the heavy atom it is bonded to (nearest heavy atom in the same
    # residue within 1.25 A) and compare at heavy-atom level.
    for hi in h_idx:
        a = atoms[hi]
        best, best_d = None, 1.25
        for j in kd.query_ball_point(coords[hi], 1.25):
            b = atoms[j]
            if (j != hi and b.element.strip().upper() not in ("H", "D")
                    and b.auth_asym_id == a.auth_asym_id and b.res_seq == a.res_seq):
                d = float(np.linalg.norm(coords[hi] - coords[j]))
                if d < best_d:
                    best, best_d = j, d
        if best is not None:
            by_key.setdefault(
                (a.auth_asym_id.strip(), a.res_seq, (a.icode or "").strip(),
                 a.atom_name.strip().upper()), best)

    out = []
    for iface in state.interfaces:
        k = _key(m["chain_id"] for m in iface.molecules)
        ri = refk.get(k)
        if ri is None:
            continue
        rec = {"pdb_id": pdb_id, "pair": "+".join(sorted(k)),
               "is_polymer_pair": "[" not in "+".join(sorted(k))}
        for kind, flag in (("hb", "hbond"), ("sb", "salt_bridge")):
            ours: Dict[frozenset, float] = {}
            for c in iface.contacts:
                if c.bond_type == flag or (
                        flag == "hbond" and c.bond_type == "salt_bridge"
                        and _pair_is_hbond(c, atoms, coords, kd)):
                    a1, a2 = atoms[c.atom1_idx], atoms[c.atom2_idx]
                    k1 = (a1.auth_asym_id.strip(), a1.res_seq, (a1.icode or "").strip(), a1.atom_name.strip().upper())
                    k2 = (a2.auth_asym_id.strip(), a2.res_seq, (a2.icode or "").strip(), a2.atom_name.strip().upper())
                    ours[frozenset((k1, k2))] = c.distance
            ref_raw = _ref_pairs(ri["h_bonds"] if kind == "hb" else ri["salt_bridges"])
            ref = {}
            for pk, d in ref_raw.items():
                ks = []
                for kk in pk:
                    i = by_key.get(kk)
                    if i is None:
                        ks.append(kk)
                    else:
                        b = atoms[i]
                        ks.append((b.auth_asym_id.strip(), b.res_seq,
                                   (b.icode or "").strip(), b.atom_name.strip().upper()))
                ref[frozenset(ks)] = d
            matched = set(ours) & set(ref)
            missed, extra = [], []
            for pk in set(ref) - matched:
                k1, k2 = tuple(pk) if len(pk) == 2 else (next(iter(pk)),) * 2
                i, j = by_key.get(k1), by_key.get(k2)
                if i is None or j is None:
                    reason, d = "atom not in model", float("nan")
                else:
                    d = float(np.linalg.norm(coords[i] - coords[j]))
                    reason = (_reject_reason(i, j, d, atoms, coords, kd)
                              if kind == "hb" else "salt criteria")
                missed.append({"a1": k1, "a2": k2, "dist_ref": ref[pk],
                               "dist_fp": d, "reason": reason})
            for pk in set(ours) - matched:
                k1, k2 = tuple(pk) if len(pk) == 2 else (next(iter(pk)),) * 2
                extra.append({"a1": k1, "a2": k2, "dist_fp": ours[pk]})
            rec[kind] = {"n_ref": len(ref), "n_fp": len(ours),
                         "n_matched": len(matched), "missed": missed, "extra": extra}
        out.append(rec)
    return out


def _pair_is_hbond(c, atoms, coords, kd) -> bool:
    """A contact labelled salt_bridge may ALSO be an H-bond (independent
    predicates); recover that flag so the H-bond audit sees it."""
    from fastpisa.interface.bonds import _hbond_assignments
    return bool(_hbond_assignments(c.atom1_idx, c.atom2_idx, c.distance,
                                   atoms, coords, kd, {}, {}, {}))


def summarize(records: Sequence[dict], kind: str = "hb",
              polymer_only: bool = True) -> dict:
    recs = [r for r in records if r["is_polymer_pair"]] if polymer_only else list(records)
    n_ref = sum(r[kind]["n_ref"] for r in recs)
    n_fp = sum(r[kind]["n_fp"] for r in recs)
    n_m = sum(r[kind]["n_matched"] for r in recs)
    reasons = Counter(m["reason"] for r in recs for m in r[kind]["missed"])
    miss_d = [m["dist_fp"] for r in recs for m in r[kind]["missed"]
              if np.isfinite(m["dist_fp"])]
    extra_d = [e["dist_fp"] for r in recs for e in r[kind]["extra"]]

    def _types(items, field_pair=("a1", "a2")):
        cnt = Counter()
        for it in items:
            cnt[tuple(sorted((it[field_pair[0]][3], it[field_pair[1]][3])))] += 1
        return cnt.most_common(12)

    return {
        "n_interfaces": len(recs),
        "n_ref": n_ref, "n_fp": n_fp, "n_matched": n_m,
        "precision": n_m / n_fp if n_fp else float("nan"),
        "recall": n_m / n_ref if n_ref else float("nan"),
        "missed_reasons": dict(reasons),
        "missed_dist_quartiles": [float(np.percentile(miss_d, q)) for q in (25, 50, 75)] if miss_d else [],
        "extra_dist_quartiles": [float(np.percentile(extra_d, q)) for q in (25, 50, 75)] if extra_d else [],
        "missed_atom_types": _types([m for r in recs for m in r[kind]["missed"]]),
        "extra_atom_types": _types([e for r in recs for e in r[kind]["extra"]]),
    }
