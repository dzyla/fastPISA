"""Compare fastPISA output against cached original-PISA reference data.

Used by ``examples/compare_vs_pisa.py`` (human-readable report) and
``tests/test_vs_pdbe_pisa.py`` (offline accuracy regression).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from fastpisa.api import analyze_interface
from fastpisa.reference.ebi_pisa import (
    load_cached_reference, identity_interfaces, cached_pdb_path,
)

#: Entries shipped in tests/data/reference. 1ppf is cached too but its
#: glycan chains were renamed by the wwPDB carbohydrate remediation after
#: PISA's run, so its sugar interfaces cannot be matched by name.
BENCHMARK_ENTRIES = (
    "1ktz 1brs 1vfb 2ptc 1acb 3hhr 4ins 1a3n 1fin 1lmb 1dfj 1ay7 1gcq "
    "1tro 3cro 1rva 9ant 2sni 1cho 1stf 1cbw"
).split()


def _key(chain_ids) -> frozenset:
    return frozenset(c.replace(" ", "") for c in chain_ids)


def compare_entry(pdb_id: str, mode: str = "pisa") -> Optional[dict]:
    """Compare one cached entry. Returns None when reference/PDB not cached.

    Result dict: ``rows`` (one per matched identity interface),
    ``ref_only`` / ``fp_only`` (unmatched chain-pair labels).
    """
    ref_all = load_cached_reference(pdb_id)
    pdb = cached_pdb_path(pdb_id)
    if ref_all is None or pdb is None:
        return None
    ref = identity_interfaces(ref_all)
    result = analyze_interface(pdb, pdb_id=pdb_id, mode=mode)

    refk = {_key(m["chain_id"] for m in i["molecules"]): i for i in ref}
    fpk = {_key(m["chain_id"] for m in i.molecules): i
           for i in result["interfaces_obj"]}

    rows = []
    for k in sorted(set(refk) & set(fpk), key=sorted):
        ri, fi = refk[k], fpk[k]
        rows.append({
            "pdb_id": pdb_id,
            "pair": "+".join(sorted(k)),
            "area_ref": ri["int_area"], "area_fp": fi.interface_area,
            "dg_ref": ri["int_solv_en"], "dg_fp": fi.solvation_energy,
            "stab_ref": ri["stab_en"], "stab_fp": fi.stabilization_energy,
            "pv_ref": ri["pvalue"], "pv_fp": fi.p_value,
            "css_ref": ri["css"], "css_fp": fi.css,
            "nhb_ref": len(ri["h_bonds"]), "nhb_fp": fi.number_hydrogen_bonds,
            "nsb_ref": len(ri["salt_bridges"]), "nsb_fp": fi.number_salt_bridges,
            "nss_ref": len(ri["ss_bonds"]), "nss_fp": fi.number_disulfide_bonds,
        })
    return {
        "pdb_id": pdb_id,
        "rows": rows,
        "ref_only": ["+".join(sorted(k)) for k in refk if k not in fpk],
        "fp_only": ["+".join(sorted(k)) for k in fpk if k not in refk],
    }


def compare_entries(pdb_ids=BENCHMARK_ENTRIES, mode: str = "pisa") -> dict:
    """Compare many entries; returns {'entries': [...], 'rows': [...]}."""
    entries, rows = [], []
    for pid in pdb_ids:
        e = compare_entry(pid, mode=mode)
        if e is None:
            continue
        entries.append(e)
        rows.extend(e["rows"])
    return {"entries": entries, "rows": rows}


def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return _pearson(ra, rb)


def summarize(rows: List[dict]) -> Dict[str, float]:
    """Headline agreement statistics over matched interface rows."""
    if not rows:
        return {}
    g = lambda f: np.array([r[f] for r in rows], dtype=float)  # noqa: E731
    area_ref, area_fp = g("area_ref"), g("area_fp")
    big = area_ref > 300
    rel_area = np.abs(area_fp - area_ref) / np.maximum(area_ref, 1.0)
    stats = {
        "n_matched": len(rows),
        "area_median_rel_err": float(np.median(rel_area)),
        "area_median_rel_err_big": float(np.median(rel_area[big])) if big.any() else float("nan"),
        "dg_pearson": _pearson(g("dg_fp"), g("dg_ref")),
        "dg_median_abs_err": float(np.median(np.abs(g("dg_fp") - g("dg_ref")))),
        "stab_pearson": _pearson(g("stab_fp"), g("stab_ref")),
        "stab_median_abs_err": float(np.median(np.abs(g("stab_fp") - g("stab_ref")))),
        "pv_median_abs_err": float(np.median(np.abs(g("pv_fp") - g("pv_ref")))),
        "pv_spearman": _spearman(g("pv_fp"), g("pv_ref")),
        "css_spearman": _spearman(g("css_fp"), g("css_ref")),
        "hb_mean_abs_diff": float(np.mean(np.abs(g("nhb_fp") - g("nhb_ref")))),
        "hb_within_1": float(np.mean(np.abs(g("nhb_fp") - g("nhb_ref")) <= 1)),
        "sb_mean_abs_diff": float(np.mean(np.abs(g("nsb_fp") - g("nsb_ref")))),
        "ss_exact": float(np.mean(g("nss_fp") == g("nss_ref"))),
    }
    return stats
