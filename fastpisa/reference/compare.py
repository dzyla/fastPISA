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
#: PISA's run, so its sugar interfaces cannot be matched by name (the same
#: renaming affects a handful of glycan/ion pairs in 1nca/1gpw/1prc).
BENCHMARK_ENTRIES = (
    "1ktz 1brs 1vfb 2ptc 1acb 3hhr 4ins 1a3n 1fin 1lmb 1dfj 1ay7 1gcq "
    "1tro 3cro 1rva 9ant 2sni 1cho 1stf 1cbw "
    "1fdl 3hfm 1nca 1cgi 1eaw 1r0r 1oph 1jck 1gpw 1tsr 1aay 1urn 1prc "
    "1f34 1e6e"
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


def compare_assembly_entry(pdb_id: str, assembly: str = "1",
                           mode: str = "pisa",
                           ligand_mode: str = "merge") -> Optional[dict]:
    """Compare one entry via the modern PDBe PISA JSON API (assembly-based).

    Fetches (with caching) the PDBe PISA JSON for the biological assembly and
    the matching RCSB assembly mmCIF, runs fastPISA on those SAME coordinates
    and matches interfaces by chain pair. Works for recent entries the frozen
    classic-CGI database does not cover. Requires network on first use.

    jsPISA analyses assemblies with bound ligands merged into their parent
    chain, so ``ligand_mode="merge"`` is the matching default here.
    """
    from fastpisa.reference.ebi_pisa import (
        fetch_pisa_assembly_json, fetch_assembly_cif, normalize_json_interfaces)

    doc = fetch_pisa_assembly_json(pdb_id, assembly)
    cif = fetch_assembly_cif(pdb_id, assembly)
    ref = normalize_json_interfaces(doc)
    result = analyze_interface(cif, pdb_id=pdb_id, mode=mode,
                               ligand_mode=ligand_mode)

    refk = {_key(m["chain_id"] for m in i["molecules"]): i for i in ref}
    fpk = {_key(m["chain_id"] for m in i.molecules): i
           for i in result["interfaces_obj"]}

    nan = float("nan")
    rows = []
    for k in sorted(set(refk) & set(fpk), key=sorted):
        ri, fi = refk[k], fpk[k]
        rows.append({
            "pdb_id": pdb_id,
            "pair": "+".join(sorted(k)),
            "area_ref": ri["int_area"], "area_fp": fi.interface_area,
            "dg_ref": ri["int_solv_en"], "dg_fp": fi.solvation_energy,
            "stab_ref": ri["stab_en"], "stab_fp": fi.stabilization_energy,
            "pv_ref": ri["pvalue"] if ri["pvalue"] is not None else nan,
            "pv_fp": fi.p_value,
            "css_ref": ri["css"] if ri["css"] is not None else nan,
            "css_fp": fi.css,
            "nhb_ref": ri["n_h_bonds"], "nhb_fp": fi.number_hydrogen_bonds,
            "nsb_ref": ri["n_salt_bridges"], "nsb_fp": fi.number_salt_bridges,
            "nss_ref": ri["n_ss_bonds"], "nss_fp": fi.number_disulfide_bonds,
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

    def _finite_pair(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return a[m], b[m]

    area_ref, area_fp = g("area_ref"), g("area_fp")
    big = area_ref > 300
    rel_area = np.abs(area_fp - area_ref) / np.maximum(area_ref, 1.0)
    pv_fp, pv_ref = _finite_pair(g("pv_fp"), g("pv_ref"))
    css_fp, css_ref = _finite_pair(g("css_fp"), g("css_ref"))
    stats = {
        "n_matched": len(rows),
        "area_median_rel_err": float(np.median(rel_area)),
        "area_median_rel_err_big": float(np.median(rel_area[big])) if big.any() else float("nan"),
        "dg_pearson": _pearson(g("dg_fp"), g("dg_ref")),
        "dg_median_abs_err": float(np.median(np.abs(g("dg_fp") - g("dg_ref")))),
        "stab_pearson": _pearson(g("stab_fp"), g("stab_ref")),
        "stab_median_abs_err": float(np.median(np.abs(g("stab_fp") - g("stab_ref")))),
        "pv_median_abs_err": float(np.median(np.abs(pv_fp - pv_ref))) if len(pv_ref) else float("nan"),
        "pv_spearman": _spearman(pv_fp, pv_ref) if len(pv_ref) > 2 else float("nan"),
        "css_spearman": _spearman(css_fp, css_ref) if len(css_ref) > 2 else float("nan"),
        "hb_mean_abs_diff": float(np.mean(np.abs(g("nhb_fp") - g("nhb_ref")))),
        "hb_within_1": float(np.mean(np.abs(g("nhb_fp") - g("nhb_ref")) <= 1)),
        "sb_mean_abs_diff": float(np.mean(np.abs(g("nsb_fp") - g("nsb_ref")))),
        "ss_exact": float(np.mean(g("nss_fp") == g("nss_ref"))),
    }

    # Polymer-polymer subset (no ligand molecule on either side): the
    # cryo-EM / predicted-model use case, and the best-calibrated regime.
    poly = [r for r in rows if "[" not in r["pair"]]
    if poly:
        gp = lambda f: np.array([r[f] for r in poly], dtype=float)  # noqa: E731
        rel_p = np.abs(gp("area_fp") - gp("area_ref")) / np.maximum(gp("area_ref"), 1.0)
        ppv_fp, ppv_ref = _finite_pair(gp("pv_fp"), gp("pv_ref"))
        pcss_fp, pcss_ref = _finite_pair(gp("css_fp"), gp("css_ref"))
        stats.update({
            "poly_n": len(poly),
            "poly_area_median_rel_err": float(np.median(rel_p)),
            "poly_dg_pearson": _pearson(gp("dg_fp"), gp("dg_ref")),
            "poly_dg_median_abs_err": float(np.median(np.abs(gp("dg_fp") - gp("dg_ref")))),
            "poly_stab_pearson": _pearson(gp("stab_fp"), gp("stab_ref")),
            "poly_pv_median_abs_err": float(np.median(np.abs(ppv_fp - ppv_ref))) if len(ppv_ref) else float("nan"),
            "poly_pv_spearman": _spearman(ppv_fp, ppv_ref) if len(ppv_ref) > 2 else float("nan"),
            "poly_css_spearman": _spearman(pcss_fp, pcss_ref) if len(pcss_ref) > 2 else float("nan"),
        })
    return stats


# ---------------------------------------------------------------------------
# Crystal mode: match PISA's FULL interface list, symmetry mates included
# ---------------------------------------------------------------------------
#: Tolerances for deciding that two crystal placements are the same contact.
#: PISA prints its per-molecule matrix to a few decimals, so an exact or
#: rounded key is wrong: a 6-fold screw translates by c/6 = 42.55 A, which
#: PISA reports as 42.6 against our 42.55. Both bounds are far below the
#: spacing between genuinely different placements (a cell edge, or a change
#: of operator).
TRANSFORM_ROTATION_TOL = 0.02
TRANSFORM_TRANSLATION_TOL = 0.8     # Angstrom

POLYMER_CLASSES = ("Protein", "NucleicAcid", "DNA", "RNA")


def _relative_transform(rot_a, tran_a, rot_b, tran_b):
    """Placement of molecule B as seen from molecule A's own frame."""
    rot_a = np.asarray(rot_a, dtype=float)
    rot_b = np.asarray(rot_b, dtype=float)
    tran_a = np.asarray(tran_a, dtype=float)
    tran_b = np.asarray(tran_b, dtype=float)
    # Orthogonal space: a crystallographic rotation IS orthogonal here, so
    # the transpose is the inverse (unlike in fractional coordinates).
    return rot_a.T @ rot_b, rot_a.T @ (tran_b - tran_a)


def compare_crystal_entry(pdb_id: str, polymer_only: bool = True,
                          allow_fetch: bool = True) -> Optional[dict]:
    """Compare ``symmetry="crystal"`` output against PISA's full interface list.

    Matches each PISA interface to one fastPISA interface on the chain pair
    and the relative crystal transform (either way round, since the two ends
    of one contact give inverse transforms). Returns ``None`` when the
    reference data is unavailable.

    ``polymer_only`` restricts both sides to polymer-polymer interfaces. That
    is the meaningful comparison for an older entry: PDB remediation has
    renamed and renumbered hetero groups since PISA's database was frozen
    (1ppf's glycans moved from chain E:401-417 to chains A/B:1-8, 1prc's HEM
    became HEC), so a ligand interface cannot be matched by name even when
    the geometry is identical.
    """
    from fastpisa.core import run_core
    from fastpisa.reference.ebi_pisa import (
        cached_pdb_path, fetch_pdb_file, fetch_pisa_xml, load_cached_reference,
        parse_pisa_xml,
    )

    reference = load_cached_reference(pdb_id)
    if reference is None:
        if not allow_fetch:
            return None
        try:
            reference = parse_pisa_xml(fetch_pisa_xml(pdb_id))
        except Exception:
            return None
    path = cached_pdb_path(pdb_id)
    if path is None:
        if not allow_fetch:
            return None
        try:
            path = fetch_pdb_file(pdb_id)
        except Exception:
            return None

    wanted = []
    for iface in reference:
        mols = iface.get("molecules") or []
        if len(mols) != 2:
            continue
        if polymer_only and not all(m.get("class") in POLYMER_CLASSES
                                    for m in mols):
            continue
        if any("rotation" not in m for m in mols):
            continue
        rot, tran = _relative_transform(
            mols[0]["rotation"], mols[0]["translation"],
            mols[1]["rotation"], mols[1]["translation"])
        wanted.append((frozenset(m["chain_id"] for m in mols), rot, tran, iface))

    state = run_core(path, mode="pisa", symmetry="crystal")
    produced = []
    for iface in state.interfaces:
        mols = iface.molecules
        if polymer_only and not all(m.get("molecule_class") in POLYMER_CLASSES
                                    for m in mols):
            continue
        rot, tran = _relative_transform(
            mols[0]["rotation"], mols[0]["translation"],
            mols[1]["rotation"], mols[1]["translation"])
        produced.append((frozenset(m["asu_molecule_id"] for m in mols),
                         rot, tran, iface))

    used, rows, missing = set(), [], []
    for chains, rot, tran, ref_iface in wanted:
        best = None
        for index, (chains2, rot2, tran2, ours) in enumerate(produced):
            if index in used or chains2 != chains:
                continue
            for cand_rot, cand_tran in ((rot2, tran2),
                                        (rot2.T, -rot2.T @ tran2)):
                if (np.abs(cand_rot - rot).max() < TRANSFORM_ROTATION_TOL
                        and np.abs(cand_tran - tran).max()
                        < TRANSFORM_TRANSLATION_TOL):
                    distance = float(np.abs(cand_tran - tran).max())
                    if best is None or distance < best[0]:
                        best = (distance, index, ours)
                    break
        if best is None:
            missing.append({
                "pair": "+".join(sorted(chains)),
                "area_ref": ref_iface.get("int_area"),
                "symop": [m.get("symop") for m in ref_iface["molecules"]],
            })
            continue
        used.add(best[1])
        ours = best[2]
        rows.append({
            "pdb_id": pdb_id,
            "pair": "+".join(sorted(chains)),
            "symop": [m.get("symop") for m in ref_iface["molecules"]],
            "area_ref": ref_iface.get("int_area"),
            "area_fp": ours.interface_area,
            "dg_ref": ref_iface.get("int_solv_en"),
            "dg_fp": ours.solvation_energy,
        })

    return {
        "pdb_id": pdb_id,
        "n_reference": len(wanted),
        "n_matched": len(rows),
        "n_reported": len(produced),
        "rows": rows,
        "missing": missing,
    }


def summarize_crystal(results: List[dict]) -> Dict[str, float]:
    """Match rate and agreement over several :func:`compare_crystal_entry`."""
    rows = [r for res in results for r in res["rows"]]
    n_ref = sum(res["n_reference"] for res in results)
    stats = {
        "n_entries": len(results),
        "n_reference": n_ref,
        "matched": sum(res["n_matched"] for res in results),
        "reported": sum(res["n_reported"] for res in results),
    }
    stats["match_rate"] = stats["matched"] / n_ref if n_ref else float("nan")
    if not rows:
        return stats
    area_ref = np.array([r["area_ref"] for r in rows], dtype=float)
    area_fp = np.array([r["area_fp"] for r in rows], dtype=float)
    big = area_ref > 100
    rel = np.abs(area_fp - area_ref) / np.maximum(area_ref, 1.0)
    stats["area_median_rel_err"] = float(np.median(rel[big])) if big.any() \
        else float(np.median(rel))
    dg_ref = np.array([r["dg_ref"] for r in rows
                       if r["dg_ref"] is not None], dtype=float)
    dg_fp = np.array([r["dg_fp"] for r in rows
                      if r["dg_ref"] is not None], dtype=float)
    stats["dg_pearson"] = _pearson(dg_fp, dg_ref)
    stats["dg_median_abs_err"] = float(np.median(np.abs(dg_fp - dg_ref)))
    return stats
