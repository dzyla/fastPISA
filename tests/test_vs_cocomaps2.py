"""Contact-map regression vs the actual COCOMAPS 2.0 standalone tool.

The reference CSVs in tests/data/reference/cocomaps2/ are the
``*_final_file.csv`` residue-pair tables produced by running the COCOMAPS 2.0
standalone code (Zenodo 10.5281/zenodo.17390665, with REDUCE for hydrogens;
HBPLUS/NACCESS unavailable, so its H-bond and BSA outputs are absent) on the
same PDB files fastPISA analyses.

Pinned here (measured 2026-08-30):
  * the residue-residue CONTACT MAP is IDENTICAL to COCOMAPS 2.0 on all
    three complexes (protein-protein 30/30, antibody-antigen 28/28,
    protein-DNA 57/57 residue pairs; interface residue sets identical);
  * COCOMAPS-convention salt bridges (Lys/Arg vs carboxylate or DNA
    phosphate, 4.5 A) match per residue pair.
"""
import csv
import os

import pytest

from fastpisa.api import analyze_interface
from fastpisa.reference.ebi_pisa import REFERENCE_DIR, cached_pdb_path

COCOMAPS_DIR = os.path.join(REFERENCE_DIR, "cocomaps2")

CASES = [
    # (pdb_id, structure path resolver, chain1, chain2, reference csv)
    ("1ktz", lambda: os.path.join(os.path.dirname(__file__), "data", "1ktz.pdb"),
     "A", "B", "1ktz_A_B.csv"),
    ("1vfb", lambda: cached_pdb_path("1vfb"), "B", "C", "1vfb_B_C.csv"),
    ("1aay", lambda: cached_pdb_path("1aay"), "A", "B", "1aay_A_B.csv"),
]


def _load_reference(name):
    path = os.path.join(COCOMAPS_DIR, name)
    if not os.path.exists(path):
        return None
    ref = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            key = tuple(sorted([(row["Chain 1"], int(row["Res. Number 1"])),
                                (row["Chain 2"], int(row["Res. Number 2"]))]))
            ref[key] = row["Type of Interactions"]
    return ref


@pytest.mark.parametrize("pdb_id,path_fn,c1,c2,ref_csv", CASES,
                         ids=[c[0] for c in CASES])
def test_contact_map_matches_cocomaps2(pdb_id, path_fn, c1, c2, ref_csv):
    ref = _load_reference(ref_csv)
    path = path_fn()
    if ref is None or path is None:
        pytest.skip("COCOMAPS 2.0 reference fixture or PDB not cached")

    result = analyze_interface(path, pdb_id=pdb_id, mode="cocomaps")
    iface = next(i for i in result["interfaces_obj"]
                 if {i.molecules[0]["chain_id"], i.molecules[1]["chain_id"]}
                 == {c1, c2})
    ours = {}
    for e in iface.cocomaps["contact_map"]:
        key = tuple(sorted([(e["residue_1_chain"], int(e["residue_1_seq"])),
                            (e["residue_2_chain"], int(e["residue_2_seq"]))]))
        ours[key] = e

    # The contact map (residue pairs at the shared 5 A atom cutoff) must be
    # IDENTICAL to COCOMAPS 2.0's.
    assert set(ours) == set(ref), (
        f"{pdb_id}: contact-map mismatch; only-ours "
        f"{sorted(set(ours) - set(ref))}, only-cocomaps "
        f"{sorted(set(ref) - set(ours))}")

    # Interface residues identical too.
    ours_res = {r for pair in ours for r in pair}
    ref_res = {r for pair in ref for r in pair}
    assert ours_res == ref_res

    # COCOMAPS-convention salt bridges agree per residue pair.
    ref_salt = {k for k, t in ref.items() if "Salt-bridge" in t}
    our_salt = {k for k, e in ours.items()
                if "salt_bridge" in e["interaction_counts"]}
    assert ref_salt == our_salt, (
        f"{pdb_id}: salt-bridge pairs differ: only-ours "
        f"{sorted(our_salt - ref_salt)}, only-cocomaps "
        f"{sorted(ref_salt - our_salt)}")

    # Pairs COCOMAPS calls purely "Proximal" (no interaction class) must be
    # dominated by proximal/no-contact classes for us too -- unless we found
    # a geometric H-bond there (the reference tool's HBPLUS step could not
    # run, so its H-bond class is systematically absent).
    ref_proximal = {k for k, t in ref.items()
                    if t.strip().strip(";").strip() == "Proximal contact"}
    disagree = {
        k for k in ref_proximal
        if ours[k]["dominant_interaction"] not in ("proximal", "hydrogen_bond")
    }
    assert len(disagree) <= max(2, len(ref_proximal) // 4), (
        f"{pdb_id}: too many proximal disagreements: {sorted(disagree)}")
