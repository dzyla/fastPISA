#!/usr/bin/env python
"""Validate assembly prediction against PISA's own predictions.

    python examples/validate_assemblies.py              # cached entries, offline
    python examples/validate_assemblies.py --record     # write the record
    python examples/validate_assemblies.py --entries 1acb 4ins

Two axes are reported. Agreement with PISA's top assembly says whether we
reproduce PISA. Agreement with the author-deposited assembly (PISA's R350
flag marks which of its assemblies the depositor asserted) is a claim about
biology, and is the number worth improving.

``--record`` rewrites tests/data/reference/assembly_validation.json with the
measured values, which tests/test_assembly_vs_pisa.py then asserts. Run it
when the algorithm changes AND you can explain the movement.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from fastpisa.reference.compare import (  # noqa: E402
    compare_assembly_entry, summarize_assembly_predictions,
)

REFERENCE_DIR = os.path.join(REPO, "tests", "data", "reference")
RECORD = os.path.join(REFERENCE_DIR, "assembly_validation.json")


def cached_entries():
    out = []
    for path in sorted(glob.glob(os.path.join(REFERENCE_DIR,
                                              "*.multimers.xml.gz"))):
        pdb_id = os.path.basename(path)[:4]
        if os.path.exists(os.path.join(REFERENCE_DIR, "pdb",
                                       f"{pdb_id}.pdb.gz")):
            out.append(pdb_id)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entries", nargs="*", default=None)
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()

    entries = ([e.lower() for e in args.entries] if args.entries
               else cached_entries())
    print(f"{len(entries)} entries\n")
    results = []
    started = time.time()
    for pdb_id in entries:
        begin = time.time()
        result = compare_assembly_entry(pdb_id, allow_fetch=False)
        if result is None:
            print(f"{pdb_id}: no reference, skipped", flush=True)
            continue
        results.append(result)
        reference = result["top_reference"]
        predicted = result["top_predicted"]
        marks = ("mm" if result["top_mmsize_match"] else "  ") + \
                ("+comp" if result["top_composition_match"] else "     ")
        print(f"{pdb_id}: PISA {reference['composition'] if reference else '-':<18}"
              f"-> ours {predicted['composition'] if predicted else '-':<18}"
              f"{marks}  recall {result['recall']:.2f}  "
              f"{time.time() - begin:5.1f}s", flush=True)

    stats = summarize_assembly_predictions(results)
    print(f"\n=== {stats['n_entries']} entries in {time.time() - started:.0f}s ===")
    print(f"top assembly, mmsize match      : "
          f"{100 * stats['top_mmsize_match_rate']:.1f}%  "
          f"({stats['n_comparable']} comparable)")
    print(f"top assembly, composition match : "
          f"{100 * stats['top_composition_match_rate']:.1f}%")
    print(f"author-deposited assembly match : "
          f"{100 * stats['author_assembly_match_rate']:.1f}%  "
          f"({stats['n_with_author_assembly']} entries)")
    print(f"recall of PISA's assembly sizes  : "
          f"{100 * stats['mean_recall']:.1f}%  "
          f"(precision {100 * stats['mean_precision']:.1f}%)")
    empty = stats["no_reference_assembly"]
    print(f"PISA predicts nothing ({empty['n_entries']} entries): we also "
          f"predict nothing stable "
          f"{100 * empty['we_predict_nothing_stable_rate']:.1f}%  "
          f"{' '.join(empty['entries'])}")

    if args.record:
        with open(RECORD, "w") as fh:
            json.dump({
                "purpose": ("Measured agreement of assembly prediction with "
                            "PISA's own predictions (multimers.pisa) and with "
                            "the author-deposited assembly (PISA's R350)."),
                "entries": [r["pdb_id"] for r in results],
                "n_entries": stats["n_entries"],
                "n_comparable": stats["n_comparable"],
                "n_with_author_assembly": stats["n_with_author_assembly"],
                "measured": {
                    "top_mmsize_match_rate": stats["top_mmsize_match_rate"],
                    "top_composition_match_rate":
                        stats["top_composition_match_rate"],
                    "author_assembly_match_rate":
                        stats["author_assembly_match_rate"],
                    "mean_recall": stats["mean_recall"],
                    "mean_precision": stats["mean_precision"],
                    "no_reference_assembly": stats["no_reference_assembly"],
                },
                "note": (
                    "Nested-interface-subset search in PISA's dissociation "
                    "order, not PISA's exhaustive search, so exact agreement "
                    "is not expected. Denominators: n_comparable excludes the "
                    "entries where PISA predicts nothing (they have no top "
                    "reference assembly); those are scored separately under "
                    "no_reference_assembly, because predicting a stable "
                    "assembly for barnase-barstar must cost something. Recall "
                    "excludes PISA's ligand-only (mmsize 0) assemblies, which "
                    "our ligand-only filter can never match by design, and is "
                    "reported with precision because recall alone rises "
                    "mechanically with how many assemblies we emit. "
                    "WHERE THE REMAINING GAP IS: composition agreement "
                    "(41.2%) lags stoichiometry (79.4%) because PISA includes "
                    "chain-bound hetero groups in its composition string that "
                    "ligand_mode='separate' keeps as separate molecules "
                    "(PISA 'EI[SO4][ACE]' vs our 'EI'), so most composition "
                    "misses are a ligand-convention difference rather than a "
                    "wrong assembly. Re-measure with "
                    "examples/validate_assemblies.py --record and explain any "
                    "movement."),
                "history": [
                    {"when": "2026-10-04, first measurement",
                     "ranking": "dG_diss descending (spec R4 as written)",
                     "top_mmsize_match_rate": 0.382,
                     "top_composition_match_rate": 0.206,
                     "author_assembly_match_rate": 0.423,
                     "mean_recall": 0.825},
                    {"when": "2026-10-04, after the ranking ruling",
                     "ranking": "largest stable first, dG_diss second",
                     "top_mmsize_match_rate": 0.500,
                     "top_composition_match_rate": 0.265,
                     "author_assembly_match_rate": 0.500,
                     "mean_recall": 0.825,
                     "superseded_by": (
                         "whole-branch review found three scoring/identity "
                         "defects that depressed these: a self-inverse 2-fold "
                         "dimer contact counted twice in dG_diss, assemblies "
                         "scored over the closing interface subset instead of "
                         "their own internal graph, and symmetry-mate hetero "
                         "groups collapsing onto one molecule id")},
                ],
            }, fh, indent=1)
            fh.write("\n")
        print(f"\nrecorded -> {RECORD}")


if __name__ == "__main__":
    main()
