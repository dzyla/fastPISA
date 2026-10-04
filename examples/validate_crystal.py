#!/usr/bin/env python
"""Validate ``symmetry="crystal"`` against original PISA, interface by interface.

    python examples/validate_crystal.py                  # cached 37 entries, offline
    python examples/validate_crystal.py --blind          # the recorded blind draw
    python examples/validate_crystal.py --entries 1acb 1urn
    python examples/validate_crystal.py --all-interfaces # include ligand interfaces

Each PISA interface is matched to one fastPISA interface on the chain pair AND
the relative crystal transform (PISA reports every molecule's orthogonal
placement), so a missing interface and a mis-sized one are distinguishable.

The ``--blind`` set is a fresh draw from the sampling frame excluding every
entry that informed a fitted constant; see
``tests/data/reference/crystal_validation.json``. Entries not already cached
are fetched once from the EBI and RCSB (network), then reused.

By default only polymer-polymer interfaces are compared. PDB remediation has
renamed and renumbered hetero groups since PISA's database was frozen -- 1ppf's
glycans moved from chain E:401-417 into chains A/B:1-8, 1prc's HEM became HEC --
so a ligand interface cannot be matched by name even when the geometry agrees.
``--all-interfaces`` shows that gap rather than hiding it.
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
    compare_crystal_entry, summarize_crystal,
)

REFERENCE_DIR = os.path.join(REPO, "tests", "data", "reference")
BLIND_FILE = os.path.join(REFERENCE_DIR, "crystal_validation.json")


def cached_entries():
    out = []
    for path in sorted(glob.glob(os.path.join(REFERENCE_DIR, "*.pisa.xml.gz"))):
        pdb_id = os.path.basename(path)[:4]
        if os.path.exists(os.path.join(REFERENCE_DIR, "pdb", f"{pdb_id}.pdb.gz")):
            out.append(pdb_id)
    return out


def blind_entries():
    with open(BLIND_FILE) as fh:
        return json.load(fh)["entries"]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blind", action="store_true",
                    help="use the recorded blind draw (network on first run)")
    ap.add_argument("--entries", nargs="*", default=None)
    ap.add_argument("--all-interfaces", action="store_true",
                    help="include ligand interfaces (expect name mismatches)")
    args = ap.parse_args()

    if args.entries:
        entries = [e.lower() for e in args.entries]
    elif args.blind:
        entries = blind_entries()
    else:
        entries = cached_entries()

    print(f"{len(entries)} entries, "
          f"{'all' if args.all_interfaces else 'polymer-polymer'} interfaces\n")
    results, skipped = [], []
    started = time.time()
    for pdb_id in entries:
        begin = time.time()
        result = compare_crystal_entry(
            pdb_id, polymer_only=not args.all_interfaces)
        if result is None:
            skipped.append(pdb_id)
            print(f"{pdb_id}: no reference data, skipped", flush=True)
            continue
        results.append(result)
        rel = [abs(r["area_fp"] - r["area_ref"]) / r["area_ref"]
               for r in result["rows"]
               if r["area_ref"] and r["area_ref"] > 100]
        median = (sorted(rel)[len(rel) // 2] if rel else float("nan"))
        flag = "" if not result["missing"] else \
            f"   <-- {len(result['missing'])} MISSING"
        print(f"{pdb_id}: {result['n_matched']:3d}/{result['n_reference']:3d} "
              f"matched (reported {result['n_reported']:3d})  "
              f"area med {100 * median:5.1f}%  "
              f"{time.time() - begin:5.1f}s{flag}", flush=True)

    stats = summarize_crystal(results)
    print(f"\n=== {stats['n_entries']} entries in {time.time() - started:.0f}s ===")
    print(f"PISA interfaces matched: {stats['matched']}/{stats['n_reference']}"
          f"  ({100 * stats['match_rate']:.1f}%); fastPISA reported "
          f"{stats['reported']}")
    print(f"interface area: median relative error "
          f"{100 * stats['area_median_rel_err']:.2f}%")
    print(f"dG solvation  : Pearson {stats['dg_pearson']:.4f}, "
          f"median |err| {stats['dg_median_abs_err']:.2f} kcal/mol")
    if skipped:
        print(f"skipped (no reference): {' '.join(skipped)}")
    worst = sorted((r for r in results if r["missing"]),
                   key=lambda r: -len(r["missing"]))
    for result in worst[:5]:
        print(f"\n{result['pdb_id']} missing {len(result['missing'])}:")
        for entry in result["missing"][:4]:
            print(f"   {entry['pair']}  area {entry['area_ref']}  "
                  f"symop {entry['symop']}")


if __name__ == "__main__":
    main()
