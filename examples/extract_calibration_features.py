#!/usr/bin/env python3
"""Distil the cached PISA reference set into a compact feature table.

Runs the fastPISA core once per entry with ``collect_calibration=True`` and
writes, per matched identity interface, the sufficient statistics for
refitting the ASP sigmas / P-value / CSS constants, together with original
PISA's targets.

    python examples/extract_calibration_features.py            # all cached
    python examples/extract_calibration_features.py --entries 1brs 1vfb

The result (``tests/data/calibration/features.json.gz``, a few hundred kB) is
what makes ``examples/calibrate.py`` reproducible offline -- the multi-tens-of-
MB coordinate/XML cache it was derived from does not have to be committed.
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from fastpisa.reference.calibrate import (
    CALIBRATION_DIR, FEATURE_TABLE, extract_entry, save_feature_table,
)
from fastpisa.reference.ebi_pisa import REFERENCE_DIR


def cached_entries():
    ids = []
    for p in sorted(glob.glob(os.path.join(REFERENCE_DIR, "*.pisa.xml.gz"))):
        pid = os.path.basename(p).split(".")[0]
        if os.path.exists(os.path.join(REFERENCE_DIR, "pdb", f"{pid}.pdb.gz")):
            ids.append(pid)
    return ids


def _safe_extract(pdb_id):
    try:
        return pdb_id, extract_entry(pdb_id), None
    except Exception as exc:  # noqa: BLE001
        return pdb_id, [], f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entries", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", default=FEATURE_TABLE)
    args = ap.parse_args()

    ids = [e.lower() for e in args.entries] if args.entries else cached_entries()
    print(f"extracting features for {len(ids)} entries "
          f"on {args.workers} workers ...", file=sys.stderr)

    records, failed = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (pid, recs, err) in enumerate(pool.map(_safe_extract, ids), 1):
            if err:
                failed.append((pid, err))
            records.extend(recs)
            if i % 25 == 0:
                print(f"  {i}/{len(ids)} entries, {len(records)} interfaces",
                      file=sys.stderr)

    save_feature_table(records, args.out)
    meta = {
        "n_entries_attempted": len(ids),
        "n_entries_with_interfaces": len({r["pdb_id"] for r in records}),
        "n_interfaces": len(records),
        "n_polymer_polymer": sum(1 for r in records if r["is_polymer_pair"]),
        "failed": failed,
    }
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    with open(os.path.join(CALIBRATION_DIR, "features_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k != "failed"}, indent=2))
    if failed:
        print(f"{len(failed)} entries failed:", file=sys.stderr)
        for pid, err in failed[:20]:
            print(f"  {pid}: {err}", file=sys.stderr)
    print(f"\nWritten: {args.out} "
          f"({os.path.getsize(args.out)/1e3:.0f} kB)", file=sys.stderr)


if __name__ == "__main__":
    main()
