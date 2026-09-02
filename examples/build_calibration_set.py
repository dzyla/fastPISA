#!/usr/bin/env python3
"""Fetch the reproducible large-scale PISA reference set (network).

Draws a non-redundant random sample of PDB entries from the frame defined in
``fastpisa.reference.sampling`` and downloads, for each, the original PISA
engine's result (EBI PISA CGI XML) and the deposited PDB file into the
reference cache.

    python examples/build_calibration_set.py --n 250 --workers 4

The cache under ``tests/data/reference/`` is gitignored for these extension
entries (it is tens of MB); what gets committed is the distilled feature
table produced by ``examples/extract_calibration_features.py``, which is
small and lets the fit be reproduced offline.

Entries that the frozen CGI has no data for, or that fail to download, are
skipped and reported -- the surviving list is written to
``tests/data/calibration/entries.json`` together with the sampling
parameters, so the exact benchmark is recoverable.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from fastpisa.reference import sampling
from fastpisa.reference.compare import BENCHMARK_ENTRIES
from fastpisa.reference.ebi_pisa import (
    REFERENCE_DIR, fetch_pdb_file, fetch_pisa_xml, load_cached_reference,
    identity_interfaces, cached_pdb_path,
)

CALIB_DIR = os.path.join(os.path.dirname(REFERENCE_DIR), "calibration")


def acquire(pdb_id: str) -> tuple:
    """Fetch one entry. Returns (pdb_id, n_identity_interfaces, error)."""
    try:
        if load_cached_reference(pdb_id) is None:
            fetch_pisa_xml(pdb_id)
        if cached_pdb_path(pdb_id) is None:
            fetch_pdb_file(pdb_id)
        ref = load_cached_reference(pdb_id)
        n = len(identity_interfaces(ref)) if ref else 0
        return (pdb_id, n, None)
    except Exception as exc:  # noqa: BLE001 - report and continue
        return (pdb_id, 0, f"{type(exc).__name__}: {exc}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=250,
                    help="target number of usable entries")
    ap.add_argument("--oversample", type=float, default=1.6,
                    help="candidate multiplier (entries with no identity "
                         "interface or no CGI record are dropped)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=sampling.SAMPLING_SEED)
    args = ap.parse_args()

    print("querying RCSB for 30%-identity cluster representatives ...",
          file=sys.stderr)
    reps = sampling.fetch_cluster_representatives()
    print(f"  {len(reps)} clusters in the frame", file=sys.stderr)

    n_cand = int(args.n * args.oversample)
    cands = sampling.sample_entries(reps, n_cand, exclude=BENCHMARK_ENTRIES,
                                    seed=args.seed)
    print(f"drawing {len(cands)} candidates (seed {args.seed}); "
          f"the {len(BENCHMARK_ENTRIES)} legacy benchmark entries are held out",
          file=sys.stderr)

    usable, skipped = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (pid, n_if, err) in enumerate(pool.map(acquire, cands), 1):
            if err:
                skipped.append((pid, err))
            elif n_if == 0:
                skipped.append((pid, "no identity interfaces"))
            else:
                usable.append(pid)
            if i % 25 == 0:
                print(f"  {i}/{len(cands)} fetched, {len(usable)} usable",
                      file=sys.stderr)

    usable = sorted(usable)[:args.n]
    os.makedirs(CALIB_DIR, exist_ok=True)
    out = os.path.join(CALIB_DIR, "entries.json")
    with open(out, "w") as fh:
        json.dump({
            "seed": args.seed,
            "frame": {
                "method": "X-RAY DIFFRACTION",
                "max_resolution": sampling.MAX_RESOLUTION,
                "max_atoms": sampling.MAX_ATOMS,
                "min_polymer_instances": 2,
                "max_release_date": sampling.MAX_RELEASE_DATE,
                "sequence_identity_cutoff": sampling.SEQUENCE_IDENTITY_CUTOFF,
                "n_clusters_in_frame": len(reps),
            },
            "legacy_benchmark": list(BENCHMARK_ENTRIES),
            "entries": usable,
            "n_skipped": len(skipped),
            "skipped": skipped[:100],
        }, fh, indent=2)
    print(f"\n{len(usable)} usable entries -> {out}", file=sys.stderr)
    print(f"{len(skipped)} skipped", file=sys.stderr)


if __name__ == "__main__":
    main()
