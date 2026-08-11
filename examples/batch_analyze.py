#!/usr/bin/env python3
"""Batch-analyse many antibody/AlphaFold complexes with fastPISA.

Demonstrates :func:`fastpisa.batch.analyze_many` (parallel, non-crashing) and
:func:`fastpisa.batch.expand_inputs`. This replaces the old throwaway
``/tmp/ab_batch`` example (item 4.1 of fastpisa_improvements.md).

Usage::

    python examples/batch_analyze.py "results/**/*.cif" -o out.jsonl --n_jobs 4

    # or analyse a directory directly:
    python examples/batch_analyze.py /path/to/models -o out.jsonl

Output ``out.jsonl`` has one JSON line per structure:
    {"path": ..., "ok": true, "n_interfaces": N, "pdb_id": ...}
"""
import argparse
import json
import os
import sys

# Make fastPISA importable even when this script is run directly (repo root is
# the parent of examples/), independent of whether the package is pip-installed.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="Paths, globs, or directories of PDB/mmCIF files")
    ap.add_argument("-o", "--output", default="batch_results.jsonl", help="Output .jsonl path")
    ap.add_argument("--mode", default="pisa", choices=["pisa", "cocomaps"])
    ap.add_argument("--n_jobs", type=int, default=1, help="Parallel workers (1=serial, -1=all CPUs)")
    ap.add_argument("--pdb_id", default=None, help="Force the same pdb_id on every file (optional)")
    ap.add_argument("--min_css", type=float, default=0.0)
    args = ap.parse_args()

    from fastpisa.batch import analyze_many, expand_inputs

    files = expand_inputs(*args.inputs)
    if not files:
        print("No structure files matched.", file=sys.stderr)
        return 1
    print(f"Analysing {len(files)} structure(s) with {args.n_jobs} worker(s)...")

    kwargs = {"mode": args.mode, "min_css": args.min_css}
    if args.pdb_id:
        kwargs["pdb_id"] = args.pdb_id

    ok = 0
    with open(args.output, "w") as out:
        for r in analyze_many(files, n_jobs=args.n_jobs, **kwargs):
            if args.pdb_id:
                pid = args.pdb_id
            else:
                pid = os.path.splitext(os.path.basename(r["path"]))[0]
            line = {
                "path": r["path"],
                "ok": r["ok"],
                "n_interfaces": r["n_interfaces"],
                "pdb_id": pid,
                "error": r.get("error"),
            }
            out.write(json.dumps(line) + "\n")
            ok += 1 if r["ok"] else 0

    print(f"Done: {ok}/{len(files)} succeeded. Results in {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
