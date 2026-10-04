#!/usr/bin/env python
"""Large-scale blind benchmark of fastPISA against original PISA.

    # draw and run 2000 entries (hours; resumable, safe to interrupt)
    python examples/benchmark_vs_pisa.py --n 2000

    # summarise whatever has finished so far, no network
    python examples/benchmark_vs_pisa.py --report

The reference is original PISA's **own published output** for each entry,
from the EBI service (``interfaces.pisa``) -- the same engine, not a local
re-run of it. That is better ground truth than a CCP4 binary for deposited
entries, and needs no CCP4 install. A local CCP4 ``pisa`` binary is still
worth having for structures PISA's frozen database does not cover (predicted
models); ``tests/test_reproduce_pisa.py`` uses it via ``FASTPISA_PISA_BIN``.

**Blind by construction.** Entries are drawn from the same sampling frame as
the calibration set (X-ray, <=3.0 A, <=12000 atoms, >=2 polymer instances,
released <=2018-06-30, 30%-identity cluster representatives) with a FRESH
seed, excluding every entry that informed a fitted constant and every entry
of the earlier 60-entry blind draw. Nothing here is refittable: the harness
only measures.

Caching: 2000 entries is 1-2 GB of PISA XML and PDB files, so it caches to
``benchmark_cache/`` (git-ignored) or wherever ``FASTPISA_BENCHMARK_CACHE``
points -- never into the committed ``tests/data/reference``.

Resumable: each finished entry is appended to ``results.jsonl`` and skipped
on a later run, and cached downloads are reused. Interrupt and restart
freely -- including with SIGKILL, which can leave a torn final line that the
resume path tolerates.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

DEFAULT_CACHE = os.environ.get("FASTPISA_BENCHMARK_CACHE") or os.path.join(
    REPO, "benchmark_cache")

# A fresh seed: 20260901 drew the calibration set, 20260903 the 60-entry
# blind draw. Reusing either would re-measure entries that are already in
# sample.
BENCHMARK_SEED = 20261004


def _calibration_entries():
    """Every entry that informed a fitted constant, plus the earlier blind draw."""
    excluded = set()
    path = os.path.join(REPO, "tests", "data", "calibration", "entries.json")
    with open(path) as fh:
        record = json.load(fh)
    # "skipped" holds [pdb_id, reason] pairs while the others hold bare ids.
    for key in ("entries", "legacy_benchmark", "skipped"):
        for item in record.get(key) or ():
            pdb_id = item[0] if isinstance(item, (list, tuple)) else item
            if isinstance(pdb_id, str):
                excluded.add(pdb_id.lower())
    for extra in ("crystal_validation.json", "assembly_validation.json"):
        p = os.path.join(REPO, "tests", "data", "reference", extra)
        if os.path.exists(p):
            with open(p) as fh:
                excluded.update(e.lower() for e in json.load(fh).get("entries", []))
    return excluded


def _drawn_entries(cache_dir, n, seed):
    """Draw once, record the draw, and reuse it on every later run."""
    path = os.path.join(cache_dir, "entries.json")
    if os.path.exists(path):
        with open(path) as fh:
            record = json.load(fh)
        if record["n"] >= n and record["seed"] == seed:
            return record["entries"][:n], record
    from fastpisa.reference.sampling import (
        fetch_cluster_representatives, sample_entries, _frame_query,
    )
    print("fetching 30%-identity cluster representatives from RCSB ...",
          flush=True)
    reps = fetch_cluster_representatives()
    excluded = _calibration_entries()
    entries = sample_entries(reps, n, exclude=sorted(excluded), seed=seed)
    record = {
        "seed": seed,
        "n": len(entries),
        "n_representatives": len(reps),
        "n_excluded_in_sample": len(excluded),
        "frame": _frame_query().get("query", "see fastpisa.reference.sampling"),
        "entries": entries,
        "note": ("blind draw: a fresh seed over the calibration sampling "
                 "frame, excluding every entry that informed a fitted "
                 "constant and the earlier 60-entry blind draw"),
    }
    with open(path, "w") as fh:
        json.dump(record, fh, indent=1)
    print(f"drew {len(entries)} entries from {len(reps)} representatives "
          f"(excluded {len(excluded)} in-sample)", flush=True)
    return entries, record


def _done(out_path):
    done = {}
    if not os.path.exists(out_path):
        return done
    with open(out_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue          # a torn last line from an interrupted run
            done[row["pdb_id"]] = row
    return done


def _fetch_one(pdb_id, cache_dir):
    """Make sure one entry's PISA reference and coordinates are on disk."""
    from fastpisa.reference.ebi_pisa import (
        cached_pdb_path, fetch_pdb_file, fetch_pisa_xml, load_cached_reference,
    )
    try:
        if load_cached_reference(pdb_id, cache_dir=cache_dir) is None:
            fetch_pisa_xml(pdb_id, cache_dir=cache_dir)
        if cached_pdb_path(pdb_id, cache_dir=cache_dir) is None:
            fetch_pdb_file(pdb_id, cache_dir=cache_dir)
        return None
    except Exception as exc:                          # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"[:200]


def _fetch_ahead(pool, pdb_ids, cache_dir):
    """Queue every fetch at once so downloading runs ahead of analysis.

    Measured on the first 43 entries: analysis is 2.7 s per entry while one
    EBI call is 5-10 s, so a fetch-batch-then-analyse-batch loop spends most
    of its wall clock waiting on the network with the CPU idle (14 s per
    entry overall). Queuing all fetches up front lets the pool stay ~one
    batch ahead, which bounds the run by whichever of the two is slower
    instead of their sum.
    """
    return {pdb_id: pool.submit(_fetch_one, pdb_id, cache_dir)
            for pdb_id in pdb_ids}


def _compact(result):
    """Keep what global statistics need, not the whole comparison."""
    return {
        "pdb_id": result["pdb_id"],
        "n_reference": result["n_reference"],
        "n_matched": result["n_matched"],
        "n_reported": result["n_reported"],
        "missing": result["missing"][:20],
        "rows": [[round(r["area_ref"], 2), round(r["area_fp"], 2),
                  round(r["dg_ref"], 3), round(r["dg_fp"], 3)]
                 for r in result["rows"]],
    }


def _expand(record):
    rows = [{"pdb_id": record["pdb_id"], "area_ref": a, "area_fp": b,
             "dg_ref": c, "dg_fp": d} for a, b, c, d in record["rows"]]
    return dict(record, rows=rows)


def summarise(records):
    from fastpisa.reference.compare import summarize_crystal
    import numpy as np

    usable = [r for r in records if r.get("n_reference")]
    # Entries PISA had no data for are counted even when NOTHING is usable:
    # a run where every fetch failed must say so rather than report an empty
    # summary that reads like "no problems found".
    n_missing = sum(1 for r in records if not r.get("n_reference"))
    if not usable:
        return {"n_entries": 0, "n_no_reference": n_missing}
    stats = summarize_crystal([_expand(r) for r in usable])
    per_entry = np.array([r["n_matched"] / r["n_reference"] for r in usable])
    stats["entries_with_every_interface_matched"] = float((per_entry == 1).mean())
    stats["worst_entries"] = [
        r["pdb_id"] for r in sorted(usable, key=lambda r: r["n_matched"] / r["n_reference"])[:10]
        if r["n_matched"] < r["n_reference"]]
    stats["n_no_reference"] = n_missing
    return stats


def _print(stats, elapsed=None):
    if not stats.get("n_entries"):
        print("nothing measured yet")
        return
    print("\n" + "=" * 64)
    print(f"{stats['n_entries']} entries, {stats['n_reference']} PISA "
          f"interfaces" + (f", {elapsed/60:.0f} min" if elapsed else ""))
    print("=" * 64)
    print(f"interfaces matched              : {stats['matched']}/"
          f"{stats['n_reference']}  ({stats['match_rate']*100:.2f}%)")
    print(f"entries with EVERY one matched  : "
          f"{stats['entries_with_every_interface_matched']*100:.1f}%")
    print(f"interfaces we report            : {stats['reported']}")
    print(f"buried area, median |rel err|   : "
          f"{stats['area_median_rel_err']*100:.2f}%")
    print(f"dG Pearson r                    : {stats['dg_pearson']:.4f}")
    print(f"dG median |err|                 : "
          f"{stats['dg_median_abs_err']:.3f} kcal/mol")
    if stats.get("worst_entries"):
        print(f"worst entries (investigate)     : "
              f"{' '.join(stats['worst_entries'])}")
    if stats.get("n_no_reference"):
        print(f"entries PISA had no data for    : {stats['n_no_reference']}")


def _run_batch(batch, fh, done, unavailable, compare, args, offset, total,
               started):
    """Analyse one prefetched batch, appending a record per entry."""
    for i, pdb_id in enumerate(batch, 1):
        t0 = time.time()
        try:
            result = compare(pdb_id, polymer_only=not args.all_interfaces,
                             allow_fetch=False, cache_dir=args.cache_dir)
        except Exception as exc:                          # noqa: BLE001
            result, error = None, f"{type(exc).__name__}: {exc}"[:300]
        else:
            error = None
        if result is None:
            record = {"pdb_id": pdb_id, "n_reference": 0, "n_matched": 0,
                      "n_reported": 0, "missing": [], "rows": [],
                      "error": error or unavailable.get(pdb_id,
                                                        "no reference data")}
        else:
            record = _compact(result)
        record["seconds"] = round(time.time() - t0, 2)
        fh.write(json.dumps(record) + "\n")
        fh.flush()                  # a 3-hour run must survive being killed
        done[pdb_id] = record

        n = offset + i
        if n % 25 == 0 or n == total:
            rate = (time.time() - started) / n
            print(f"  {n}/{total}  {rate:.1f} s/entry  "
                  f"eta {(total - n) * rate / 60:.0f} min", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2000, help="entries to draw")
    ap.add_argument("--seed", type=int, default=BENCHMARK_SEED)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent EBI/RCSB fetches. Kept modest on "
                         "purpose: this is a free public service and the "
                         "run is bounded by analysis anyway.")
    ap.add_argument("--batch", type=int, default=100,
                    help="entries per prefetch+analyse batch")
    ap.add_argument("--all-interfaces", action="store_true",
                    help="include ligand interfaces, which PDB remediation "
                         "makes unmatchable by name for older entries")
    ap.add_argument("--report", action="store_true",
                    help="summarise finished entries only; no network")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many NEW entries this run")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    # Fetches land in the benchmark cache, never in the committed set.
    os.environ["FASTPISA_REFERENCE_DIR"] = args.cache_dir
    out_path = os.path.join(args.cache_dir, "results.jsonl")
    done = _done(out_path)

    if args.report:
        _print(summarise(list(done.values())))
        print(f"\n(from {out_path}, {len(done)} entries recorded)")
        return 0

    entries, draw = _drawn_entries(args.cache_dir, args.n, args.seed)
    todo = [e for e in entries if e not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(entries)} drawn, {len(done)} already done, "
          f"{len(todo)} to go (cache {args.cache_dir})", flush=True)

    from fastpisa.reference.compare import compare_crystal_entry

    unavailable = {}
    started = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = _fetch_ahead(pool, todo, args.cache_dir)
        with open(out_path, "a") as fh:
            for start in range(0, len(todo), args.batch):
                batch = todo[start:start + args.batch]
                for pdb_id in batch:
                    error = pending[pdb_id].result()
                    if error:
                        unavailable[pdb_id] = error
                _run_batch(batch, fh, done, unavailable,
                           compare_crystal_entry, args, start, len(todo),
                           started)

    stats = summarise(list(done.values()))
    _print(stats, time.time() - started)
    summary_path = os.path.join(args.cache_dir, "summary.json")
    with open(summary_path, "w") as fh:
        json.dump({"draw": {k: v for k, v in draw.items() if k != "entries"},
                   "n_done": len(done), "measured": stats}, fh, indent=1)
    print(f"\nper-entry records : {out_path}\nsummary           : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
