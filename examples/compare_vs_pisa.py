#!/usr/bin/env python3
"""Compare fastPISA against the original PISA engine (EBI PDBe PISA service).

For every benchmark entry this runs fastPISA on the deposited PDB file and
compares each identity (asymmetric-unit) interface against original PISA's
values: interface area, solvation dG, stab energy, P-value, CSS and the
H-bond / salt-bridge / disulfide counts.

Usage:
    python examples/compare_vs_pisa.py                 # cached benchmark
    python examples/compare_vs_pisa.py --fetch 1abc …  # add entries (network)
    python examples/compare_vs_pisa.py --json out.json # machine-readable dump

Reference data is cached under tests/data/reference/ so the comparison (and
the accuracy regression test tests/test_vs_pdbe_pisa.py) runs offline.
"""
import argparse
import json
import sys

from fastpisa.reference.compare import BENCHMARK_ENTRIES, compare_entries, summarize


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", nargs="*", default=None, metavar="PDBID",
                    help="fetch these entries (EBI PISA XML + RCSB PDB) into "
                         "the reference cache before comparing")
    ap.add_argument("--entries", nargs="*", default=None, metavar="PDBID",
                    help="restrict the comparison to these entries")
    ap.add_argument("--mode", default="pisa",
                    choices=["pisa", "cocomaps", "combined"])
    ap.add_argument("--json", default=None, metavar="PATH",
                    help="also write rows + summary as JSON")
    args = ap.parse_args()

    ids = list(BENCHMARK_ENTRIES)
    if args.fetch:
        from fastpisa.reference.ebi_pisa import fetch_pisa_xml, fetch_pdb_file
        for pid in args.fetch:
            print(f"fetching {pid} ...", file=sys.stderr)
            fetch_pisa_xml(pid)
            fetch_pdb_file(pid)
            if pid.lower() not in ids:
                ids.append(pid.lower())
    if args.entries:
        ids = [p.lower() for p in args.entries]

    res = compare_entries(ids, mode=args.mode)
    rows = res["rows"]

    hdr = (f"{'pdb':6} {'interface':24} {'area fp/ref':>16} {'dG fp/ref':>16} "
           f"{'stab fp/ref':>16} {'pv fp/ref':>12} {'hb':>7} {'sb':>7} {'ss':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['pdb_id']:6} {r['pair'][:24]:24} "
              f"{r['area_fp']:7.1f}/{r['area_ref']:7.1f} "
              f"{r['dg_fp']:7.2f}/{r['dg_ref']:7.2f} "
              f"{r['stab_fp']:7.2f}/{r['stab_ref']:7.2f} "
              f"{r['pv_fp']:5.2f}/{r['pv_ref']:5.2f} "
              f"{r['nhb_fp']:3d}/{r['nhb_ref']:3d} "
              f"{r['nsb_fp']:3d}/{r['nsb_ref']:3d} "
              f"{r['nss_fp']:2d}/{r['nss_ref']:2d}")

    unmatched = [(e["pdb_id"], e["ref_only"], e["fp_only"])
                 for e in res["entries"] if e["ref_only"] or e["fp_only"]]
    if unmatched:
        print("\nUnmatched interfaces:")
        for pid, ro, fo in unmatched:
            if ro:
                print(f"  {pid}: PISA-only  {ro}")
            if fo:
                print(f"  {pid}: fastPISA-only  {fo}")

    s = summarize(rows)
    print(f"\n=== Summary over {s['n_matched']} matched identity interfaces ===")
    print(f"interface area : median rel err {s['area_median_rel_err']*100:.1f}% "
          f"(interfaces >300 A^2: {s['area_median_rel_err_big']*100:.1f}%)")
    print(f"dG solvation   : Pearson {s['dg_pearson']:.3f}, "
          f"median |err| {s['dg_median_abs_err']:.2f} kcal/mol")
    print(f"stab energy    : Pearson {s['stab_pearson']:.3f}, "
          f"median |err| {s['stab_median_abs_err']:.2f} kcal/mol")
    print(f"P-value        : median |err| {s['pv_median_abs_err']:.3f}, "
          f"Spearman {s['pv_spearman']:.3f}")
    print(f"CSS            : Spearman {s['css_spearman']:.3f}")
    print(f"H-bonds        : mean |diff| {s['hb_mean_abs_diff']:.2f}, "
          f"{s['hb_within_1']*100:.0f}% within +-1")
    print(f"salt bridges   : mean |diff| {s['sb_mean_abs_diff']:.2f}")
    print(f"disulfides     : {s['ss_exact']*100:.0f}% exact")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "summary": s}, fh, indent=2)
        print(f"\nWritten: {args.json}")


if __name__ == "__main__":
    main()
