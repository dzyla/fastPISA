#!/usr/bin/env python3
"""Audit and refit fastPISA's fitted constants against original PISA.

Runs entirely offline from the distilled feature table written by
``examples/extract_calibration_features.py``.

    python examples/calibrate.py                      # full report
    python examples/calibrate.py --holdout-legacy     # blind test of the
                                                      # shipped constants on
                                                      # entries they never saw
    python examples/calibrate.py --emit-sigma         # print a SIGMA block

What it reports, and why each part is there:

1. **Dataset + per-class support.** A sigma is only a fitted constant if
   something fitted it; classes seen in three interfaces are flagged.
2. **Blind evaluation of the shipped constants** on entries excluded from
   their calibration. This is the only number that may be quoted as
   out-of-sample accuracy.
3. **Grouped 10-fold cross-validation** of a refit (folds never split a PDB
   entry, because interfaces within an entry are correlated).
4. **Refit on everything**, with standard errors and the design-matrix
   condition number, so ill-determined classes are visible.
5. **The shipped model** -- the residue-level hierarchical fit (class sigma +
   shrunk per-atom-type deviation) re-derived from the committed
   ``residue_fit.json.gz`` and cross-validated at interface level. This is
   the fit ``asp_table.py`` carries; steps 3-4 are the class-level
   diagnostics.

Correlation is reported alongside bias, slope and R^2 about the 1:1 line:
Pearson r is inflated by the two-orders-of-magnitude spread of interface
size and will look excellent even when the scale is systematically wrong.
"""
import argparse
import json
import sys

import numpy as np

from fastpisa.energy.asp_table import DELTA, SIGMA, class_of_fine
from fastpisa.reference.calibrate import (
    CLASSES, FINE_TYPE_RIDGE, class_support, cross_validate_sigma, dg_metrics,
    entry_folds, fit_css, fit_p_scale, fit_sigma, fit_sigma_residue_level,
    fitted_classes, fitted_fine_types, load_feature_table,
    load_residue_fit_rows, predict_css, predict_dg, predict_p,
)
from fastpisa.reference.compare import BENCHMARK_ENTRIES
from fastpisa.scoring.scoring import P_VALUE_Z_SCALE

CSS_COEF = (-6.9088, -0.1699, 0.8485)


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def _print_dg(title, m):
    if m.get("n", 0) < 3:
        print(f"{title:<34} (n={m.get('n', 0)}, too few)")
        return
    print(f"{title:<34} n={m['n']:4d}  r={m['pearson']:.3f}  "
          f"R2(1:1)={m['r2_identity']:+.3f}  med|err|={m['median_abs_err']:.2f}  "
          f"bias={m['mean_err']:+.2f}  slope={m['slope']:.3f}  "
          f"rmse={m['rmse']:.2f}")


def report_pv_css(records, sigma, z_scale, css_coef, label):
    dg = predict_dg(records, sigma)
    pv_ref = np.array([r["pv_ref"] if r["pv_ref"] is not None else np.nan
                       for r in records], float)
    css_ref = np.array([r["css_ref"] if r["css_ref"] is not None else np.nan
                        for r in records], float)
    pv = predict_p(records, sigma, z_scale, dg=dg)
    css = predict_css(records, css_coef, dg=dg)
    poly = np.array([r["is_polymer_pair"] for r in records])
    for name, mask in (("all", np.ones(len(records), bool)), ("polymer", poly)):
        ok = mask & np.isfinite(pv_ref)
        print(f"  P-value [{label}/{name}]  med|err|={np.median(np.abs(pv[ok]-pv_ref[ok])):.3f}"
              f"  Spearman={_spearman(pv[ok], pv_ref[ok]):.3f}  n={int(ok.sum())}")
        ok = mask & np.isfinite(css_ref)
        print(f"  CSS     [{label}/{name}]  MAE={np.mean(np.abs(css[ok]-css_ref[ok])):.3f}"
              f"  Spearman={_spearman(css[ok], css_ref[ok]):.3f}  n={int(ok.sum())}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=None)
    ap.add_argument("--k", type=int, default=10, help="CV folds (grouped by entry)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ridge", type=float, default=0.0)
    ap.add_argument("--holdout-legacy", action="store_true",
                    help="evaluate the SHIPPED constants only on entries that "
                         "were not part of their calibration set")
    ap.add_argument("--emit-sigma", action="store_true",
                    help="print a ready-to-paste SIGMA dict from the full refit")
    ap.add_argument("--fit-on", choices=["all", "sampled"], default="all",
                    help="'sampled' fits ONLY on the randomly-drawn entries "
                         "and keeps the 36 hand-picked legacy entries as a "
                         "held-out test set -- the design that lets "
                         "tests/test_vs_pdbe_pisa.py stay an honest "
                         "out-of-sample regression after a refit")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    records = load_feature_table(args.features) if args.features else load_feature_table()
    if not records:
        sys.exit("no feature table; run examples/extract_calibration_features.py first")

    legacy = {p.lower() for p in BENCHMARK_ENTRIES}
    all_records = records
    entries = sorted({r["pdb_id"] for r in all_records})
    new = [r for r in all_records if r["pdb_id"] not in legacy]
    held = [r for r in all_records if r["pdb_id"] in legacy]

    print(f"=== dataset ===")
    ap_ = np.array([r["is_polymer_pair"] for r in all_records])
    print(f"entries {len(entries)}  interfaces {len(all_records)}  "
          f"polymer-polymer {int(ap_.sum())}  ligand-involving {int((~ap_).sum())}")
    print(f"of which NOT in the shipped calibration set: "
          f"{len({r['pdb_id'] for r in new})} entries / {len(new)} interfaces")

    if args.fit_on == "sampled":
        records = new
        print(f"\nFIT SET: the {len({r['pdb_id'] for r in new})} randomly "
              f"sampled entries ({len(new)} interfaces).")
        print(f"HELD-OUT TEST SET: the {len({r['pdb_id'] for r in held})} "
              f"hand-picked legacy entries ({len(held)} interfaces) -- never "
              f"seen by this fit.")
    poly = np.array([r["is_polymer_pair"] for r in records])

    print("\n=== per-class evidence (buried area informing each sigma) ===")
    sup = class_support(records)
    print(f"{'class':6} {'sigma':>10} {'n_iface':>8} {'n_entry':>8} {'tot_area':>12}")
    for c in CLASSES:
        s = sup[c]
        flag = "  <-- thin" if s["n_entries"] < 5 else ""
        print(f"{c:6} {SIGMA[c]:>10.5f} {s['n_interfaces']:>8d} "
              f"{s['n_entries']:>8d} {s['total_area']:>12.0f}{flag}")

    # ---- 2. blind evaluation of the shipped constants --------------------
    target = np.array([r["dg_ref"] for r in records], float)
    dg_ship = predict_dg(records, SIGMA)
    print("\n=== shipped constants ===")
    _print_dg("dG in-sample (all)", dg_metrics(dg_ship, target))
    _print_dg("dG in-sample (polymer)", dg_metrics(dg_ship, target, poly))
    if new:
        tn = np.array([r["dg_ref"] for r in new], float)
        pn = np.array([r["is_polymer_pair"] for r in new])
        dn = predict_dg(new, SIGMA)
        print("  -- BLIND (entries the shipped fit never saw) --")
        _print_dg("dG blind (all)", dg_metrics(dn, tn))
        _print_dg("dG blind (polymer)", dg_metrics(dn, tn, pn))
        report_pv_css(new, SIGMA, P_VALUE_Z_SCALE, CSS_COEF, "blind")
    if args.holdout_legacy:
        return

    # ---- 3. grouped CV of a refit ---------------------------------------
    print(f"\n=== refit, grouped {args.k}-fold CV (folds never split an entry) ===")
    cv = cross_validate_sigma(records, k=args.k, seed=args.seed, ridge=args.ridge)
    _print_dg("dG out-of-fold (all)", dg_metrics(cv["pred"], cv["target"]))
    _print_dg("dG out-of-fold (polymer)", dg_metrics(cv["pred"], cv["target"], poly))

    # ---- 4. full refit ---------------------------------------------------
    sigma_new, diag = fit_sigma(records, ridge=args.ridge)
    print(f"\n=== full refit (n={diag['n_interfaces']} interfaces, "
          f"{diag['n_entries']} entries) ===")
    print(f"design-matrix condition number {diag['condition_number']:.1f}"
          f"   residual RMS {diag['residual_rms']:.2f} kcal/mol")
    if diag["held"]:
        print(f"classes with no data (incumbent kept): {diag['held']}")
    print(f"{'class':6} {'shipped':>10} {'refit':>10} {'std_err':>10} {'z':>7}")
    for c in diag["columns"]:
        se = diag["std_err"][c]
        z = sigma_new[c] / se if se and np.isfinite(se) and se > 0 else float("nan")
        print(f"{c:6} {SIGMA[c]:>10.5f} {sigma_new[c]:>10.5f} {se:>10.5f} {z:>7.1f}")

    dg_new = predict_dg(records, sigma_new)
    _print_dg("dG in-sample refit (all)", dg_metrics(dg_new, target))
    _print_dg("dG in-sample refit (polymer)", dg_metrics(dg_new, target, poly))

    if args.fit_on == "sampled" and held:
        th = np.array([r["dg_ref"] for r in held], float)
        ph = np.array([r["is_polymer_pair"] for r in held])
        print("  -- refit evaluated on the HELD-OUT legacy entries --")
        _print_dg("dG held-out shipped (all)", dg_metrics(predict_dg(held, SIGMA), th))
        _print_dg("dG held-out refit (all)", dg_metrics(predict_dg(held, sigma_new), th))
        _print_dg("dG held-out shipped (poly)", dg_metrics(predict_dg(held, SIGMA), th, ph))
        _print_dg("dG held-out refit (poly)", dg_metrics(predict_dg(held, sigma_new), th, ph))

    # ---- 5. the SHIPPED model: residue-level hierarchical refit ----------
    rows = load_residue_fit_rows()
    if rows:
        classes_r = fitted_classes(rows)
        fine_r = fitted_fine_types(rows)
        sig_r, delta_r, diag_r = fit_sigma_residue_level(
            rows, class_of_fine, classes_r, fine_r, FINE_TYPE_RIDGE,
            polymer_only=False)
        print(f"\n=== shipped model: residue-level hierarchical refit "
              f"({diag_r['n_residues']} residues, {len(classes_r)} classes + "
              f"{len(fine_r)} shrunk fine types, ridge {FINE_TYPE_RIDGE:g}) ===")
        dmax_s = max(abs(sig_r[c] - SIGMA[c]) for c in classes_r)
        dmax_d = max(abs(delta_r[t] - DELTA.get(t, 0.0)) for t in fine_r)
        print(f"max |refit - shipped|: sigma {dmax_s:.2e}, delta {dmax_d:.2e}  "
              f"(residual RMS per residue {diag_r['residual_rms']:.3f} kcal/mol)")
        # honest interface-level CV of the shipped model
        folds = entry_folds(records, k=args.k, seed=args.seed)
        by_entry = {}
        for pid, r in rows:
            by_entry.setdefault(pid, []).append((pid, r))
        oof = np.full(len(records), np.nan)
        for fold in folds:
            test_ent = {records[i]["pdb_id"] for i in fold}
            train_rows = [row for pid, rs in by_entry.items() if pid not in test_ent for row in rs]
            sg, dl, _ = fit_sigma_residue_level(train_rows, class_of_fine, classes_r, fine_r,
                                                FINE_TYPE_RIDGE, polymer_only=False)
            sg = dict(sg); sg.update({c: 0.0 for c in ("P", "NA_P")})
            oof[fold] = predict_dg([records[i] for i in fold], sg, dl)
        _print_dg("SHIPPED dG out-of-fold (all)", dg_metrics(oof, target))
        _print_dg("SHIPPED dG out-of-fold (polymer)", dg_metrics(oof, target, poly))
        sigma_new, z_sigma = dict(SIGMA), None

    # ---- P-value and CSS -------------------------------------------------
    print("\n=== P-value / CSS ===")
    report_pv_css(records, SIGMA, P_VALUE_Z_SCALE, CSS_COEF, "shipped")
    z_new, z_loss = fit_p_scale(records, sigma_new)
    css_new, css_diag = fit_css(records, sigma_new)
    print(f"  refit P_VALUE_Z_SCALE = {z_new:.4f} (shipped {P_VALUE_Z_SCALE})"
          f"  median|err| {z_loss:.3f}")
    print(f"  refit CSS coefficients = ({css_new[0]:.4f}, {css_new[1]:.4f}, "
          f"{css_new[2]:.4f})  (shipped {CSS_COEF})")
    report_pv_css(records, sigma_new, z_new, css_new, "refit")
    if args.fit_on == "sampled" and held:
        report_pv_css(held, SIGMA, P_VALUE_Z_SCALE, CSS_COEF, "held-out shipped")
        report_pv_css(held, sigma_new, z_new, css_new, "held-out refit")

    # honest CV for the P-value scale and CSS as well
    folds = entry_folds(records, k=args.k, seed=args.seed)
    pv_oof = np.full(len(records), np.nan)
    css_oof = np.full(len(records), np.nan)
    for fold in folds:
        test = set(fold)
        train = [r for i, r in enumerate(records) if i not in test]
        sig_f, _ = fit_sigma(train, ridge=args.ridge)
        z_f, _ = fit_p_scale(train, sig_f)
        c_f, _ = fit_css(train, sig_f)
        held = [records[i] for i in fold]
        dg_f = predict_dg(held, sig_f)
        pv_oof[fold] = predict_p(held, sig_f, z_f, dg=dg_f)
        css_oof[fold] = predict_css(held, c_f, dg=dg_f)
    pv_ref = np.array([r["pv_ref"] if r["pv_ref"] is not None else np.nan
                       for r in records], float)
    css_ref = np.array([r["css_ref"] if r["css_ref"] is not None else np.nan
                        for r in records], float)
    for name, mask in (("all", np.ones(len(records), bool)), ("polymer", poly)):
        ok = mask & np.isfinite(pv_ref) & np.isfinite(pv_oof)
        print(f"  P-value [out-of-fold/{name}]  med|err|="
              f"{np.median(np.abs(pv_oof[ok]-pv_ref[ok])):.3f}"
              f"  Spearman={_spearman(pv_oof[ok], pv_ref[ok]):.3f}")
        ok = mask & np.isfinite(css_ref) & np.isfinite(css_oof)
        print(f"  CSS     [out-of-fold/{name}]  MAE="
              f"{np.mean(np.abs(css_oof[ok]-css_ref[ok])):.3f}"
              f"  Spearman={_spearman(css_oof[ok], css_ref[ok]):.3f}")

    if args.emit_sigma:
        src = sig_r if rows else sigma_new
        print("\nSIGMA = {")
        for c in SIGMA:
            v = src.get(c, SIGMA[c])
            print(f'    "{c}":{" " * (12 - len(c))}{v:>9.5f},')
        print("}")
        if rows:
            print("DELTA = {")
            for t in sorted(delta_r):
                print(f'    "{t}": {delta_r[t]:+.5f},')
            print("}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"sigma_refit": sigma_new, "diagnostics": diag,
                       "p_value_z_scale": z_new, "css_coef": css_new,
                       "css_diag": css_diag}, fh, indent=2)
        print(f"\nWritten: {args.json}")


if __name__ == "__main__":
    main()
