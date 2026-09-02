"""Out-of-sample accuracy regression on the sampled 674-entry benchmark.

``tests/test_vs_pdbe_pisa.py`` is a *regression* test: it re-runs the whole
pipeline on 36 entries whose interfaces informed the fitted constants, so it
protects against breakage but cannot measure generalisation. This module is
the complement. It works from the committed calibration tables

* ``features.json.gz``  -- 6904 matched identity interfaces from 674 PDB
  entries (400 a seeded random draw from a stated sampling frame), each
  with its buried area per fine atom type and PISA's targets;
* ``residue_fit.json.gz`` -- 119,078 PISA-matched interface residues with
  PISA's per-residue solvation energy and buried area;

and asserts **grouped cross-validated** accuracy: every fold is fitted
without the entries it is scored on, so nothing here is in-sample.

Thresholds sit just under the values measured at calibration time
(2026-09-01, residue-level hierarchical model, NACCESS radii, grouped
10-fold CV at interface level):

    polymer-polymer (n=2314): Pearson 0.987, R^2 about the 1:1 line 0.975,
    median |error| 0.33 kcal/mol, bias -0.01; P-value median |error| 0.060,
    Spearman 0.88.
    protein-nucleic acid (n=241): R^2 0.96, median 0.30.
    all interfaces (n=6904): R^2 ~0.69.

The ligand-involving regime is deliberately held to a much weaker bar. That
is the measured truth (see the "ligand" test below and the note in
fastpisa/energy/asp_table.py): ion and small-additive interfaces have a 9%
median interface-AREA error against PISA before any energy constant is
applied, so their energies cannot be better than that.

The tables store sufficient statistics, not coordinates, so this runs in
seconds and needs neither the ~100 MB coordinate cache nor network.
"""
import numpy as np
import pytest

from fastpisa.energy.asp_table import DELTA, SIGMA, class_of_fine
from fastpisa.reference.calibrate import (
    FINE_TYPE_RIDGE, PINNED_CLASSES, dg_metrics, entry_folds, fit_p_scale,
    fit_sigma_residue_level, fitted_classes, fitted_fine_types,
    load_feature_table, load_residue_fit_rows, predict_dg, predict_p,
    scheme_b,
)

_records = load_feature_table()
_rows = load_residue_fit_rows()

pytestmark = pytest.mark.skipif(
    not _records or not _rows,
    reason="calibration tables not present (build them with "
           "examples/build_calibration_set.py + "
           "examples/extract_calibration_features.py [--residues])")


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


@pytest.fixture(scope="module")
def shipped_model():
    classes = fitted_classes(_rows)
    fine = fitted_fine_types(_rows)
    sigma, delta, diag = fit_sigma_residue_level(
        _rows, class_of_fine, classes, fine, FINE_TYPE_RIDGE, polymer_only=False)
    return {"classes": classes, "fine": fine, "sigma": sigma,
            "delta": delta, "diag": diag}


@pytest.fixture(scope="module")
def oof(shipped_model):
    """Out-of-fold dG and P-value of the SHIPPED model, grouped by entry.

    Each fold's sigma/delta are fitted at residue level on the other
    entries' residues, then evaluated on the held-out interfaces' full
    per-type buried areas -- exactly what the pipeline would compute.
    """
    recs = _records
    classes, fine = shipped_model["classes"], shipped_model["fine"]
    by_entry = {}
    for pid, r in _rows:
        by_entry.setdefault(pid, []).append((pid, r))
    folds = entry_folds(recs, k=10, seed=0)
    dg = np.full(len(recs), np.nan)
    pv = np.full(len(recs), np.nan)
    for fold in folds:
        test_ent = {recs[i]["pdb_id"] for i in fold}
        train = [row for pid, rows in by_entry.items() if pid not in test_ent
                 for row in rows]
        sigma, delta, _ = fit_sigma_residue_level(
            train, class_of_fine, classes, fine, FINE_TYPE_RIDGE, polymer_only=False)
        sigma = dict(sigma)
        sigma.update({c: 0.0 for c in PINNED_CLASSES})
        held = [recs[i] for i in fold]
        d = predict_dg(held, sigma, delta)
        z, _ = fit_p_scale([recs[i] for i in range(len(recs)) if i not in set(fold)],
                           sigma)
        dg[fold] = d
        pv[fold] = predict_p(held, sigma, z, dg=d)
    NA = ("A", "G", "C", "U", "DA", "DG", "DC", "DT", "DU")
    poly = np.array([r["is_polymer_pair"] for r in recs])
    nuc = np.array([any(t.split(":")[0] in NA for t in r["bsa_by_type"])
                    for r in recs]) & poly
    return {
        "dg": dg, "pv": pv,
        "dg_ref": np.array([r["dg_ref"] for r in recs], float),
        "pv_ref": np.array([r["pv_ref"] if r["pv_ref"] is not None else np.nan
                            for r in recs], float),
        "poly": poly, "nuc": nuc,
    }


def test_benchmark_is_large_and_mostly_unseen():
    """Guard the benchmark itself: a shrunken table would silently weaken
    every threshold below."""
    from fastpisa.reference.compare import BENCHMARK_ENTRIES

    entries = {r["pdb_id"] for r in _records}
    legacy = {p.lower() for p in BENCHMARK_ENTRIES}
    assert len(_records) >= 6000
    assert len(entries) >= 600
    assert len(entries - legacy) >= 550
    assert sum(1 for r in _records if r["is_polymer_pair"]) >= 2000
    assert len(_rows) >= 100000


def test_solvation_energy_out_of_sample_polymer(oof):
    m = dg_metrics(oof["dg"], oof["dg_ref"], oof["poly"])
    assert m["pearson"] > 0.985
    assert m["r2_identity"] > 0.965
    assert m["median_abs_err"] < 0.40
    assert abs(m["mean_err"]) < 0.3


def test_solvation_energy_out_of_sample_protein_nucleic_acid(oof):
    m = dg_metrics(oof["dg"], oof["dg_ref"], oof["nuc"])
    assert m["n"] >= 150
    assert m["r2_identity"] > 0.92
    assert m["median_abs_err"] < 0.45


def test_solvation_energy_out_of_sample_all(oof):
    m = dg_metrics(oof["dg"], oof["dg_ref"])
    assert m["pearson"] > 0.80
    assert m["r2_identity"] > 0.62
    assert m["median_abs_err"] < 1.3


def test_p_value_out_of_sample_polymer(oof):
    ok = oof["poly"] & np.isfinite(oof["pv"]) & np.isfinite(oof["pv_ref"])
    err = np.abs(oof["pv"][ok] - oof["pv_ref"][ok])
    assert np.median(err) < 0.075
    assert _spearman(oof["pv"][ok], oof["pv_ref"][ok]) > 0.84


def test_shipped_constants_match_a_full_refit(shipped_model):
    """The shipped SIGMA / DELTA must be the fit this repository reproduces.

    Without this the constants and the calibration data can drift apart
    silently -- exactly the state this benchmark was built to end.
    """
    for c in shipped_model["classes"]:
        assert shipped_model["sigma"][c] == pytest.approx(SIGMA[c], abs=1e-4), (
            f"SIGMA[{c}] refits to {shipped_model['sigma'][c]:.5f} but "
            f"{SIGMA[c]:.5f} is shipped; re-run examples/calibrate.py --emit-sigma")
    for t in shipped_model["fine"]:
        assert shipped_model["delta"][t] == pytest.approx(DELTA.get(t, 0.0), abs=1e-4), t
    for c in PINNED_CLASSES:
        assert SIGMA[c] == 0.0
    # every shipped deviation belongs to a type the data actually supports
    assert set(DELTA) == set(shipped_model["fine"])


def test_shipped_scheme_is_the_specified_scheme_b():
    """asp_table.class_of_fine must stay identical to the scheme the CV
    experiments were run under (calibrate.scheme_b)."""
    types = {t for _, r in _rows[:20000] for t in r["bsa_by_type"]}
    for t in types:
        assert class_of_fine(t) == scheme_b(t), t


def test_every_fitted_class_is_determined():
    """Each shipped class sigma must be supported by its data, not by noise
    (residue-level OLS standard errors, class level only)."""
    classes = fitted_classes(_rows)
    cidx = {c: j for j, c in enumerate(classes)}
    X = np.zeros((len(_rows), len(classes)))
    y = np.empty(len(_rows))
    for i, (_, r) in enumerate(_rows):
        y[i] = r["solv_ref"]
        for t, a in r["bsa_by_type"].items():
            j = cidx.get(class_of_fine(t))
            if j is not None:
                X[i, j] += a
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = float(resid @ resid) / (len(y) - len(classes))
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    assert np.linalg.cond(X) < 300
    for c in classes:
        z = abs(beta[cidx[c]] / se[cidx[c]])
        assert z > 3.0, f"class {c}: sigma {beta[cidx[c]]:.5f} +- {se[cidx[c]]:.5f}"


def test_residue_buried_area_matches_pisa():
    """Per-residue buried area vs PISA (NACCESS radii): the geometry that the
    solvation fit rests on. Was 6.1% median before the radius change."""
    rel = []
    for _, r in _rows:
        if r["is_polymer"] and r["bsa_ref"] > 5.0:
            bsa_fp = sum(r["bsa_by_type"].values())
            rel.append(abs(bsa_fp - r["bsa_ref"]) / r["bsa_ref"])
    rel = np.array(rel)
    assert len(rel) > 50000
    assert np.median(rel) < 0.025
    assert np.percentile(rel, 90) < 0.12


def test_ligand_interface_area_error_is_documented_not_hidden():
    """Ligand-pair interface AREA is much less accurate than polymer-pair.

    This asserts the documented weakness so the docs cannot quietly become
    wrong: if ligand geometry is ever fixed, this test fails and the claims
    in asp_table.py / README get revisited.
    """
    af = np.array([r["area_fp"] for r in _records], float)
    ar = np.array([r["area_ref"] for r in _records], float)
    poly = np.array([r["is_polymer_pair"] for r in _records])
    rel = np.abs(af - ar) / np.maximum(ar, 1.0)
    assert np.median(rel[poly]) < 0.025
    assert np.median(rel[~poly]) > 0.05, (
        "ligand-pair interface areas now agree with PISA much better than "
        "documented -- update the accuracy claims in asp_table.py and README")
    big = ar > 300
    assert np.median(rel[big]) < 0.025
