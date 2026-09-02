"""Out-of-sample accuracy regression on the sampled 674-entry benchmark.

``tests/test_vs_pdbe_pisa.py`` is a *regression* test: it re-runs the whole
pipeline on 36 entries whose interfaces informed the fitted constants, so it
protects against breakage but cannot measure generalisation. This module is
the complement. It works from the committed feature table
(``tests/data/calibration/features.json.gz``, 674 PDB entries / 6881 matched
identity interfaces, 400 of them a seeded random draw from a stated sampling
frame) and asserts the **grouped cross-validated** accuracy: every fold is
fitted without the entries it is scored on, so nothing here is in-sample.

Thresholds sit just under the values measured at calibration time
(2026-09-01, grouped 10-fold CV):

    polymer-polymer (n=2303): Pearson 0.971, R^2 about the 1:1 line 0.940,
    median |error| 0.74 kcal/mol, bias +0.15; P-value median |error| 0.067,
    Spearman 0.851.
    all interfaces (n=6881): Pearson 0.81, R^2 0.65, median |error| 1.53.

The ligand-involving regime is deliberately held to a much weaker bar. That
is not laziness -- it is the measured truth (see the "ligand" test below and
the note in fastpisa/energy/asp_table.py): ion and small-additive interfaces
have a 12% median interface-AREA error against PISA before any energy
constant is applied, so their energies cannot be better than that.

The feature table stores sufficient statistics, not coordinates, so this
runs in seconds and needs neither the ~100 MB coordinate cache nor network.
"""
import numpy as np
import pytest

from fastpisa.energy.asp_table import SIGMA
from fastpisa.reference.calibrate import (
    dg_metrics, entry_folds, fit_p_scale, fit_sigma, load_feature_table,
    predict_dg, predict_p,
)

_records = load_feature_table()

pytestmark = pytest.mark.skipif(
    not _records,
    reason="calibration feature table not present (build it with "
           "examples/build_calibration_set.py + "
           "examples/extract_calibration_features.py)")

# Classes actually fitted: P is pinned at 0 (see asp_table.SIGMA).
_FIT_CLASSES = tuple(c for c in SIGMA if c not in ("X", "P"))


@pytest.fixture(scope="module")
def oof():
    """Out-of-fold dG and P-value predictions, grouped by PDB entry."""
    recs = _records
    folds = entry_folds(recs, k=10, seed=0)
    dg = np.full(len(recs), np.nan)
    pv = np.full(len(recs), np.nan)
    for fold in folds:
        test = set(fold)
        train = [r for i, r in enumerate(recs) if i not in test]
        sigma, _ = fit_sigma(train, classes=_FIT_CLASSES)
        sigma["P"] = 0.0
        z, _ = fit_p_scale(train, sigma)
        held = [recs[i] for i in fold]
        d = predict_dg(held, sigma)
        dg[fold] = d
        pv[fold] = predict_p(held, sigma, z, dg=d)
    poly = np.array([r["is_polymer_pair"] for r in recs])
    return {
        "dg": dg,
        "pv": pv,
        "dg_ref": np.array([r["dg_ref"] for r in recs], float),
        "pv_ref": np.array([r["pv_ref"] if r["pv_ref"] is not None else np.nan
                            for r in recs], float),
        "poly": poly,
    }


def test_benchmark_is_large_and_mostly_unseen():
    """Guard the benchmark itself: a shrunken table would silently weaken
    every threshold below."""
    from fastpisa.reference.compare import BENCHMARK_ENTRIES

    entries = {r["pdb_id"] for r in _records}
    legacy = {p.lower() for p in BENCHMARK_ENTRIES}
    assert len(_records) >= 6000
    assert len(entries) >= 600
    # the fit must rest mainly on the randomly sampled entries, not the
    # hand-picked legacy ones
    assert len(entries - legacy) >= 550
    assert sum(1 for r in _records if r["is_polymer_pair"]) >= 2000


def test_solvation_energy_out_of_sample_polymer(oof):
    m = dg_metrics(oof["dg"], oof["dg_ref"], oof["poly"])
    assert m["pearson"] > 0.96
    assert m["r2_identity"] > 0.92
    assert m["median_abs_err"] < 0.85
    # no systematic offset: a biased dG shifts every downstream score
    assert abs(m["mean_err"]) < 0.5


def test_solvation_energy_out_of_sample_all(oof):
    m = dg_metrics(oof["dg"], oof["dg_ref"])
    assert m["pearson"] > 0.78
    assert m["r2_identity"] > 0.60
    assert m["median_abs_err"] < 1.8


def test_p_value_out_of_sample_polymer(oof):
    ok = oof["poly"] & np.isfinite(oof["pv"]) & np.isfinite(oof["pv_ref"])
    err = np.abs(oof["pv"][ok] - oof["pv_ref"][ok])
    assert np.median(err) < 0.09
    ra = np.argsort(np.argsort(oof["pv"][ok])).astype(float)
    rb = np.argsort(np.argsort(oof["pv_ref"][ok])).astype(float)
    assert float(np.corrcoef(ra, rb)[0, 1]) > 0.80


def test_shipped_constants_match_a_full_refit():
    """The shipped SIGMA must be the fit this repository can reproduce.

    Without this, the constants and the calibration data can drift apart
    silently -- which is exactly the state this benchmark was built to end.
    """
    sigma, diag = fit_sigma(_records, classes=_FIT_CLASSES)
    for c in diag["columns"]:
        assert sigma[c] == pytest.approx(SIGMA[c], abs=1e-4), (
            f"sigma[{c}] refits to {sigma[c]:.5f} but {SIGMA[c]:.5f} is "
            f"shipped; re-run examples/calibrate.py --emit-sigma")
    assert SIGMA["P"] == 0.0
    # A well-conditioned design matrix is what makes the individual sigmas
    # interpretable rather than a collinear trade-off.
    assert diag["condition_number"] < 150


def test_every_fitted_class_is_determined():
    """Each shipped sigma must be supported by its data, not by noise."""
    _, diag = fit_sigma(_records, classes=_FIT_CLASSES)
    for c in diag["columns"]:
        se = diag["std_err"][c]
        assert np.isfinite(se) and se > 0
        assert abs(SIGMA[c] / se) > 3.0, (
            f"sigma[{c}] = {SIGMA[c]:.5f} +- {se:.5f} is not distinguishable "
            f"from zero; it should be pinned, not fitted")


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
    assert np.median(rel[~poly]) > 0.06, (
        "ligand-pair interface areas now agree with PISA much better than "
        "documented -- update the accuracy claims in asp_table.py and README")
    # Large interfaces are the well-behaved regime in every group.
    big = ar > 300
    assert np.median(rel[big]) < 0.025
