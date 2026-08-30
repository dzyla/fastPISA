"""Accuracy regression vs the ORIGINAL PISA engine (EBI PDBe PISA service).

Runs fully offline against the reference data cached in
tests/data/reference/ (EBI PISA XML + RCSB PDB files for 21 entries,
117 matched identity interfaces). These thresholds sit just under the
accuracy measured at calibration time (2026-08-29):

    area 1.5% / 1.3% median rel err; dG Pearson 0.950 (median 0.94
    kcal/mol); stab Pearson 0.973; P-value median |err| 0.125; CSS
    Spearman 0.80; H-bonds 89% within +-1; salt bridges mean |diff| 0.13;
    disulfides 100% exact.

Refresh the picture with ``python examples/compare_vs_pisa.py``.
"""
import os

import pytest

from fastpisa.reference.compare import BENCHMARK_ENTRIES, compare_entries, summarize
from fastpisa.reference.ebi_pisa import REFERENCE_DIR, cached_pdb_path, load_cached_reference

_have_reference = all(
    load_cached_reference(pid) is not None and cached_pdb_path(pid) is not None
    for pid in BENCHMARK_ENTRIES
) if os.path.isdir(REFERENCE_DIR) else False

pytestmark = pytest.mark.skipif(
    not _have_reference, reason="reference benchmark cache not present")


@pytest.fixture(scope="module")
def benchmark():
    res = compare_entries(BENCHMARK_ENTRIES, mode="pisa")
    return res, summarize(res["rows"])


def test_identity_interfaces_are_found(benchmark):
    res, _ = benchmark
    n_ref = sum(len(e["rows"]) + len(e["ref_only"]) for e in res["entries"])
    n_matched = sum(len(e["rows"]) for e in res["entries"])
    assert n_matched >= 110
    assert n_matched / n_ref >= 0.97, (
        f"matched {n_matched}/{n_ref} identity interfaces; unmatched: "
        f"{[(e['pdb_id'], e['ref_only']) for e in res['entries'] if e['ref_only']]}")


def test_interface_area_accuracy(benchmark):
    _, s = benchmark
    assert s["area_median_rel_err"] < 0.03
    assert s["area_median_rel_err_big"] < 0.03


def test_solvation_energy_accuracy(benchmark):
    _, s = benchmark
    assert s["dg_pearson"] > 0.90
    assert s["dg_median_abs_err"] < 1.5


def test_stabilization_energy_accuracy(benchmark):
    _, s = benchmark
    assert s["stab_pearson"] > 0.90
    assert s["stab_median_abs_err"] < 1.6


def test_p_value_accuracy(benchmark):
    _, s = benchmark
    assert s["pv_median_abs_err"] < 0.18
    assert s["pv_spearman"] > 0.55


def test_css_rank_agreement(benchmark):
    _, s = benchmark
    assert s["css_spearman"] > 0.65


def test_bond_count_accuracy(benchmark):
    _, s = benchmark
    assert s["hb_mean_abs_diff"] < 1.3
    assert s["hb_within_1"] > 0.80
    assert s["sb_mean_abs_diff"] < 0.4
    assert s["ss_exact"] == 1.0
