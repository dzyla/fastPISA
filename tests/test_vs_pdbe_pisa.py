"""Accuracy regression vs the ORIGINAL PISA engine (EBI PDBe PISA service).

Runs fully offline against the reference data cached in
tests/data/reference/ (EBI PISA XML + RCSB PDB files for 36 entries,
262 matched identity interfaces).

SCOPE, read this before quoting any number here: these 36 entries are
hand-picked AND their interfaces are part of the set the shipped constants
were fitted on, so this file measures **in-sample** agreement. It is a
regression test -- it catches a pipeline that has broken -- not a measure of
how fastPISA does on a structure it has never seen. For that, see
``tests/test_calibration_benchmark.py``, which asserts grouped
cross-validated accuracy over 674 entries / 6881 interfaces.

Values measured after the 2026-09-01 recalibration:

    area 2.2% (1.3% >300 A^2); dG Pearson 0.938 (median 1.34 kcal/mol);
    stab Pearson 0.974; P-value median |err| 0.104; H-bonds 91% within +-1;
    salt bridges mean |diff| 0.08; disulfides 100% exact; CSS Spearman 0.67.
    Polymer-polymer subset (the cryo-EM / predicted-model regime, n=153):
    area 1.5%, dG Pearson 0.969 (median 1.03 kcal/mol), stab 0.990, P-value
    median |err| 0.075 (Spearman 0.79), CSS Spearman 0.70.

Against the previous constants the overall dG Pearson on THIS set fell
(0.956 -> 0.938) while the median error on polymer pairs improved
(1.28 -> 1.03 kcal/mol) and the P-value improved sharply (0.109 -> 0.075).
That trade is expected and wanted: the old constants were fitted on these
262 interfaces alone, so their correlation here was partly memorised. The
new constants are fitted on 26x more, unbiasedly sampled data and are
better out of sample -- which is what the companion test measures.

P-values of small-ligand/ion interfaces follow different PISA statistics
and are not asserted. Refresh with ``python examples/compare_vs_pisa.py``.
"""
import os

import pytest

from fastpisa.reference.compare import BENCHMARK_ENTRIES, compare_entries, summarize
from fastpisa.reference.ebi_pisa import REFERENCE_DIR, cached_pdb_path, load_cached_reference

_have_reference = all(
    load_cached_reference(pid) is not None and cached_pdb_path(pid) is not None
    for pid in BENCHMARK_ENTRIES
) if os.path.isdir(REFERENCE_DIR) else False

from fastpisa.surface.freesasa_backend import available as _freesasa_available

pytestmark = [
    pytest.mark.skipif(not _have_reference,
                       reason="reference benchmark cache not present"),
    pytest.mark.skipif(not _freesasa_available(),
                       reason="21-entry benchmark needs the FreeSASA C "
                              "backend (pure-Python ASA is too slow here)"),
]


@pytest.fixture(scope="module")
def benchmark():
    res = compare_entries(BENCHMARK_ENTRIES, mode="pisa")
    return res, summarize(res["rows"])


def test_identity_interfaces_are_found(benchmark):
    res, _ = benchmark
    n_ref = sum(len(e["rows"]) + len(e["ref_only"]) for e in res["entries"])
    n_matched = sum(len(e["rows"]) for e in res["entries"])
    assert n_matched >= 255
    # Most unmatched pairs are wwPDB remediation renamings (glycans / HEM->HEC
    # / re-chained PO4) where PISA's cached run predates the current file.
    assert n_matched / n_ref >= 0.88, (
        f"matched {n_matched}/{n_ref} identity interfaces; unmatched: "
        f"{[(e['pdb_id'], e['ref_only']) for e in res['entries'] if e['ref_only']]}")


def test_interface_area_accuracy(benchmark):
    _, s = benchmark
    assert s["area_median_rel_err"] < 0.04
    assert s["area_median_rel_err_big"] < 0.03
    assert s["poly_area_median_rel_err"] < 0.025


def test_solvation_energy_accuracy(benchmark):
    _, s = benchmark
    assert s["dg_pearson"] > 0.92
    assert s["dg_median_abs_err"] < 1.5
    assert s["poly_dg_pearson"] > 0.95


def test_stabilization_energy_accuracy(benchmark):
    _, s = benchmark
    assert s["stab_pearson"] > 0.94
    assert s["stab_median_abs_err"] < 1.8
    assert s["poly_stab_pearson"] > 0.96


def test_p_value_accuracy(benchmark):
    _, s = benchmark
    assert s["pv_median_abs_err"] < 0.13
    # Ligand/ion p-values follow different PISA statistics; the rank
    # agreement is asserted on the polymer-polymer regime.
    assert s["poly_pv_spearman"] > 0.72


def test_css_rank_agreement(benchmark):
    _, s = benchmark
    assert s["css_spearman"] > 0.60
    assert s["poly_css_spearman"] > 0.65


def test_bond_count_accuracy(benchmark):
    _, s = benchmark
    assert s["hb_mean_abs_diff"] < 1.0
    assert s["hb_within_1"] > 0.85
    assert s["sb_mean_abs_diff"] < 0.3
    assert s["ss_exact"] == 1.0
