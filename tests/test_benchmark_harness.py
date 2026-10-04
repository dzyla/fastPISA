"""The parts of the large benchmark that can silently corrupt a 3-hour run.

Three failure modes, none of which show up as a crash:

* **A non-blind draw.** If the exclusion list misses the calibration
  entries, the headline number is measured on entries that set the
  constants, and it reads high for the wrong reason.
* **A broken resume.** If finished entries are not recognised, a restart
  re-fetches and re-analyses everything; if a torn final line from a killed
  run aborts the read, the whole record is lost.
* **Lossy compaction.** Only four numbers per interface are stored, and the
  global statistics are recomputed from them, so the round trip has to be
  exact.
"""

from __future__ import annotations

import importlib.util
import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _harness():
    path = os.path.join(REPO, "examples", "benchmark_vs_pisa.py")
    spec = importlib.util.spec_from_file_location("_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_draw_excludes_every_calibration_entry():
    """Otherwise the benchmark is not blind."""
    bench = _harness()
    excluded = bench._calibration_entries()
    with open(os.path.join(REPO, "tests", "data", "calibration",
                           "entries.json")) as fh:
        record = json.load(fh)
    for pdb_id in record["entries"]:
        assert pdb_id.lower() in excluded, pdb_id
    for pdb_id in record["legacy_benchmark"]:
        assert pdb_id.lower() in excluded, pdb_id
    # The 400-entry draw plus 36 legacy plus the skipped pair plus the
    # earlier blind draw: several hundred, not a handful.
    assert len(excluded) > 400


def test_skipped_entries_are_id_reason_pairs_not_bare_ids():
    """The shape that broke the first run: [["1a21", "no interfaces"], ...]."""
    bench = _harness()
    excluded = bench._calibration_entries()
    assert "1a21" in excluded and "1bxy" in excluded
    # The reason strings must not have been mistaken for entry ids.
    assert not any(" " in e for e in excluded)


def test_the_benchmark_seed_differs_from_every_fitted_draw():
    bench = _harness()
    with open(os.path.join(REPO, "tests", "data", "calibration",
                           "entries.json")) as fh:
        assert bench.BENCHMARK_SEED != json.load(fh)["seed"]


def test_resume_reads_finished_entries(tmp_path):
    bench = _harness()
    path = tmp_path / "results.jsonl"
    path.write_text('{"pdb_id": "1abc", "n_reference": 3}\n'
                    '{"pdb_id": "2def", "n_reference": 4}\n')
    done = bench._done(str(path))
    assert set(done) == {"1abc", "2def"}
    assert done["2def"]["n_reference"] == 4


def test_a_torn_final_line_does_not_lose_the_record(tmp_path):
    """A run killed mid-write leaves a partial last line."""
    bench = _harness()
    path = tmp_path / "results.jsonl"
    path.write_text('{"pdb_id": "1abc", "n_reference": 3}\n'
                    '{"pdb_id": "2def", "n_ref')
    done = bench._done(str(path))
    assert set(done) == {"1abc"}, "the intact entries must survive"


def test_a_missing_record_resumes_from_scratch(tmp_path):
    bench = _harness()
    assert bench._done(str(tmp_path / "nothing.jsonl")) == {}


def test_compaction_round_trips_the_statistics():
    """Global stats are recomputed from the four stored numbers."""
    pytest.importorskip("gemmi")
    from fastpisa.reference.compare import (
        compare_crystal_entry, summarize_crystal,
    )
    bench = _harness()

    full = compare_crystal_entry("1acb", allow_fetch=False)
    assert full is not None
    direct = summarize_crystal([full])
    restored = summarize_crystal([bench._expand(bench._compact(full))])
    for key in ("match_rate", "area_median_rel_err", "dg_pearson",
                "dg_median_abs_err"):
        assert restored[key] == pytest.approx(direct[key], abs=5e-3), key


def test_the_summary_reports_per_entry_agreement_not_only_the_pooled_rate():
    """A pooled rate hides which entries failed; the worst must be named."""
    pytest.importorskip("gemmi")
    from fastpisa.reference.compare import compare_crystal_entry
    bench = _harness()

    records = [bench._compact(compare_crystal_entry(p, allow_fetch=False))
               for p in ("1acb", "1brs")]
    stats = bench.summarise(records)
    assert "entries_with_every_interface_matched" in stats
    assert "worst_entries" in stats
    assert stats["n_entries"] == 2


def test_an_entry_with_no_reference_is_recorded_not_dropped():
    """An unavailable entry must be visible in the denominator discussion."""
    bench = _harness()
    stats = bench.summarise([
        {"pdb_id": "1abc", "n_reference": 0, "n_matched": 0,
         "n_reported": 0, "rows": [], "error": "no reference data"},
    ])
    assert stats["n_entries"] == 0
    assert stats["n_no_reference"] == 1
