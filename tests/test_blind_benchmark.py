"""The large blind benchmark against original PISA, checked from the record.

``examples/benchmark_vs_pisa.py --n 2000`` compares fastPISA's crystal mode
against original PISA's *own published output* over a fresh-seed draw from
the calibration sampling frame, excluding every entry that informed a fitted
constant. That run needs the network and ~2.3 h, so its per-entry outcome is
frozen, coordinate-free, in ``tests/data/reference/blind_benchmark.json.gz``
(~330 kB) and this module re-derives the published statistics from it in
milliseconds.

Two things are asserted, and they are different claims:

1. the record is INTERNALLY CONSISTENT -- recomputing the summary from the
   per-entry rows reproduces the summary stored beside them, so the numbers
   quoted in README.md and CLAUDE.md cannot drift away from the measurement
   they came from;
2. the draw was BLIND -- no entry in it appears in the calibration set, the
   36 legacy entries, or the earlier crystal/assembly validation sets.

Point 2 is what makes point 1 worth quoting. Re-measure with
``python examples/benchmark_vs_pisa.py --n 2000`` then ``--record``; do not
edit the record or relax the thresholds here.
"""

from __future__ import annotations

import gzip
import json
import os

import pytest

from fastpisa.reference.compare import summarize_crystal

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RECORD = os.path.join(DATA, "reference", "blind_benchmark.json.gz")


def _record():
    if not os.path.exists(RECORD):
        pytest.skip("blind benchmark record not present")
    with gzip.open(RECORD, "rt") as fh:
        return json.load(fh)


def _rows(entry):
    return [{"pdb_id": entry["pdb_id"], "area_ref": a, "area_fp": b,
             "dg_ref": c, "dg_fp": d} for a, b, c, d in entry["rows"]]


def _usable(record):
    return [dict(e, rows=_rows(e)) for e in record["entries"]
            if e.get("n_reference")]


def test_the_record_covers_the_whole_draw():
    record = _record()
    assert record["n_drawn"] == 2000
    assert len(record["entries"]) == 2000
    usable = _usable(record)
    # PISA's frozen database has no data for 2 of the 2000. They are kept in
    # the record and counted, not dropped: a run where every fetch failed
    # must read as a failure, not as an empty success.
    assert len(usable) == 1998
    assert record["measured"]["n_no_reference"] == 2000 - len(usable)


def test_recomputing_the_summary_reproduces_the_recorded_one():
    """The published numbers come from the per-entry rows, not from prose."""
    record = _record()
    stats = summarize_crystal(_usable(record))
    recorded = record["measured"]
    for key in ("n_entries", "n_reference", "matched", "reported"):
        assert stats[key] == recorded[key]
    for key in ("match_rate", "area_median_rel_err", "dg_pearson",
                "dg_median_abs_err"):
        assert stats[key] == pytest.approx(recorded[key], rel=1e-9)


def test_the_measured_accuracy_is_what_the_readme_claims():
    """README.md and CLAUDE.md quote these; they are measured, not targets."""
    record = _record()
    stats = summarize_crystal(_usable(record))
    assert stats["n_entries"] == 1998
    assert stats["n_reference"] == 26183
    assert stats["matched"] == 25985
    assert stats["match_rate"] == pytest.approx(0.9924, abs=0.0005)
    assert stats["area_median_rel_err"] == pytest.approx(0.0120, abs=0.0005)
    assert stats["dg_pearson"] == pytest.approx(0.9927, abs=0.0005)
    assert stats["dg_median_abs_err"] == pytest.approx(0.151, abs=0.005)


def test_the_draw_was_blind():
    """Nothing in it informed a constant -- the whole point of the number.

    An in-sample entry here would make 99.2% a statement about fitting, not
    about generalisation, so the exclusion is asserted rather than trusted
    to the draw-time ``exclude`` argument.
    """
    record = _record()
    assert record["draw"]["seed"] == 20261004
    assert record["draw"]["n_excluded_in_sample"] >= 436   # 400 + 36 legacy

    in_sample = set()
    with open(os.path.join(DATA, "calibration", "entries.json")) as fh:
        calibration = json.load(fh)
    for key in ("entries", "legacy_benchmark", "skipped"):
        for item in calibration.get(key) or ():
            if isinstance(item, (list, tuple)):
                item = item[0]
            if isinstance(item, str):
                in_sample.add(item.lower())
    assert len(in_sample) >= 400, "calibration entry list did not load"
    for name in ("crystal_validation.json", "assembly_validation.json"):
        path = os.path.join(DATA, "reference", name)
        if os.path.exists(path):
            with open(path) as fh:
                in_sample.update(e.lower()
                                 for e in json.load(fh).get("entries", []))

    drawn = {e["pdb_id"].lower() for e in record["entries"]}
    assert not drawn & in_sample


def test_the_record_carries_no_coordinates():
    """It ships in the repository; it must stay a measurement, not a cache."""
    assert os.path.getsize(RECORD) < 2 * 1024 * 1024
    record = _record()
    entry = record["entries"][0]
    assert set(entry) <= {"pdb_id", "n_reference", "n_matched", "n_reported",
                          "missing", "rows", "seconds", "error"}
    assert all(len(row) == 4 for row in entry["rows"])
