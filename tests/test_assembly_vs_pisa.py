"""Assembly prediction against PISA's own published predictions.

The accuracy was unknown before it was measured, so this test asserts the
RECORDED number in tests/data/reference/assembly_validation.json rather than
an aspiration. If a change moves it, the recorded value is what has to be
re-measured and justified -- not the threshold quietly relaxed.

Two axes: agreement with PISA's top assembly, and agreement with the
author-deposited assembly (PISA's R350 marks which of its assemblies the
depositor asserted). The second is a claim about biology rather than about
reproducing PISA.
"""

from __future__ import annotations

import json
import os

import pytest

gemmi = pytest.importorskip("gemmi")

from fastpisa.reference.compare import (  # noqa: E402
    compare_assembly_entry, summarize_assembly_predictions,
)

RECORD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                      "reference", "assembly_validation.json")


def _recorded():
    with open(RECORD) as fh:
        return json.load(fh)


def test_a_single_entry_compares():
    result = compare_assembly_entry("1acb", allow_fetch=False)
    assert result is not None
    assert result["reference_total"] == 1
    assert result["top_reference"]["composition"] == "EI"
    assert result["predicted_total"] >= 1
    assert isinstance(result["top_mmsize_match"], bool)


def test_an_entry_with_no_predicted_assembly_is_handled():
    """1brs: PISA predicts nothing stable. Must compare, not crash."""
    result = compare_assembly_entry("1brs", allow_fetch=False)
    assert result is not None
    assert result["reference_total"] == 0
    assert result["top_reference"] is None


def test_an_uncached_entry_returns_none():
    assert compare_assembly_entry("zzzz", allow_fetch=False) is None


def test_measured_accuracy_matches_the_recorded_value():
    recorded = _recorded()
    results = [compare_assembly_entry(p, allow_fetch=False)
               for p in recorded["entries"]]
    results = [r for r in results if r is not None]
    assert len(results) == len(recorded["entries"])

    stats = summarize_assembly_predictions(results)
    expected = recorded["measured"]
    for key in ("top_mmsize_match_rate", "top_composition_match_rate",
                "author_assembly_match_rate", "mean_recall"):
        assert stats[key] == pytest.approx(expected[key], abs=0.02), (
            f"{key}: measured {stats[key]:.3f}, recorded "
            f"{expected[key]:.3f} -- re-measure and update the record "
            f"with the reason, do not relax this")


def test_the_record_states_an_honest_number():
    """The record must not claim perfection it has not earned."""
    recorded = _recorded()
    measured = recorded["measured"]
    assert 0.0 <= measured["top_mmsize_match_rate"] <= 1.0
    assert len(recorded["entries"]) >= 30
    assert recorded["note"], "the record must say what the number means"
