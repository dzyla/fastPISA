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


def test_the_record_states_its_denominators():
    """I5: the measured rates drop three entries; the record must say so.

    1brs, 1ay7 and 1gpw have PISA ``total_asm = 0``, so they have no top
    reference assembly and leave the match-rate denominator -- yet those are
    our clearest FALSE POSITIVES (we predict 9, 1 and 16 assemblies for
    them). A rate over 34 entries described as "over the 37 cached entries"
    hides exactly the cases where we are most wrong.
    """
    recorded = _recorded()
    assert recorded["n_entries"] == 37
    assert recorded["n_comparable"] == 34
    assert recorded["n_with_author_assembly"] >= 20
    assert recorded["n_comparable"] < recorded["n_entries"]


def test_the_record_scores_the_entries_pisa_calls_empty():
    """I5: an entry where PISA predicts nothing must cost something.

    Otherwise predicting a stable assembly for barnase-barstar is free.
    """
    recorded = _recorded()
    empty = recorded["measured"]["no_reference_assembly"]
    assert empty["n_entries"] >= 3
    assert 0.0 <= empty["we_predict_nothing_stable_rate"] <= 1.0
    results = [compare_assembly_entry(p, allow_fetch=False)
               for p in recorded["entries"]]
    results = [r for r in results if r is not None]
    stats = summarize_assembly_predictions(results)
    assert stats["no_reference_assembly"]["n_entries"] == empty["n_entries"]
    assert stats["no_reference_assembly"][
        "we_predict_nothing_stable_rate"] == pytest.approx(
        empty["we_predict_nothing_stable_rate"], abs=0.02)


def test_recall_is_reported_with_a_precision_counterpart():
    """I6: recall alone rises mechanically with the number of assemblies emitted.

    For 1urn we emit 11 against PISA's 25 sets, for 1brs 9 against 0. The
    README's headline reading ("usually generated, merely not ranked first")
    rests on recall, so precision has to sit beside it.
    """
    recorded = _recorded()
    assert "mean_precision" in recorded["measured"]
    assert 0.0 <= recorded["measured"]["mean_precision"] <= 1.0
    results = [compare_assembly_entry(p, allow_fetch=False)
               for p in recorded["entries"]]
    results = [r for r in results if r is not None]
    stats = summarize_assembly_predictions(results)
    assert stats["mean_precision"] == pytest.approx(
        recorded["measured"]["mean_precision"], abs=0.02)


def test_recall_excludes_ligand_only_reference_assemblies():
    """I6: PISA reports [ZN] as an assembly; our filter can never match it.

    Seven entries have an mmsize=0 reference assembly. Dropping ligand-only
    components is a deliberate deviation, so those sizes leave the recall
    denominator rather than silently capping it.
    """
    result = compare_assembly_entry("4ins", allow_fetch=False)
    assert result is not None
    sizes = {a["mmsize_ref"] for a in result["rows"]}
    assert 0 in sizes, "4ins must have a ligand-only reference assembly"
    assert result["recall"] == pytest.approx(result["recall_detail"]["recall"])
    assert 0 not in result["recall_detail"]["reference_sizes"]


def test_the_record_states_an_honest_number():
    """The record must not claim perfection it has not earned."""
    recorded = _recorded()
    measured = recorded["measured"]
    assert 0.0 <= measured["top_mmsize_match_rate"] <= 1.0
    assert len(recorded["entries"]) >= 30
    assert recorded["note"], "the record must say what the number means"
