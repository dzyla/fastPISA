"""The crystal comparison harness itself, so the headline claim is re-runnable.

``compare_crystal_entry`` matches fastPISA's crystal-mode interfaces against
PISA's by chain pair AND relative crystal transform. Matching has to be
tolerant, not exact: PISA prints its per-molecule matrices to a few
decimals, so a 6-fold screw's c/6 = 42.55 A comes back as 42.6 and a
rounded key puts real interfaces on the boundary.
"""

from __future__ import annotations

import pytest

gemmi = pytest.importorskip("gemmi")

from fastpisa.reference.compare import (  # noqa: E402
    compare_crystal_entry, summarize_crystal,
)


def test_a_cached_entry_matches_every_pisa_interface():
    result = compare_crystal_entry("1acb")
    assert result is not None
    assert result["n_reference"] == 9
    assert result["n_matched"] == 9
    assert not result["missing"]
    assert result["rows"]
    for row in result["rows"]:
        assert row["area_ref"] > 0 and row["area_fp"] > 0
        assert set(row) >= {"pdb_id", "pair", "symop", "area_ref", "area_fp",
                            "dg_ref", "dg_fp"}


def test_polymer_only_filtering_excludes_hetero_groups():
    """Hetero-group names diverge from PISA after PDB remediation.

    1ppf's glycans moved from chain E (401-417) into chains A/B (1-8), so
    name-based matching of ligand interfaces is impossible for such entries
    and the polymer-polymer subset is the meaningful target.
    """
    everything = compare_crystal_entry("1ppf", polymer_only=False)
    polymer = compare_crystal_entry("1ppf", polymer_only=True)
    assert polymer["n_reference"] < everything["n_reference"]
    assert polymer["n_matched"] == polymer["n_reference"] == 8
    assert everything["n_matched"] < everything["n_reference"]


def test_summary_reports_match_rate_and_agreement():
    results = [compare_crystal_entry(p) for p in ("1acb", "1stf", "1ktz")]
    stats = summarize_crystal([r for r in results if r])
    assert stats["n_entries"] == 3
    assert stats["matched"] == stats["n_reference"]
    assert stats["match_rate"] == 1.0
    assert stats["area_median_rel_err"] < 0.05
    assert stats["dg_pearson"] > 0.95


def test_an_entry_without_cached_reference_data_returns_none():
    assert compare_crystal_entry("zzzz", allow_fetch=False) is None
