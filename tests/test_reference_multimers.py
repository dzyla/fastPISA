"""PISA's own assembly predictions, the ground truth for assembly validation.

Published per entry at multimers.pisa (verified 2026-10-04). Two fields make
this more than a self-comparison: ``r350`` is non-zero when PISA's assembly
matches the author-deposited REMARK-350 assembly, and ``score`` is PISA's
text verdict. ``total_asm == 0`` is a legitimate answer -- PISA finds no
stable assembly for barnase-barstar (1brs) -- and must not look like a
fetch failure.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastpisa.reference.ebi_pisa import load_cached_multimers, parse_pisa_multimers

MINIMAL = """<pisa_multimers>
  <status>Ok</status>
  <pdb_entry>
    <pdb_code>1acb</pdb_code>
    <status>Ok</status>
    <total_asm>1</total_asm>
    <asm_set>
      <ser_no>1</ser_no>
      <assembly>
        <id>1</id><size>2</size><mmsize>2</mmsize>
        <score>This assembly appears to be stable in solution.</score>
        <diss_energy>5.5696952172</diss_energy>
        <entropy>10.679620819</entropy>
        <diss_area>770.846279</diss_area>
        <int_energy>-12.252981565</int_energy>
        <n_diss>2</n_diss><symNumber>1</symNumber><R350>1</R350>
        <formula>AB</formula><composition>EI</composition>
        <molecule>
          <chain_id>E</chain_id>
          <rxx>1</rxx><rxy>0</rxy><rxz>0</rxz><tx>0</tx>
          <ryx>0</ryx><ryy>1</ryy><ryz>0</ryz><ty>0</ty>
          <rzx>0</rzx><rzy>0</rzy><rzz>1</rzz><tz>0</tz>
        </molecule>
      </assembly>
    </asm_set>
  </pdb_entry>
</pisa_multimers>"""

EMPTY = """<pisa_multimers>
  <status>Ok</status>
  <pdb_entry>
    <pdb_code>1brs</pdb_code><status>Ok</status><total_asm>0</total_asm>
  </pdb_entry>
</pisa_multimers>"""


def test_parses_one_assembly():
    doc = parse_pisa_multimers(MINIMAL)
    assert doc["pdb_id"] == "1acb"
    assert doc["total_asm"] == 1
    assert len(doc["assemblies"]) == 1
    asm = doc["assemblies"][0]
    assert asm["set_no"] == 1 and asm["id"] == 1
    assert asm["size"] == 2 and asm["mmsize"] == 2
    assert asm["formula"] == "AB" and asm["composition"] == "EI"
    assert asm["diss_energy"] == pytest.approx(5.5696952172)
    assert asm["entropy"] == pytest.approx(10.679620819)
    assert asm["symmetry_number"] == 1
    assert asm["r350"] == 1
    assert "stable in solution" in asm["score"]
    assert np.allclose(asm["molecules"][0]["rotation"], np.eye(3))
    assert asm["molecules"][0]["chain_id"] == "E"


def test_no_predicted_assembly_is_a_valid_answer():
    """1brs: PISA predicts nothing stable. Zero must round-trip."""
    doc = parse_pisa_multimers(EMPTY)
    assert doc["total_asm"] == 0
    assert doc["assemblies"] == []


def test_cached_reference_is_available_offline():
    doc = load_cached_multimers("1acb")
    assert doc is not None, "commit tests/data/reference/1acb.multimers.xml.gz"
    assert doc["total_asm"] >= 1
    assert doc["assemblies"][0]["composition"]


def test_a_missing_entry_returns_none():
    assert load_cached_multimers("zzzz") is None


def test_every_cached_entry_parses():
    import glob
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    paths = glob.glob(os.path.join(here, "data", "reference", "*.multimers.xml.gz"))
    assert len(paths) >= 30, f"only {len(paths)} cached multimer references"
    for path in paths:
        pdb_id = os.path.basename(path)[:4]
        doc = load_cached_multimers(pdb_id)
        assert doc is not None and doc["pdb_id"] == pdb_id
        assert len(doc["assemblies"]) >= 0
        for asm in doc["assemblies"]:
            assert asm["size"] >= 1
            assert asm["mmsize"] <= asm["size"]
            assert asm["molecules"], f"{pdb_id} assembly {asm['id']} has no molecules"
