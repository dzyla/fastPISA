"""PISA-fidelity regressions added with the residue-level calibration.

* atom-level H-bond / salt-bridge agreement with PISA's own bond lists
  (pair-by-pair precision / recall, not just counts) on the committed
  36-entry reference set;
* the hydrophobic / polar split of the solvation energy;
* negative residue numbers survive parsing (they used to collapse onto 0,
  which silently broke residue matching for DNA numbered about a centre
  and for expression tags).
"""
import os

import pytest

from fastpisa.reference.compare import BENCHMARK_ENTRIES
from fastpisa.reference.ebi_pisa import (
    REFERENCE_DIR, cached_pdb_path, load_cached_reference,
)
from fastpisa.surface.freesasa_backend import available as _freesasa_available

_have_reference = all(
    load_cached_reference(pid) is not None and cached_pdb_path(pid) is not None
    for pid in BENCHMARK_ENTRIES
) if os.path.isdir(REFERENCE_DIR) else False

needs_reference = pytest.mark.skipif(
    not (_have_reference and _freesasa_available()),
    reason="needs the cached 36-entry reference set and the FreeSASA backend")


@pytest.fixture(scope="module")
def bond_audit():
    from fastpisa.reference.bonds_audit import audit_entry

    recs = []
    for pid in BENCHMARK_ENTRIES:
        recs.extend(audit_entry(pid))
    return recs


@needs_reference
def test_hbond_pairs_match_pisa(bond_audit):
    """Measured 2026-09-01 on the full 674-entry set (polymer pairs):
    precision 0.958, recall 0.952. The 36-entry set is held to a floor
    just under that."""
    from fastpisa.reference.bonds_audit import summarize

    s = summarize(bond_audit, "hb", polymer_only=True)
    assert s["n_ref"] > 500
    assert s["precision"] > 0.92
    assert s["recall"] > 0.92


@needs_reference
def test_salt_bridge_pairs_match_pisa(bond_audit):
    from fastpisa.reference.bonds_audit import summarize

    s = summarize(bond_audit, "sb", polymer_only=True)
    assert s["n_ref"] > 100
    # full 674-entry set: precision 0.985 / recall 0.979; the 36-entry
    # subset measured 0.98 / 0.96
    assert s["precision"] > 0.95
    assert s["recall"] > 0.94


def test_solvation_energy_splits_into_apolar_and_polar():
    from fastpisa.core import run_core

    st = run_core(os.path.join(os.path.dirname(__file__), "data", "1ktz.pdb"),
                  mode="pisa")
    assert st.interfaces
    for i in st.interfaces:
        assert i.solvation_energy_apolar + i.solvation_energy_polar == \
            pytest.approx(i.solvation_energy, abs=0.02)
        # burying carbon is favourable; the polar side costs energy
        assert i.solvation_energy_apolar < 0
        assert i.solvation_energy_polar > 0


def test_apolar_polar_fields_are_written_to_json(tmp_path):
    from fastpisa.api import PISAInterfaceAnalyzer

    ana = PISAInterfaceAnalyzer(
        os.path.join(os.path.dirname(__file__), "data", "1ktz.pdb"), pdb_id="1ktz")
    ana.analyze()
    ana.write_json(str(tmp_path))
    import json
    path = next(tmp_path.glob("*-interfaces.json"))
    doc = json.load(open(path))

    def _find_interfaces(node):
        if isinstance(node, list) and node and isinstance(node[0], dict) \
                and "solvation_energy" in node[0]:
            return node
        if isinstance(node, dict):
            for v in node.values():
                hit = _find_interfaces(v)
                if hit:
                    return hit
        if isinstance(node, list):
            for v in node:
                hit = _find_interfaces(v)
                if hit:
                    return hit
        return None

    first = _find_interfaces(doc)[0]
    assert "solvation_energy_apolar" in first and "solvation_energy_polar" in first


def test_negative_residue_numbers_are_parsed(tmp_path):
    from fastpisa.parser.pdb_parser import parse_pdb

    lines = [
        "ATOM      1  N   GLY B  -1      10.000  10.000  10.000  1.00 20.00           N",
        "ATOM      2  CA  GLY B  -1      11.000  10.000  10.000  1.00 20.00           C",
        "ATOM      3  N   HIS B   0      12.000  10.000  10.000  1.00 20.00           N",
        "END",
    ]
    p = tmp_path / "neg.pdb"
    p.write_text("\n".join(lines) + "\n")
    st = parse_pdb(str(p))
    assert sorted({a.res_seq for a in st.atoms}) == [-1, 0]
