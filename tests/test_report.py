"""fastpisa.report: manuscript digests of the interface between chain groups."""
import os

import pytest

import fastpisa
from fastpisa.report import chain_inventory, group_interface, one_letter
from fastpisa.reference.ebi_pisa import cached_pdb_path

_BRS = cached_pdb_path("1brs")
pytestmark = pytest.mark.skipif(_BRS is None, reason="1brs not cached")


@pytest.fixture(scope="module")
def res():
    return fastpisa.analyze(_BRS, pdb_id="1brs")


def test_chain_inventory_lists_polymers(res):
    inv = chain_inventory(res)
    labels = {r["label"] for r in inv}
    assert {"A", "B", "C", "D", "E", "F"} <= labels
    prot = [r for r in inv if r["label"] == "A"][0]
    assert prot["class"] == "Protein" and prot["n_residues"] > 100


def test_single_pair_digest_matches_the_interface(res):
    gi = group_interface(res, ["A"], ["D"], "barnase", "barstar")
    assert [p.chains for p in gi.pairs] == [("A", "D")]
    p = gi.pairs[0]
    assert gi.interface_area == pytest.approx(p.interface_area, abs=0.01)
    # PISA convention: the two sides add up to twice the interface area
    assert gi.buried_total == pytest.approx(2 * p.interface_area, abs=0.5)
    assert gi.dg_solv == pytest.approx(p.solvation_energy, abs=0.01)
    assert gi.dg_apolar + gi.dg_polar == pytest.approx(gi.dg_solv, abs=0.05)
    assert gi.n_hbonds == p.number_hydrogen_bonds
    assert gi.n_salt_bridges == p.number_salt_bridges
    assert gi.n_hbonds >= 10 and gi.n_salt_bridges >= 5   # barnase-barstar is bond-rich


def test_group_digest_is_additive_over_pairs(res):
    gi = group_interface(res, ["A", "B", "C"], ["D", "E", "F"], "barnase", "barstar")
    pairs = {tuple(sorted(p.chains)) for p in gi.pairs}
    assert {("A", "D"), ("B", "E"), ("C", "F")} <= pairs
    assert gi.interface_area == pytest.approx(sum(p.interface_area for p in gi.pairs), abs=0.05)
    assert gi.n_hbonds == sum(p.number_hydrogen_bonds for p in gi.pairs)
    assert gi.stab_energy == pytest.approx(sum(p.stabilization_energy for p in gi.pairs), abs=0.05)
    # no chain in both groups
    with pytest.raises(ValueError):
        group_interface(res, ["A"], ["A", "D"])


def test_residue_lists_are_the_interface_residues(res):
    gi = group_interface(res, ["A"], ["D"], "barnase", "barstar")
    ep = gi.residues_side1
    assert ep and all(r.chain == "A" and r.bsa > 0 for r in ep)
    assert [int(r.seq) for r in ep] == sorted(int(r.seq) for r in ep)
    names = {f"{r.name}{r.seq}" for r in ep}
    assert {"ARG59", "HIS102", "ARG83"} <= names          # the known barnase hot spots
    total = sum(r.bsa for r in ep)
    assert total == pytest.approx(gi.buried_side1, abs=0.5)
    assert gi.residue_string(1).startswith(("K27", "Q", "R", "S", "A"))
    assert one_letter("ARG") == "R" and one_letter("DA") == "A" and one_letter("ZN") == "X"


def test_tables_and_prose(res):
    gi = group_interface(res, ["A"], ["D"], "barnase", "barstar")
    bt = gi.bonds_table()
    # each contact appears once with its dominant label; PISA's counts are
    # independent predicates (a charged H-bonded pair counts in both), so the
    # table is bounded by the counts, not equal to their sum
    assert gi.n_salt_bridges + gi.n_disulfides <= len(bt) <= gi.n_hbonds + gi.n_salt_bridges + gi.n_disulfides
    assert set(bt["type"]) <= {"hydrogen bond", "salt bridge", "disulfide"}
    assert (bt["chain 1"] == "A").all() and (bt["chain 2"] == "D").all()
    assert len(gi.pair_table()) == 1
    assert len(gi.residue_table(2)) == len(gi.residues_side2)
    cm = gi.contact_map_table()
    assert len(cm) == gi.n_residue_pairs and "barnase" in cm.columns
    text = gi.results_paragraph()
    assert f"{gi.buried_total:,.0f}" in text and "hydrogen bond" in text
    assert "barnase" in text and "barstar" in text
    assert "PISA" in gi.methods_paragraph() and "COCOMAPS" in gi.methods_paragraph()
    d = gi.to_dict()
    assert d["n_hbonds"] == gi.n_hbonds and d["pairs"] == ["A + D"]


def test_viewer_commands(res):
    gi = group_interface(res, ["A"], ["D"])
    cx = gi.chimerax_command()
    assert cx.startswith("name side1 /A:") and "name side2 /D:" in cx
    pm = gi.pymol_command()
    assert "select side1, (chain A and resi " in pm and "select side2, (chain D and resi " in pm


def test_empty_group_interface(res):
    gi = group_interface(res, ["A"], ["F"])       # barnase A does not touch barstar F
    assert gi.empty and gi.buried_total == 0 and "No interface" in gi.results_paragraph()
