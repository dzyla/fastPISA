"""App-layer components that carry no Streamlit calls: figures, the Mol*
MolViewSpec builder, the comparison engine, interpretation, and (when
pdb_align is installed) chain detection / superposition."""
import json
import os
import sys

import pytest

import fastpisa
from fastpisa.report import ComplexEntry, compare, group_interface, interpret
from fastpisa.reference.ebi_pisa import cached_pdb_path

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")
sys.path.insert(0, APP)

_BRS = cached_pdb_path("1brs")
pytestmark = pytest.mark.skipif(_BRS is None, reason="1brs not cached")


@pytest.fixture(scope="module")
def res():
    return fastpisa.analyze(_BRS, pdb_id="1brs")


@pytest.fixture(scope="module")
def gi(res):
    return group_interface(res, ["A"], ["D"], "barnase", "barstar")


@pytest.fixture(scope="module")
def cmp(res, gi):
    g2 = group_interface(res, ["B"], ["E"], "barnase", "barstar")
    return compare([ComplexEntry("A:D", gi, res), ComplexEntry("B:E", g2, res)])


def test_interpretation_flags(gi):
    flags = interpret(gi)
    levels = {f["level"] for f in flags}
    assert levels <= {"info", "note", "warning"}
    text = " ".join(f["text"] for f in flags)
    assert "779" in text and "hot-spot" in text and "Arg59" in text
    from fastpisa.report import GroupInterface
    empty = GroupInterface("x", "y", ["A"], ["F"])
    assert interpret(empty)[0]["level"] == "warning"


def test_figures_render(res, gi, cmp):
    import figures as F
    from fastpisa.report import chain_residue_axis
    axis = {"A": chain_residue_axis(res, ["A"])}
    assert len(axis["A"]) == 108 and axis["A"][0][3] in "ACDEFGHIKLMNPQRSTVWY"   # barnase A, polymer only
    pair_maps = F.contact_maps_per_pair(gi)
    assert len(pair_maps) == 1 and pair_maps[0][0] == "A + D"
    full = F.full_contact_map(gi, chain_residue_axis(res, ["A"]), chain_residue_axis(res, ["D"]))
    for fig in (F.footprint(gi, 1, axis), F.footprint(gi, 2), F.residue_bars(gi), F.composition(gi),
                F.bond_network(gi), pair_maps[0][1], full, F.compare_bars(cmp),
                F.compare_footprints(cmp, 1), F.compare_heatmap(cmp, 1)):
        png = F.fig_bytes(fig, "png", dpi=50)
        assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 1000
        assert b"<svg" in F.fig_bytes(fig, "svg")[:300]


def test_mvs_document_is_well_formed(gi, res):
    from molstar_view import build_mvs, interface_view_html, comparison_view_html, MOLSTAR_JS
    import gzip
    txt = gzip.open(_BRS, "rt").read()
    mvs = build_mvs(txt, "pdb", gi, show_surface=True, show_bonds=True, show_labels=True, other_chains=["B", "C"])
    assert mvs["root"]["kind"] == "root"
    dl = mvs["root"]["children"][0]
    assert dl["kind"] == "download" and dl["params"]["url"].startswith("data:chemical/x-pdb;base64,")
    structure = dl["children"][0]["children"][0]
    kinds = [c["kind"] for c in structure["children"]]
    assert kinds.count("component") >= 5 and "primitives" in kinds
    prims = [c for c in structure["children"] if c["kind"] == "primitives"][0]["children"]
    assert len(prims) == len(gi.bonds())
    assert all(p["params"]["kind"] == "distance_measurement" for p in prims)
    # opacity lives in its own node, never as a representation param
    for c in structure["children"]:
        for r in c.get("children", []):
            if r.get("kind") == "representation":
                assert "opacity" not in r["params"]
    html = interface_view_html(txt, "pdb", gi, other_chains=["B"])
    assert MOLSTAR_JS in html and "loadMvsData" in html
    json.loads(html.split("const mvs = ", 1)[1].split(";\n", 1)[0])       # embedded JSON parses
    g2 = group_interface(res, ["B"], ["E"], "barnase", "barstar")
    html2 = comparison_view_html([{"name": "a", "text": txt, "fmt": "pdb", "gi": gi, "color": "#000"},
                                  {"name": "b", "text": txt, "fmt": "pdb", "gi": g2, "color": "#f00"}])
    assert html2.count('"kind": "download"') == 2


def test_comparison_engine(cmp):
    st = cmp.summary_table()
    assert list(st["complex"]) == ["A:D", "B:E"] and st["H-bonds"].iloc[0] == 15
    ov = cmp.overlap_table()
    assert len(ov) == 1 and ov["Jaccard"].iloc[0] >= 0.9          # B/E are copies of A/D
    mat = cmp.residue_matrix(1)
    assert mat.shape[1] == 2 and (mat > 0).any().all()
    assert all(k.startswith("A:") for k in mat.index)             # mapped onto the reference chain
    prose = cmp.prose()
    assert "Relative to A:D" in prose and "Jaccard" in prose
    with pytest.raises(ValueError):
        compare(cmp.entries[:1])


def test_sequence_alignment_maps_renumbered_chains(res, gi):
    """Residue matching by sequence when numbering differs (forced)."""
    g2 = group_interface(res, ["B"], ["E"], "barnase", "barstar")
    c = compare([ComplexEntry("ref", gi, res), ComplexEntry("mob", g2, res)], align="sequence")
    mat = c.residue_matrix(1)
    assert set(mat.index) and all(k.startswith("A:") for k in mat.index)
    assert c.overlap_table()["shared"].iloc[0] >= 20


pdb_align = pytest.importorskip("pdb_align")


def test_chain_detection_and_superposition():
    from alignment import detect_shared_chains, superpose
    m = detect_shared_chains(_BRS, _BRS, ["A"], ["B", "C", "D", "E", "F"])
    assert len(m) == 1 and m[0].ref_chain == "A" and m[0].mob_chain in ("B", "C") and m[0].identity > 95
    sp = superpose(_BRS, _BRS, ["A"], ["B"])
    assert sp.rmsd < 1.0 and sp.n_aligned > 100 and "ATOM" in sp.aligned_text


def test_proximity_and_render_scripts(res):
    from fastpisa.report import chimerax_render_script, proximity_flags, pymol_render_script
    gi = group_interface(res, ["A", "B"], ["D", "E"], "barnase", "barstar")
    prox = gi.proximity_table()
    assert set(prox["molecule 1"]) == {"A", "B"} and len(prox) == 4
    touching = prox[prox["has_interface"]]
    assert {tuple(x) for x in touching[["molecule 1", "molecule 2"]].values} == {("A", "D"), ("B", "E")}
    far = prox[~prox["has_interface"]]
    assert (far["min_distance"] > 5).all()                    # A+E / B+D are not omissions: they do not touch
    flags = proximity_flags(gi.pair_proximity, "barnase", "barstar")
    assert flags and "2 of 4" in flags[0]["text"]
    none = group_interface(res, ["A"], ["F"])
    assert proximity_flags(none.pair_proximity, "a", "b")[0]["level"] == "warning"
    cx = chimerax_render_script(gi)
    assert "color side1 & C #E69F00" in cx and "distance #1/" in cx and "lighting soft" in cx
    pm = pymol_render_script(gi)
    assert "util.cnc('side1')" in pm and "distance bond0" in pm
