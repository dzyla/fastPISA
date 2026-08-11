"""Tests for AlphaFold confidence filtering (4.4), visualisation (4.3) and the
batch module (4.1)."""
import importlib.util
import json
import os
import subprocess
import sys

import pytest

from fastpisa.api import PISAInterfaceAnalyzer
from fastpisa.pae import build_pae_index_map, interface_pae_score, load_pae

from conftest import KTZ, REPO_ROOT

_HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def _make_pae_json(structure, value=5.0, iptm=None, tmp_path=None):
    n = max(build_pae_index_map(structure).values()) + 1
    matrix = [[value] * n for _ in range(n)]
    data = {"predicted_aligned_error": matrix, "max_predicted_aligned_error": value}
    if iptm is not None:
        data["iptm"] = iptm
    path = os.path.join(str(tmp_path), "pae.json")
    with open(path, "w") as fh:
        json.dump(data, fh)
    return path, n


class TestPaeFiltering:
    def test_load_pae_and_scores(self, tmp_path):
        ana = PISAInterfaceAnalyzer(KTZ, pdb_id="1ktz", mode="pisa")
        ana.analyze()
        pae_path, n = _make_pae_json(ana._parsed_structure(), value=5.0, iptm=0.9,
                                     tmp_path=tmp_path)
        ana.load_pae(pae_path)
        assert ana.pae_data.iptm == 0.9
        assert ana.pae_data.n_residues == n > 0
        scores = ana.pae_scores()
        assert len(scores) == 1
        assert scores[1] == pytest.approx(5.0)

    def test_filter_by_pae(self, tmp_path):
        ana = PISAInterfaceAnalyzer(KTZ, pdb_id="1ktz", mode="pisa")
        ana.load_pae(
            _make_pae_json(ana._parsed_structure(), value=5.0, iptm=0.9,
                           tmp_path=tmp_path)[0])
        assert len(ana.filter_by_pae(max_pae=5.0)) == 1   # <= 5 kept
        assert len(ana.filter_by_pae(max_pae=0.1)) == 0   # all dropped

    def test_filter_by_iptm(self, tmp_path):
        ana = PISAInterfaceAnalyzer(KTZ, pdb_id="1ktz", mode="pisa")
        ana.load_pae(
            _make_pae_json(ana._parsed_structure(), iptm=0.5, tmp_path=tmp_path)[0])
        assert ana.filter_by_iptm(min_iptm=0.8) == []     # model unreliable -> none
        ana.load_pae(
            _make_pae_json(ana._parsed_structure(), iptm=0.95, tmp_path=tmp_path)[0])
        assert len(ana.filter_by_iptm(min_iptm=0.8)) == 1

    def test_filter_requires_load_pae(self):
        ana = PISAInterfaceAnalyzer(KTZ, pdb_id="1ktz", mode="pisa")
        with pytest.raises(ValueError):
            ana.filter_by_pae(max_pae=5.0)


class TestPlddtFromBfactor:
    def test_load_plddt_and_scores(self):
        ana = PISAInterfaceAnalyzer(KTZ, pdb_id="1ktz", mode="pisa")
        model_before = ana.model_plddt()          # None (not loaded)
        assert model_before is None
        ana.load_plddt()
        assert ana.has_plddt
        model = ana.model_plddt()
        assert model is not None and model > 0.0
        scores = ana.plddt_scores()
        assert len(scores) == 1
        s = scores[1]
        assert s is not None and 0.0 <= s <= 100.0

    def test_filter_by_plddt(self):
        ana = PISAInterfaceAnalyzer(KTZ, pdb_id="1ktz", mode="pisa")
        ana.load_plddt()
        assert len(ana.filter_by_plddt(min_plddt=0.0)) == 1     # everything
        assert len(ana.filter_by_plddt(min_plddt=200.0)) == 0   # nothing

    def test_filter_requires_load_plddt(self):
        ana = PISAInterfaceAnalyzer(KTZ, pdb_id="1ktz", mode="pisa")
        with pytest.raises(ValueError):
            ana.filter_by_plddt(min_plddt=70.0)

    def test_constant_bfactor_raises(self, tmp_path):
        # Build a model whose B-factors are all identical -> no pLDDT signal.
        st = _constant_bfactor_structure()
        pdb = os.path.join(str(tmp_path), "flat.pdb")
        _write_structure_pdb(st, pdb)
        ana = PISAInterfaceAnalyzer(pdb, pdb_id="flat", mode="pisa")
        with pytest.raises(ValueError):
            ana.load_plddt()


def _constant_bfactor_structure():
    from fastpisa.parser.pdb_parser import Atom, Chain, PDBStructure
    chain = Chain(auth_asym_id="A", label_asym_id="A")
    for seq in (1, 2):
        for aname, el in (("N", "N"), ("CA", "C")):
            chain.atoms.append(Atom(
                atom_name=aname, altloc="", res_name="ALA", chain_id="A",
                res_seq=seq, icode="", x=0.0, y=0.0, z=0.0, occupancy=1.0,
                bfactor=0.0, element=el, label_asym_id="A", label_seq_id=seq,
                label_comp_id="ALA", auth_asym_id="A", auth_seq_id=seq))
    return PDBStructure(chains=[chain])


def _write_structure_pdb(structure, path):
    # Emit raw PDB text (no gemmi needed) so all B-factors land in the file.
    lines = ["CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1"]
    serial = 1
    for chain in structure.chains:
        for a in chain.atoms:
            lines.append(
                "ATOM  %5d %-4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s"
                % (serial, a.atom_name[:4], a.res_name[:3], chain.auth_asym_id,
                   a.res_seq, a.x, a.y, a.z, 1.0, a.bfactor, a.element)
            )
            serial += 1
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


class TestPymolMolstar:
    def test_write_pymol_script(self, tmp_path):
        ana = PISAInterfaceAnalyzer(KTZ, pdb_id="1ktz", mode="pisa")
        out = ana.write_pymol_script(str(tmp_path / "iface.pml"))
        text = open(out).read()
        assert text.lstrip().startswith("# fastPISA PyMOL script")
        assert f"load {os.path.abspath(KTZ)}" in text
        assert "color" in text and "chain " in text

    def test_write_molstar_html(self, tmp_path):
        ana = PISAInterfaceAnalyzer(KTZ, pdb_id="1ktz", mode="pisa")
        out = ana.write_molstar_html(str(tmp_path / "iface.html"))
        text = open(out).read()
        assert "molstar" in text.lower()
        assert "ifaceSelections" in text
        assert "ball-and-stick" in text


@pytest.mark.skipif(_HAS_MATPLOTLIB, reason="matplotlib installed; ImportError path not applicable")
class TestHeatmapOptional:
    def test_heatmap_requires_matplotlib(self, tmp_path):
        from fastpisa.viz import plot_contact_heatmap
        ana = PISAInterfaceAnalyzer(KTZ, pdb_id="1ktz", mode="pisa")
        ana.analyze()
        with pytest.raises(ImportError):
            plot_contact_heatmap(ana.interfaces[0], ana._parsed_atoms(),
                                 out_path=str(tmp_path / "hm.png"))


class TestBatch:
    def test_analyze_many_serial(self):
        from fastpisa.batch import analyze_many, expand_inputs
        res = analyze_many([KTZ], n_jobs=1)
        assert len(res) == 1
        assert res[0]["ok"] is True
        assert res[0]["n_interfaces"] == 1

    def test_expand_inputs_glob_and_dupes(self):
        from fastpisa.batch import expand_inputs
        files = expand_inputs(os.path.join(KTZ), os.path.join(KTZ))
        assert files == [KTZ]  # de-duplicated, order kept