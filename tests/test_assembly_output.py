"""Predicted assemblies must be reachable and serialisable.

Opt-in: the default run is unchanged, so no existing caller pays for the
search. Every number that leaves the package has to survive json.dump --
numpy scalars in the crystal placements have broken that before.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

gemmi = pytest.importorskip("gemmi")

import fastpisa  # noqa: E402
from fastpisa.api import PISAInterfaceAnalyzer  # noqa: E402
from fastpisa.core import run_core  # noqa: E402

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data", "reference")
ACB = os.path.join(REF, "pdb", "1acb.pdb.gz")


def test_prediction_is_off_by_default():
    state = run_core(ACB, mode="pisa", symmetry="crystal")
    assert state.assemblies == []


def test_prediction_populates_core_state():
    state = run_core(ACB, mode="pisa", symmetry="crystal",
                     predict_assemblies=True)
    assert state.assemblies
    assert state.assemblies[0].rank == 1


def test_assembly_json_carries_the_predictions():
    result = fastpisa.analyze(ACB, pdb_id="1acb", mode="pisa",
                              symmetry="crystal", predict_assemblies=True)
    doc = result.assembly_json["assembly"]
    assert "predicted_assemblies" in doc
    entries = doc["predicted_assemblies"]
    assert entries
    first = entries[0]
    assert set(first) >= {"rank", "size", "mmsize", "composition", "formula",
                          "dissociation_energy", "entropy", "n_interfaces",
                          "interface_ids", "molecules"}
    assert first["rank"] == 1
    for molecule in first["molecules"]:
        assert set(molecule) >= {"asu_molecule_id", "symop_no", "cell"}
    # Must survive serialisation: numpy scalars have broken this before.
    json.dumps(result.assembly_json)


def test_the_key_is_absent_when_prediction_is_off():
    result = fastpisa.analyze(ACB, pdb_id="1acb", mode="pisa",
                              symmetry="crystal")
    assert "predicted_assemblies" not in result.assembly_json["assembly"]


def test_analyzer_exposes_assemblies_and_provenance():
    analyzer = PISAInterfaceAnalyzer(ACB, pdb_id="1acb", mode="pisa",
                                     symmetry="crystal",
                                     predict_assemblies=True)
    analyzer.analyze()
    assert analyzer.assemblies
    assert analyzer.assemblies[0].mmsize == 2
    assert analyzer.analysis_provenance()["predict_assemblies"] is True


def test_cli_predicts_assemblies(tmp_path):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run(
        [sys.executable, "-m", "fastpisa.cli", ACB, "--pdb_id", "1acb",
         "--mode", "pisa", "--symmetry", "crystal", "--predict-assemblies",
         "-o", str(tmp_path)],
        capture_output=True, text=True, cwd=repo, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "assembl" in out.stdout.lower()
    doc = json.load(open(tmp_path / "1acb-assembly1.json"))
    assert doc["assembly"]["predicted_assemblies"][0]["mmsize"] == 2
