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


# ---------------------------------------------------------------------------
# The output must be reconstructible (review C2) and name real molecules (I4)
# ---------------------------------------------------------------------------
def test_each_assembly_molecule_carries_its_exact_placement():
    """C2: the JSON must let a consumer rebuild the assembly.

    ``symop_no`` was found by a linear scan requiring exact equality of the
    translation INCLUDING its lattice part, so a composed placement outside
    the first shell fell through to a default of 1. On 1urn, 9 of 39 output
    placements claimed symop 1 while their rotation was not the identity --
    rank 1 among them -- so rebuilding placed molecules on top of each other.
    The exact fractional placement is now emitted alongside, which makes the
    document reconstructible whatever the lookup does.
    """
    import numpy as np

    result = fastpisa.analyze(os.path.join(REF, "pdb", "1urn.pdb.gz"),
                              pdb_id="1urn", mode="pisa", symmetry="crystal",
                              predict_assemblies=True)
    entries = result.assembly_json["assembly"]["predicted_assemblies"]
    assert entries

    identity = np.eye(3, dtype=int)
    checked = 0
    for entry in entries:
        for molecule in entry["molecules"]:
            assert set(molecule) >= {"asu_molecule_id", "symop_no", "cell",
                                     "frac_rotation", "frac_translation",
                                     "frac_denominator"}
            rotation = np.asarray(molecule["frac_rotation"], dtype=int)
            assert rotation.shape == (3, 3)
            assert abs(int(round(float(np.linalg.det(rotation))))) == 1
            # A non-identity rotation may not be labelled symop 1.
            if not np.array_equal(rotation, identity):
                assert molecule["symop_no"] != 1, (
                    f"{molecule['asu_molecule_id']} has rotation "
                    f"{rotation.tolist()} but claims symop 1")
            # The lattice part of the exact translation is the reported cell.
            den = molecule["frac_denominator"]
            lattice = [v // den for v in molecule["frac_translation"]]
            assert lattice == list(molecule["cell"])
            checked += 1
    assert checked > 10


def test_an_unmatched_placement_is_not_silently_labelled_symop_one():
    """A miss must be visible, not disguised as the identity operation."""
    from fastpisa.assembly.graph import Placement
    from fastpisa.core import _symop_of

    state = run_core(os.path.join(REF, "pdb", "1acb.pdb.gz"),
                     mode="pisa", symmetry="crystal")
    invented = Placement(rotation=((0, 1, 0), (0, 0, 1), (1, 0, 0)),
                         translation=(5, 7, 11), denominator=24)
    assert _symop_of(state, "E", invented) is None


def test_assembly_molecule_ids_name_one_physical_molecule():
    """I4: a mate's hetero group must keep its residue number.

    A mate ligand's chain id is ``[SO4]H~4_555:623``, and splitting it on the
    mate separator from the right took the residue number into the label --
    leaving ``[SO4]H``, which collapses every SO4 of chain H onto one node.
    1a3n held both ``[HEM]A`` and ``[HEM]A:142``; in 1prc three distinct SO4
    residues became one, and two different interfaces resolved to the same
    node.
    """
    from fastpisa.assembly.predict import _molecule_index

    for pdb_id in ("1a3n", "1prc", "4ins"):
        state = run_core(os.path.join(REF, "pdb", f"{pdb_id}.pdb.gz"),
                         mode="pisa", symmetry="crystal")
        index = _molecule_index(state)
        ligands = [key for key, mol in index.items()
                   if mol.get("molecule_class") == "Ligand"]
        assert ligands, pdb_id
        for key in ligands:
            assert key.startswith("["), f"{pdb_id}: {key!r}"
            assert ":" in key, (
                f"{pdb_id}: ligand id {key!r} has no residue number, so every "
                "hetero group of that CCD in that chain collapses onto it")
        # No id may be a truncation of another.
        for key in ligands:
            assert not any(other != key and other.startswith(key + ":")
                           for other in ligands), (
                f"{pdb_id}: {key!r} is a truncated form of another molecule id")
