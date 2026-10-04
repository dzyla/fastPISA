"""Enumerating the assemblies a crystal can form, then ranking them.

Candidates come from NESTED interface subsets: sort the crystal's interfaces
most-stabilising first and, for k = 1..n, grow components using only the top
k. That is PISA's dissociation ordering, it is bounded at n iterations, and it
avoids the 2^n subset search. Each survivor is scored with the same
minimum-cut dissociation machinery the assembly document already uses.
"""

from __future__ import annotations

import os

import pytest

gemmi = pytest.importorskip("gemmi")

from fastpisa.assembly.predict import Assembly, predict_assemblies  # noqa: E402
from fastpisa.core import run_core  # noqa: E402

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data", "reference")


def _crystal(pdb_id):
    return run_core(os.path.join(REF, "pdb", f"{pdb_id}.pdb.gz"),
                    mode="pisa", symmetry="crystal")


def test_1acb_predicts_the_heterodimer_pisa_predicts():
    """PISA: one assembly, composition EI, mmsize 2, matching REMARK 350."""
    assemblies = predict_assemblies(_crystal("1acb"))
    assert assemblies
    top = assemblies[0]
    assert isinstance(top, Assembly)
    assert top.mmsize == 2
    assert set(top.composition.replace("[", "").replace("]", "")) >= {"E", "I"}
    assert top.dissociation_energy > 0


def test_assemblies_are_ranked_by_dissociation_energy():
    assemblies = predict_assemblies(_crystal("1a3n"))
    energies = [a.dissociation_energy for a in assemblies]
    assert energies == sorted(energies, reverse=True)
    assert [a.rank for a in assemblies] == list(range(1, len(assemblies) + 1))


def test_duplicate_assemblies_are_collapsed():
    assemblies = predict_assemblies(_crystal("1acb"))
    seen = {tuple(sorted(a.composition)) + (a.size,) for a in assemblies}
    assert len(seen) == len(assemblies)


def test_no_assembly_is_infinite_or_oversized():
    for pdb_id in ("1acb", "1ktz", "2ptc"):
        for assembly in predict_assemblies(_crystal(pdb_id)):
            assert 1 <= assembly.size <= 512
            assert assembly.size == len(assembly.nodes)
            assert assembly.mmsize <= assembly.size


def test_a_model_without_a_cell_predicts_from_its_own_coordinates():
    """Review Focus 1: a predicted model has no symmetry; do not raise."""
    state = run_core(os.path.join(os.path.dirname(REF), "1ktz.pdb"),
                     mode="pisa")
    assemblies = predict_assemblies(state)
    assert isinstance(assemblies, list)
    for assembly in assemblies:
        assert assembly.size >= 1


def test_a_lone_molecule_yields_the_monomer(tmp_path):
    """Review Focus 4: no interfaces at all must not crash enumeration."""
    path = tmp_path / "single.pdb"
    path.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00"
        "           C\nEND\n")
    assemblies = predict_assemblies(run_core(str(path), mode="pisa"))
    assert len(assemblies) == 1
    assert assemblies[0].size == 1
    assert assemblies[0].mmsize == 0 or assemblies[0].mmsize == 1


def test_a_ligand_only_component_does_not_outrank_a_polymer_assembly():
    """Review Focus 5: a lone ion is not an assembly worth reporting."""
    assemblies = predict_assemblies(_crystal("2ptc"))
    assert assemblies
    top = assemblies[0]
    assert top.mmsize >= 2, (
        f"top assembly is {top.composition} with mmsize {top.mmsize}")
    assert all(a.mmsize >= 1 for a in assemblies), (
        "a component of ligands only must not be reported as an assembly")


def test_composition_and_formula_follow_pisa_shape():
    """PISA writes 'E[4]I[4][CA][4]' and formula 'A4B4a4'."""
    assemblies = predict_assemblies(_crystal("2ptc"))
    top = assemblies[0]
    assert top.composition
    assert top.formula
    assert top.formula[0].isalpha()
    # A homo-oligomer repeats a chain; the count appears in the composition.
    if top.size > top.mmsize:
        assert "[" in top.composition


def test_interface_ids_point_back_at_the_interfaces_used():
    state = _crystal("1acb")
    valid = {i.interface_id for i in state.interfaces}
    for assembly in predict_assemblies(state):
        assert set(assembly.interface_ids) <= valid
        assert assembly.n_interfaces == len(assembly.interface_ids)
