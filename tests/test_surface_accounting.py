"""Regression tests for heavy-atom and whole-residue surface accounting."""

from __future__ import annotations

import pytest

import fastpisa
from fastpisa.core import run_core
from fastpisa.parser.pdb_parser import Atom
from fastpisa.reference.ebi_pisa import cached_pdb_path
from fastpisa.report import group_interface
from fastpisa.surface.per_residue import compute_per_residue_surface


def _pdb_atom(serial, name, chain, x, element):
    return (
        f"{'ATOM':<6}{serial:>5} {name:>4} {'ALA':>3} {chain:1}{1:>4}    "
        f"{x:>8.3f}{0.0:>8.3f}{0.0:>8.3f}{1.0:>6.2f}{20.0:>6.2f}"
        f"          {element:>2}\n"
    )


def test_explicit_hydrogens_do_not_change_surface_totals(tmp_path):
    heavy_path = tmp_path / "heavy.pdb"
    hydrogen_path = tmp_path / "hydrogen.pdb"
    heavy = _pdb_atom(1, "C", "A", 0.0, "C") + _pdb_atom(2, "C", "B", 3.0, "C")
    hydrogens = _pdb_atom(3, "H", "A", 20.0, "H") + _pdb_atom(4, "H", "B", -20.0, "H")
    heavy_path.write_text(heavy + "END\n")
    hydrogen_path.write_text(heavy + hydrogens + "END\n")

    without_h = run_core(str(heavy_path), mode="pisa", point_density=120)
    with_h = run_core(str(hydrogen_path), mode="pisa", point_density=120)

    assert with_h.assembly_asa == pytest.approx(without_h.assembly_asa, abs=0.05)
    assert with_h.assembly_bsa == pytest.approx(without_h.assembly_bsa, abs=0.05)
    assert with_h.interfaces[0].interface_area == pytest.approx(
        without_h.interfaces[0].interface_area, abs=0.05
    )


def _atom(name: str, x: float) -> Atom:
    return Atom(
        atom_name=name,
        altloc=" ",
        res_name="ALA",
        chain_id="A",
        res_seq=1,
        icode="",
        x=x,
        y=0.0,
        z=0.0,
        occupancy=1.0,
        bfactor=20.0,
        element="C" if name != "N" else "N",
        label_asym_id="A",
        label_seq_id=1,
        label_comp_id="ALA",
        auth_asym_id="A",
        auth_seq_id=1,
        group="ATOM",
    )


def test_interface_residue_asa_includes_all_residue_atoms():
    atoms = [_atom("CA", 0.0), _atom("N", 1.0)]

    result = compute_per_residue_surface(
        atoms,
        atom_asa_combined={0: 10.0, 1: 20.0},
        atom_bsa_buried={0: 4.0},
        interface_atom_indices={0},
        mol_atoms=[0, 1],
    )

    assert result["accessible_surface_areas"] == [30.0]
    assert result["buried_surface_areas"] == [4.0]


_VFB = cached_pdb_path("1vfb")


@pytest.mark.skipif(_VFB is None, reason="1vfb reference PDB not cached")
def test_group_residue_keeps_whole_isolated_asa_across_pairs():
    result = fastpisa.analyze(_VFB, pdb_id="1vfb")
    digest = group_interface(result, ["A"], ["B", "C"])
    tyr50 = next(r for r in digest.residues_side1 if (r.chain, r.seq) == ("A", "50"))

    pair_asa = []
    pair_bsa = []
    for interface in digest.pairs:
        molecule = next(m for m in interface.molecules if m["chain_id"] == "A")
        for seq, asa, bsa in zip(
            molecule["residue_seq_ids"],
            molecule["accessible_surface_areas"],
            molecule["buried_surface_areas"],
        ):
            if str(seq) == "50":
                pair_asa.append(float(asa))
                pair_bsa.append(float(bsa))

    assert len(pair_asa) == 2
    assert tyr50.asa == pytest.approx(max(pair_asa), abs=0.01)
    assert tyr50.bsa == pytest.approx(sum(pair_bsa), abs=0.01)


def test_surface_backend_info_reports_the_runtime_algorithm(monkeypatch):
    import fastpisa.surface.freesasa_backend as backend

    monkeypatch.setattr(backend, "_HAVE_FREESASA", False)
    assert backend.surface_backend_info() == {
        "backend": "python",
        "algorithm": "Shrake-Rupley",
        "version": None,
    }

    try:
        import freesasa
    except ImportError:
        return
    monkeypatch.setattr(backend, "_HAVE_FREESASA", True)
    info = backend.surface_backend_info()
    assert info["backend"] == "FreeSASA"
    assert info["algorithm"] == freesasa.Parameters().algorithm()
