"""Regression tests for coordinate model, altloc, and identifier parsing."""

from __future__ import annotations

import gzip

import pytest

from fastpisa.core import run_core
from fastpisa.parser.pdb_parser import parse_mmcif, parse_pdb


def _pdb_atom(
    serial: int,
    atom_name: str,
    *,
    altloc: str = " ",
    occupancy: float = 1.0,
    x: float = 1.0,
    element: str = "C",
) -> str:
    return (
        f"{'ATOM':<6}{serial:>5} {atom_name:>4}{altloc:1}{'ALA':>3} "
        f"{'A':1}{1:>4}{' ':1}   {x:>8.3f}{2.0:>8.3f}{3.0:>8.3f}"
        f"{occupancy:>6.2f}{20.0:>6.2f}          {element:>2}\n"
    )


def test_pdb_uses_first_model_and_resolves_altlocs(tmp_path):
    path = tmp_path / "models.pdb"
    path.write_text(
        "MODEL        1\n"
        + _pdb_atom(1, "CA", altloc="A", occupancy=0.40, x=1.0)
        + _pdb_atom(2, "CA", altloc="B", occupancy=0.60, x=2.0)
        + _pdb_atom(3, "N", altloc=" ", occupancy=0.20, x=3.0, element="N")
        + _pdb_atom(4, "N", altloc="A", occupancy=0.80, x=4.0, element="N")
        + "ENDMDL\nMODEL        2\n"
        + _pdb_atom(5, "C", x=99.0)
        + "ENDMDL\nEND\n"
    )

    atoms = parse_pdb(str(path)).atoms

    assert [(a.atom_name, a.altloc, a.x) for a in atoms] == [
        ("CA", "B", 2.0),
        ("N", " ", 3.0),
    ]


def test_pdb_altloc_occupancy_tie_uses_alphabetically_first(tmp_path):
    path = tmp_path / "tie.pdb"
    path.write_text(
        _pdb_atom(1, "CA", altloc="B", occupancy=0.50, x=2.0)
        + _pdb_atom(2, "CA", altloc="A", occupancy=0.50, x=1.0)
        + "END\n"
    )

    atom = parse_pdb(str(path)).atoms[0]

    assert (atom.altloc, atom.x) == ("A", 1.0)


def test_pdb_rejects_missing_element_column(tmp_path):
    path = tmp_path / "missing_element.pdb"
    path.write_text(_pdb_atom(1, "CA", element="") + "END\n")

    with pytest.raises(ValueError, match=r"line 1.*columns 77-78"):
        parse_pdb(str(path))


_MMCIF = """data_altloc
_entry.id ALTLOC
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_ins_code
_atom_site.pdbx_PDB_model_num
ATOM 1 C CA B ALA AA 3 2.0 0.0 0.0 0.30 21.0 7 ALA A CA B 1
ATOM 2 C CA A ALA AA 3 1.0 0.0 0.0 0.70 22.0 7 ALA A CA B 1
ATOM 3 N N  . ALA AA 3 0.0 0.0 0.0 1.00 23.0 7 ALA A N  B 1
ATOM 4 C C  . ALA AA 3 99.0 0.0 0.0 1.00 24.0 7 ALA A C  B 2
"""


def _write_mmcif(tmp_path, suffix: str = ".mmcif"):
    path = tmp_path / f"model{suffix}"
    if suffix.endswith(".gz"):
        with gzip.open(path, "wt") as handle:
            handle.write(_MMCIF)
    else:
        path.write_text(_MMCIF)
    return path


def test_mmcif_preserves_atom_site_identifiers_and_first_model(tmp_path):
    atoms = parse_mmcif(str(_write_mmcif(tmp_path))).atoms

    assert len(atoms) == 2
    ca = next(a for a in atoms if a.atom_name == "CA")
    assert (ca.auth_asym_id, ca.label_asym_id) == ("A", "AA")
    assert (ca.auth_seq_id, ca.label_seq_id, ca.icode, ca.altloc) == (7, 3, "B", "A")
    assert (ca.occupancy, ca.bfactor, ca.group) == (0.7, 22.0, "ATOM")


@pytest.mark.parametrize("suffix", [".mmcif", ".mmcif.gz"])
def test_core_dispatches_mmcif_suffixes(tmp_path, suffix):
    path = _write_mmcif(tmp_path, suffix)

    state = run_core(str(path), mode="pisa", point_density=24)

    assert len(state.atoms) == 2
    assert state.structure.source == "ALTLOC"
