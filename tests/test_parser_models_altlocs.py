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
    """Conformer A wins this residue on TOTAL occupancy (1.20 vs 0.60).

    Per-atom selection would have taken CA from B (0.60 > 0.40) and so mixed
    two conformers in one residue; selection is per residue instead, see
    ``test_a_residue_keeps_one_conformer_not_a_per_atom_chimera``. The
    blank-altloc N is shared by both conformers and is kept regardless of
    its occupancy.
    """
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
        ("CA", "A", 1.0),
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
    # mmCIF parsing needs gemmi, which the no-optional-deps CI leg omits on
    # purpose: numpy + scipy are the only hard requirements fastPISA claims.
    pytest.importorskip("gemmi")
    atoms = parse_mmcif(str(_write_mmcif(tmp_path))).atoms

    assert len(atoms) == 2
    ca = next(a for a in atoms if a.atom_name == "CA")
    assert (ca.auth_asym_id, ca.label_asym_id) == ("A", "AA")
    assert (ca.auth_seq_id, ca.label_seq_id, ca.icode, ca.altloc) == (7, 3, "B", "A")
    assert (ca.occupancy, ca.bfactor, ca.group) == (0.7, 22.0, "ATOM")


@pytest.mark.parametrize("suffix", [".mmcif", ".mmcif.gz"])
def test_core_dispatches_mmcif_suffixes(tmp_path, suffix):
    pytest.importorskip("gemmi")
    path = _write_mmcif(tmp_path, suffix)

    state = run_core(str(path), mode="pisa", point_density=24)

    assert len(state.atoms) == 2
    assert state.structure.source == "ALTLOC"


# ---------------------------------------------------------------------------
# Alternate conformers must be chosen as a CONSISTENT SET per residue
# ---------------------------------------------------------------------------
def _pdb_res_atom(serial, name, res, *, altloc=" ", occupancy=1.0, x=1.0,
                  element="C", seq=10, chain="A"):
    return (
        f"{'ATOM':<6}{serial:>5} {name:<4}{altloc:1}{res:>3} "
        f"{chain:1}{seq:>4}{' ':1}   {x:>8.3f}{0.0:>8.3f}{0.0:>8.3f}"
        f"{occupancy:>6.2f}{20.0:>6.2f}          {element:>2}\n"
    )


def test_microheterogeneity_keeps_only_the_dominant_residue(tmp_path):
    """Altlocs with DIFFERENT residue names are one residue, not two.

    Partially occupied residues (a modelled point mutation, a mixed
    population) appear as altloc A = SER / altloc B = ALA at one site. The
    selection key included the residue name, so both conformers survived --
    leaving two atoms at identical coordinates that then occlude each other
    in the isolated-monomer ASA and corrupt that residue's ASA, BSA and dG.
    """
    path = tmp_path / "micro.pdb"
    path.write_text(
        _pdb_res_atom(1, "N", "SER", altloc="A", occupancy=0.6, x=10.0, element="N")
        + _pdb_res_atom(2, "CA", "SER", altloc="A", occupancy=0.6, x=11.0)
        + _pdb_res_atom(3, "CB", "SER", altloc="A", occupancy=0.6, x=11.5)
        + _pdb_res_atom(4, "OG", "SER", altloc="A", occupancy=0.6, x=12.5, element="O")
        + _pdb_res_atom(5, "N", "ALA", altloc="B", occupancy=0.4, x=10.0, element="N")
        + _pdb_res_atom(6, "CA", "ALA", altloc="B", occupancy=0.4, x=11.0)
        + _pdb_res_atom(7, "CB", "ALA", altloc="B", occupancy=0.4, x=11.4)
        + "END\n"
    )

    atoms = parse_pdb(str(path)).atoms

    assert {a.res_name for a in atoms} == {"SER"}
    assert sorted(a.atom_name for a in atoms) == ["CA", "CB", "N", "OG"]
    # No two atoms may share a position.
    positions = [(round(a.x, 3), round(a.y, 3), round(a.z, 3)) for a in atoms]
    assert len(set(positions)) == len(positions)


def test_a_residue_keeps_one_conformer_not_a_per_atom_chimera(tmp_path):
    """Conformer labels are a set: mixing A and B fabricates geometry.

    Picking the highest-occupancy atom INDEPENDENTLY per atom name can take
    CA from conformer B and CB from conformer A, which is a molecule that was
    never modelled.
    """
    path = tmp_path / "chimera.pdb"
    path.write_text(
        # Conformer A dominates the residue overall (0.7 + 0.4 vs 0.3 + 0.6)
        # even though its CB carries the lower occupancy of the two CBs.
        _pdb_res_atom(1, "CA", "VAL", altloc="A", occupancy=0.7, x=1.0)
        + _pdb_res_atom(2, "CB", "VAL", altloc="A", occupancy=0.4, x=2.0)
        + _pdb_res_atom(3, "CA", "VAL", altloc="B", occupancy=0.3, x=3.0)
        + _pdb_res_atom(4, "CB", "VAL", altloc="B", occupancy=0.6, x=4.0)
        + "END\n"
    )

    atoms = {a.atom_name: a for a in parse_pdb(str(path)).atoms}

    assert {a.altloc for a in atoms.values()} == {"A"}
    assert atoms["CA"].x == pytest.approx(1.0)
    assert atoms["CB"].x == pytest.approx(2.0)


def test_blank_altloc_atoms_are_always_kept(tmp_path):
    """The shared backbone of a residue with a split side chain stays."""
    path = tmp_path / "partial.pdb"
    path.write_text(
        _pdb_res_atom(1, "N", "SER", occupancy=1.0, x=0.0, element="N")
        + _pdb_res_atom(2, "CA", "SER", occupancy=1.0, x=1.0)
        + _pdb_res_atom(3, "OG", "SER", altloc="A", occupancy=0.3, x=2.0, element="O")
        + _pdb_res_atom(4, "OG", "SER", altloc="B", occupancy=0.7, x=3.0, element="O")
        + "END\n"
    )

    atoms = {a.atom_name: a for a in parse_pdb(str(path)).atoms}

    assert sorted(atoms) == ["CA", "N", "OG"]
    assert atoms["OG"].altloc == "B"      # the better-occupied conformer
    assert atoms["N"].altloc == " "


def test_altloc_selection_is_per_residue_not_per_chain(tmp_path):
    """Residue 10 may prefer conformer B while residue 11 prefers A."""
    path = tmp_path / "two_res.pdb"
    path.write_text(
        _pdb_res_atom(1, "CA", "ALA", altloc="A", occupancy=0.3, x=1.0, seq=10)
        + _pdb_res_atom(2, "CA", "ALA", altloc="B", occupancy=0.7, x=2.0, seq=10)
        + _pdb_res_atom(3, "CA", "ALA", altloc="A", occupancy=0.8, x=3.0, seq=11)
        + _pdb_res_atom(4, "CA", "ALA", altloc="B", occupancy=0.2, x=4.0, seq=11)
        + "END\n"
    )

    atoms = {a.res_seq: a for a in parse_pdb(str(path)).atoms}

    assert atoms[10].altloc == "B"
    assert atoms[11].altloc == "A"


# ---------------------------------------------------------------------------
# CRYST1: the inputs any symmetry work needs
# ---------------------------------------------------------------------------
def test_cryst1_is_read_with_the_pdb_column_widths(tmp_path):
    """Cell and space group must be read at PDB v3.3 field widths.

    The record is fixed-width: a/b/c are 9 columns each (7-15, 16-24, 25-33),
    the angles 7 each (34-40, 41-47, 48-54), the space group 56-66 and Z
    67-70. They were being read as 5-column fields, which turned 1acb's
    ``55.300 59.400 42.500 90.00 99.10 90.00 P 1 21 1`` into a = 55.0,
    b = 0.3, c = 59.0, alpha = 400.0 and a space group of
    "0.00  99.10  90.00 P 1 21 1". Nothing consumed it yet, so nothing was
    visibly wrong -- but ``PDBStructure.space_group`` is public, and this is
    exactly the input symmetry-mate generation would be built on.
    """
    path = tmp_path / "cryst.pdb"
    path.write_text(
        "CRYST1   55.300   59.400   42.500  90.00  99.10  90.00 P 1 21 1      2\n"
        + _pdb_res_atom(1, "CA", "ALA", x=1.0)
        + "END\n"
    )

    structure = parse_pdb(str(path))

    assert structure.space_group == "P 1 21 1"
    info = structure.crystal_info
    assert info["a"] == pytest.approx(55.300)
    assert info["b"] == pytest.approx(59.400)
    assert info["c"] == pytest.approx(42.500)
    assert info["alpha"] == pytest.approx(90.00)
    assert info["beta"] == pytest.approx(99.10)
    assert info["gamma"] == pytest.approx(90.00)
    assert info["z"] == 2


def test_cryst1_with_a_missing_z_still_yields_the_cell(tmp_path):
    """Z is optional in the wild; the cell must survive without it."""
    path = tmp_path / "noz.pdb"
    path.write_text(
        "CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 P 1\n"
        + _pdb_res_atom(1, "CA", "ALA", x=1.0)
        + "END\n"
    )

    structure = parse_pdb(str(path))

    assert structure.space_group == "P 1"
    assert structure.crystal_info["a"] == pytest.approx(100.0)
    assert structure.crystal_info["gamma"] == pytest.approx(90.0)
    assert structure.crystal_info.get("z") is None


def test_a_structure_with_no_cryst1_has_no_crystal_info(tmp_path):
    """Predicted models carry no cell; that must not be an error."""
    path = tmp_path / "model.pdb"
    path.write_text(_pdb_res_atom(1, "CA", "ALA", x=1.0) + "END\n")

    structure = parse_pdb(str(path))

    assert structure.crystal_info == {}
    assert structure.space_group == ""
