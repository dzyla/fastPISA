"""Crystallographic symmetry-mate generation.

60% of the interfaces original PISA reports for a deposited entry involve a
symmetry mate (322 of 802 across the 37 cached references are
identity-identity). Reproducing them means expanding the asymmetric unit by
the space group's operations and the neighbouring cells, and keeping the
copies that can bury surface against it.

Two things this must get right, both verified against PISA's own reported
transforms in ``test_reference_symmetry.py``:

* the contact screen uses the SHADOW cutoff (r1 + r2 + 2*probe ~ 6.6 A), not
  a 5 A atom-contact cutoff -- 1acb's interface 9 is a real 14.2 A^2
  interface whose closest atom pair is 5.02 A apart;
* the cell search range must be derived from the cell and the molecule, not
  fixed at +-1: a thin cell puts contacting images two cells away.
"""

from __future__ import annotations

import numpy as np
import pytest

gemmi = pytest.importorskip("gemmi")

from fastpisa.assembly.crystal import (  # noqa: E402
    SymmetryCopy, cell_search_range, crystal_operators, symmetry_copies,
)

CUBIC = {"a": 50.0, "b": 50.0, "c": 50.0,
         "alpha": 90.0, "beta": 90.0, "gamma": 90.0, "space_group": "P 1"}
MONOCLINIC = {"a": 55.3, "b": 59.4, "c": 42.5,
              "alpha": 90.0, "beta": 99.1, "gamma": 90.0,
              "space_group": "P 1 21 1"}


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
def test_p1_has_only_the_identity_operator():
    ops = crystal_operators("P 1")
    assert len(ops) == 1
    assert np.allclose(ops[0].rotation, np.eye(3))
    assert np.allclose(ops[0].translation, 0.0)
    assert ops[0].triplet == "x,y,z"


def test_p21_has_the_screw_axis():
    ops = crystal_operators("P 1 21 1")
    assert len(ops) == 2
    triplets = {op.triplet for op in ops}
    assert "x,y,z" in triplets
    assert "-x,y+1/2,-z" in triplets


def test_centred_lattices_include_the_centring_operators():
    """C2 has 4 general positions, not 2: centring must not be dropped."""
    ops = crystal_operators("C 1 2 1")
    assert len(ops) == 4
    assert any(np.allclose(op.translation, [0.5, 0.5, 0.0]) for op in ops)


def test_an_unknown_space_group_is_reported_not_guessed():
    with pytest.raises(ValueError, match="space group"):
        crystal_operators("NOT A GROUP")


# ---------------------------------------------------------------------------
# Cell search range
# ---------------------------------------------------------------------------
def test_search_range_covers_a_molecule_wider_than_one_cell():
    """A molecule spanning 1.5 cells needs +-2, not +-1.

    The bound is deliberately isotropic: a symmetry operation can ROTATE the
    copy, so a molecule 30 A long along x extends 30 A along y in a 90-degree
    image. Using the per-axis extent would miss that image. The bounding-box
    diagonal is the conservative bound, and the contact screen prunes
    whatever it over-generates.
    """
    cell = gemmi.UnitCell(20.0, 20.0, 20.0, 90.0, 90.0, 90.0)
    span = np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
    assert cell_search_range(cell, span, cutoff=6.6) == (2, 2, 2)


def test_search_range_is_at_least_one_cell_each_way():
    cell = gemmi.UnitCell(200.0, 200.0, 200.0, 90.0, 90.0, 90.0)
    span = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    assert cell_search_range(cell, span, cutoff=6.6) == (1, 1, 1)


# ---------------------------------------------------------------------------
# Copies
# ---------------------------------------------------------------------------
def _two_atoms(x=0.0):
    return np.array([[x, 0.0, 0.0], [x + 2.0, 0.0, 0.0]])


def test_a_lone_molecule_in_a_large_p1_cell_has_no_mates():
    """Nothing to contact: every image is a whole cell away."""
    copies = symmetry_copies(_two_atoms(), CUBIC, cutoff=6.6)
    assert copies == []


def test_a_molecule_spanning_a_small_cell_contacts_its_own_images():
    # Atoms at x = 0 and 2 in an 8 A cell: the +1 image sits at x = 8, so the
    # closest pair is 6 A apart and inside the 6.6 A shadow cutoff. (A 12 A
    # cell would leave them 10 A apart -- genuinely no contact.)
    small = dict(CUBIC, a=8.0, b=8.0, c=8.0)
    copies = symmetry_copies(_two_atoms(), small, cutoff=6.6)
    assert copies
    assert all(isinstance(c, SymmetryCopy) for c in copies)
    # Pure translations in P1: rotation stays the identity.
    for c in copies:
        assert np.allclose(c.rotation, np.eye(3))
        assert any(c.cell)


def test_the_identity_copy_is_never_returned():
    """The asymmetric unit is not its own symmetry mate."""
    small = dict(MONOCLINIC, a=20.0, b=20.0, c=20.0)
    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    for c in symmetry_copies(coords, small, cutoff=6.6):
        moved = coords @ np.asarray(c.rotation).T + np.asarray(c.translation)
        assert not np.allclose(moved, coords, atol=1e-6)
        assert not (c.symop_no == 1 and c.cell == (0, 0, 0))


def test_copies_are_rigid_motions():
    small = dict(MONOCLINIC, a=25.0, b=25.0, c=25.0)
    coords = np.array([[0.0, 0.0, 0.0], [5.0, 1.0, 2.0], [2.0, 7.0, 1.0]])
    before = np.linalg.norm(coords[0] - coords[1])
    for c in symmetry_copies(coords, small, cutoff=8.0):
        rot = np.asarray(c.rotation)
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
        moved = coords @ rot.T + np.asarray(c.translation)
        assert np.linalg.norm(moved[0] - moved[1]) == pytest.approx(before)


def test_copies_are_unique():
    small = dict(MONOCLINIC, a=22.0, b=22.0, c=22.0)
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]])
    copies = symmetry_copies(coords, small, cutoff=6.6)
    keys = [(c.symop_no, c.cell) for c in copies]
    assert len(keys) == len(set(keys))


def test_missing_cell_parameters_yield_no_copies():
    """A predicted model has no CRYST1; that is not an error."""
    assert symmetry_copies(_two_atoms(), {}, cutoff=6.6) == []
    assert symmetry_copies(_two_atoms(), {"space_group": "P 1"}, cutoff=6.6) == []


# ---------------------------------------------------------------------------
# Directly against PISA's own transforms
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pdb_id", ["1acb", "1aay", "1a3n", "1cgi", "1oph"])
def test_generated_copies_contain_every_transform_pisa_used(pdb_id):
    """The headline requirement: no PISA mate may be missing.

    For each cached entry, every symmetry-mate transform appearing in PISA's
    interface list must be reproduced by the generator, matched on the
    orthogonal rotation and translation PISA itself reports.
    """
    import gzip
    import os

    from fastpisa.parser.pdb_parser import parse_pdb
    from fastpisa.reference.ebi_pisa import parse_pisa_xml

    ref = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "reference")
    structure = parse_pdb(os.path.join(ref, "pdb", f"{pdb_id}.pdb.gz"))
    coords = np.array([[a.x, a.y, a.z] for a in structure.atoms
                       if a.element.strip().upper() not in ("H", "D")
                       and a.res_name.strip().upper() not in ("HOH", "WAT")],
                      dtype=float)
    copies = symmetry_copies(coords, structure.crystal_info, cutoff=6.65)

    with gzip.open(os.path.join(ref, f"{pdb_id}.pisa.xml.gz"), "rt") as fh:
        interfaces = parse_pisa_xml(fh.read())

    wanted = set()
    for iface in interfaces:
        for mol in iface["molecules"]:
            rot = np.asarray(mol["rotation"], dtype=float)
            tr = np.asarray(mol["translation"], dtype=float)
            if np.allclose(rot, np.eye(3)) and np.allclose(tr, 0.0):
                continue
            wanted.add((tuple(np.round(rot.ravel(), 4)),
                        tuple(np.round(tr, 2))))

    have = {(tuple(np.round(np.asarray(c.rotation).ravel(), 4)),
             tuple(np.round(np.asarray(c.translation), 2)))
            for c in copies}

    missing = [w for w in wanted if w not in have]
    assert not missing, (
        f"{pdb_id}: {len(missing)} of {len(wanted)} PISA transforms not "
        f"generated (generated {len(copies)} copies). First missing: "
        f"rot={missing[0][0]} tr={missing[0][1]}")


# ---------------------------------------------------------------------------
# Exact placement arithmetic
# ---------------------------------------------------------------------------
def test_copies_carry_exact_integer_fractional_placements():
    """Symmetry equivalence must not be decided by rounded floats.

    A crystallographic operation is exact in fractional coordinates: the
    rotation is an integer matrix and the translation a rational with a small
    denominator. Deciding whether two placements are the same by rounding
    orthogonal-space translations to a fixed number of decimals puts real
    cases on the boundary -- 1urn's 6-fold screw translates by c/6 = 42.55 A,
    which rounds to 42.5 or 42.6 depending on the last bit, so an interface
    and its mirror image failed to collapse.
    """
    # 10 A cell: both the identity translations and the 2-fold screw bring
    # images into contact (a 24 A cell leaves this molecule isolated).
    small = dict(MONOCLINIC, a=10.0, b=10.0, c=10.0)
    coords = np.array([[0.0, 0.0, 0.0], [4.0, 2.0, 1.0], [1.0, 6.0, 3.0]])
    copies = symmetry_copies(coords, small, cutoff=6.6)
    assert len(copies) >= 4
    assert {c.symop_no for c in copies} == {1, 2}
    for c in copies:
        rot = np.asarray(c.frac_rotation)
        tran = np.asarray(c.frac_translation)
        assert rot.dtype.kind == "i", rot.dtype
        assert tran.dtype.kind == "i", tran.dtype
        assert rot.shape == (3, 3) and tran.shape == (3,)
        assert abs(int(round(np.linalg.det(rot)))) == 1


def test_fractional_placement_includes_the_cell_shift():
    """The integer placement must be the FULL placement, cell included.

    Otherwise two copies of one operation in different cells look identical.
    """
    small = dict(CUBIC, a=8.0, b=8.0, c=8.0)
    copies = symmetry_copies(_two_atoms(), small, cutoff=6.6)
    by_cell = {c.cell: c for c in copies}
    assert len(by_cell) == len(copies)
    den = copies[0].frac_denominator
    for cell, c in by_cell.items():
        assert tuple(np.asarray(c.frac_translation) // den) == cell


def test_equivalence_key_separates_distinct_placements():
    """Different lattice directions are different interfaces.

    A tripod spanning 6 A in an 8 A P1 cell touches its images along 12
    distinct cell offsets. Each offset and its negative are the SAME
    interface seen from either end, so 12 placements must give exactly 6
    distinct keys -- no more (over-reporting) and no fewer (silently merging
    different packing contacts).
    """
    from fastpisa.core import _equivalence_key

    small = dict(CUBIC, a=8.0, b=8.0, c=8.0)
    coords = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0],
                       [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]])
    copies = symmetry_copies(coords, small, cutoff=6.6)
    assert len(copies) == 12
    asu = {"asu_molecule_id": "A", "frac_rotation": np.eye(3, dtype=int),
           "frac_translation": np.zeros(3, dtype=int), "frac_denominator": 24}
    keys = {
        _equivalence_key(asu, {
            "asu_molecule_id": "A",
            "frac_rotation": c.frac_rotation,
            "frac_translation": c.frac_translation,
            "frac_denominator": c.frac_denominator,
        })
        for c in copies
    }
    assert len(keys) == 6


def test_equivalence_key_collapses_an_interface_and_its_reverse():
    """A + mate(B, S) and B + mate(A, S^-1) are one interface."""
    from fastpisa.core import _equivalence_key

    den = 24
    rot = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    tran = np.array([0, den // 2, den])          # -x, y+1/2, -z+1
    forward = (
        {"asu_molecule_id": "A", "frac_rotation": np.eye(3, dtype=int),
         "frac_translation": np.zeros(3, dtype=int), "frac_denominator": den},
        {"asu_molecule_id": "B", "frac_rotation": rot,
         "frac_translation": tran, "frac_denominator": den},
    )
    inv_rot = np.rint(np.linalg.inv(rot)).astype(int)
    reverse = (
        {"asu_molecule_id": "B", "frac_rotation": np.eye(3, dtype=int),
         "frac_translation": np.zeros(3, dtype=int), "frac_denominator": den},
        {"asu_molecule_id": "A", "frac_rotation": inv_rot,
         "frac_translation": -inv_rot @ tran, "frac_denominator": den},
    )
    assert _equivalence_key(*forward) == _equivalence_key(*reverse)


def test_equivalence_key_collapses_a_three_fold_and_its_inverse():
    """A fractional rotation is NOT orthogonal: transpose is not inverse.

    In hexagonal/trigonal axes the 3-fold is ``-y, x-y, z``:

        R  = [[0,-1,0],[1,-1,0],[0,0,1]]
        R^-1 = R^2 = [[-1,1,0],[-1,0,0],[0,0,1]]     (the other 3-fold)
        R^T  = [[0,1,0],[-1,-1,0],[0,0,1]]           -- a different matrix

    Using the transpose as the inverse made ``B + D[3-fold]`` and
    ``D + B[inverse 3-fold]`` -- one physical contact, identical buried area
    to 0.01 A^2 -- produce different keys, so 4ins reported 48 interfaces
    where PISA reports 29, every extra one an exact twin of another.

    A monoclinic 2-fold is its own inverse AND its own transpose, which is
    why the orthorhombic cases missed this.
    """
    from fastpisa.core import _equivalence_key

    den = 24
    three_fold = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]])
    inverse = three_fold @ three_fold          # R^2 == R^-1 for a 3-fold
    assert not np.array_equal(inverse, three_fold.T)
    assert np.array_equal(three_fold @ inverse, np.eye(3, dtype=int))

    identity = {"frac_rotation": np.eye(3, dtype=int),
                "frac_translation": np.zeros(3, dtype=int),
                "frac_denominator": den}
    forward = ({**identity, "asu_molecule_id": "B"},
               {"asu_molecule_id": "D", "frac_rotation": three_fold,
                "frac_translation": np.zeros(3, dtype=int),
                "frac_denominator": den})
    reverse = ({**identity, "asu_molecule_id": "D"},
               {"asu_molecule_id": "B", "frac_rotation": inverse,
                "frac_translation": np.zeros(3, dtype=int),
                "frac_denominator": den})
    assert _equivalence_key(*forward) == _equivalence_key(*reverse)


def test_equivalence_key_collapses_a_screw_with_a_translation():
    """The translation must be carried through the true inverse too."""
    from fastpisa.core import _equivalence_key

    den = 24
    rot = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]])
    inv = rot @ rot
    tran = np.array([0, 0, den // 3])            # 3-fold screw: z + 1/3
    identity = {"frac_rotation": np.eye(3, dtype=int),
                "frac_translation": np.zeros(3, dtype=int),
                "frac_denominator": den}
    forward = ({**identity, "asu_molecule_id": "A"},
               {"asu_molecule_id": "B", "frac_rotation": rot,
                "frac_translation": tran, "frac_denominator": den})
    reverse = ({**identity, "asu_molecule_id": "B"},
               {"asu_molecule_id": "A", "frac_rotation": inv,
                "frac_translation": -inv @ tran, "frac_denominator": den})
    assert _equivalence_key(*forward) == _equivalence_key(*reverse)


# ---------------------------------------------------------------------------
# The cell window must follow the operation, not the origin
# ---------------------------------------------------------------------------
def test_mates_are_found_for_a_molecule_far_from_the_cell_origin():
    """A symmetry operation can displace a copy many cells away.

    Deposited coordinates are not centred on the cell origin. In P 1 21 1
    the 2-fold maps fractional x to -x, so a molecule sitting at x ~ 5 cells
    lands at x ~ -5 and needs a shift of ~10 cells to come back next to the
    original. Searching a fixed window around the ORIGIN cannot reach it,
    however wide the cap: the window has to be centred on the offset that
    brings the transformed copy back to the molecule.

    Measured cost of getting this wrong, on a blind 60-entry draw: four
    PISA interfaces missed in 1cq1, 1wuf and 4o45, every one of them needing
    a cell component PISA reports as 2 or 3.
    """
    cell_edge = 12.0
    # Fractional x ~ 5.0: five cells from the origin along a. The 2-fold maps
    # that to x ~ -5, so the contacting image needs a shift of TEN cells --
    # far outside any window centred on the origin.
    offset = 5.0 * cell_edge
    coords = np.array([[offset, 1.0, 1.0], [offset + 3.0, 2.0, 1.0],
                       [offset + 1.0, 1.0, 4.0]])
    info = {"a": cell_edge, "b": cell_edge, "c": cell_edge,
            "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
            "space_group": "P 1 21 1"}

    copies = symmetry_copies(coords, info, cutoff=6.6)

    assert copies, "no mates found for a molecule far from the cell origin"
    assert any(c.symop_no == 2 for c in copies), (
        "the 2-fold screw mate was not generated: the cell window is centred "
        "on the origin instead of on the operation's displacement")
    assert all(abs(c.cell[0]) >= 9 for c in copies if c.symop_no == 2), (
        "the screw mate must sit about ten cells out")
    # And every returned copy really is in contact.
    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    for c in copies:
        moved = coords @ np.asarray(c.rotation).T + np.asarray(c.translation)
        assert tree.query(moved)[0].min() <= 6.6 + 1e-6


def test_the_window_is_relative_to_the_operation_for_every_operator():
    """Same requirement in a higher-symmetry group with several operators."""
    from scipy.spatial import cKDTree

    cell_edge = 11.0
    base = 3.0 * cell_edge
    coords = np.array([[base, base, 1.0], [base + 3.0, base + 1.0, 2.0],
                       [base + 1.0, base + 3.0, 1.0]])
    info = {"a": cell_edge, "b": cell_edge, "c": cell_edge,
            "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
            "space_group": "P 21 21 21"}

    copies = symmetry_copies(coords, info, cutoff=6.6)

    assert len({c.symop_no for c in copies}) >= 2, (
        f"only operators {sorted({c.symop_no for c in copies})} produced mates")
    # Each operator's mates sit around ITS own displacement, far from 0.
    assert any(max(abs(v) for v in c.cell) >= 3
               for c in copies if c.symop_no != 1)
    tree = cKDTree(coords)
    for c in copies:
        moved = coords @ np.asarray(c.rotation).T + np.asarray(c.translation)
        assert tree.query(moved)[0].min() <= 6.6 + 1e-6
