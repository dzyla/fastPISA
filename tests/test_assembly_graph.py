"""The crystal contact graph: nodes are placed molecules, edges are interfaces.

An edge records the RELATIVE placement between its two molecules, so it can be
applied at any node to reach that node's partner -- which is what lets a finite
assembly be grown from a seed. Placement arithmetic is exact integer work on
fractional coordinates: a fractional rotation is not orthogonal, so its
inverse is not its transpose (in hexagonal axes the 3-fold's inverse is the
other 3-fold), and rounding orthogonal translations puts a 6-fold screw's
c/6 = 42.55 A on a rounding boundary.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

gemmi = pytest.importorskip("gemmi")

from fastpisa.assembly.graph import (  # noqa: E402
    IDENTITY_PLACEMENT, ContactEdge, Placement, contact_edges, neighbours,
)
from fastpisa.core import run_core  # noqa: E402

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data", "reference")

THREE_FOLD = Placement(rotation=((0, -1, 0), (1, -1, 0), (0, 0, 1)),
                       translation=(0, 0, 8), denominator=24)


def test_identity_placement_is_neutral():
    assert THREE_FOLD.compose(IDENTITY_PLACEMENT) == THREE_FOLD
    assert IDENTITY_PLACEMENT.compose(THREE_FOLD) == THREE_FOLD


def test_inverse_uses_the_integer_inverse_not_the_transpose():
    """R^-1 of the hexagonal 3-fold is the other 3-fold, not R^T."""
    inverse = THREE_FOLD.inverse()
    rot = np.asarray(THREE_FOLD.rotation)
    assert not np.array_equal(np.asarray(inverse.rotation), rot.T)
    assert THREE_FOLD.compose(inverse) == IDENTITY_PLACEMENT
    assert inverse.compose(THREE_FOLD) == IDENTITY_PLACEMENT


def test_composition_is_exact_and_associative():
    squared = THREE_FOLD.compose(THREE_FOLD)
    cubed = squared.compose(THREE_FOLD)
    assert cubed.rotation == IDENTITY_PLACEMENT.rotation
    # Three applications of a +1/3 screw advance exactly one cell.
    assert cubed.translation == (0, 0, 24)
    assert all(isinstance(v, int) for v in cubed.translation)


def test_key_separates_the_rotation_from_the_lattice_part():
    """Growth needs 'same operation, different cell' to be detectable."""
    shifted = Placement(rotation=THREE_FOLD.rotation,
                        translation=(24, 0, 8), denominator=24)
    assert shifted.key() != THREE_FOLD.key()
    assert shifted.rotation == THREE_FOLD.rotation
    assert shifted.cell() != THREE_FOLD.cell()
    assert shifted.cell() == (1, 0, 0)


def test_edges_come_from_crystal_interfaces_sorted_by_stability():
    state = run_core(os.path.join(REF, "pdb", "1acb.pdb.gz"),
                     mode="pisa", symmetry="crystal")
    edges = contact_edges(state)
    assert len(edges) == len(state.interfaces)
    assert all(isinstance(e, ContactEdge) for e in edges)
    stabilities = [e.stabilization for e in edges]
    assert stabilities == sorted(stabilities)
    assert all(e.placement.denominator == 24 for e in edges)
    assert {e.molecule_a for e in edges} <= {"E", "I"}


def test_an_edge_applied_at_a_node_reaches_its_partner():
    state = run_core(os.path.join(REF, "pdb", "1acb.pdb.gz"),
                     mode="pisa", symmetry="crystal")
    edges = contact_edges(state)
    seed = ("E", IDENTITY_PLACEMENT)
    found = neighbours(seed, edges)
    assert found, "chain E must have crystal neighbours"
    for (molecule, placement), edge in found:
        assert molecule in ("E", "I")
        assert isinstance(placement, Placement)
    # Applying an edge then its inverse returns to the seed.
    (molecule, placement), edge = found[0]
    back = neighbours((molecule, placement), edges)
    assert any(node == seed for node, _ in back)


def test_a_structure_without_symmetry_yields_only_asu_edges():
    state = run_core(os.path.join(os.path.dirname(REF), "1ktz.pdb"),
                     mode="pisa")
    edges = contact_edges(state)
    assert edges
    assert all(e.placement == IDENTITY_PLACEMENT for e in edges)
