"""Growing a connected component, and knowing when it never stops.

A set of interfaces either closes into a finite assembly or tiles the crystal.
The discriminator: if growth reaches the same molecule under the same
ROTATION but at a different lattice cell, the component repeats
translationally -- it is a lattice, a sheet or a fibre, not an assembly. A
missed infinite case is an unbounded loop, so a hard node cap backs the test
up and says which guard stopped it.
"""

from __future__ import annotations

from fastpisa.assembly.graph import (
    IDENTITY_PLACEMENT, ContactEdge, Placement,
)
from fastpisa.assembly.predict import MAX_ASSEMBLY_NODES, grow

DEN = 24


def _placement(rotation, translation):
    return Placement(rotation=rotation, translation=translation,
                     denominator=DEN)


#: A 2-fold: -x, y, -z. Self-inverse, so A + mate(A) closes into a dimer.
TWO_FOLD = _placement(((-1, 0, 0), (0, 1, 0), (0, 0, -1)), (0, 0, 0))

#: A pure lattice translation along a: tiles the crystal forever.
TRANSLATION = _placement(((1, 0, 0), (0, 1, 0), (0, 0, 1)), (DEN, 0, 0))

#: A 3-fold: closes into a trimer after three applications.
THREE_FOLD = _placement(((0, -1, 0), (1, -1, 0), (0, 0, 1)), (0, 0, 0))


def _edge(a, b, placement, stab=-10.0):
    return ContactEdge(molecule_a=a, molecule_b=b, placement=placement,
                       stabilization=stab, area=500.0, interface_id=1)


def test_no_edges_gives_the_lone_seed():
    result = grow(("A", IDENTITY_PLACEMENT), [])
    assert result.finite is True
    assert len(result.nodes) == 1
    assert result.nodes[0] == ("A", IDENTITY_PLACEMENT)
    assert result.edges == ()


def test_a_two_fold_closes_into_a_dimer():
    result = grow(("A", IDENTITY_PLACEMENT), [_edge("A", "A", TWO_FOLD)])
    assert result.finite is True
    assert len(result.nodes) == 2
    assert {m for m, _ in result.nodes} == {"A"}


def test_a_three_fold_closes_into_a_trimer():
    result = grow(("A", IDENTITY_PLACEMENT), [_edge("A", "A", THREE_FOLD)])
    assert result.finite is True
    assert len(result.nodes) == 3


def test_a_heterodimer_interface_closes():
    result = grow(("A", IDENTITY_PLACEMENT),
                  [_edge("A", "B", IDENTITY_PLACEMENT)])
    assert result.finite is True
    assert {m for m, _ in result.nodes} == {"A", "B"}
    assert len(result.nodes) == 2


def test_a_pure_lattice_translation_is_infinite():
    """The core discrimination: same molecule, same rotation, other cell."""
    result = grow(("A", IDENTITY_PLACEMENT), [_edge("A", "A", TRANSLATION)])
    assert result.finite is False
    assert "translation" in result.reason
    assert len(result.nodes) <= MAX_ASSEMBLY_NODES


def test_a_screw_axis_is_infinite():
    """A 2-fold screw advances along its axis forever."""
    screw = _placement(((-1, 0, 0), (0, 1, 0), (0, 0, -1)), (0, DEN // 2, 0))
    result = grow(("A", IDENTITY_PLACEMENT), [_edge("A", "A", screw)])
    assert result.finite is False


def test_a_finite_assembly_plus_a_lattice_edge_is_infinite():
    """One bridging contact turns a closed dimer into a sheet."""
    result = grow(("A", IDENTITY_PLACEMENT),
                  [_edge("A", "A", TWO_FOLD), _edge("A", "A", TRANSLATION)])
    assert result.finite is False


def test_the_node_cap_is_the_backstop_not_the_primary_guard():
    """An infinite case must be caught by the rotation test, not the cap."""
    result = grow(("A", IDENTITY_PLACEMENT), [_edge("A", "A", TRANSLATION)],
                  max_nodes=MAX_ASSEMBLY_NODES)
    assert result.finite is False
    assert "translation" in result.reason
    assert len(result.nodes) < 10, (
        "the cap stopped growth instead of the translational test")


def test_the_cap_still_rejects_an_unexpectedly_large_component():
    """Defence in depth: a huge finite component is reported, not returned."""
    edges = [_edge("A", "A", THREE_FOLD)]
    result = grow(("A", IDENTITY_PLACEMENT), edges, max_nodes=2)
    assert result.finite is False
    assert "cap" in result.reason


def test_growth_is_deterministic():
    edges = [_edge("A", "B", IDENTITY_PLACEMENT), _edge("A", "A", TWO_FOLD)]
    first = grow(("A", IDENTITY_PLACEMENT), edges)
    second = grow(("A", IDENTITY_PLACEMENT), edges)
    assert first.nodes == second.nodes
    assert first.finite == second.finite


def test_the_edges_used_are_reported():
    edges = [_edge("A", "B", IDENTITY_PLACEMENT, stab=-20.0),
             _edge("A", "A", TWO_FOLD, stab=-5.0)]
    result = grow(("A", IDENTITY_PLACEMENT), edges)
    assert result.finite is True
    assert len(result.edges) >= 1
    assert all(isinstance(e, ContactEdge) for e in result.edges)
