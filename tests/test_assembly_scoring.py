"""Assembly scoring: each internal contact counted once, and all of them.

Two defects found in whole-branch review, both of which moved reported
numbers:

* **C1** -- for a self-inverse operation (any pure 2-fold, inversion centre
  or mirror, i.e. the commonest crystallographic dimer) ``neighbours``
  emits both directions of a homomolecular edge and both land on the SAME
  partner node, so the contact entered the dissociation cut twice.
  ``assembly_dissociation`` sums repeated pairs, so 1ktz's A[2] scored
  dG_diss 20.87 where PISA reports 5.63.
* **I3** -- an assembly was scored over the interface subset that happened
  to close it (the top-k of the nested search), not over its own internal
  interface graph as the spec requires, which systematically understates
  dG_diss and feeds the ranking's first key.
"""

from __future__ import annotations

import os

import pytest

gemmi = pytest.importorskip("gemmi")

from fastpisa.assembly.graph import (  # noqa: E402
    IDENTITY_PLACEMENT, ContactEdge, Placement, contact_edges,
)
from fastpisa.assembly.predict import _score, predict_assemblies  # noqa: E402
from fastpisa.core import run_core  # noqa: E402
from fastpisa.energy.entropy import dissociation_entropy  # noqa: E402

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data", "reference")
DEN = 24

#: -x, y, -z. Self-inverse: applying it twice returns the original.
TWO_FOLD = Placement(rotation=((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
                     translation=(0, 0, 0), denominator=DEN)
#: A 3-fold, whose inverse is a DIFFERENT operation.
THREE_FOLD = Placement(rotation=((0, -1, 0), (1, -1, 0), (0, 0, 1)),
                       translation=(0, 0, 0), denominator=DEN)


def test_a_two_fold_homodimer_counts_its_single_contact_once():
    """dG_diss must be -stab - T*dS, with stab entering exactly once."""
    mass = 20_000.0
    stab = -10.0
    nodes = (("A", IDENTITY_PLACEMENT), ("A", IDENTITY_PLACEMENT.compose(TWO_FOLD)))
    edges = [ContactEdge(molecule_a="A", molecule_b="A", placement=TWO_FOLD,
                         stabilization=stab, area=600.0, interface_id=1)]

    diss, entropy = _score(nodes, edges, {}, {"A": mass})

    expected_entropy = dissociation_entropy([mass, mass])
    assert entropy == pytest.approx(expected_entropy)
    assert diss == pytest.approx(-stab - expected_entropy, abs=1e-9), (
        "the 2-fold contact was counted twice: a self-inverse operation "
        "reaches the same partner node from both directions")


def test_a_heterodimer_is_unaffected():
    """The control: A-B emits one direction per node, so it was always right."""
    stab = -12.0
    nodes = (("A", IDENTITY_PLACEMENT), ("B", IDENTITY_PLACEMENT))
    edges = [ContactEdge(molecule_a="A", molecule_b="B",
                         placement=IDENTITY_PLACEMENT, stabilization=stab,
                         area=600.0, interface_id=1)]

    diss, entropy = _score(nodes, edges, {}, {"A": 20_000.0, "B": 25_000.0})

    assert diss == pytest.approx(-stab - entropy, abs=1e-9)


def test_a_three_fold_trimer_keeps_all_three_contacts():
    """Do NOT fix C1 by dropping the reverse branch: here it is a real contact.

    In a 3-fold trimer node 0 touches both node 1 and node 2, and those are
    different node pairs. Breaking the trimer into a monomer plus a dimer
    cuts two of the three contacts.
    """
    mass = 20_000.0
    stab = -9.0
    second = IDENTITY_PLACEMENT.compose(THREE_FOLD)
    third = second.compose(THREE_FOLD)
    nodes = (("A", IDENTITY_PLACEMENT), ("A", second), ("A", third))
    edges = [ContactEdge(molecule_a="A", molecule_b="A", placement=THREE_FOLD,
                         stabilization=stab, area=600.0, interface_id=1)]

    diss, entropy = _score(nodes, edges, {}, {"A": mass})

    # dG_diss is the MINIMUM over pathways, and a 3-cycle of equal contacts
    # has two: peel one monomer off (2 contacts, 5.45) or release all three
    # (3 contacts, 2.16). Releasing all three is cheaper because the entropy
    # gained from a third free body outweighs the extra contact.
    peel_one = -2 * stab - dissociation_entropy([mass, 2 * mass])
    release_all = -3 * stab - dissociation_entropy([mass, mass, mass])
    assert diss == pytest.approx(min(peel_one, release_all), abs=1e-9)
    # All three contacts must be present for that to be reachable at all.
    assert release_all < peel_one


def test_an_assembly_is_scored_over_every_one_of_its_internal_interfaces():
    """I3: the interfaces used must be ALL those internal to the node set.

    Not merely the top-k subset that happened to close the component during
    the nested search.
    """
    state = run_core(os.path.join(REF, "pdb", "1a3n.pdb.gz"),
                     mode="pisa", symmetry="crystal")
    all_edges = contact_edges(state)
    assemblies = predict_assemblies(state)
    assert assemblies

    for assembly in assemblies:
        members = {(molecule, placement.key())
                   for molecule, placement in assembly.nodes}
        # Independently: an edge is internal when applying it at a member
        # node lands on another member node. Forward only when the edge's
        # first molecule is this node's, reverse only when its second is --
        # the same contract `neighbours` implements, written out here so the
        # expectation does not come from the code under test.
        internal = set()
        for molecule, placement in assembly.nodes:
            for edge in all_edges:
                reached = []
                if edge.molecule_a == molecule:
                    reached.append((edge.molecule_b,
                                    placement.compose(edge.placement)))
                if edge.molecule_b == molecule:
                    reached.append(
                        (edge.molecule_a,
                         placement.compose(edge.placement.inverse())))
                for partner, partner_placement in reached:
                    if ((partner, partner_placement.key()) in members
                            and (partner, partner_placement.key())
                            != (molecule, placement.key())):
                        internal.add(edge.interface_id)
        assert set(assembly.interface_ids) == internal, (
            f"assembly {assembly.rank} ({assembly.composition}) was scored "
            f"over {sorted(assembly.interface_ids)} but its internal "
            f"interfaces are {sorted(internal)}")
