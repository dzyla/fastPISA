"""The core's public options must actually change the computation.

A knob that is accepted, echoed into provenance and then ignored is worse
than one that does not exist: it invites a reviewer to believe a sensitivity
analysis was done when nothing varied.
"""

from __future__ import annotations

import pytest

from fastpisa.core import run_core
from fastpisa.reference.ebi_pisa import cached_pdb_path

_BRS = cached_pdb_path("1brs")
pytestmark = pytest.mark.skipif(_BRS is None, reason="1brs not cached")


def _contacts(cutoff):
    state = run_core(_BRS, mode="pisa", interface_cutoff=cutoff)
    return [c for iface in state.interfaces for c in iface.contacts]


def test_interface_cutoff_bounds_the_reported_contacts():
    """No contact may be reported beyond the requested cutoff.

    The cutoff used to reach only the interface-ATOM selection while
    ``find_contacts`` kept its own 5.0 A default, so ``interface_cutoff=3.0``
    still reported pairs at 4.96 A.
    """
    contacts = _contacts(3.0)
    assert contacts, "1brs has contacts within 3 A"
    assert max(c.distance for c in contacts) < 3.0


def test_a_wider_interface_cutoff_finds_more_contacts():
    """Raising the cutoff above 5 A used to be a silent no-op."""
    narrow = _contacts(5.0)
    wide = _contacts(6.0)
    assert len(wide) > len(narrow)
    assert max(c.distance for c in wide) > 5.0
    assert max(c.distance for c in wide) < 6.0


# ---------------------------------------------------------------------------
# Bond tables in the PDBe-shaped document
# ---------------------------------------------------------------------------
def test_interfaces_json_carries_the_bond_tables():
    """PDBe PISA's interface entries list the bonds; ours omitted them.

    ``Interface.to_bond_dict`` built exactly these tables but nothing ever
    called it, so a consumer diffing a fastPISA document against a PDBe
    response found the bond lists missing even though every bond had been
    detected and counted.
    """
    import fastpisa

    result = fastpisa.analyze(_BRS, pdb_id="1brs", mode="pisa")
    entries = result.interfaces_json["assembly"]["interfaces"]
    assert entries

    best = max(entries, key=lambda e: e["number_hydrogen_bonds"])
    for key in ("hydrogen_bonds", "salt_bridges", "disulfide_bonds",
                "covalent_bonds"):
        assert key in best, key
        assert "bond_distances" in best[key]

    hbonds = best["hydrogen_bonds"]
    n = best["number_hydrogen_bonds"]
    assert n > 0
    # One entry per counted bond, in every parallel list.
    for field, values in hbonds.items():
        assert len(values) == n, field
    assert all(0.0 < d <= 3.89 for d in hbonds["bond_distances"])
    assert len(hbonds["atom_site_1_chains"]) == n
    assert set(hbonds["atom_site_1_chains"]) | set(hbonds["atom_site_2_chains"])

    sb = best["salt_bridges"]
    assert len(sb["bond_distances"]) == best["number_salt_bridges"]


def test_bond_tables_are_empty_lists_not_nulls_when_absent():
    """A consumer can iterate without null checks."""
    import fastpisa

    result = fastpisa.analyze(_BRS, pdb_id="1brs", mode="pisa")
    for entry in result.interfaces_json["assembly"]["interfaces"]:
        assert isinstance(entry["disulfide_bonds"]["bond_distances"], list)
        if entry["number_disulfide_bonds"] == 0:
            assert entry["disulfide_bonds"]["bond_distances"] == []


# ---------------------------------------------------------------------------
# Distinct molecule pairs must never be merged
# ---------------------------------------------------------------------------
def test_every_molecule_pair_that_buries_area_is_reported_once():
    """One interface per molecule PAIR -- no pair may be collapsed onto another.

    1ppf carries two N-glycan trees (eight sugars each, in chains A and B),
    and PISA reports every sugar as its own monomer: adjacent sugars are
    covalently linked 1.4 A apart and bury real area against each other.
    Identifying a molecule by its chain LETTER makes every chain-A sugar
    pair look like the same ("A", "A") interface, so all but the first are
    dropped -- which is how ``[NAG]A:2 + [BMA]A:3`` and every ``E + sugar``
    beyond the first went missing.
    """
    import os

    from fastpisa.core import run_core

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "reference", "pdb", "1ppf.pdb.gz")
    state = run_core(path, mode="pisa")

    pairs = [(i.molecule1_id, i.molecule2_id) for i in state.interfaces]
    assert len(pairs) == len(set(pairs)), "a molecule pair reported twice"

    labels = {tuple(sorted(m["chain_id"] for m in i.molecules))
              for i in state.interfaces}
    # The glycan tree: covalently linked neighbours, 1.4-1.5 A apart.
    for expected in (("[BMA]A:3", "[NAG]A:2"),
                     ("[BMA]A:3", "[MAN]A:4"),
                     ("[NAG]A:1", "[NAG]A:2"),
                     ("[FUC]A:8", "[NAG]A:1")):
        assert tuple(sorted(expected)) in labels, f"missing {expected}"
    # Both glycan trees contact the protein at more than one sugar.
    protein_sugar = [l for l in labels if "E" in l and any("[" in x for x in l)]
    assert len(protein_sugar) >= 4, sorted(protein_sugar)
    # Chain A and chain B trees are independent: both must appear.
    assert any("]A:" in x for l in labels for x in l)
    assert any("]B:" in x for l in labels for x in l)
