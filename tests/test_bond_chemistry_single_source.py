"""One hydrogen-bond chemistry table, used by every consumer.

fastPISA carried three donor/acceptor definitions: the calibrated one in
:mod:`fastpisa.interface.bonds`, a second table in
:mod:`fastpisa.cocomaps.interactions` (``HBOND_ATOMS_AA``), and
``contacts.is_hydrogen_bond`` built on the second. They disagreed -- and
because ``find_contacts`` classified every contact with the second table
before :func:`fastpisa.interface.bonds.detect_bond_flags` overwrote the
verdict, the disagreement was invisible while still being computed.

These tests pin the invariant rather than the implementation: whatever
answers "is this atom a donor / an acceptor", it must answer the same way
everywhere.
"""

from __future__ import annotations

import pytest

from fastpisa.interface import bonds
from fastpisa.interface.contacts import is_hydrogen_bond

# (residue, atom, element) triples that exercise the places the tables
# disagreed: charged side chains, proline's N (no H to donate), the ribose
# 2'-OH (donates AND accepts), sulfur acceptors, and nucleobase edges.
PROBES = [
    ("ARG", "NE", "N"), ("ARG", "NH1", "N"), ("LYS", "NZ", "N"),
    ("ASP", "OD1", "O"), ("GLU", "OE2", "O"), ("ASN", "ND2", "N"),
    ("GLN", "OE1", "O"), ("SER", "OG", "O"), ("THR", "OG1", "O"),
    ("TYR", "OH", "O"), ("TRP", "NE1", "N"), ("HIS", "ND1", "N"),
    ("CYS", "SG", "S"), ("MET", "SD", "S"), ("MSE", "SE", "SE"),
    ("ALA", "N", "N"), ("ALA", "O", "O"), ("ALA", "OXT", "O"),
    ("PRO", "N", "N"),
    ("DA", "N6", "N"), ("DA", "N1", "N"), ("DG", "O6", "O"),
    ("DT", "N3", "N"), ("DC", "N4", "N"),
    ("DA", "OP1", "O"), ("DA", "O2'", "O"), ("DA", "O3'", "O"),
    ("A", "O2'", "O"),
    ("HOH", "O", "O"), ("GOL", "O1", "O"), ("NAG", "N2", "N"),
]


def test_is_hydrogen_bond_uses_the_calibrated_role_table():
    """``is_hydrogen_bond`` must agree with the detector's own chemistry.

    It used to consult a second table in which proline's N was a donor and
    the ribose 2'-OH could not donate.
    """
    close = 3.0
    for res1, a1, e1 in PROBES:
        for res2, a2, e2 in PROBES:
            r1 = bonds.hb_roles(res1, a1, e1)
            r2 = bonds.hb_roles(res2, a2, e2)
            expected = (("donor" in r1 and "acceptor" in r2)
                        or ("donor" in r2 and "acceptor" in r1))
            got = is_hydrogen_bond(res1, a1, e1, res2, a2, e2, close)
            assert got is expected, (
                f"{res1}:{a1} vs {res2}:{a2}: is_hydrogen_bond={got} but the "
                f"calibrated roles are {sorted(r1)} / {sorted(r2)}")


def test_is_hydrogen_bond_still_respects_the_distance_cutoff():
    assert is_hydrogen_bond("ASN", "ND2", "N", "ASP", "OD1", "O", 3.0) is True
    assert is_hydrogen_bond("ASN", "ND2", "N", "ASP", "OD1", "O", 4.5) is False


def test_cocomaps_classifier_has_no_private_hbond_table():
    """The second donor/acceptor table must be gone, not merely unused.

    A dormant duplicate is a future divergence: the COCOMAPS classifier's
    ``is_hbond=None`` path would silently reintroduce it.
    """
    from fastpisa.cocomaps import interactions

    assert not hasattr(interactions, "HBOND_ATOMS_AA")
    assert not hasattr(interactions, "_atoms_roles")


def test_cocomaps_classifier_falls_back_to_the_shared_chemistry():
    """With no geometric verdict supplied, the shared table decides."""
    from fastpisa.cocomaps.interactions import classify_atom_pair

    kwargs = dict(el1="N", el2="O", dist=3.0, vdw_radius1=1.55,
                  vdw_radius2=1.52)
    donor_acceptor = classify_atom_pair(
        res1="ASN", atom1="ND2", res2="ASP", atom2="OD1", **kwargs)
    assert donor_acceptor == "hydrogen_bond"

    # Proline's N has no hydrogen to donate: not an H-bond under the
    # calibrated table, whatever the old one said.
    no_donor = classify_atom_pair(
        res1="PRO", atom1="N", res2="ASP", atom2="OD1", **kwargs)
    assert no_donor != "hydrogen_bond"


def test_find_contacts_leaves_classification_to_the_detector():
    """Geometry and chemistry are separate steps, run once each.

    ``find_contacts`` used to label every pair with the duplicate table; the
    labels were then thrown away. It now reports geometry only.
    """
    import numpy as np

    from fastpisa.interface.contacts import find_contacts
    from fastpisa.parser.pdb_parser import Atom

    def atom(name, res, el, x):
        return Atom(atom_name=name, altloc=" ", res_name=res, chain_id="A",
                    res_seq=1, icode="", x=x, y=0.0, z=0.0, occupancy=1.0,
                    bfactor=0.0, element=el, label_asym_id="A", label_seq_id=1,
                    label_comp_id=res, auth_asym_id="A", auth_seq_id=1)

    atoms = [atom("ND2", "ASN", "N", 0.0), atom("OD1", "ASP", "O", 2.9)]
    contacts = find_contacts(atoms, [0], [1], contact_cutoff=5.0)
    assert len(contacts) == 1
    assert contacts[0].bond_type == "other"
    assert contacts[0].bond_types == ()


# ---------------------------------------------------------------------------
# Ring membership must agree with the ring geometry it gates
# ---------------------------------------------------------------------------
def test_aromatic_atom_sets_contain_only_real_ring_atoms():
    """``AROMATIC_RING_ATOMS`` gates the pi classes that ``rings.py`` verifies.

    It listed Tyr OH (a hydroxyl substituent) and thymine C7 (the 5-methyl)
    as ring atoms. Neither is in a ring, and neither appears in
    ``rings.RING_GROUPS``, so the two tables disagreed about what a ring is.
    """
    from fastpisa.cocomaps.interactions import (
        AROMATIC_RING_ATOMS, NUCLEOBASE_AROMATIC,
    )
    from fastpisa.cocomaps.rings import RING_GROUPS

    for res, names in AROMATIC_RING_ATOMS.items():
        geometric = {a for group in RING_GROUPS[res] for a in group}
        assert names <= geometric, f"{res}: {sorted(names - geometric)}"

    for res, names in NUCLEOBASE_AROMATIC.items():
        geometric = {a for group in RING_GROUPS[res] for a in group}
        assert names <= geometric, f"{res}: {sorted(names - geometric)}"


def test_the_apolar_class_predicate_has_no_dead_branch():
    """``cls != "X"`` was unreachable: no C/S/NA_C class is named ``X``."""
    from fastpisa.energy.asp_table import SIGMA, is_apolar_class

    assert is_apolar_class("X") is False
    apolar = {c for c in SIGMA if is_apolar_class(c)}
    assert "X" not in apolar
    assert {"C_CA", "C_ali", "C_aro", "S_met", "S_cys", "NA_C_base"} <= apolar
    assert not apolar & {"N_bb", "O_bb", "NA_OP", "MET", "ZN", "HAL", "P"}
