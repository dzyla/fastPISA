"""Cross-molecule neighbour screening.

Interface detection asked "which atoms of molecule i are near molecule j?"
once per molecule PAIR, rebuilding that molecule's index list, coordinate
array and two KD-trees every time. On an assembly with many molecules
(ordered water included, a ribosome, a crystal-packing shell) the O(N^2)
pair loop over per-molecule rebuilds dominated everything else: 82% of the
runtime on 1brs with water (519 molecules), where only 2% of pairs turned
out to be in contact at all.

The replacement answers the question for every pair in ONE pass over the
atoms, so the cost scales with the number of atoms and their contacts rather
than with the square of the molecule count. These tests pin its semantics
exactly, because interface detection -- and therefore every downstream
energy -- depends on it.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from fastpisa.interface.contacts import cross_molecule_neighbors


def _screen(coords, mol_of_atom, cutoff):
    coords = np.asarray(coords, dtype=float)
    return cross_molecule_neighbors(
        coords, np.asarray(mol_of_atom, dtype=int), cKDTree(coords), cutoff)


def test_distant_molecules_produce_no_pair():
    out = _screen([[0, 0, 0], [50, 0, 0]], [0, 1], 5.0)
    assert out == {}


def test_a_touching_pair_reports_the_near_atoms_on_each_side():
    coords = [[0, 0, 0], [0, 0, 20],      # molecule 0: one near, one far
              [3, 0, 0], [40, 0, 0]]      # molecule 1: one near, one far
    out = _screen(coords, [0, 0, 1, 1], 5.0)
    assert set(out) == {(0, 1)}
    near0, near1 = out[(0, 1)]
    assert near0 == [0]
    assert near1 == [2]


def test_the_cutoff_is_inclusive_of_atoms_exactly_within_it():
    coords = [[0, 0, 0], [4.9, 0, 0]]
    assert set(_screen(coords, [0, 1], 5.0)) == {(0, 1)}
    assert _screen(coords, [0, 1], 4.0) == {}


def test_atoms_excluded_from_every_molecule_are_ignored():
    """``-1`` marks atoms outside the interface search (hydrogens, water)."""
    coords = [[0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]]
    out = _screen(coords, [0, -1, 1], 5.0)
    assert set(out) == {(0, 1)}
    assert out[(0, 1)] == ([0], [2])


def test_pairs_are_keyed_low_molecule_first():
    coords = [[0, 0, 0], [2.0, 0, 0]]
    out = _screen(coords, [3, 1], 5.0)
    assert set(out) == {(1, 3)}
    near_low, near_high = out[(1, 3)]
    assert near_low == [1]   # the atom belonging to molecule 1
    assert near_high == [0]  # the atom belonging to molecule 3


def test_near_atom_lists_are_sorted_and_unique():
    rng = np.random.default_rng(7)
    coords = rng.random((200, 3)) * 12.0
    mol = np.where(coords[:, 0] < 6.0, 0, 1)
    out = _screen(coords, mol, 5.0)
    near0, near1 = out[(0, 1)]
    assert near0 == sorted(set(near0))
    assert near1 == sorted(set(near1))


def test_screening_agrees_with_the_brute_force_answer():
    """Exhaustive equivalence on random coordinates and 6 molecules."""
    rng = np.random.default_rng(11)
    coords = rng.random((300, 3)) * 20.0
    mol = rng.integers(0, 6, size=300)
    cutoff = 4.0

    expected: dict = {}
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            mi, mj = int(mol[i]), int(mol[j])
            if mi == mj:
                continue
            if np.linalg.norm(coords[i] - coords[j]) > cutoff:
                continue
            key = (min(mi, mj), max(mi, mj))
            lo, hi = expected.setdefault(key, (set(), set()))
            if mi < mj:
                lo.add(i), hi.add(j)
            else:
                lo.add(j), hi.add(i)

    got = _screen(coords, mol, cutoff)
    assert set(got) == set(expected)
    for key, (lo, hi) in expected.items():
        assert got[key] == (sorted(lo), sorted(hi))


# ---------------------------------------------------------------------------
# End-to-end equivalence on the case the screening targets
# ---------------------------------------------------------------------------
def test_water_inclusive_run_is_unchanged_by_the_screening():
    """Golden values recorded with the pre-screening (O(N^2)) code path.

    Ordered water turns 1ktz into 165 molecules / 13,530 candidate pairs,
    which is the regime the one-pass screen exists for. What the screen
    decides -- WHICH molecule pairs form an interface -- must be identical,
    so the molecule count, the interface count and the full set of pairs are
    pinned exactly. The areas are the surface engine's business (they differ
    by 0.045% between the two ASA backends, which is why they carry a
    relative tolerance here rather than a hardcoded value); they are held to
    closed-form geometry in ``test_sasa_backends.py`` and against PISA in
    ``test_vs_pdbe_pisa.py``.
    """
    from fastpisa.core import run_core

    state = run_core("tests/data/1ktz.pdb", mode="pisa", exclude_water=False)
    assert len(state.molecules) == 165
    assert len(state.interfaces) == 474

    pairs = sorted((i.molecule1_id, i.molecule2_id) for i in state.interfaces)
    assert len(set(pairs)) == 474, "a molecule pair must yield one interface"
    assert pairs[0] == (0, 1)
    assert pairs[1] == (0, 2)
    assert pairs[-1] == (158, 161)

    by_pair = {(i.molecule1_id, i.molecule2_id): i for i in state.interfaces}
    # An absolute 0.5 A^2 on these individual water contacts: a small
    # interface is a difference of two large ASAs, so the two quadratures
    # scatter ~0.2 A^2 on it regardless of its size. Still far tighter than
    # any detection regression (the burial-test bug inflated ASA 6x).
    assert by_pair[(0, 1)].interface_area == pytest.approx(46.71, abs=0.5)
    assert by_pair[(0, 2)].interface_area == pytest.approx(33.47, abs=0.5)
    assert by_pair[(158, 161)].interface_area == pytest.approx(12.30, abs=0.5)
    assert by_pair[(0, 1)].number_hydrogen_bonds == 1
    assert by_pair[(0, 2)].number_hydrogen_bonds == 0

    total_area = sum(i.interface_area for i in state.interfaces)
    total_dg = sum(i.solvation_energy for i in state.interfaces)
    assert total_area == pytest.approx(12473.69, rel=1e-3)
    assert total_dg == pytest.approx(58.55, rel=1e-2)
