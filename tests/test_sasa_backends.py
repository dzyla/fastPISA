"""Numerical correctness of the two ASA backends, and their equivalence.

The pure-Python Shrake-Rupley implementation is the fallback used whenever
``freesasa`` is not installed, so it carries the full weight of the fitted
solvation model on its own. These tests pin it to CLOSED-FORM geometry
(a lone sphere, and the exact spherical-cap area two overlapping probe
spheres hide from each other) rather than to the other backend alone, so a
bug cannot hide behind an agreeing pair.

Both backends must run the SAME algorithm (Shrake-Rupley) with the SAME
point count, because the shipped ASP sigmas were fitted to one of them and
are applied by whichever happens to be installed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

from fastpisa.parser.pdb_parser import Atom, parse_pdb
from fastpisa.surface import freesasa_backend as fsb
from fastpisa.surface.shrake_rupley import (
    calculate_asa_python, surface_radius,
)

from conftest import KTZ

PROBE = 1.4
# Tolerance for a 4096-point Shrake-Rupley quadrature against closed form.
QUAD_REL = 0.01


def _atom(x, y, z, element="C", name="CB", res="ALA"):
    """One heavy atom at a position, with a definite surface radius."""
    return Atom(
        atom_name=name, altloc=" ", res_name=res, chain_id="A", res_seq=1,
        icode="", x=x, y=y, z=z, occupancy=1.0, bfactor=20.0, element=element,
        label_asym_id="A", label_seq_id=1, label_comp_id=res,
        auth_asym_id="A", auth_seq_id=1,
    )


def _python_asa(atoms, point_density=4096, probe_radius=PROBE):
    """Pure-Python ASA over a self-contained atom list."""
    ids = list(range(len(atoms)))
    coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=float)
    radii = np.array([surface_radius(a) for a in atoms], dtype=float)
    return calculate_asa_python(
        atoms=atoms, atom_indices=ids, probe_radius=probe_radius,
        point_density=point_density, combined_coords=coords,
        combined_radii=radii, kd_tree=cKDTree(coords),
    )


def _cap_area(r_self, r_other, d, probe=PROBE):
    """Closed-form area of ``r_self``'s probe sphere hidden inside ``r_other``'s.

    A probe centre on atom i's accessible sphere (radius ``r_i + probe``) is
    inaccessible exactly when it lies within ``r_j + probe`` of atom j. The
    hidden region is a spherical cap; for sphere radii R1, R2 at separation d
    the cap on sphere 1 has height ``R1 - (d^2 + R1^2 - R2^2) / (2 d)`` and
    area ``2 * pi * R1 * h``.
    """
    r1, r2 = r_self + probe, r_other + probe
    if d >= r1 + r2:
        return 0.0
    h = r1 - (d * d + r1 * r1 - r2 * r2) / (2.0 * d)
    return 2.0 * math.pi * r1 * h


# ---------------------------------------------------------------------------
# Closed-form geometry
# ---------------------------------------------------------------------------
def test_lone_atom_asa_is_the_whole_probe_sphere():
    atoms = [_atom(0.0, 0.0, 0.0)]
    r = surface_radius(atoms[0])
    expected = 4.0 * math.pi * (r + PROBE) ** 2
    assert _python_asa(atoms)[0] == pytest.approx(expected, rel=1e-9)


def test_two_atoms_hide_exactly_the_analytic_spherical_cap():
    """The burial test must use ``r_j + probe``, not ``r_j``.

    Omitting the probe radius leaves the probe sphere occluded only by the
    bare van-der-Waals sphere, which overstates ASA several-fold.
    """
    d = 3.2
    atoms = [_atom(0.0, 0.0, 0.0), _atom(d, 0.0, 0.0)]
    r = surface_radius(atoms[0])
    full = 4.0 * math.pi * (r + PROBE) ** 2
    expected = full - _cap_area(r, r, d)
    asa = _python_asa(atoms)
    assert asa[0] == pytest.approx(expected, rel=QUAD_REL)
    assert asa[1] == pytest.approx(expected, rel=QUAD_REL)


def test_occluders_beyond_twice_the_radius_are_not_truncated():
    """The neighbour search must reach ``r_i + r_j + 2 * probe``.

    Two potassium ions (surface radius 2.75 A) still shadow each other at
    7.95 A, which a ``2 * r_max + probe + 1`` cutoff would miss entirely.
    """
    d = 7.95
    atoms = [_atom(0.0, 0.0, 0.0, element="K", name="K", res="K"),
             _atom(d, 0.0, 0.0, element="K", name="K", res="K")]
    r = surface_radius(atoms[0])
    assert d < 2.0 * r + 2.0 * PROBE, "test premise: the pair still overlaps"
    full = 4.0 * math.pi * (r + PROBE) ** 2
    expected = full - _cap_area(r, r, d)
    assert _python_asa(atoms)[0] == pytest.approx(expected, rel=QUAD_REL)
    assert _python_asa(atoms)[0] < full


def test_buried_atom_inside_a_shell_has_zero_asa():
    """An atom fully enclosed by neighbours must report no accessible area."""
    centre = _atom(0.0, 0.0, 0.0)
    shell = []
    golden = math.pi * (3.0 - 5.0 ** 0.5)
    for i in range(60):
        y = 1.0 - 2.0 * i / 59.0
        rad = math.sqrt(max(1.0 - y * y, 0.0))
        th = golden * i
        shell.append(_atom(3.0 * math.cos(th) * rad, 3.0 * y,
                           3.0 * math.sin(th) * rad))
    asa = _python_asa([centre] + shell, point_density=1024)
    assert asa[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Backend equivalence
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not fsb.available(), reason="freesasa not installed")
def test_python_and_freesasa_agree_on_a_real_structure():
    """Same algorithm, same points: the backends must be interchangeable."""
    structure = parse_pdb(KTZ)
    atoms = [a for a in structure.atoms
             if a.element.strip().upper() not in ("H", "D")
             and a.res_name.strip().upper() not in ("HOH", "WAT")]
    ids = list(range(len(atoms)))

    py = _python_asa(atoms, point_density=480)
    fs = fsb.calculate_asa_freesasa(atoms=atoms, atom_indices=ids,
                                    point_density=480)
    a = np.array([py[i] for i in ids])
    b = np.array([fs[i] for i in ids])

    assert a.sum() == pytest.approx(b.sum(), rel=0.002)
    diff = np.abs(a - b)
    assert np.median(diff) < 0.5
    assert diff.max() < 5.0


def test_point_density_drives_the_python_quadrature():
    """``point_density`` is a real knob, not a silently ignored argument."""
    structure = parse_pdb(KTZ)
    atoms = [a for a in structure.atoms
             if a.element.strip().upper() not in ("H", "D")][:200]
    coarse = sum(_python_asa(atoms, point_density=42).values())
    fine = sum(_python_asa(atoms, point_density=4096).values())
    assert coarse != fine
    assert coarse == pytest.approx(fine, rel=0.03)


@pytest.mark.skipif(not fsb.available(), reason="freesasa not installed")
def test_freesasa_algorithm_is_pinned_explicitly(monkeypatch):
    """Never inherit FreeSASA's default: it has changed, and S&R is unsafe.

    ``surface_backend_info`` must report the algorithm actually configured,
    so provenance in a paper's methods matches what ran. The engine is pinned
    here rather than inherited from the environment: this test is about the
    FreeSASA path, and the ambient ``FASTPISA_SASA_BACKEND`` may select the
    other one.
    """
    monkeypatch.setenv("FASTPISA_SASA_BACKEND", "freesasa")
    assert fsb.FREESASA_ALGORITHM == "LeeRichards"
    params = fsb.lee_richards_parameters(fsb.LEE_RICHARDS_SLICES, 1.4)
    assert str(params.algorithm()) == fsb.FREESASA_ALGORITHM
    assert params.nSlices() == fsb.LEE_RICHARDS_SLICES
    info = fsb.surface_backend_info()
    assert info["algorithm"] == fsb.FREESASA_ALGORITHM
    assert info["quadrature"] == f"{fsb.LEE_RICHARDS_SLICES} slices"


@pytest.mark.skipif(not fsb.available(), reason="freesasa not installed")
def test_freesasa_survives_spatially_sparse_subsets():
    """Guards the reason FreeSASA is pinned to Lee-Richards.

    FreeSASA 2.2.1's Shrake-Rupley kernel SEGFAULTS on inputs whose atoms are
    sparse in a large bounding box -- e.g. two atoms 20 A apart, which the
    per-molecule and per-pair ASA calls produce routinely. A segfault cannot
    be caught, so it would abort a user's run mid-analysis. Switching this
    backend to Shrake-Rupley makes this test crash the whole suite, which is
    exactly the alarm we want.
    """
    atoms = [_atom(0.0, 0.0, 0.0), _atom(20.0, 0.0, 0.0)]
    r = surface_radius(atoms[0])
    expected = 4.0 * math.pi * (r + PROBE) ** 2
    out = fsb.calculate_asa_freesasa(atoms=atoms, atom_indices=[0, 1])
    assert out[0] == pytest.approx(expected, rel=0.02)
    assert out[1] == pytest.approx(expected, rel=0.02)


@pytest.mark.skipif(not fsb.available(), reason="freesasa not installed")
def test_freesasa_single_atom_is_the_analytic_sphere():
    """A lone ion is a real input (``[ZN]A:301``) and must not crash."""
    atoms = [_atom(0.0, 0.0, 0.0, element="ZN", name="ZN", res="ZN")]
    r = surface_radius(atoms[0])
    expected = 4.0 * math.pi * (r + PROBE) ** 2
    got = fsb.calculate_asa_freesasa(atoms=atoms, atom_indices=[0])[0]
    assert got == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
def test_backend_can_be_forced_to_python(monkeypatch):
    """Reproducibility: a run must be able to pin the surface engine.

    The fitted sigmas are applied by whichever backend is installed, so
    comparing them -- and calibrating against one of them -- requires an
    explicit switch rather than "whatever is importable".
    """
    from fastpisa.surface import shrake_rupley as sr

    monkeypatch.setenv("FASTPISA_SASA_BACKEND", "python")
    assert sr.active_backend() == "python"
    monkeypatch.setenv("FASTPISA_SASA_BACKEND", "auto")
    assert sr.active_backend() == ("freesasa" if fsb.available() else "python")


def test_unknown_backend_name_is_rejected(monkeypatch):
    from fastpisa.surface import shrake_rupley as sr

    monkeypatch.setenv("FASTPISA_SASA_BACKEND", "nonsense")
    with pytest.raises(ValueError, match="FASTPISA_SASA_BACKEND"):
        sr.active_backend()


@pytest.mark.skipif(not fsb.available(), reason="freesasa not installed")
def test_provenance_follows_the_pinned_backend(monkeypatch):
    """Reported provenance must describe the engine that will RUN.

    ``surface_backend_info`` keyed off "is freesasa importable", so pinning
    ``FASTPISA_SASA_BACKEND=python`` on a machine with freesasa installed ran
    the Shrake-Rupley engine while the methods paragraph claimed FreeSASA /
    Lee-Richards -- the same class of mislabelling this module exists to stop.
    """
    monkeypatch.setenv("FASTPISA_SASA_BACKEND", "python")
    info = fsb.surface_backend_info(point_density=480)
    assert info["backend"] == "python"
    assert info["algorithm"] == "Shrake-Rupley"
    assert info["quadrature"] == "480 sphere points"

    monkeypatch.setenv("FASTPISA_SASA_BACKEND", "freesasa")
    info = fsb.surface_backend_info()
    assert info["backend"] == "FreeSASA"
    assert info["algorithm"] == fsb.FREESASA_ALGORITHM


def test_python_asa_is_independent_of_atom_order():
    """Guards the nearest-neighbour-first early exit.

    The Python engine sorts each atom's neighbours by distance and stops once
    no test point survives, which is ~2.2x faster and bit-identical. Both the
    sort and the early exit could make a result depend on the order atoms
    arrive in; it must not.
    """
    import random

    structure = parse_pdb(KTZ)
    atoms = [a for a in structure.atoms
             if a.element.strip().upper() not in ("H", "D")
             and a.res_name.strip().upper() not in ("HOH", "WAT")][:300]

    straight = _python_asa(atoms, point_density=480)

    order = list(range(len(atoms)))
    random.Random(4).shuffle(order)
    shuffled_atoms = [atoms[i] for i in order]
    shuffled = _python_asa(shuffled_atoms, point_density=480)

    for position, original in enumerate(order):
        assert shuffled[position] == pytest.approx(straight[original],
                                                   abs=1e-9), original


def test_a_fully_buried_atom_short_circuits_to_zero():
    """The early exit must yield exactly zero, not a small residue."""
    centre = _atom(0.0, 0.0, 0.0)
    shell = []
    golden = math.pi * (3.0 - 5.0 ** 0.5)
    for i in range(80):
        y = 1.0 - 2.0 * i / 79.0
        radius = math.sqrt(max(1.0 - y * y, 0.0))
        theta = golden * i
        shell.append(_atom(2.8 * math.cos(theta) * radius, 2.8 * y,
                           2.8 * math.sin(theta) * radius))
    asa = _python_asa([centre] + shell, point_density=1024)
    assert asa[0] == 0.0
