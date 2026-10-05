"""Crystal-mode interface detection, matched against PISA interface by interface.

``symmetry="crystal"`` expands the asymmetric unit by its space group and
reports every interface a molecule makes in the crystal -- what original
PISA does for a deposited entry, and the 60% of its output fastPISA could
not produce.

Matching is exact, not by rank: each interface is keyed by its two chains
and the RELATIVE crystal transform between them, computed the same way on
both sides (PISA reports each molecule's orthogonal transform directly). So
a missing interface and a mismatched one are distinguishable.
"""

from __future__ import annotations

import gzip
import os

import numpy as np
import pytest

gemmi = pytest.importorskip("gemmi")

from fastpisa.core import run_core  # noqa: E402
from fastpisa.reference.ebi_pisa import parse_pisa_xml  # noqa: E402

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "reference")


def _canonical(chain_a, rot_a, tr_a, chain_b, rot_b, tr_b):
    """Chain pair + relative transform, invariant to which side is listed first.

    Everything is reduced to "chain_a in the asymmetric unit, chain_b placed
    by R" by applying the inverse of A's own placement to both. The reverse
    listing gives the inverse transform, so the two orderings are compared
    and the lexicographically smaller one is returned.
    """
    rot_a, rot_b = np.asarray(rot_a, float), np.asarray(rot_b, float)
    tr_a, tr_b = np.asarray(tr_a, float), np.asarray(tr_b, float)

    def rel(r1, t1, r2, t2):
        inv = r1.T
        return inv @ r2, inv @ (t2 - t1)

    r_ab, t_ab = rel(rot_a, tr_a, rot_b, tr_b)
    r_ba, t_ba = rel(rot_b, tr_b, rot_a, tr_a)
    one = (chain_a, chain_b, tuple(np.round(r_ab.ravel(), 2)),
           tuple(np.round(t_ab, 1)))
    two = (chain_b, chain_a, tuple(np.round(r_ba.ravel(), 2)),
           tuple(np.round(t_ba, 1)))
    return min(one, two)


def pisa_keys(pdb_id):
    with gzip.open(os.path.join(REF, f"{pdb_id}.pisa.xml.gz"), "rt") as fh:
        interfaces = parse_pisa_xml(fh.read())
    out = {}
    for iface in interfaces:
        mols = iface["molecules"]
        if len(mols) != 2:
            continue
        key = _canonical(mols[0]["chain_id"], mols[0]["rotation"],
                         mols[0]["translation"],
                         mols[1]["chain_id"], mols[1]["rotation"],
                         mols[1]["translation"])
        out[key] = iface
    return out


def fastpisa_keys(pdb_id, **kwargs):
    state = run_core(os.path.join(REF, "pdb", f"{pdb_id}.pdb.gz"),
                     mode="pisa", symmetry="crystal", **kwargs)
    out = {}
    for iface in state.interfaces:
        m1, m2 = iface.molecules
        key = _canonical(m1["asu_molecule_id"], m1["rotation"], m1["translation"],
                         m2["asu_molecule_id"], m2["rotation"], m2["translation"])
        out[key] = iface
    return out, state


# ---------------------------------------------------------------------------
# Behaviour of the mode itself
# ---------------------------------------------------------------------------
def test_crystal_mode_finds_more_interfaces_than_the_asu_alone():
    path = os.path.join(REF, "pdb", "1acb.pdb.gz")
    asu = run_core(path, mode="pisa")
    crystal = run_core(path, mode="pisa", symmetry="crystal")
    assert len(crystal.interfaces) > len(asu.interfaces)


@pytest.mark.parametrize("pdb_id", ["1acb", "1ppf", "1prc"])
def test_crystal_mode_is_a_superset_of_the_asu_analysis(pdb_id):
    """Expanding the crystal may only ADD interfaces, never drop one.

    1ppf and 1prc are the sharp cases: their chains carry many separate
    hetero groups (a glycan tree, a dozen cofactors), and PISA reports each
    as its own monomer -- 39 of 1ppf's 75 interfaces are ligand-ligand. If
    the symmetry-equivalence key identifies a molecule by its chain letter
    alone, every ligand in chain E collapses onto chain E and those
    interfaces are silently merged away.
    """
    path = os.path.join(REF, "pdb", f"{pdb_id}.pdb.gz")
    asu = run_core(path, mode="pisa")
    crystal = run_core(path, mode="pisa", symmetry="crystal")

    def labels(state):
        return {tuple(sorted(m["chain_id"] for m in i.molecules))
                for i in state.interfaces}

    before, after = labels(asu), labels(crystal)
    assert before
    assert len(crystal.interfaces) >= len(asu.interfaces)
    assert not (before - after), (
        f"{pdb_id}: crystal mode lost {len(before - after)} of "
        f"{len(before)} ASU interfaces, e.g. {sorted(before - after)[:3]}")


def test_asu_interfaces_are_unchanged_by_crystal_mode():
    """Adding mates must not perturb an interface that was already there.

    Interface area is isolated-ASA minus pair-ASA over the two molecules
    only, so a third molecule cannot change it. If this drifts, the pair ASA
    is leaking the surroundings.
    """
    path = os.path.join(REF, "pdb", "1acb.pdb.gz")
    asu = run_core(path, mode="pisa")
    crystal = run_core(path, mode="pisa", symmetry="crystal")

    def by_chains(state):
        return {
            tuple(sorted(m["chain_id"] for m in i.molecules)): i
            for i in state.interfaces
        }

    before = by_chains(asu)
    after = by_chains(crystal)
    assert before
    for chains, iface in before.items():
        assert chains in after, chains
        assert after[chains].interface_area == pytest.approx(
            iface.interface_area, abs=0.01)
        assert after[chains].solvation_energy == pytest.approx(
            iface.solvation_energy, abs=0.01)
        assert after[chains].number_hydrogen_bonds == iface.number_hydrogen_bonds


def test_a_structure_without_a_cell_is_unaffected(tmp_path):
    """A predicted model has no CRYST1: crystal mode must be a no-op.

    (Built by stripping CRYST1 from a deposited file rather than reusing one
    -- every cached entry is a crystal structure and therefore has a cell.)
    """
    source = os.path.join(os.path.dirname(REF), "1ktz.pdb")
    stripped = tmp_path / "no_cell.pdb"
    with open(source) as fh:
        stripped.write_text("".join(
            line for line in fh if not line.startswith("CRYST1")))

    plain = run_core(str(stripped), mode="pisa")
    crystal = run_core(str(stripped), mode="pisa", symmetry="crystal")
    assert len(crystal.interfaces) == len(plain.interfaces) >= 1
    assert all(m["cell"] == (0, 0, 0)
               for i in crystal.interfaces for m in i.molecules)


def test_a_cell_without_a_space_group_is_a_no_op(tmp_path):
    """Some converted files carry a cell but no usable symbol."""
    source = os.path.join(os.path.dirname(REF), "1ktz.pdb")
    patched = tmp_path / "no_sg.pdb"
    lines = []
    with open(source) as fh:
        for line in fh:
            if line.startswith("CRYST1"):
                line = line[:55] + "\n"
            lines.append(line)
    patched.write_text("".join(lines))

    plain = run_core(str(patched), mode="pisa")
    crystal = run_core(str(patched), mode="pisa", symmetry="crystal")
    assert len(crystal.interfaces) == len(plain.interfaces)


def test_unknown_symmetry_option_is_rejected():
    with pytest.raises(ValueError, match="symmetry"):
        run_core(os.path.join(REF, "pdb", "1acb.pdb.gz"), symmetry="nonsense")


def test_every_interface_names_its_crystal_transform():
    state = run_core(os.path.join(REF, "pdb", "1acb.pdb.gz"),
                     mode="pisa", symmetry="crystal")
    for iface in state.interfaces:
        for mol in iface.molecules:
            assert "symop" in mol and "cell" in mol
            assert "asu_chain_id" in mol
            rot = np.asarray(mol["rotation"], dtype=float)
            assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-6)
        # At least one side must be the asymmetric unit itself: every
        # mate-mate contact is symmetry-equivalent to an ASU-mate one.
        assert any(m["cell"] == (0, 0, 0) and m["symop_no"] == 1
                   for m in iface.molecules)


def test_symmetry_equivalent_interfaces_are_reported_once():
    """A + mate(B, S) and B + mate(A, S-inverse) are one interface."""
    keys, state = fastpisa_keys("1acb")
    assert len(keys) == len(state.interfaces), (
        "duplicate symmetry-equivalent interfaces in the output")


# ---------------------------------------------------------------------------
# Against PISA
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pdb_id", ["1acb", "1cgi", "1oph", "1stf", "1ay7", "1ktz"])
def test_pisa_interfaces_are_all_found(pdb_id):
    """Every PISA interface must be matched by chains AND relative transform."""
    ref = pisa_keys(pdb_id)
    got, state = fastpisa_keys(pdb_id)
    missing = sorted(set(ref) - set(got))
    assert not missing, (
        f"{pdb_id}: {len(missing)}/{len(ref)} PISA interfaces not found "
        f"(fastPISA reported {len(got)}). First missing: chains "
        f"{missing[0][0]}+{missing[0][1]} area "
        f"{ref[missing[0]]['int_area']:.1f} A^2")


@pytest.mark.parametrize("pdb_id", ["1acb", "1cgi", "1oph", "1stf", "1ay7", "1ktz"])
def test_matched_interface_areas_agree_with_pisa(pdb_id):
    ref = pisa_keys(pdb_id)
    got, _ = fastpisa_keys(pdb_id)
    shared = sorted(set(ref) & set(got))
    assert len(shared) >= max(1, int(0.8 * len(ref)))
    errors = []
    for key in shared:
        a, b = ref[key]["int_area"], got[key].interface_area
        if a and a > 100:                      # relative error is meaningless below this
            errors.append(abs(b - a) / a)
    assert errors
    assert float(np.median(errors)) < 0.05, (
        f"{pdb_id}: median area error {np.median(errors):.1%} over "
        f"{len(errors)} interfaces > 100 A^2")


# ---------------------------------------------------------------------------
# Reachable from the public surfaces
# ---------------------------------------------------------------------------
def test_analyzer_exposes_crystal_mode():
    from fastpisa.api import PISAInterfaceAnalyzer

    path = os.path.join(REF, "pdb", "1acb.pdb.gz")
    asu = PISAInterfaceAnalyzer(path, pdb_id="1acb", mode="pisa")
    crystal = PISAInterfaceAnalyzer(path, pdb_id="1acb", mode="pisa",
                                    symmetry="crystal")
    asu.analyze()
    crystal.analyze()
    assert len(crystal.interfaces) > len(asu.interfaces)
    assert crystal.analysis_provenance()["symmetry"] == "crystal"
    assert asu.analysis_provenance()["symmetry"] == "none"
    assert "symmetry" in asu.analysis_provenance()["coordinate_scope"].lower() \
        or "no symmetry" in asu.analysis_provenance()["coordinate_scope"]


def test_analyzer_rejects_an_unknown_symmetry_mode():
    from fastpisa.api import PISAInterfaceAnalyzer

    path = os.path.join(REF, "pdb", "1acb.pdb.gz")
    with pytest.raises(ValueError, match="symmetry"):
        PISAInterfaceAnalyzer(path, symmetry="nope").analyze()


def test_cli_accepts_crystal_symmetry(tmp_path):
    import subprocess
    import sys

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run(
        [sys.executable, "-m", "fastpisa.cli",
         os.path.join(REF, "pdb", "1acb.pdb.gz"), "--pdb_id", "1acb",
         "--mode", "pisa", "--symmetry", "crystal", "-o", str(tmp_path)],
        capture_output=True, text=True, cwd=repo, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "Interfaces found" in out.stdout
    import json
    doc = json.load(open(tmp_path / "1acb-assembly1-interfaces.json"))
    assert doc["assembly"]["interface_count"] >= 9


def test_json_records_the_symmetry_operation_of_each_molecule():
    """A crystal interface is only interpretable with its operation."""
    import fastpisa

    result = fastpisa.analyze(os.path.join(REF, "pdb", "1acb.pdb.gz"),
                              pdb_id="1acb", mode="pisa", symmetry="crystal")
    entries = result.interfaces_json["assembly"]["interfaces"]
    assert entries
    mates = 0
    for entry in entries:
        for mol in entry["molecules"]:
            assert "symop" in mol and "cell" in mol and "symop_no" in mol
            if mol["symop_no"] != 1 or tuple(mol["cell"]) != (0, 0, 0):
                mates += 1
    assert mates > 0, "1acb's packing interfaces must name their operators"
    assert any(e.get("n_occurrences", 1) >= 1 for e in entries)


# ---------------------------------------------------------------------------
# Offline regression over diverse space groups
# ---------------------------------------------------------------------------
#: One cached entry per distinct space-group setting, so the lattice
#: arithmetic is exercised across primitive, centred, screw-axis,
#: rhombohedral and hexagonal settings rather than on one crystal form.
#: 1ktz is H 3 2 (18 operations) and 1prc P 43 21 2; 4ins is H 3, where the
#: fractional 3-fold's inverse is NOT its transpose, which is what caught
#: the equivalence-key bug.
SPACE_GROUP_SAMPLE = [
    "1acb",   # P 1 21 1
    "1brs",   # C 1 2 1
    "1aay",   # C 2 2 21
    "1urn",   # P 65 2 2
    "1eaw",   # P 1
    "1ppf",   # P 21 21 21
    "1cgi",   # P 41 21 2
    "1dfj",   # I 4
    "1ktz",   # H 3 2
    "1prc",   # P 43 21 2
    "1stf",   # P 31 2 1
    "2ptc",   # I 2 2 2
    "4ins",   # H 3
    "9ant",   # P 2 2 21
]


def _polymer(molecules):
    return all(m.get("molecule_class") in ("Protein", "NucleicAcid")
               for m in molecules)


def _relative(rot_a, tr_a, rot_b, tr_b):
    rot_a, rot_b = np.asarray(rot_a, float), np.asarray(rot_b, float)
    tr_a, tr_b = np.asarray(tr_a, float), np.asarray(tr_b, float)
    return rot_a.T @ rot_b, rot_a.T @ (tr_b - tr_a)


@pytest.mark.parametrize("pdb_id", SPACE_GROUP_SAMPLE)
def test_all_pisa_polymer_interfaces_are_reproduced(pdb_id):
    """Every polymer-polymer interface PISA reports must be found and sized.

    Matching is on the chain pair AND the relative crystal transform, with a
    tolerance rather than rounded floats: PISA prints its matrices to a few
    decimals, so a 6-fold screw's c/6 = 42.55 A appears as 42.6 against our
    42.55.

    Measured over all 37 cached entries: 566/566 matched, area median error
    1.20%, dG Pearson 0.995. Over a blind draw of 60 random entries from the
    sampling frame: 732/732 matched, area median 1.16%, dG Pearson 0.991.
    """
    with gzip.open(os.path.join(REF, f"{pdb_id}.pisa.xml.gz"), "rt") as fh:
        reference = parse_pisa_xml(fh.read())
    wanted = []
    for iface in reference:
        mols = iface["molecules"]
        if len(mols) != 2:
            continue
        if not all(m["class"] in ("Protein", "NucleicAcid", "DNA", "RNA")
                   for m in mols):
            continue
        rot, tran = _relative(mols[0]["rotation"], mols[0]["translation"],
                              mols[1]["rotation"], mols[1]["translation"])
        wanted.append((frozenset(m["chain_id"] for m in mols), rot, tran,
                       iface))
    assert wanted, f"{pdb_id} has no polymer-polymer PISA interfaces"

    state = run_core(os.path.join(REF, "pdb", f"{pdb_id}.pdb.gz"),
                     mode="pisa", symmetry="crystal")
    produced = []
    for iface in state.interfaces:
        if not _polymer(iface.molecules):
            continue
        a, b = iface.molecules
        rot, tran = _relative(a["rotation"], a["translation"],
                              b["rotation"], b["translation"])
        produced.append((frozenset((a["asu_molecule_id"],
                                    b["asu_molecule_id"])), rot, tran, iface))

    used, errors, missing = set(), [], []
    for chains, rot, tran, iface in wanted:
        best = None
        for index, (chains2, rot2, tran2, ours) in enumerate(produced):
            if index in used or chains2 != chains:
                continue
            for candidate_rot, candidate_tran in (
                    (rot2, tran2), (rot2.T, -rot2.T @ tran2)):
                if (np.abs(candidate_rot - rot).max() < 0.02
                        and np.abs(candidate_tran - tran).max() < 0.8):
                    distance = np.abs(candidate_tran - tran).max()
                    if best is None or distance < best[0]:
                        best = (distance, index, ours)
                    break
        if best is None:
            missing.append(iface)
            continue
        used.add(best[1])
        if iface["int_area"] and iface["int_area"] > 100:
            errors.append(abs(best[2].interface_area - iface["int_area"])
                          / iface["int_area"])

    assert not missing, (
        f"{pdb_id}: {len(missing)}/{len(wanted)} PISA interfaces not found; "
        f"first is {sorted(missing[0]['molecules'][0]['chain_id'])} area "
        f"{missing[0]['int_area']:.1f} A^2")
    if errors:
        assert float(np.median(errors)) < 0.06, (
            f"{pdb_id}: median area error {np.median(errors):.1%}")


def test_crystal_mode_cost_is_bounded():
    """Expansion must stay affordable: only mates that TOUCH are built.

    1prc is 10k atoms in P 43 21 2 (8 operations) and its asymmetric unit
    holds 27 molecules, so a cell window of 3x3x3 per operation screens
    thousands of candidate placements. What has to stay small is how many of
    them are built and analysed: measured 2026-10-04, 135 mates against 27
    ASU molecules, 5 of the 8 operations contributing, 4 distinct cell
    shifts.

    This assertion used to be wall clock alone (``< 60 s``, measured 5.1 s),
    which is not a property of the pruning. The pure-Python surface engine is
    ~13x slower than FreeSASA by design, so on CI's runners the identical
    work took 65-67 s and every ``FASTPISA_SASA_BACKEND=python`` leg failed
    against a budget written on the accelerated one -- a red suite for a
    hardware difference the repository documents. The structural bound is the
    real invariant; the timing check stays as a runaway tripwire, with a
    budget per backend.
    """
    import time

    from fastpisa.surface.shrake_rupley import active_backend

    path = os.path.join(REF, "pdb", "1prc.pdb.gz")
    asu = run_core(path, mode="pisa", symmetry="none")

    start = time.monotonic()
    state = run_core(path, mode="pisa", symmetry="crystal")
    elapsed = time.monotonic() - start

    assert state.interfaces

    def _is_mate(molecule):
        return (molecule.get("symop_no", 1) != 1
                or tuple(molecule.get("cell") or (0, 0, 0)) != (0, 0, 0))

    mates = [m for m in state.molecules if _is_mate(m)]
    identity = [m for m in state.molecules if not _is_mate(m)]

    # The asymmetric unit itself must come through unchanged...
    assert len(identity) == len(asu.molecules)
    # ...and the mates built must be the touching few, not the window. An
    # expansion that stopped pruning would build 8 operations x 27 cells.
    assert len(mates) < 10 * len(asu.molecules), (
        f"{len(mates)} mates built from {len(asu.molecules)} ASU molecules "
        f"-- pruning looks lost")

    budget = 60.0 if active_backend() == "freesasa" else 300.0
    assert elapsed < budget, (
        f"crystal mode on 1prc took {elapsed:.1f}s on the "
        f"{active_backend()} backend (budget {budget:.0f}s)")
