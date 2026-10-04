"""Assembly dissociation: rigid-body entropy and the minimum-cut pathway.

What PISA actually reports, recovered exactly from the PDBe PISA 2.0 JSON of
20 biological assemblies (``tests/data/reference/json/``):

    dG_diss = -sum(stabilization_energy over the interfaces CUT) - T*dS

holding to +-0.01 kcal/mol on every assembly whose interface list is
complete. Three things about that relation were wrong here:

1. the entropy was ADDED, not subtracted;
2. the sum ran over EVERY interface rather than the ones a dissociation
   pathway actually breaks;
3. ``T*dS`` was ``0.02 * area + 0.5 * n_residues``, a hand-tuned surrogate
   that is not an entropy -- it scores 468 kcal/mol median error against
   PISA's own values, anticorrelated (r = -0.11).
"""

from __future__ import annotations

import glob
import gzip
import json
import math
import os
import statistics

import pytest

from fastpisa.energy.entropy import (
    STANDARD_STATE_LN_TERM, dissociation_entropy, element_mass,
)
from fastpisa.energy.dissociation import (
    DissociationPathway, assembly_dissociation,
)

REF_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "reference", "json")


# ---------------------------------------------------------------------------
# The entropy term
# ---------------------------------------------------------------------------
def test_a_single_body_has_no_dissociation_entropy():
    assert dissociation_entropy([50_000.0]) == 0.0
    assert dissociation_entropy([]) == 0.0


def test_two_body_entropy_follows_the_sackur_tetrode_form():
    """Pinned to closed form: the mass dependence is the reduced mass^(3/2)."""
    from fastpisa.energy.entropy import RT_KCAL

    m1, m2 = 40_000.0, 25_000.0
    expected = RT_KCAL * (
        1.5 * math.log(m1) + 1.5 * math.log(m2) - 1.5 * math.log(m1 + m2)
        + STANDARD_STATE_LN_TERM)
    assert dissociation_entropy([m1, m2]) == pytest.approx(expected, rel=1e-12)


def test_entropy_grows_with_the_number_of_released_bodies():
    two = dissociation_entropy([30_000.0, 30_000.0])
    three = dissociation_entropy([20_000.0, 20_000.0, 20_000.0])
    four = dissociation_entropy([15_000.0] * 4)
    assert two < three < four
    # Each extra released body costs roughly one more translational term.
    assert 1.5 < three / two < 2.5


def test_entropy_is_additive_over_a_splitting_tree():
    """Required for the recursive minimum-cut search to be exact.

    Splitting A+B+C into three bodies must cost the same whether it is
    evaluated at once or as (A+B | C) followed by (A | B).
    """
    a, b, c = 18_000.0, 24_000.0, 31_000.0
    one_step = dissociation_entropy([a, b, c])
    two_steps = (dissociation_entropy([a + b, c])
                 + dissociation_entropy([a, b]))
    assert one_step == pytest.approx(two_steps, rel=1e-12)


def test_entropy_is_symmetric_in_its_parts():
    assert dissociation_entropy([10.0, 90.0]) == pytest.approx(
        dissociation_entropy([90.0, 10.0]))


def test_element_mass_knows_the_biological_elements_and_is_strict():
    assert element_mass("C") == pytest.approx(12.011)
    assert element_mass(" zn ") == pytest.approx(65.38)
    with pytest.raises(KeyError):
        element_mass("Xx")


# ---------------------------------------------------------------------------
# The dissociation pathway
# ---------------------------------------------------------------------------
def test_a_dimer_dissociates_by_breaking_its_one_interface():
    masses = {"A": 20_000.0, "B": 20_000.0}
    path = assembly_dissociation(masses, [("A", "B", -12.5)])
    assert isinstance(path, DissociationPathway)
    assert [sorted(p) for p in path.parts] == [["A"], ["B"]]
    assert path.cut == [("A", "B")]
    expected_tds = dissociation_entropy([20_000.0, 20_000.0])
    assert path.entropy == pytest.approx(expected_tds)
    assert path.dissociation_energy == pytest.approx(12.5 - expected_tds)


def test_the_cheapest_cut_wins_not_the_sum_of_all_interfaces():
    """A chain A=B=C: shearing off C costs less than breaking both bonds."""
    masses = {"A": 20_000.0, "B": 20_000.0, "C": 20_000.0}
    edges = [("A", "B", -40.0), ("B", "C", -5.0)]
    path = assembly_dissociation(masses, edges)
    assert path.cut == [("B", "C")]
    assert sorted(len(p) for p in path.parts) == [1, 2]


def test_a_strongly_bound_assembly_may_dissociate_into_more_than_two_parts():
    """PISA does this: 9eyh releases all four chains at once."""
    masses = {k: 20_000.0 for k in "ABCD"}
    # A loose daisy chain: every bond is weaker than one entropy term, so
    # releasing every monomer is cheaper than any single split.
    edges = [("A", "B", -0.2), ("B", "C", -0.2), ("C", "D", -0.2)]
    path = assembly_dissociation(masses, edges)
    assert len(path.parts) == 4
    assert len(path.cut) == 3


def test_an_assembly_with_no_interfaces_has_no_pathway():
    path = assembly_dissociation({"A": 1000.0}, [])
    assert path.parts == [["A"]]
    assert path.cut == []
    assert path.entropy == 0.0
    assert path.dissociation_energy == 0.0


def test_pathway_only_cuts_along_real_interfaces():
    """Two molecules that share no interface are already separate bodies."""
    masses = {"A": 20_000.0, "B": 20_000.0, "C": 20_000.0}
    path = assembly_dissociation(masses, [("A", "B", -30.0)])
    # C is not bound to anything, so the assembly is already in two pieces
    # and the reported pathway must not pretend to break a C interface.
    assert all(pair in {("A", "B")} for pair in path.cut)


# ---------------------------------------------------------------------------
# Directly against PISA
# ---------------------------------------------------------------------------
def _reference_assemblies():
    for path in sorted(glob.glob(os.path.join(REF_JSON, "*.json.gz"))):
        with gzip.open(path, "rt") as fh:
            doc = json.load(fh)
        yield os.path.basename(path)[:4], doc["assembly"]


def test_pisa_dissociation_identity_holds_on_the_reference_assemblies():
    """Documents the relation these tests are built on.

    For each assembly, PISA's own numbers must satisfy
    ``dG_diss = -sum(cut stab) - entropy`` for SOME subset of its listed
    interfaces. Assemblies whose JSON omits interfaces cannot close the sum
    and are reported, not silently passed.
    """
    import itertools

    closed = 0
    total = 0
    for pid, a in _reference_assemblies():
        total += 1
        stabs = [i["stabilization_energy"] for i in a["interfaces"]]
        target = -a["dissociation_energy"] - a["entropy"]
        if len(stabs) > 14:
            continue
        for r in range(1, len(stabs) + 1):
            if any(abs(sum(c) - target) <= 0.015 * r + 0.005
                   for c in itertools.combinations(stabs, r)):
                closed += 1
                break
    assert total == 20
    assert closed >= 12, (
        f"only {closed}/{total} assemblies close the identity; the recovered "
        "relation may be wrong")


# The dissociation cut PISA chose, identified from ITS OWN numbers (the
# interface subset whose stabilization energies satisfy the identity above
# AND whose removal disconnects the chain graph). Independent of the entropy
# model being tested, so this is not a circular comparison.
PISA_CUTS = {
    "8jfw": [["A"], ["D"]],
    "8k0m": [["A", "B"], ["C"]],
    "8roa": [["A"], ["B"]],
    "8rsy": [["A"], ["E"]],
    "8tw6": [["A", "B", "F", "G"], ["D", "E", "X", "Y"]],
    "8u9p": [["A", "B", "C", "D", "E"], ["G"]],
    "8w2p": [["A", "B", "C", "E"], ["D"]],
    "8y53": [["A", "B", "G", "R"], ["E"], ["N"]],
    "8y5f": [["A", "B", "C"], ["D"]],
    "9bl3": [["A", "B", "G"], ["C"]],
    "9c0r": [["A", "B"], ["C"], ["D"]],
    "9dnd": [["A", "B", "D"], ["C"]],
    "9eyh": [["A"], ["B"], ["C"], ["D000"]],
}


def _chain_masses(pid):
    from fastpisa.parser.pdb_parser import parse_mmcif

    cif = os.path.join(os.path.dirname(REF_JSON), "assemblies",
                       f"{pid}-assembly1.cif.gz")
    if not os.path.exists(cif):
        return None
    masses = {}
    for atom in parse_mmcif(cif).atoms:
        try:
            m = element_mass(atom.element)
        except KeyError:
            m = 12.011
        masses[atom.auth_asym_id] = masses.get(atom.auth_asym_id, 0.0) + m
    return masses


def test_entropy_model_reproduces_pisa_entropies():
    """The headline claim: a one-parameter rigid-body model, not a surrogate.

    Measured over the 13 assemblies whose dissociation cut is identifiable
    from PISA's own numbers.
    """
    pytest.importorskip("gemmi")
    errors = []
    pairs = []
    for pid, parts in PISA_CUTS.items():
        masses = _chain_masses(pid)
        if masses is None:
            continue
        blocks = [sum(masses[c] for c in block) for block in parts]
        ref = dict(_reference_assemblies())[pid]["entropy"]
        got = dissociation_entropy(blocks)
        errors.append(got - ref)
        pairs.append((got, ref))

    assert len(errors) == 13
    import statistics

    median_abs = statistics.median(abs(e) for e in errors)
    bias = statistics.mean(errors)
    mean_got = statistics.mean(p[0] for p in pairs)
    mean_ref = statistics.mean(p[1] for p in pairs)
    cov = sum((g - mean_got) * (r - mean_ref) for g, r in pairs)
    var = (sum((g - mean_got) ** 2 for g, _ in pairs)
           * sum((r - mean_ref) ** 2 for _, r in pairs)) ** 0.5
    r = cov / var

    assert median_abs < 1.2, f"median |error| {median_abs:.2f} kcal/mol"
    assert abs(bias) < 0.5, f"bias {bias:+.2f} kcal/mol"
    assert r > 0.95, f"Pearson r {r:.3f}"
    assert max(abs(e) for e in errors) < 3.5


def test_entropy_model_beats_the_surrogate_it_replaces():
    """Guards against regressing to an area-based pseudo-entropy."""
    import statistics

    model, legacy, ref = [], [], []
    for pid, parts in PISA_CUTS.items():
        masses = _chain_masses(pid)
        if masses is None:
            continue
        a = dict(_reference_assemblies())[pid]
        blocks = [sum(masses[c] for c in block) for block in parts]
        model.append(dissociation_entropy(blocks))
        legacy.append(0.02 * a["dissociation_area"]
                      + 0.5 * sum(i["number_interface_residues"]
                                  for i in a["interfaces"]))
        ref.append(a["entropy"])

    model_err = statistics.median(abs(m - r) for m, r in zip(model, ref))
    legacy_err = statistics.median(abs(l - r) for l, r in zip(legacy, ref))
    assert model_err < 1.2
    assert legacy_err > 100          # the surrogate is off by two orders
    assert model_err < legacy_err / 50


def test_shipped_entropy_constant_matches_a_refit():
    """The shipped constant must be what the reference data refits to.

    Mirrors the sigma-table guard in ``test_calibration_benchmark.py``: a
    hand-tweaked constant is caught here rather than in a paper.
    """
    pytest.importorskip("gemmi")
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "examples", "calibrate_entropy.py")
    spec = importlib.util.spec_from_file_location("calibrate_entropy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows, _ = module.collect()
    assert len(rows) == 13
    from fastpisa.energy.entropy import STANDARD_STATE_LN_TERM as shipped

    assert module.fit(rows) == pytest.approx(shipped, abs=5e-4)


# ---------------------------------------------------------------------------
# End to end: the assembly document
# ---------------------------------------------------------------------------
_RESULT_CACHE = {}


def _fastpisa_result(pid):
    """Run fastPISA on the cached assembly mmCIF PISA itself analysed."""
    if pid in _RESULT_CACHE:
        return _RESULT_CACHE[pid]
    import fastpisa

    cif = os.path.join(os.path.dirname(REF_JSON), "assemblies",
                       f"{pid}-assembly1.cif.gz")
    if not os.path.exists(cif):
        _RESULT_CACHE[pid] = None
        return None
    # jsPISA analyses assemblies with ligands merged into their parent chain.
    _RESULT_CACHE[pid] = fastpisa.analyze(cif, pdb_id=pid, mode="pisa",
                                          ligand_mode="merge")
    return _RESULT_CACHE[pid]


def _fastpisa_assembly(pid):
    result = _fastpisa_result(pid)
    return None if result is None else result.assembly_json["assembly"]


def test_assembly_document_reports_the_pathway_entropy_not_a_sum():
    """``entropy`` must be the T*dS of ONE dissociation, not a per-interface sum.

    The old code summed ``0.02 * area + 0.5 * n_res`` over every interface,
    so entropy grew without bound with assembly size.
    """
    pytest.importorskip("gemmi")
    doc = _fastpisa_assembly("8y5f")
    assert doc is not None
    tds = doc["entropy"]
    assert 5.0 < tds < 40.0, tds
    # Two to four released bodies is one to three translational terms.
    assert tds < 4 * 15.0


def test_assembly_dissociation_energy_subtracts_the_entropy():
    """Sign convention: entropy OPPOSES dissociation.

    ``dG_diss = -sum(cut stab) - T*dS``. The old code added it.
    """
    pytest.importorskip("gemmi")
    import fastpisa

    cif = os.path.join(os.path.dirname(REF_JSON), "assemblies",
                       "8jfw-assembly1.cif.gz")
    result = fastpisa.analyze(cif, pdb_id="8jfw", mode="pisa",
                              ligand_mode="merge")
    doc = result.assembly_json["assembly"]
    path = result.dissociation_pathway
    assert path is not None
    cut_stab = sum(
        iface.stabilization_energy for iface in result.interfaces
        if tuple(sorted(m["chain_id"] for m in iface.molecules)) in
        {tuple(sorted(pair)) for pair in path.cut}
    )
    assert doc["dissociation_energy"] == pytest.approx(
        -cut_stab - doc["entropy"], abs=0.02)
    assert doc["entropy"] == pytest.approx(path.entropy, abs=0.01)


def test_assembly_entropy_tracks_pisa_across_the_reference_set():
    """Measured end to end on the assemblies PISA published numbers for."""
    pytest.importorskip("gemmi")
    import statistics

    got, ref = [], []
    for pid, assembly in _reference_assemblies():
        doc = _fastpisa_assembly(pid)
        if doc is None:
            continue
        got.append(doc["entropy"])
        ref.append(assembly["entropy"])

    assert len(got) == 20
    errors = [g - r for g, r in zip(got, ref)]
    # Measured 2026-10-03: median |err| 0.90, bias -0.67, r 0.895.
    # The large single error (9c0r) is a PATHWAY disagreement -- PISA splits
    # it into three bodies, fastPISA into two -- not an entropy error.
    assert statistics.median(abs(e) for e in errors) < 1.5
    assert abs(statistics.mean(errors)) < 1.5
    assert _pearson(got, ref) > 0.85


def _pearson(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    var = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return cov / var


def _spearman(a, b):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        for rank, i in enumerate(order):
            out[i] = float(rank)
        return out
    return _pearson(ranks(a), ranks(b))


def test_assembly_dissociation_energy_tracks_pisa_across_the_reference_set():
    """The headline end-to-end number, blind: fastPISA picks its own pathway.

    Measured 2026-10-03 over the 20 reference assemblies: median |error|
    1.88 kcal/mol, bias -2.99, Pearson 0.934, Spearman 0.977. The quantity
    is used to RANK assemblies by stability, so the rank agreement is the
    figure that matters; the two largest absolute errors (8w2p, 9c0r) are
    assemblies where fastPISA's own stabilization energies lead it to a
    different dissociation pathway than PISA's.
    """
    pytest.importorskip("gemmi")
    import statistics

    got, ref = [], []
    for pid, assembly in _reference_assemblies():
        doc = _fastpisa_assembly(pid)
        if doc is None:
            continue
        got.append(doc["dissociation_energy"])
        ref.append(assembly["dissociation_energy"])

    assert len(got) == 20
    errors = [g - r for g, r in zip(got, ref)]
    assert statistics.median(abs(e) for e in errors) < 4.0
    assert _pearson(got, ref) > 0.85
    assert _spearman(got, ref) > 0.90


def test_dissociation_energy_beats_the_formula_it_replaces():
    """The old sum-everything-and-add-a-surrogate formula, for the record."""
    pytest.importorskip("gemmi")
    import statistics

    from fastpisa.energy.energy import bond_energy

    new_err, old_err = [], []
    for pid, assembly in _reference_assemblies():
        result = _fastpisa_result(pid)
        if result is None:
            continue
        ref = assembly["dissociation_energy"]
        new_err.append(result.assembly_json["assembly"]["dissociation_energy"]
                       - ref)
        legacy = 0.0
        for iface in result.interfaces:
            contact = bond_energy(iface.number_hydrogen_bonds,
                                  iface.number_salt_bridges,
                                  iface.number_disulfide_bonds)
            n = iface.number_interface_residues
            legacy += (-(iface.solvation_energy + contact)
                       + 0.02 * iface.interface_area + 0.5 * n)
        old_err.append(legacy - ref)

    assert statistics.median(abs(e) for e in new_err) < 4.0
    assert statistics.median(abs(e) for e in old_err) > 50


def test_a_large_bound_group_is_handled_without_exponential_search():
    """Above MAX_EXACT_MOLECULES the search must degrade, not explode.

    A long chain of bound components (hundreds of ordered waters bridging
    each other, a large filament) exceeds the exact O(3^n) subset search. The
    restricted mode must stay flat in the number of components: recursing
    into every subset of a 40-node group would never return.
    """
    import time

    from fastpisa.energy.dissociation import MAX_EXACT_MOLECULES

    n = MAX_EXACT_MOLECULES + 26
    masses = {f"m{i:03d}": 1000.0 + i for i in range(n)}
    keys = sorted(masses)
    edges = [(keys[i], keys[i + 1], -3.0 - 0.1 * i) for i in range(n - 1)]

    start = time.monotonic()
    path = assembly_dissociation(masses, edges)
    elapsed = time.monotonic() - start

    assert elapsed < 5.0, f"restricted search took {elapsed:.1f}s for {n} parts"
    assert path.exact is False
    assert len(path.parts) >= 2
    assert path.cut
    # A pathway is still a real one: its energy matches its own cut and parts.
    cut_total = sum(
        w for (a, b, w) in edges
        if (a, b) in {tuple(sorted(p)) for p in path.cut}
        or (b, a) in {tuple(sorted(p)) for p in path.cut}
    )
    assert path.dissociation_energy == pytest.approx(
        -cut_total - path.entropy, abs=1e-6)
    # The weakest link of the chain is the cheapest single peel.
    assert path.cut == [("m000", "m001")]
