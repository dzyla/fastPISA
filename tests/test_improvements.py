"""Regression tests for the 0.2.0 improvements.

Guards the "single source of truth" claims:
  * PISA and COCOMAPS modes report IDENTICAL contact counts (H-bonds, salt
    bridges, disulfides) on the same structure -- item 3.2 of
    fastpisa_improvements.md. Requires the shared chemistry changes
    (HBOND_DISTANCE alignment + removal of the blanket generic-covalent rule).
  * No bogus generic "covalent" bond type (crystal self-copies are not bonds).
  * Modified nucleotides are recognised as polymer nucleic acids.
  * calculate_bsa (old BSA convention) is deprecated.
  * hot_spot_residues() and CLI --version work.
"""
import os
import subprocess
import sys

import pytest

from fastpisa.api import PISAInterfaceAnalyzer
from fastpisa.parser.pdb_parser import Atom, Chain, PDBStructure
from fastpisa.interface.contacts import get_molecules, NUCLEIC_ACIDS

from conftest import KTZ, EXTERNAL_CIF, needs_external_cif, REPO_ROOT


# ---------------------------------------------------------------------------
# PISA == COCOMAPS contact-count agreement (single source of truth)
# ---------------------------------------------------------------------------
def _assert_counts_agree(path):
    p = PISAInterfaceAnalyzer(path, pdb_id="x", mode="pisa").analyze()["interfaces_obj"]
    c = PISAInterfaceAnalyzer(path, pdb_id="x", mode="cocomaps").analyze()["interfaces_obj"]
    assert len(p) == len(c) >= 1
    for pi, ci in zip(p, c):
        assert pi.interface_id == ci.interface_id
        for attr in ("number_hydrogen_bonds", "number_salt_bridges",
                     "number_disulfide_bonds"):
            assert getattr(pi, attr) == getattr(ci, attr), (
                f"interface {pi.interface_id} {attr}: PISA={getattr(pi, attr)} "
                f"!= COCOMAPS={getattr(ci, attr)}"
            )


class TestModeCountAgreement:
    def test_1ktz_counts_agree(self):
        _assert_counts_agree(KTZ)

    @needs_external_cif
    def test_hfree_counts_agree(self):
        _assert_counts_agree(EXTERNAL_CIF)


# ---------------------------------------------------------------------------
# No generic "covalent" bond type (the old blanket d < 2.2 A rule)
# ---------------------------------------------------------------------------
class TestNoGenericCovalent:
    def test_no_covalent_bond_type_in_pisa_mode(self):
        p = PISAInterfaceAnalyzer(KTZ, pdb_id="x", mode="pisa").analyze()
        for iface in p["interfaces_obj"]:
            assert iface.number_covalent_bonds == 0
            assert all(ct.bond_type != "covalent" for ct in iface.contacts)

    @needs_external_cif
    def test_no_covalent_on_hfree(self):
        p = PISAInterfaceAnalyzer(EXTERNAL_CIF, pdb_id="x", mode="pisa").analyze()
        for iface in p["interfaces_obj"]:
            assert iface.number_covalent_bonds == 0


# ---------------------------------------------------------------------------
# Modified nucleotides are recognised as polymer nucleic acids (item 2.4)
# ---------------------------------------------------------------------------
def _atom(name, res, seq, auth_seq, chain="X"):
    return Atom(
        atom_name=name, altloc="", res_name=res, chain_id=chain, res_seq=seq,
        icode="", x=0.0, y=0.0, z=0.0, occupancy=1.0, bfactor=0.0,
        element=name[0], label_asym_id=chain, label_seq_id=seq,
        label_comp_id=res, auth_asym_id=chain, auth_seq_id=auth_seq,
    )


def _na_structure(residue, n_res=3):
    chain = Chain(auth_asym_id="X", label_asym_id="X")
    for seq in range(1, n_res + 1):
        for aname in ("P", "O5'", "C1'"):
            chain.atoms.append(_atom(aname, residue, seq, seq))
    return PDBStructure(chains=[chain])


class TestNucleicAcidRecognition:
    @pytest.mark.parametrize("code", ["5MC", "PSU", "7MG", "2MG", "H2U", "DA"])
    def test_modified_nucleotide_is_polymer(self, code):
        assert code in NUCLEIC_ACIDS
        mols = get_molecules(_na_structure(code))
        # exactly one polymer NucleicAcid molecule, no per-residue ligands
        assert len(mols) == 1
        assert mols[0]["chain_type"] == "polymer"
        assert mols[0]["molecule_class"] == "NucleicAcid"


# ---------------------------------------------------------------------------
# calculate_bsa is deprecated in favour of compute_buried_surface
# ---------------------------------------------------------------------------
class TestCalculateBsaDeprecated:
    def test_emits_deprecation_warning(self):
        from fastpisa.parser.pdb_parser import parse_pdb
        from fastpisa.surface.shrake_rupley import calculate_bsa
        atoms = parse_pdb(KTZ).atoms
        with pytest.warns(DeprecationWarning):
            calculate_bsa({0: 10.0}, atoms[:2])


# ---------------------------------------------------------------------------
# hot_spot_residues + CLI --version
# ---------------------------------------------------------------------------
class TestHotSpotsAndVersion:
    def test_hot_spot_residues_sorted(self):
        a = PISAInterfaceAnalyzer(KTZ, pdb_id="1ktz", mode="pisa")
        a.analyze()
        hs = a.hot_spot_residues(top_n=3)
        assert len(hs) <= 3
        assert all(hs[i]["bsa"] >= hs[i + 1]["bsa"] for i in range(len(hs) - 1))
        assert all(set(h["interfaces"]) for h in hs)

    def test_hot_spot_residues_solv_rank(self):
        a = PISAInterfaceAnalyzer(KTZ, pdb_id="1ktz", mode="pisa")
        a.analyze()
        hs = a.hot_spot_residues(top_n=5, by="solv")
        assert all(hs[i]["solvation_energy"] <= hs[i + 1]["solvation_energy"]
                   for i in range(len(hs) - 1))

    def test_cli_version(self):
        r = subprocess.run(
            [sys.executable, "-m", "fastpisa.cli", "--version"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip().startswith("fastpisa ")
