"""Integration tests for fastPISA: mode invariants, energy/BSA, min_css, CLI.

Uses 1ktz.pdb (chain A/B dimer, the canonical small test case) and, when the
external AlphaFold-style structure is available (FASTPISA_EXTERNAL_CIF), an H-free complex.
"""
import json
import os
import subprocess
import sys

import pytest

from fastpisa.api import PISAInterfaceAnalyzer
from fastpisa.energy.energy import (
    calculate_binding_energy, calculate_contact_energy, calculate_solvation_energy,
)

from conftest import KTZ, EXTERNAL_CIF, needs_external_cif, REPO_ROOT


# ---------------------------------------------------------------------------
# Interface invariant: both modes MUST detect identical interface IDs
# ---------------------------------------------------------------------------
class TestInterfaceInvariant:
    def test_1ktz_modes_agree(self):
        p = PISAInterfaceAnalyzer(KTZ, pdb_id="1ktz", mode="pisa").analyze()
        c = PISAInterfaceAnalyzer(KTZ, pdb_id="1ktz", mode="cocomaps").analyze()
        pi = [i["interface_id"] for i in p["interfaces"]["assembly"]["interfaces"]]
        ci = [i["interface_id"] for i in c["interfaces"]["assembly"]["interfaces"]]
        assert sorted(pi) == sorted(ci)
        assert len(pi) >= 1

    @needs_external_cif
    def test_hfree_structure_modes_agree(self):
        p = PISAInterfaceAnalyzer(EXTERNAL_CIF, pdb_id="ext", mode="pisa").analyze()
        c = PISAInterfaceAnalyzer(EXTERNAL_CIF, pdb_id="ext", mode="cocomaps").analyze()
        pi = [i["interface_id"] for i in p["interfaces"]["assembly"]["interfaces"]]
        ci = [i["interface_id"] for i in c["interfaces"]["assembly"]["interfaces"]]
        assert sorted(pi) == sorted(ci)
        assert len(pi) >= 1


# ---------------------------------------------------------------------------
# Disulfide chemistry: no bogus disulfides, every one is Cys-Sg..Cys-Sg
# ---------------------------------------------------------------------------
class TestDisulfideOnStructures:
    @needs_external_cif
    def test_no_bogus_disulfides_hfree(self):
        p = PISAInterfaceAnalyzer(EXTERNAL_CIF, pdb_id="ext", mode="pisa").analyze()
        bad = 0
        for iface in p["interfaces_obj"]:
            for c in iface.contacts:
                if c.bond_type == "disulfide":
                    ok = (c.atom1_residue == "CYS" and c.atom2_residue == "CYS"
                          and c.atom1_name.strip() == "SG"
                          and c.atom2_name.strip() == "SG")
                    if not ok:
                        bad += 1
        assert bad == 0

    @needs_external_cif
    def test_hfree_structure_reports_hbonds(self):
        # H-free structure must still report H-bonds (donor/acceptor rule,
        # not explicit-H based). Regression for the old 2-vs-22 bug.
        p = PISAInterfaceAnalyzer(EXTERNAL_CIF, pdb_id="ext", mode="pisa").analyze()
        big = max(p["interfaces_obj"], key=lambda i: i.interface_area)
        assert big.number_hydrogen_bonds > 0


# ---------------------------------------------------------------------------
# Salt bridges restricted to charged side-chain pairs
# ---------------------------------------------------------------------------
class TestSaltBridgeOnStructures:
    CHARGED = {("ARG", "NH1"), ("ARG", "NH2"), ("ARG", "CZ"),
               ("LYS", "NZ"), ("HIS", "ND1"), ("HIS", "NE2"),
               ("ASP", "OD1"), ("ASP", "OD2"),
               ("GLU", "OE1"), ("GLU", "OE2")}

    @needs_external_cif
    def test_all_salt_bridges_are_charged_side_chains(self):
        p = PISAInterfaceAnalyzer(EXTERNAL_CIF, pdb_id="ext", mode="pisa").analyze()
        bad = []
        for iface in p["interfaces_obj"]:
            for c in iface.contacts:
                if c.bond_type == "salt_bridge":
                    if ((c.atom1_residue, c.atom1_name.strip().upper()) not in self.CHARGED
                            or (c.atom2_residue, c.atom2_name.strip().upper()) not in self.CHARGED):
                        bad.append((c.atom1_residue, c.atom1_name, c.atom2_residue, c.atom2_name))
        assert bad == [], f"non-charged salt bridges: {bad[:5]}"


# ---------------------------------------------------------------------------
# Energy: binding energy must equal solv + contact (no double counting)
# ---------------------------------------------------------------------------
class TestEnergy:
    @needs_external_cif
    def test_binding_energy_no_double_count(self):
        p = PISAInterfaceAnalyzer(EXTERNAL_CIF, pdb_id="ext", mode="pisa").analyze()
        iface = max(p["interfaces_obj"], key=lambda i: i.interface_area)
        ce, _, _, _, _ = calculate_contact_energy(iface.contacts)
        binding = calculate_binding_energy(iface.solvation_energy, iface.contacts)
        assert abs(binding - (iface.solvation_energy + ce)) < 1e-6

    @needs_external_cif
    def test_bsa_less_than_asa(self):
        # Correct BSA convention -> assembly BSA must be < assembly ASA.
        p = PISAInterfaceAnalyzer(EXTERNAL_CIF, pdb_id="ext", mode="pisa").analyze()
        asm = p["assembly"]["assembly"]
        assert asm["buried_surface_area"] < asm["accessible_surface_area"]


# ---------------------------------------------------------------------------
# min_css significance filter
# ---------------------------------------------------------------------------
class TestMinCss:
    @needs_external_cif
    def test_filter_reduces_interfaces_and_agrees_across_modes(self):
        p = PISAInterfaceAnalyzer(EXTERNAL_CIF, pdb_id="ext", mode="pisa").analyze()
        pf = PISAInterfaceAnalyzer(EXTERNAL_CIF, pdb_id="ext", mode="pisa",
                                   min_css=0.5).analyze()
        cf = PISAInterfaceAnalyzer(EXTERNAL_CIF, pdb_id="ext", mode="cocomaps",
                                   min_css=0.5).analyze()
        n_all = len(p["interfaces_obj"])
        n_filt = len(pf["interfaces_obj"])
        assert n_filt < n_all
        assert [i.interface_id for i in pf["interfaces_obj"]] == \
               [i.interface_id for i in cf["interfaces_obj"]]
        assert all(i.css >= 0.5 for i in pf["interfaces_obj"])


# ---------------------------------------------------------------------------
# Python API + CLI
# ---------------------------------------------------------------------------
class TestAPIAndCLI:
    def test_summary(self):
        a = PISAInterfaceAnalyzer(KTZ, pdb_id="1ktz", mode="pisa")
        a.analyze()
        s = a.summary()
        assert "fastPISA" in s and "interfaces" in s

    def test_write_json(self, tmp_path):
        a = PISAInterfaceAnalyzer(KTZ, pdb_id="1ktz", mode="pisa")
        a.analyze()
        paths = a.write_json(str(tmp_path))
        assert os.path.exists(paths["interfaces"])
        assert os.path.exists(paths["assembly"])
        d = json.load(open(paths["interfaces"]))
        assert d["assembly"]["interface_count"] == 1

    def test_cli_pisa(self, tmp_path):
        r = subprocess.run(
            [sys.executable, "-m", "fastpisa.cli", KTZ, "--pdb_id", "1ktz",
             "-o", str(tmp_path), "--json-summary"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert r.returncode == 0, r.stderr
        assert os.path.exists(os.path.join(str(tmp_path), "1ktz-assembly1-interfaces.json"))

    def test_cli_cocomaps(self, tmp_path):
        r = subprocess.run(
            [sys.executable, "-m", "fastpisa.cli", KTZ, "--pdb_id", "1ktz",
             "--mode", "cocomaps", "-o", str(tmp_path)],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert r.returncode == 0, r.stderr
        assert os.path.exists(os.path.join(str(tmp_path), "1ktz-assembly1-interfaces.json"))
