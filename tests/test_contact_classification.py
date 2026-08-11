"""Unit tests for the fixed atom-chemistry contact classifier.

These guard the three bugs that previously made PISA-mode output scientifically
unsound: bogus disulfides (any <3.0 A pair), H-bonds requiring explicit H atoms,
and salt bridges flagged from any N-O pair.
"""
import pytest

from fastpisa.interface.contacts import (
    is_disulfide, is_salt_bridge, is_hydrogen_bond,
)


# ---------------------------------------------------------------------------
# Disulfide
# ---------------------------------------------------------------------------
class TestDisulfide:
    def test_real_cys_ss(self):
        # Two Cys S-gamma atoms, 2.05 A, short distance -> disulfide
        assert is_disulfide("CYS", "CYS", "S", "S", 2.05) is True

    def test_s_s_within_3_but_not_cys(self):
        # sulfur-sulfur but Met SD ... Met SD is NOT a disulfide
        assert is_disulfide("MET", "MET", "S", "S", 2.5) is False

    def test_cys_but_one_s_only(self):
        # Cys-Cys but only one is sulfur (shouldn't happen, guard anyway)
        assert is_disulfide("CYS", "CYS", "S", "C", 2.2) is False

    def test_no_carbon_carbon_false_positive(self):
        # The old bug: any short pair counted as disulfide
        assert is_disulfide("LEU", "ASP", "C", "O", 2.5) is False

    def test_too_far(self):
        assert is_disulfide("CYS", "CYS", "S", "S", 3.5) is False

    def test_case_insensitive(self):
        assert is_disulfide("cys", "Cys", "s", "S", 2.0) is True


# ---------------------------------------------------------------------------
# Salt bridge
# ---------------------------------------------------------------------------
class TestSaltBridge:
    def test_lys_glu_salt_bridge(self):
        # Lys NZ(+) ... Glu OE2(-) is a genuine salt bridge
        assert is_salt_bridge("LYS", "NZ", "GLU", "OE2", 2.8) is True

    def test_arg_asp_salt_bridge(self):
        assert is_salt_bridge("ARG", "NH1", "ASP", "OD1", 3.2) is True

    def test_not_backbone_pair(self):
        # A backbone amide N ... carbonyl O is an H-bond, NOT a salt bridge.
        # This was the old bug (any N-O < 4.0 A).
        assert is_salt_bridge("LEU", "N", "GLN", "O", 3.0) is False

    def test_same_charge_rejected(self):
        assert is_salt_bridge("LYS", "NZ", "ARG", "NH1", 2.9) is False

    def test_too_far(self):
        assert is_salt_bridge("LYS", "NZ", "GLU", "OE2", 4.5) is False

    def test_unrecognized_atom_rejected(self):
        # Gly has no charged side-chain atom
        assert is_salt_bridge("GLY", "CA", "GLU", "OE2", 2.8) is False


# ---------------------------------------------------------------------------
# Hydrogen bond (no explicit H required)
# ---------------------------------------------------------------------------
class TestHydrogenBond:
    def test_asn_donor_asp_acceptor(self):
        # Asn ND2 (donor) ... Asp OD1 (acceptor)
        assert is_hydrogen_bond("ASN", "ND2", "N", "ASP", "OD1", "O", 3.0) is True

    def test_ser_og_backbone_o(self):
        # Ser OG is donor+acceptor; pair with a backbone carbonyl O
        assert is_hydrogen_bond("SER", "OG", "O", "XXX", "O", "O", 3.1) is True

    def test_two_acceptor_oxygens_not_hbond(self):
        # Thr OG1 ... Thr OG1 (both donor+acceptor) is fine, but the classic
        # bug case: two acceptor-only backbone carbonyl O's are NOT an H-bond.
        # Backbone O is acceptor-only; a contact of O(generic backbone) with
        # another O acceptor-only must be rejected when neither is a donor.
        assert is_hydrogen_bond("ALA", "O", "O", "ALA", "O", "O", 3.0) is False

    def test_non_no_pair_rejected(self):
        assert is_hydrogen_bond("LEU", "CG", "C", "LEU", "CG", "C", 3.0) is False

    def test_too_far(self):
        assert is_hydrogen_bond("ASN", "ND2", "N", "ASP", "OD1", "O", 4.5) is False
