"""Regression tests for three defects found by calibrating against CCP4 PISA v2.2.0.

Ground truth for the calibration was an original CCP4 PISA binary run as
`-analyse` then `-list interfaces`, over 7 antibody-antigen reference
structures (63 interfaces matched to fastPISA's own by chain pair).

What that calibration showed, and what these tests pin:

  1. fastpisa/output/ was absent from the repository -- .gitignore's `output/` rule,
     meant for runtime output directories, also matched the SOURCE package -- so
     `from fastpisa.api import ...` raised ModuleNotFoundError from any clean clone.
  2. The P-value saturated at its clamp floor (0.001) for EVERY interface, because the
     model's mean grew linearly in interface area while its spread grew as sqrt(area).
     A constant field carries no information, and CSS silently inherited that.
  3. asp_table.py and energy.py documented OPPOSITE sign conventions.

Not pinned here, and deliberately: absolute agreement with PISA's dG. fastPISA's
energies are uncalibrated by design (see the warnings in scoring.py and asp_table.py);
these tests check internal consistency and information content, not external accuracy.
"""
import numpy as np
import pytest

from fastpisa.scoring.scoring import calculate_p_value
from fastpisa.energy.asp_table import SIGMA, get_asp


class TestOutputPackageImportable:
    """The blocker: the package must import from a clean checkout."""

    def test_json_output_module_is_importable(self):
        from fastpisa.output.json_output import (
            build_interfaces_json,
            build_assembly_json,
        )
        assert callable(build_interfaces_json)
        assert callable(build_assembly_json)

    def test_api_import_does_not_raise(self):
        from fastpisa.api import PISAInterfaceAnalyzer
        assert PISAInterfaceAnalyzer is not None

    def test_documents_carry_the_keys_every_consumer_indexes(self):
        """cli.py, batch.py, api.summary() and the existing suite pin this layout."""
        from fastpisa.api import PISAInterfaceAnalyzer
        import os

        ktz = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "tests", "data", "1ktz.pdb")
        a = PISAInterfaceAnalyzer(ktz, pdb_id="1ktz", mode="pisa")
        a.analyze()
        asm = a.assembly_json["assembly"]
        for key in ("accessible_surface_area", "buried_surface_area",
                    "dissociation_energy"):
            assert key in asm, f"cli.py and api.summary() read {key}"
        ij = a.interfaces_json["assembly"]
        assert "interface_count" in ij, "batch.py reads interface_count"
        assert isinstance(ij["interfaces"], list), "cocomaps pipeline zips over interfaces"
        assert asm["buried_surface_area"] < asm["accessible_surface_area"]


class TestPValueIsNotDegenerate:
    """The P-value must vary with interface chemistry, not clamp for everything."""

    def test_p_value_varies_across_realistic_interfaces(self):
        # Interfaces spanning the range seen in the references (24 - 3700 A^2),
        # each with a plausible solvation energy for its size.
        cases = [(-2.9, 24.0), (-29.5, 316.0), (-22.7, 468.0),
                 (-19.2, 736.0), (-176.6, 3692.0)]
        ps = [calculate_p_value(solv, area, 20000.0) for solv, area in cases]
        assert len(set(np.round(ps, 4))) > 1, (
            "every P-value is identical -- the model has saturated, which is the "
            "defect this test exists to catch (all 63 calibration interfaces "
            "returned exactly 0.001)")

    def test_p_value_does_not_saturate_with_interface_size(self):
        """A large interface must not be forced to the clamp purely by being large."""
        # Same energy DENSITY, increasing area: p should stay comparable, not collapse.
        density = -0.05  # kcal/mol/A^2
        ps = [calculate_p_value(density * a, a, 20000.0) for a in (100, 500, 1000, 3000)]
        assert max(ps) > 0.002, (
            f"p collapsed to the clamp for a constant energy density: {ps}")
        assert max(ps) - min(ps) < 0.5, (
            f"p should be roughly scale-free in area at fixed density, got {ps}")

    def test_hydrophobic_interface_is_more_significant_than_polar(self):
        area = 800.0
        p_hydrophobic = calculate_p_value(-60.0, area, 20000.0)
        p_polar = calculate_p_value(-2.0, area, 20000.0)
        assert p_hydrophobic < p_polar, (
            "a strongly buried interface must score MORE significant (lower p) "
            f"than a barely buried one of the same area: {p_hydrophobic} vs {p_polar}")

    def test_p_value_stays_in_unit_interval(self):
        for solv in (-500.0, -1.0, 0.0, 25.0):
            for area in (1.0, 250.0, 5000.0):
                p = calculate_p_value(solv, area, 20000.0)
                assert 0.0 <= p <= 1.0


class TestAspCalibratedConvention:
    """The PISA-calibrated sigma table: dG_solv = sum sigma*BSA, negative =
    favourable, exactly like PISA's int_solv_en."""

    def test_hydrophobic_burial_favourable_charged_burial_costly(self):
        assert SIGMA["C"] < 0, "carbon burial must be favourable"
        assert SIGMA["S"] < 0, "sulfur burial must be favourable"
        assert SIGMA["N+"] > 0, "burying charged N must cost energy"
        assert SIGMA["O-"] > 0, "burying carboxylate O must cost energy"

    def test_burying_a_hydrophobic_patch_gives_negative_energy(self):
        from fastpisa.energy.energy import calculate_solvation_energy

        class _Atom:
            def __init__(self, name, element, res="LEU"):
                self.atom_name = name
                self.element = element
                self.res_name = res

        atoms = [_Atom("CD1", "C"), _Atom("CD2", "C")]
        bsa = {0: 40.0, 1: 30.0}
        e = calculate_solvation_energy({0, 1}, bsa, atoms)
        assert e < 0, (
            "burying hydrophobic carbons must yield a negative (favourable) "
            "solvation energy, matching PISA's int_solv_en sign")

    def test_element_fallback_for_unknown_atom_names(self):
        for element in ("C", "S"):
            assert get_asp("ZZ9", element) == pytest.approx(SIGMA[element])

    def test_charged_class_requires_the_residue(self):
        # Lys NZ is charged (its own class, a desolvation PENALTY); the same
        # atom name without residue context is a neutral hetero N.
        from fastpisa.energy.asp_table import DELTA, atom_class
        assert atom_class("NZ", "N", "LYS") == "N_lys"
        assert get_asp("NZ", "N", "LYS") == pytest.approx(
            SIGMA["N_lys"] + DELTA.get("LYS:NZ", 0.0))
        assert get_asp("NZ", "N", "LYS") > 0 > get_asp("CB", "C", "LYS")
        assert get_asp("NZ", "N") == pytest.approx(SIGMA["N"])

    def test_hydrogens_carry_no_solvation(self):
        assert get_asp("H", "H") == 0.0
