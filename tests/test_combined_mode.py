"""Combined mode + unified-core invariants.

All three modes run the same shared core (fastpisa/core.py), so they must
find identical interfaces (IDs, areas, energetics). Combined mode must carry
BOTH the PISA energetics and the COCOMAPS contact map on every interface.
"""
import pytest

from fastpisa.api import PISAInterfaceAnalyzer, analyze_interface

from conftest import KTZ


@pytest.fixture(scope="module")
def results():
    return {
        mode: analyze_interface(KTZ, pdb_id="1ktz", mode=mode)
        for mode in ("pisa", "cocomaps", "combined")
    }


def test_all_modes_identical_interfaces(results):
    ids = {m: [i.interface_id for i in r["interfaces_obj"]]
           for m, r in results.items()}
    assert ids["pisa"] == ids["cocomaps"] == ids["combined"]
    areas = {m: [i.interface_area for i in r["interfaces_obj"]]
             for m, r in results.items()}
    assert areas["pisa"] == areas["cocomaps"] == areas["combined"]
    solv = {m: [i.solvation_energy for i in r["interfaces_obj"]]
            for m, r in results.items()}
    assert solv["pisa"] == solv["cocomaps"] == solv["combined"]


def test_all_modes_same_dissociation_energy(results):
    """The COCOMAPS-mode dG_diss sign/formula bug is fixed: one shared formula."""
    vals = {m: r["assembly"]["assembly"]["dissociation_energy"]
            for m, r in results.items()}
    assert vals["pisa"] == vals["cocomaps"] == vals["combined"]


def test_combined_carries_both_reports(results):
    for iface in results["combined"]["interfaces_obj"]:
        # PISA side
        assert iface.interface_area > 0
        assert iface.solvation_energy != 0
        # COCOMAPS side
        assert iface.cocomaps
        assert "contact_map" in iface.cocomaps
        assert "interaction_population" in iface.cocomaps
        assert iface.cocomaps["num_residue_pairs"] > 0

    docs = results["combined"]["interfaces"]["assembly"]["interfaces"]
    for d in docs:
        assert "interface_contact_map" in d
        assert d["number_hydrogen_bonds"] >= 0


def test_combined_uses_pisa_bond_counts(results):
    """Combined mode reports PISA atom-contact bond counts (not populations)."""
    for pi, ci in zip(results["pisa"]["interfaces_obj"],
                      results["combined"]["interfaces_obj"]):
        assert pi.number_hydrogen_bonds == ci.number_hydrogen_bonds
        assert pi.number_salt_bridges == ci.number_salt_bridges
        assert pi.number_disulfide_bonds == ci.number_disulfide_bonds
        assert pi.number_other_bonds == ci.number_other_bonds


def test_combined_is_default_mode():
    ana = PISAInterfaceAnalyzer(KTZ, pdb_id="1ktz")
    assert ana.mode == "combined"


def test_unknown_mode_raises():
    ana = PISAInterfaceAnalyzer(KTZ, pdb_id="1ktz", mode="bogus")
    with pytest.raises(ValueError, match="Unknown mode"):
        ana.analyze()
