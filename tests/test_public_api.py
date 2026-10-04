"""The one-liner a scientist actually wants, and what it hands back.

Getting interfaces out of a structure should be one call returning objects
you can iterate, index and print -- not a dict to spelunk or a two-step
construct-then-analyze dance. ``fastpisa.analyze`` was the only thing at
package level, and it returns the raw JSON documents.
"""

from __future__ import annotations

import os

import pytest

import fastpisa

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data", "reference")
BRS = os.path.join(REF, "pdb", "1brs.pdb.gz")
KTZ = os.path.join(os.path.dirname(REF), "1ktz.pdb")


def test_one_call_returns_interface_objects():
    found = fastpisa.interfaces(KTZ)
    assert found
    assert all(isinstance(i, fastpisa.Interface) for i in found)
    first = found[0]
    assert first.interface_area > 0
    assert isinstance(first.label, str) and "+" in first.label


def test_the_result_is_a_plain_sequence():
    """Iterable, indexable, sized -- no wrapper to learn."""
    found = fastpisa.interfaces(KTZ)
    assert len(found) == len([i for i in found])
    assert found[0] is list(found)[0]
    assert found[-1] is found[len(found) - 1]


def test_interfaces_come_back_largest_first():
    """The interface a user cares about should be ``[0]``."""
    found = fastpisa.interfaces(BRS)
    areas = [i.interface_area for i in found]
    assert areas == sorted(areas, reverse=True)
    assert len(found) > 1


def test_the_bonds_are_reachable_as_objects():
    found = fastpisa.interfaces(BRS)
    biggest = found[0]
    assert biggest.number_hydrogen_bonds > 0
    bonds = biggest.hydrogen_bonds
    assert len(bonds) == biggest.number_hydrogen_bonds
    bond = bonds[0]
    assert isinstance(bond, fastpisa.AtomContact)
    assert " -- " in bond.label
    assert 0.0 < bond.distance <= 3.89


def test_repr_is_informative_enough_to_debug_with():
    found = fastpisa.interfaces(KTZ)
    text = repr(found[0])
    assert "Interface" in text
    assert "A^2" in text and "kcal/mol" in text


def test_options_pass_through():
    """``mode`` reaches the core: only the COCOMAPS modes build a contact map.

    (``ligand_mode`` would be the obvious thing to vary, but it leaves
    1brs's polymer interfaces untouched -- its ions sit away from them -- so
    it proves nothing here.)
    """
    combined = fastpisa.interfaces(KTZ, mode="combined")
    pisa_only = fastpisa.interfaces(KTZ, mode="pisa")
    assert combined and pisa_only
    assert combined[0].contact_map, "combined mode must carry a contact map"
    assert pisa_only[0].contact_map == [], "pisa mode must not"
    # Same interfaces either way -- one shared core, by construction.
    assert ([i.interface_area for i in combined]
            == [i.interface_area for i in pisa_only])


def test_ligand_mode_changes_which_molecules_exist():
    """1a3n's hemes are separate monomers by default, part of the chain merged."""
    hemoglobin = os.path.join(REF, "pdb", "1a3n.pdb.gz")
    separate = fastpisa.interfaces(hemoglobin, ligand_mode="separate")
    merged = fastpisa.interfaces(hemoglobin, ligand_mode="merge")
    assert any("[HEM]" in i.label for i in separate)
    assert not any("[HEM]" in i.label for i in merged)
    assert len(separate) != len(merged)


def test_a_bad_option_is_rejected_not_ignored():
    with pytest.raises(ValueError, match="mode"):
        fastpisa.interfaces(KTZ, mode="nonsense")


def test_a_missing_file_says_so():
    with pytest.raises(FileNotFoundError):
        fastpisa.interfaces("does_not_exist.pdb")


def test_the_package_exports_what_it_documents():
    assert set(fastpisa.__all__) >= {
        "analyze", "interfaces", "Interface", "AtomContact",
        "PISAInterfaceAnalyzer", "__version__"}
    for name in fastpisa.__all__:
        assert hasattr(fastpisa, name), name


def test_analyze_still_returns_the_documents():
    """The existing entry point keeps working unchanged."""
    result = fastpisa.analyze(KTZ, pdb_id="1ktz")
    assert "assembly" in result.assembly_json
    assert result.interfaces
