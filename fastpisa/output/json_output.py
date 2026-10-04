"""Build the two PDBe-PISA-shaped JSON documents.

Both builders return a dict with a single top-level "assembly" key, matching the PDBe
PISA API layout that fastPISA reproduces, because every consumer indexes it that way:
`api.summary()` reads assembly_json["assembly"]["accessible_surface_area"], `cli.py`
reads the same three assembly fields, `batch.py` reads
interfaces["assembly"]["interface_count"], the cocomaps pipeline zips over
interfaces_json["assembly"]["interfaces"], and tests/test_pipeline.py asserts on
`d["assembly"]["interface_count"]` and `p["assembly"]["assembly"]["buried_surface_area"]`.
Changing the nesting therefore breaks the CLI, the batch runner and the test suite at once.

Interface entries are serialised from the `Interface` dataclass. Field names follow the
PDBe vocabulary (`interface_area`, `solvation_energy`, `number_hydrogen_bonds`, ...) so a
document can be diffed against a PDBe response.

CALIBRATION STATUS. Do not quote numbers from this file: it is the serialiser, and
its copy of the benchmark went stale once (it advertised the 2026-08-29 figures from
117 interfaces / 21 entries long after the 674-entry recalibration). The single source
of truth for accuracy is CLAUDE.md's "Validation status" section, asserted by
tests/test_calibration_benchmark.py (grouped 10-fold CV, offline) and
tests/test_vs_pdbe_pisa.py (the legacy in-sample entries).

Assembly-level fields are NOT per-interface sums:

  dissociation_energy  -sum(stabilization_energy over the interfaces CUT) - T*dS,
                       along the cheapest dissociation pathway
                       (fastpisa.energy.dissociation). PISA's own relation,
                       recovered from the PDBe PISA 2.0 assembly JSON.
  entropy              T*dS of that one dissociation (rigid-body translational
                       term, fastpisa.energy.entropy) -- positive, and SUBTRACTED
                       from the dissociation energy.

Each interface entry also carries PDBe-shaped bond tables (hydrogen_bonds,
salt_bridges, disulfide_bonds, covalent_bonds), whose lengths equal the matching
number_* counts by construction.
"""
from typing import Any, Dict, List


def _interface_entry(iface: Any) -> Dict[str, Any]:
    """Serialise one Interface dataclass into a PDBe-shaped dict."""
    mol1, mol2 = iface.molecules
    entry: Dict[str, Any] = {
        "interface_id": iface.interface_id,
        "molecule_1_id": iface.molecule1_id,
        "molecule_2_id": iface.molecule2_id,
        "molecules": [mol1, mol2],
        "interface_area": iface.interface_area,
        "solvation_energy": iface.solvation_energy,
        # fastPISA extension: hydrophobic / polar split of solvation_energy
        "solvation_energy_apolar": iface.solvation_energy_apolar,
        "solvation_energy_polar": iface.solvation_energy_polar,
        "stabilization_energy": iface.stabilization_energy,
        "p_value": iface.p_value,
        "css": iface.css,
        "number_interface_residues": iface.number_interface_residues,
        "number_hydrogen_bonds": iface.number_hydrogen_bonds,
        "number_salt_bridges": iface.number_salt_bridges,
        "number_disulfide_bonds": iface.number_disulfide_bonds,
        "number_covalent_bonds": iface.number_covalent_bonds,
        "number_other_bonds": iface.number_other_bonds,
    }
    # Bond tables, PDBe-shaped: one parallel-list block per bond class, built
    # from the SAME independent predicates the counts come from
    # (fastpisa.interface.bonds.detect_bond_flags), so
    # len(hydrogen_bonds["bond_distances"]) == number_hydrogen_bonds by
    # construction. Always present, empty lists when a class has no bonds, so
    # a consumer can iterate without null checks.
    entry["hydrogen_bonds"] = iface.to_bond_dict("hbond")
    entry["salt_bridges"] = iface.to_bond_dict("salt_bridge")
    entry["disulfide_bonds"] = iface.to_bond_dict("disulfide")
    entry["covalent_bonds"] = iface.to_bond_dict("covalent")

    # COCOMAPS mode attaches a contact map; omit the key entirely in PISA mode rather
        # than emitting a null, so a consumer can test membership.
    cocomaps = getattr(iface, "cocomaps", None)
    if cocomaps:
        entry["interface_contact_map"] = cocomaps
    return entry


def build_interfaces_json(
    pdb_id: str,
    assembly_id: Any,
    assembly_mmsize: str,
    assembly_dissociation_energy: float,
    assembly_asa: float,
    assembly_bsa: float,
    assembly_entropy: float,
    assembly_dissociation_area: float,
    assembly_solvation_energy_gain: float,
    assembly_formula: str,
    assembly_composition: str,
    interfaces: List[Any],
    total_atoms: int,
    total_asa: float,
) -> Dict[str, Any]:
    """The per-interface document: assembly-level totals plus one entry per interface."""
    return {
        "assembly": {
            "pdb_id": pdb_id,
            "assembly_id": str(assembly_id),
            "mmsize": assembly_mmsize,
            "dissociation_energy": assembly_dissociation_energy,
            "accessible_surface_area": assembly_asa,
            "buried_surface_area": assembly_bsa,
            "entropy": assembly_entropy,
            "dissociation_area": assembly_dissociation_area,
            "solvation_energy_gain": assembly_solvation_energy_gain,
            "formula": assembly_formula,
            "composition": assembly_composition,
            "total_atoms": total_atoms,
            "total_accessible_surface_area": total_asa,
            "interface_count": len(interfaces),
            "interfaces": [_interface_entry(i) for i in interfaces],
        }
    }


def build_assembly_json(
    pdb_id: str,
    assembly_id: Any,
    assembly_size: str,
    assembly_mmsize: str,
    assembly_dissociation_energy: float,
    assembly_asa: float,
    assembly_bsa: float,
    assembly_entropy: float,
    assembly_dissociation_area: float,
    assembly_solvation_energy_gain: float,
    assembly_formula: str,
    assembly_composition: str,
) -> Dict[str, Any]:
    """The assembly-level document, without per-interface detail."""
    return {
        "assembly": {
            "pdb_id": pdb_id,
            "assembly_id": str(assembly_id),
            "size": assembly_size,
            "mmsize": assembly_mmsize,
            "dissociation_energy": assembly_dissociation_energy,
            "accessible_surface_area": assembly_asa,
            "buried_surface_area": assembly_bsa,
            "entropy": assembly_entropy,
            "dissociation_area": assembly_dissociation_area,
            "solvation_energy_gain": assembly_solvation_energy_gain,
            "formula": assembly_formula,
            "composition": assembly_composition,
        }
    }
