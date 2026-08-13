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

NOTE ON ENERGIES. `solvation_energy` and anything derived from it are NOT calibrated
against CCP4 PISA's dG. Measured against PISA v2.2.0 over 63 matched interfaces from 7
antibody-antigen complexes, fastPISA's solvation energy relates to PISA's dG with
Spearman -0.408 (p=0.015, n=35) on ANTIBODY-ANTIGEN interfaces while giving +0.596 on
antibody-antibody ones -- i.e. the pooled correlation (+0.324) is a Simpson's paradox and
the sign is class-dependent. Magnitudes run ~6x larger than PISA's. Interface AREA, by
contrast, agrees at Pearson 0.9996 (median error 1.9%) and salt-bridge counts at Spearman
0.998. Treat area and salt bridges as quantitative; treat energies as uncalibrated.
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
