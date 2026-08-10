"""
Per-residue surface area calculation for PISA interfaces.

Computes per-residue accessible and buried surface areas by aggregating
per-atom ASA/BSA values from the combined structure calculation.
"""

import numpy as np
from typing import Dict, List, Tuple
from fastpisa.parser.pdb_parser import Atom
from fastpisa.surface.shrake_rupley import get_vdw_radius


def compute_per_residue_surface(
    atoms: List[Atom],
    atom_asa_combined: Dict[int, float],
    interface_atom_indices: set,
    mol_atoms: List[int],
) -> Dict:
    """Compute per-residue ASA and BSA for interface residues.

    Parameters
    ----------
    atoms : list of Atom
        All atoms in the structure.
    atom_asa_combined : dict
        Accessible surface area per atom from the combined calculation.
    interface_atom_indices : set
        Set of atom indices that are at the interface.
    mol_atoms : list
        Atom indices belonging to this molecule.

    Returns
    -------
    dict
        Per-residue data with lists for each residue.
    """
    # Group interface atoms by residue
    residues = {}  # (auth_asym_id, seq_id, icode) -> list of atom indices
    for idx in interface_atom_indices:
        atom = atoms[idx]
        res_key = (atom.auth_asym_id, atom.res_seq, atom.icode)
        if res_key not in residues:
            residues[res_key] = []
        residues[res_key].append(idx)

    result = {
        "residue_label_comp_ids": [],
        "residue_seq_ids": [],
        "residue_label_seq_ids": [],
        "residue_ins_codes": [],
        "residue_bonds": [],
        "solvation_energies": [],
        "accessible_surface_areas": [],
        "buried_surface_areas": [],
    }

    for res_key, atom_indices in sorted(residues.items()):
        res_name = atoms[atom_indices[0]].label_comp_id
        res_seq = atoms[atom_indices[0]].res_seq
        res_icode = atoms[atom_indices[0]].icode or None

        # Per-atom ASA and BSA
        asa_list = []
        bsa_list = []
        solv_list = []

        for idx in atom_indices:
            atom = atoms[idx]
            r = get_vdw_radius(atom.element)
            total_surf = 4 * np.pi * r ** 2
            asa = atom_asa_combined.get(idx, 0.0)
            bsa = total_surf - asa
            asa_list.append(round(asa, 2))
            bsa_list.append(round(bsa, 2))

            # Solvation energy for this atom
            from fastpisa.energy.asp_table import get_asp
            asp = get_asp(atom.atom_name, atom.element)
            solv_list.append(round(asp * bsa, 4))

        result["residue_label_comp_ids"].append(res_name)
        result["residue_seq_ids"].append(str(res_seq))
        result["residue_label_seq_ids"].append(str(res_seq))
        result["residue_ins_codes"].append(res_icode)
        result["residue_bonds"].append(None)
        result["solvation_energies"].append(round(sum(solv_list), 4))
        result["accessible_surface_areas"].append(round(sum(asa_list), 2))
        result["buried_surface_areas"].append(round(sum(bsa_list), 2))

    return result