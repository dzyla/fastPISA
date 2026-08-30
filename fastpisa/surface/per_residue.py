"""
Per-residue surface area calculation for PISA interfaces.

Computes per-residue accessible and buried surface areas by aggregating
per-atom ASA/BSA values from the combined structure calculation.
"""

import numpy as np
from typing import Dict, List, Tuple
from fastpisa.parser.pdb_parser import Atom
from fastpisa.surface.shrake_rupley import get_vdw_radius


def compute_buried_surface(
    asa_alone: Dict[Tuple[int, int], float],
    asa_combined: Dict[int, float],
    n_atoms: int,
    masks: List[np.ndarray],
) -> Tuple[Dict[int, float], float, float]:
    """Compute the physically meaningful per-atom buried surface.

    For a given atom, buried surface = (isolated-ASA of the atom's molecule)
    - (combined-structure ASA). This is the area hidden by intermolecular
    contact, and is the correct quantity for solvation-energy and per-residue
    BSA. It replaces the old 4*pi*r_vdw^2 - ASA convention which overstated
    BSA enormously.

    Also returns the assembly-level totals:
      assembly_asa = sum of combined ASA
      assembly_bsa = sum of all isolated ASAs - combined ASA

    Returns
    -------
    (per_atom_buried dict, assembly_asa, assembly_bsa)
    """
    atom_mol = {}
    for mol_idx, mask in enumerate(masks):
        for i in range(n_atoms):
            if mask[i]:
                atom_mol[i] = mol_idx

    per_atom_buried: Dict[int, float] = {}
    for i in range(n_atoms):
        mi = atom_mol.get(i)
        if mi is None:
            per_atom_buried[i] = 0.0
        else:
            per_atom_buried[i] = max(
                asa_alone.get((mi, i), 0.0) - asa_combined.get(i, 0.0), 0.0
            )

    assembly_asa = float(sum(asa_combined.values()))
    total_isolated = float(sum(asa_alone.values()))
    assembly_bsa = max(total_isolated - assembly_asa, 0.0)
    return per_atom_buried, assembly_asa, assembly_bsa


def compute_per_residue_surface(
    atoms: List[Atom],
    atom_asa_combined: Dict[int, float],
    atom_bsa_buried: Dict[int, float],
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
    atom_bsa_buried : dict
        Buried surface area per atom = isolated-molecule ASA - combined ASA
        (the physically meaningful buried area, NOT 4*pi*r^2 - ASA).
    interface_atom_indices : set
        Set of atom indices that are at the interface.
    mol_atoms : list
        Atom indices belonging to this molecule (unused beyond grouping).

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
            asa = atom_asa_combined.get(idx, 0.0)
            bsa = max(atom_bsa_buried.get(idx, 0.0), 0.0)
            asa_list.append(round(asa, 2))
            bsa_list.append(round(bsa, 2))

            # Solvation energy for this atom
            from fastpisa.energy.asp_table import get_asp
            asp = get_asp(atom.atom_name, atom.element, atom.res_name)
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