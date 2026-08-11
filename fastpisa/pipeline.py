"""
Main analysis pipeline for fastPISA.

This module orchestrates the full PISA analysis:
1. Parse PDB/mmCIF input
2. Calculate solvent-accessible surface areas (ASA) for isolated molecules
3. Calculate ASA for the combined structure
4. Detect interfaces between all molecule pairs
5. Find atom-atom contacts (H-bonds, salt bridges, disulfides, other bonds)
6. Calculate solvation/binding energy, entropy, P-value, CSS for each interface
7. Predict assemblies using crystallographic symmetry
8. Generate assembly.json and interfaces.json output

The output matches the PDBe PISA JSON schema.
"""

import json
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from fastpisa.parser.pdb_parser import parse_pdb, parse_mmcif, PDBStructure
from fastpisa.surface.shrake_rupley import (
    calculate_asa, calculate_asa_batched, calculate_bsa, get_vdw_radius,
)
from fastpisa.interface.contacts import (
    find_interface_atoms, find_contacts, get_molecules, get_molecule_masks,
    filter_water_molecules, Interface, AtomContact,
)
from fastpisa.interface.interface import calculate_interface_area
from fastpisa.energy.asp_table import get_asp
from fastpisa.energy.energy import (
    calculate_solvation_energy, calculate_contact_energy, calculate_binding_energy,
    calculate_entropy, calculate_dissociation_energy, calculate_stabilization_energy,
    calculate_assembly_dissociation_energy,
)
from fastpisa.scoring.scoring import calculate_p_value, calculate_css
from fastpisa.output.json_output import build_interfaces_json, build_assembly_json
from fastpisa.surface.per_residue import (
    compute_per_residue_surface, compute_buried_surface,
)


def analyze_structure(
    input_file: str,
    pdb_id: str = "unknown",
    assembly_id: str = "1",
    probe_radius: float = 1.4,
    point_density: int = 480,
    interface_cutoff: float = 5.0,
    asis: bool = False,
    extended_data: bool = False,
    exclude_water: bool = True,
    min_css: float = 0.0,
) -> dict:
    """Run the full PISA analysis on a structure.

    Parameters
    ----------
    input_file : str
        Path to PDB or mmCIF file.
    pdb_id : str
        PDB identifier (e.g. "6nxr").
    assembly_id : str
        Assembly ID for output.
    probe_radius : float
        Probe sphere radius for ASA calculation (default 1.4 A).
    point_density : int
        Number of points on the probe sphere (default 480).
    interface_cutoff : float
        Distance cutoff for interface atom detection (default 5.0 A).
    asis : bool
        If True, only calculate interfaces (no assembly prediction).
    extended_data : bool
        If True, include extended -list data.
    min_css : float
        If > 0, only interfaces with CSS >= min_css are kept (a
        significance filter that drops weak/artifact crystal-packing
        contacts). Default 0.0 keeps everything.

    Returns
    -------
    dict
        interfaces.json and assembly.json data.
    """
    global atoms_global

    # 1. Parse input file
    if input_file.endswith(".cif") or input_file.endswith(".cif.gz"):
        structure = parse_mmcif(input_file)
    else:
        structure = parse_pdb(input_file)

    atoms_global = structure.atoms
    atoms = structure.atoms

    if not atoms:
        raise ValueError(f"No atoms found in {input_file}")

    n_atoms = len(atoms)
    print(f"Parsed {n_atoms} atoms from {input_file}")

    # 2. Get molecules and masks (exclude ordered water from interface search)
    molecules = get_molecules(structure)
    molecules = filter_water_molecules(molecules, exclude_water=exclude_water)
    masks = get_molecule_masks(atoms, molecules)
    n_molecules = len(molecules)
    print(f"Found {n_molecules} molecules")

    # 3. Build KD-tree for neighbor lookup using all atoms
    from scipy.spatial import cKDTree
    all_coords = np.array([[a.x, a.y, a.z] for a in atoms])
    all_radii = np.array([get_vdw_radius(a.element) for a in atoms])
    neighbor_cutoff = 2.0 * all_radii.max() + probe_radius + 1.0
    kd_tree = cKDTree(all_coords)

    # 4. Calculate ASA for the COMBINED structure ONCE (cached)
    print("Calculating combined-structure ASA...")
    asa_combined = calculate_asa(
        atoms=atoms,
        probe_radius=probe_radius,
        point_density=point_density,
        kd_tree=kd_tree,
        combined_coords=all_coords,
        combined_radii=all_radii,
        neighbor_cutoff=neighbor_cutoff,
    )

    # 5. Calculate isolated ASA for each molecule (needed for the proper
    #    buried-surface convention: buried = isolated - combined)
    asa_alone = {}
    for mol_idx in range(n_molecules):
        mask = masks[mol_idx]
        mol_atom_indices = [i for i, m in enumerate(mask) if m]
        mol_atoms = [atoms[i] for i in mol_atom_indices]
        if mol_atoms:
            asa = calculate_asa(
                atoms=mol_atoms,
                probe_radius=probe_radius,
                point_density=point_density,
                atom_indices=mol_atom_indices,
                kd_tree=kd_tree,
                combined_coords=all_coords,
                combined_radii=all_radii,
                neighbor_cutoff=neighbor_cutoff,
            )
            for local_i, global_i in enumerate(mol_atom_indices):
                asa_alone[(mol_idx, global_i)] = asa.get(global_i, 0.0)

    # 6. Compute the physically meaningful buried surface per atom
    #    (buried = isolated ASA - combined ASA), replacing the old
    #    (4*pi*r_vdw^2 - ASA) convention that vastly overstated BSA.
    bsa_combined, assembly_asa, assembly_bsa = compute_buried_surface(
        asa_alone, asa_combined, n_atoms, masks
    )
    print(f"Combined ASA: {assembly_asa:.1f} A^2, BSA: {assembly_bsa:.1f} A^2")

    total_asa_alone = {}
    for mol_idx in range(n_molecules):
        mask = masks[mol_idx]
        total_asa_alone[mol_idx] = sum(
            asa_alone.get((mol_idx, i), 0.0) for i in range(n_atoms) if mask[i]
        )

    # 7. Detect interfaces between all molecule pairs
    interfaces = []
    interface_id = 0

    for mol1 in range(n_molecules):
        for mol2 in range(mol1 + 1, n_molecules):
            mask1 = masks[mol1]
            mask2 = masks[mol2]

            # Find interface atoms
            idx1, idx2 = find_interface_atoms(atoms, mask1, mask2, interface_cutoff)

            if len(idx1) == 0 or len(idx2) == 0:
                continue

            # Find contacts between interface atoms
            contacts = find_contacts(
                atoms, mask1, mask2,
                [i for i in range(n_atoms) if mask1[i]],
                [i for i in range(n_atoms) if mask2[i]],
                idx1, idx2,
            )

            if len(contacts) == 0:
                continue

            interface_id += 1

            # Calculate interface area
            mol_atoms_12 = [i for i in range(n_atoms) if mask1[i] or mask2[i]]
            asa_12 = calculate_asa(
                atoms=[atoms[i] for i in mol_atoms_12],
                probe_radius=probe_radius,
                point_density=point_density,
                atom_indices=mol_atoms_12,
                kd_tree=kd_tree,
                combined_coords=all_coords,
                combined_radii=all_radii,
                neighbor_cutoff=neighbor_cutoff,
            )
            # Get total ASA for the two-molecule complex
            total_asa_12 = sum(asa_12.get(gi, 0.0) for gi in mol_atoms_12)
            total_asa_mol1_alone = total_asa_alone.get(mol1, 0.0)
            total_asa_mol2_alone = total_asa_alone.get(mol2, 0.0)

            interface_area = max(
                (total_asa_mol1_alone + total_asa_mol2_alone - total_asa_12) / 2.0, 0.0
            )

            # Calculate solvation energy using BSA from combined structure
            interface_atom_set = set(idx1) | set(idx2)
            solv_energy = calculate_solvation_energy(
                interface_atom_set, bsa_combined, atoms
            )

            # Calculate binding energy
            binding_energy = calculate_binding_energy(solv_energy, contacts)

            # Count contacts by type
            n_hbonds = sum(1 for c in contacts if c.bond_type == "hbond")
            n_salt = sum(1 for c in contacts if c.bond_type == "salt_bridge")
            n_disulfide = sum(1 for c in contacts if c.bond_type == "disulfide")
            n_covalent = sum(1 for c in contacts if c.bond_type == "covalent")
            n_other = sum(1 for c in contacts if c.bond_type == "other")

            # Calculate entropy
            n_res1 = len(set((atoms[i].res_seq, atoms[i].icode) for i in idx1))
            n_res2 = len(set((atoms[i].res_seq, atoms[i].icode) for i in idx2))
            entropy = calculate_entropy(interface_area, n_res1, n_res2)

            # Calculate P-value
            p_value = calculate_p_value(
                solv_energy, interface_area,
                assembly_asa if assembly_asa > 0 else 1.0,
            )

            # Calculate CSS
            css = calculate_css(
                interface_area, solv_energy, p_value,
                len(contacts), n_res1 + n_res2,
                assembly_asa if assembly_asa > 0 else 1.0,
            )

            # Stabilization energy
            stab_energy = calculate_stabilization_energy(solv_energy, contacts)

            # Build interface
            iface = Interface(
                interface_id=interface_id,
                molecule1_id=mol1,
                molecule2_id=mol2,
                interface_area=round(interface_area, 2),
                solvation_energy=round(solv_energy, 2),
                stabilization_energy=round(stab_energy, 2),
                p_value=round(p_value, 3),
                css=round(css, 3),
                number_interface_residues=n_res1 + n_res2,
                number_hydrogen_bonds=n_hbonds,
                number_covalent_bonds=n_covalent,
                number_disulfide_bonds=n_disulfide,
                number_salt_bridges=n_salt,
                number_other_bonds=n_other,
                contacts=contacts,
            )

            # Build molecule info with per-residue surface data
            mol_info_1 = molecules[mol1].copy()
            mol_info_1["int_natoms"] = len(idx1)
            mol_info_1["int_nres"] = n_res1
            res_data_1 = compute_per_residue_surface(
                atoms, asa_combined, bsa_combined,
                set(idx1), [i for i in range(n_atoms) if mask1[i]],
            )
            mol_info_1.update(res_data_1)
            mol_info_2 = molecules[mol2].copy()
            mol_info_2["int_natoms"] = len(idx2)
            mol_info_2["int_nres"] = n_res2
            res_data_2 = compute_per_residue_surface(
                atoms, asa_combined, bsa_combined,
                set(idx2), [i for i in range(n_atoms) if mask2[i]],
            )
            mol_info_2.update(res_data_2)

            iface.molecules = [mol_info_1, mol_info_2]
            interfaces.append(iface)

    # 7b. Optional significance filter (drop weak / artifact interfaces)
    if min_css > 0:
        interfaces = [i for i in interfaces if i.css >= min_css]
        for idx, iface in enumerate(interfaces):
            iface.interface_id = idx + 1

    # 8. Calculate assembly-level statistics
    n_interfaces = len(interfaces)
    total_interface_area = sum(iface.interface_area for iface in interfaces)
    total_solv_energy = sum(iface.solvation_energy for iface in interfaces)

    # Assembly dissociation energy
    solv_energies = [iface.solvation_energy for iface in interfaces]
    contact_energies = []
    for iface in interfaces:
        ce, _, _, _, _ = calculate_contact_energy(iface.contacts)
        contact_energies.append(ce)
    entropies = []
    for iface in interfaces:
        e = calculate_entropy(iface.interface_area,
                              iface.number_interface_residues // 2,
                              iface.number_interface_residues - iface.number_interface_residues // 2)
        entropies.append(e)

    diss_energy = calculate_assembly_dissociation_energy(
        [iface.interface_area for iface in interfaces],
        solv_energies, contact_energies, entropies,
    )

    # Assembly formula and composition
    formula = _build_formula(molecules)
    composition = _build_composition(molecules)

    # Number of macromolecular units (polymers only)
    n_macromolecular = sum(1 for m in molecules if m.get("chain_type") == "polymer")

    # 9. Generate output JSON
    interfaces_json = build_interfaces_json(
        pdb_id=pdb_id,
        assembly_id=assembly_id,
        assembly_mmsize=str(n_macromolecular),
        assembly_dissociation_energy=round(diss_energy, 2),
        assembly_asa=round(assembly_asa, 2),
        assembly_bsa=round(assembly_bsa, 2),
        assembly_entropy=round(sum(entropies), 2),
        assembly_dissociation_area=round(total_interface_area, 2),
        assembly_solvation_energy_gain=round(total_solv_energy, 2),
        assembly_formula=formula,
        assembly_composition=composition,
        interfaces=interfaces,
        total_atoms=n_atoms,
        total_asa=assembly_asa,
    )

    assembly_json = build_assembly_json(
        pdb_id=pdb_id,
        assembly_id=assembly_id,
        assembly_size=str(len(molecules)),
        assembly_mmsize=str(n_macromolecular),
        assembly_dissociation_energy=round(diss_energy, 2),
        assembly_asa=round(assembly_asa, 2),
        assembly_bsa=round(assembly_bsa, 2),
        assembly_entropy=round(sum(entropies), 2),
        assembly_dissociation_area=round(total_interface_area, 2),
        assembly_solvation_energy_gain=round(total_solv_energy, 2),
        assembly_formula=formula,
        assembly_composition=composition,
    )

    return {
        "interfaces": interfaces_json,
        "assembly": assembly_json,
        "interfaces_obj": interfaces,
    }


def _build_formula(molecules):
    """Build the formula string (e.g., 'A(2)a(2)b(2)')."""
    from collections import Counter
    class_counts = Counter(m.get("molecule_class", "Other") for m in molecules)
    formula_parts = []
    for cls, count in sorted(class_counts.items()):
        if cls == "Protein":
            letter = "A"
        elif cls == "NucleicAcid":
            letter = "a"
        elif cls == "Ligand":
            letter = "b"
        else:
            letter = "x"
        if count == 1:
            formula_parts.append(letter)
        else:
            formula_parts.append(f"{letter}({count})")
    return "".join(formula_parts)


def _build_composition(molecules):
    """Build the composition string (e.g., 'A-2A[NA](2)[GOL](2)')."""
    parts = []
    for mol in molecules:
        cls = mol.get("molecule_class", "Other")
        chain_id = mol.get("auth_asym_id", "")
        if cls == "Ligand":
            ccd = mol.get("ccd_id", "")
            parts.append(f"[{ccd}]({chain_id})")
        else:
            parts.append(chain_id)
    return "-".join(parts)