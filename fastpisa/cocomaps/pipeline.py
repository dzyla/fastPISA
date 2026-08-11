"""
COCOMAPS analysis pipeline for fastPISA.

Runs the same core analysis as the PISA pipeline (parse structure, detect
molecules, compute ASA/BSA, find interface atom pairs at the same 5 A cutoff)
but reports the interface in COCOMAPS style: a residue-residue contact map
with per-contact interaction-type classification.

The output is a dict identical in shape to the PISA pipeline's result
(interfaces / assembly JSON) so that the two modes are interchangeable, with
the COCOMAPS contact-map and interaction-count fields added.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from fastpisa.parser.pdb_parser import parse_pdb, parse_mmcif
from fastpisa.surface.shrake_rupley import calculate_asa, get_vdw_radius
from fastpisa.interface.contacts import (
    find_interface_atoms, find_contacts, get_molecules, get_molecule_masks,
    filter_water_molecules, Interface, AtomContact,
)
from fastpisa.energy.energy import (
    calculate_solvation_energy, calculate_binding_energy, calculate_entropy,
)
from fastpisa.scoring.scoring import calculate_p_value, calculate_css
from fastpisa.surface.per_residue import (
    compute_per_residue_surface, compute_buried_surface,
)
from fastpisa.cocomaps.contact_map import (
    build_residue_contact_map, aggregate_residue_pairs,
)


def analyze_structure_cocomaps(
    input_file: str,
    pdb_id: str = "unknown",
    assembly_id: str = "1",
    probe_radius: float = 1.4,
    point_density: int = 480,
    interface_cutoff: float = 5.0,
    asis: bool = False,
    extended_data: bool = False,
    interaction_cutoff: float = 5.0,
    exclude_water: bool = True,
    min_css: float = 0.0,
) -> dict:
    """Run COCOMAPS analysis on a structure.

    Returns a dict with keys:
      - "interfaces": JSON-compatible doc, same schema as PISA mode plus
        COCOMAPS-specific "interface_contact_map" per interface.
      - "assembly": assembly JSON doc.
      - "interfaces_obj": list of Interface objects extended with COCOMAPS
        attributes (contact_map, interaction_population).

    min_css : float
        If > 0, only interfaces with CSS >= min_css are kept (a
        significance filter that drops weak/artifact crystal-packing
        contacts). Default 0.0 keeps everything (PISA-compatible).
    """
    # 1. Parse input file
    if input_file.endswith(".cif") or input_file.endswith(".cif.gz"):
        structure = parse_mmcif(input_file)
    else:
        structure = parse_pdb(input_file)

    atoms = structure.atoms
    if not atoms:
        raise ValueError(f"No atoms found in {input_file}")

    n_atoms = len(atoms)
    print(f"Parsed {n_atoms} atoms from {input_file}")

    # 2. Molecules and masks (exclude ordered water from interface search)
    molecules = get_molecules(structure)
    molecules = filter_water_molecules(molecules, exclude_water=exclude_water)
    masks = get_molecule_masks(atoms, molecules)
    n_molecules = len(molecules)
    print(f"Found {n_molecules} molecules")

    # 3. KD-tree + combined ASA (shared with PISA mode)
    global atoms_global
    atoms_global = atoms
    all_coords = np.array([[a.x, a.y, a.z] for a in atoms])
    all_radii = np.array([get_vdw_radius(a.element) for a in atoms])
    neighbor_cutoff = 2.0 * all_radii.max() + probe_radius + 1.0
    kd_tree = cKDTree(all_coords)

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

    # 4. Per-molecule isolated ASA (needed for the proper buried-surface
    #    convention: buried = isolated - combined)
    asa_alone = {}
    for mol_idx in range(n_molecules):
        mask = masks[mol_idx]
        mol_atom_indices = [i for i, m in enumerate(mask) if m]
        if not mol_atom_indices:
            continue
        asa = calculate_asa(
            atoms=[atoms[i] for i in mol_atom_indices],
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

    # 5. Compute the physically meaningful buried surface per atom
    #    (buried = isolated ASA - combined ASA), replacing the old
    #    (4*pi*r_vdw^2 - ASA) convention that vastly overstated BSA.
    bsa_combined, assembly_asa, assembly_bsa = compute_buried_surface(
        asa_alone, asa_combined, n_atoms, masks
    )
    print(f"Combined ASA: {assembly_asa:.1f} A^2, BSA: {assembly_bsa:.1f} A^2")

    total_asa_alone = {
        mol_idx: sum(asa_alone.get((mol_idx, i), 0.0)
                     for i in range(n_atoms) if masks[mol_idx][i])
        for mol_idx in range(n_molecules)
    }

    # 5. Detect interfaces (identical cutoff to PISA => same interfaces)
    interfaces = []
    interface_id = 0

    for mol1 in range(n_molecules):
        for mol2 in range(mol1 + 1, n_molecules):
            mask1, mask2 = masks[mol1], masks[mol2]
            mol1_ids = [i for i in range(n_atoms) if mask1[i]]
            mol2_ids = [i for i in range(n_atoms) if mask2[i]]

            idx1, idx2 = find_interface_atoms(atoms, mask1, mask2, interface_cutoff)

            if len(idx1) == 0 or len(idx2) == 0:
                continue

            # COCOMAPS residue contact map (5 A cutoff = same as interface)
            residue_contacts = build_residue_contact_map(
                atoms, mask1, mask2, mol1_ids, mol2_ids, interaction_cutoff,
            )
            if not residue_contacts:
                continue

            # PISA-style atom contacts (for area / energy compatibility)
            contacts = find_contacts(
                atoms, mask1, mask2, mol1_ids, mol2_ids, idx1, idx2,
            )

            interface_id += 1

            # Interface area (same formula as PISA)
            mol_atoms_12 = mol1_ids + mol2_ids
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
            total_asa_12 = sum(asa_12.get(gi, 0.0) for gi in mol_atoms_12)
            interface_area = max(
                (total_asa_alone[mol1] + total_asa_alone[mol2] - total_asa_12) / 2.0, 0.0
            )

            interface_atom_set = set(idx1) | set(idx2)
            bsa_iface = {i: bsa_combined.get(i, 0.0) for i in interface_atom_set}
            solv_energy = calculate_solvation_energy(interface_atom_set, bsa_combined, atoms)
            binding_energy = calculate_binding_energy(solv_energy, contacts)

            n_res1 = len(set((atoms[i].res_seq, atoms[i].icode) for i in idx1))
            n_res2 = len(set((atoms[i].res_seq, atoms[i].icode) for i in idx2))
            entropy = calculate_entropy(interface_area, n_res1, n_res2)

            # COCOMAPS interaction population
            interaction_population = _count_interactions(residue_contacts)

            # P-value and CSS (same model as PISA mode)
            p_value = calculate_p_value(
                solv_energy, interface_area,
                assembly_asa if assembly_asa > 0 else 1.0,
            )
            css = calculate_css(
                interface_area, solv_energy, p_value,
                len(contacts), n_res1 + n_res2,
                assembly_asa if assembly_asa > 0 else 1.0,
            )

            # Build an Interface object compatible with the PISA output builder
            iface = Interface(
                interface_id=interface_id,
                molecule1_id=mol1,
                molecule2_id=mol2,
                interface_area=round(interface_area, 2),
                solvation_energy=round(solv_energy, 2),
                stabilization_energy=round(binding_energy, 2),
                p_value=round(p_value, 3),
                css=round(css, 3),
                number_interface_residues=n_res1 + n_res2,
                number_hydrogen_bonds=interaction_population.get("hydrogen_bond", 0),
                number_covalent_bonds=0,
                number_disulfide_bonds=interaction_population.get("disulfide", 0),
                number_salt_bridges=interaction_population.get("salt_bridge", 0),
                number_other_bonds=interaction_population.get("apolar_vdw", 0)
                + interaction_population.get("polar_vdw", 0),
                contacts=contacts,
            )

            # COCOMAPS extensions
            contact_map_entries = aggregate_residue_pairs(residue_contacts, atoms)
            iface.cocomaps = {
                "interaction_population": interaction_population,
                "contact_map": contact_map_entries,
                "num_residue_pairs": len(contact_map_entries),
            }

            # Per-residue surface (same as PISA)
            mol_info_1 = molecules[mol1].copy()
            mol_info_1["int_natoms"] = len(idx1)
            mol_info_1["int_nres"] = n_res1
            mol_info_1.update(compute_per_residue_surface(
                atoms, asa_combined, bsa_combined, set(idx1), mol1_ids))
            mol_info_2 = molecules[mol2].copy()
            mol_info_2["int_natoms"] = len(idx2)
            mol_info_2["int_nres"] = n_res2
            mol_info_2.update(compute_per_residue_surface(
                atoms, asa_combined, bsa_combined, set(idx2), mol2_ids))
            iface.molecules = [mol_info_1, mol_info_2]

            interfaces.append(iface)

    # 5b. Optional significance filter (drop weak / artifact interfaces)
    if min_css > 0:
        interfaces = [i for i in interfaces if i.css >= min_css]
        for idx, iface in enumerate(interfaces):
            iface.interface_id = idx + 1

    # 6. Assembly statistics (same structure as PISA)
    from fastpisa.pipeline import _build_formula, _build_composition
    formula = _build_formula(molecules)
    composition = _build_composition(molecules)
    n_macromolecular = sum(1 for m in molecules if m.get("chain_type") == "polymer")

    total_interface_area = sum(i.interface_area for i in interfaces)
    total_solv = sum(i.solvation_energy for i in interfaces)
    entropies = [calculate_entropy(i.interface_area,
                                   i.number_interface_residues // 2,
                                   i.number_interface_residues - i.number_interface_residues // 2)
                 for i in interfaces]
    diss_energy = sum(solv + stab for solv, stab in
                      zip([i.solvation_energy for i in interfaces],
                          [i.stabilization_energy for i in interfaces]))

    # 7. Output
    from fastpisa.output.json_output import build_interfaces_json, build_assembly_json
    interfaces_json = build_interfaces_json(
        pdb_id=pdb_id,
        assembly_id=assembly_id,
        assembly_mmsize=str(n_macromolecular),
        assembly_dissociation_energy=round(diss_energy, 2),
        assembly_asa=round(assembly_asa, 2),
        assembly_bsa=round(assembly_bsa, 2),
        assembly_entropy=round(sum(entropies), 2),
        assembly_dissociation_area=round(total_interface_area, 2),
        assembly_solvation_energy_gain=round(total_solv, 2),
        assembly_formula=formula,
        assembly_composition=composition,
        interfaces=interfaces,
        total_atoms=n_atoms,
        total_asa=assembly_asa,
    )
    # Add COCOMAPS contact map data to each interface doc
    for d, iface in zip(interfaces_json["assembly"]["interfaces"], interfaces):
        d["interface_contact_map"] = iface.cocomaps

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
        assembly_solvation_energy_gain=round(total_solv, 2),
        assembly_formula=formula,
        assembly_composition=composition,
    )

    return {
        "interfaces": interfaces_json,
        "assembly": assembly_json,
        "interfaces_obj": interfaces,
    }


def _count_interactions(residue_contacts) -> Dict[str, int]:
    """Count atoms by interaction type across the contact map."""
    from collections import Counter
    counter = Counter(c.interaction_type for c in residue_contacts)
    return dict(counter)