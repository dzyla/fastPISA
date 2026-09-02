"""Shared analysis core for all fastPISA modes.

Every mode (``pisa``, ``cocomaps``, ``combined``) runs the exact same physics
exactly once through :func:`run_core`:

  parse -> molecules/masks -> combined ASA -> per-molecule isolated ASA ->
  per-atom buried surface -> interface detection -> atom contacts ->
  per-interface area / energies / P-value / CSS.

The modes differ only in decoration:

  * ``pisa``      -- bond counts from the PISA atom-contact classifier.
  * ``cocomaps``  -- adds the residue-residue contact map + interaction
                     populations; bond counts come from those populations.
  * ``combined``  -- PISA bond counts AND the COCOMAPS contact map on the
                     same interface objects (one unified report).

Because the interfaces are detected once, the historical invariant "all modes
find identical interfaces" is now true by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import logging
import numpy as np
from scipy.spatial import cKDTree

from fastpisa.parser.pdb_parser import parse_pdb, parse_mmcif, PDBStructure
from fastpisa.surface.shrake_rupley import calculate_asa, surface_radius
from fastpisa.interface.contacts import (
    find_interface_atoms, find_contacts, get_molecules, get_molecule_masks,
    filter_water_molecules, Interface,
)
from fastpisa.interface.bonds import detect_bond_flags
from fastpisa.energy.energy import (
    calculate_solvation_energy, solvation_energy_components, bond_energy,
    calculate_entropy, calculate_assembly_dissociation_energy,
)
from fastpisa.scoring.scoring import calculate_p_value_pisa, calculate_css_pisa
from fastpisa.surface.per_residue import (
    compute_per_residue_surface, compute_buried_surface,
)
from fastpisa.output.json_output import build_interfaces_json, build_assembly_json

logger = logging.getLogger(__name__)

MODES = ("pisa", "cocomaps", "combined")


@dataclass
class CoreState:
    """Everything the shared core computed for one structure."""
    structure: PDBStructure
    atoms: list
    molecules: List[dict]
    masks: List[np.ndarray]
    asa_combined: Dict[int, float]
    asa_alone: Dict[Tuple[int, int], float]
    bsa_combined: Dict[int, float]
    assembly_asa: float
    assembly_bsa: float
    total_asa_alone: Dict[int, float]
    interfaces: List[Interface] = field(default_factory=list)


def run_core(
    input_file: str,
    probe_radius: float = 1.4,
    point_density: int = 480,
    interface_cutoff: float = 5.0,
    exclude_water: bool = True,
    min_css: float = 0.0,
    mode: str = "combined",
    interaction_cutoff: float = 5.0,
    ligand_mode: str = "separate",
    collect_calibration: bool = False,
) -> CoreState:
    """Run the shared analysis once and return the populated interfaces.

    ``ligand_mode``: ``"separate"`` (classic PISA -- every bound hetero group
    is its own monomer) or ``"merge"`` (jsPISA-on-assembly convention -- a
    chain's bound ligands/cofactors belong to that chain's molecule).

    ``collect_calibration``: also record, on each interface's ``calibration``
    dict, the sufficient statistics for refitting the ASP sigmas and the
    P-value model (buried area per solvation class, surface-area per class,
    and the buried-patch moments). Off by default; it costs one extra pass
    over the interface atoms and adds no physics.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode!r} (expected one of {MODES})")
    if ligand_mode not in ("separate", "merge"):
        raise ValueError(
            f"Unknown ligand_mode: {ligand_mode!r} (expected 'separate' or 'merge')")
    want_cocomaps = mode in ("cocomaps", "combined")

    # 1. Parse input file
    if str(input_file).endswith((".cif", ".cif.gz")):
        structure = parse_mmcif(input_file)
    else:
        structure = parse_pdb(input_file)

    atoms = structure.atoms
    if not atoms:
        raise ValueError(f"No atoms found in {input_file}")
    n_atoms = len(atoms)
    logger.info("Parsed %d atoms from %s", n_atoms, input_file)

    # 2. Molecules and masks (exclude ordered water from interface search)
    molecules = get_molecules(structure, merge_ligands=(ligand_mode == "merge"))
    molecules = filter_water_molecules(molecules, exclude_water=exclude_water)
    masks = get_molecule_masks(atoms, molecules)
    n_molecules = len(molecules)
    logger.info("Found %d molecules", n_molecules)

    # Surfaces, interfaces and contacts are computed over HEAVY atoms only --
    # the PISA convention. Explicit hydrogens (when a model has them) stay in
    # the parsed atom list so the H-bond detector can use their geometry, but
    # they carry no surface area and are not contact partners.
    heavy = np.array([a.element.strip().upper() not in ("H", "D")
                      for a in atoms])
    masks = [m & heavy for m in masks]

    # 3. KD-tree over all atoms (shared by every ASA call)
    all_coords = np.array([[a.x, a.y, a.z] for a in atoms])
    all_radii = np.array([surface_radius(a) for a in atoms])
    neighbor_cutoff = 2.0 * all_radii.max() + probe_radius + 1.0
    kd_tree = cKDTree(all_coords)

    asa_kwargs = dict(
        probe_radius=probe_radius,
        point_density=point_density,
        kd_tree=kd_tree,
        combined_coords=all_coords,
        combined_radii=all_radii,
        neighbor_cutoff=neighbor_cutoff,
    )

    # 4. Combined-structure ASA (once)
    logger.info("Calculating combined-structure ASA...")
    asa_combined = calculate_asa(atoms=atoms, **asa_kwargs)

    # 5. Isolated ASA per molecule (buried = isolated - combined)
    mol_atom_ids = [np.flatnonzero(mask).tolist() for mask in masks]
    asa_alone: Dict[Tuple[int, int], float] = {}
    for mol_idx in range(n_molecules):
        ids = mol_atom_ids[mol_idx]
        if not ids:
            continue
        asa = calculate_asa(
            atoms=[atoms[i] for i in ids], atom_indices=ids, **asa_kwargs)
        for gi in ids:
            asa_alone[(mol_idx, gi)] = asa.get(gi, 0.0)

    # 6. Per-atom buried surface + assembly totals
    bsa_combined, assembly_asa, assembly_bsa = compute_buried_surface(
        asa_alone, asa_combined, n_atoms, masks
    )
    logger.info("Combined ASA: %.1f A^2, BSA: %.1f A^2", assembly_asa, assembly_bsa)

    total_asa_alone = {
        mol_idx: sum(asa_alone.get((mol_idx, i), 0.0) for i in mol_atom_ids[mol_idx])
        for mol_idx in range(n_molecules)
    }

    # Per-atom ASP sigma (for solvation & the P-value surface statistics)
    from fastpisa.energy.asp_table import get_asp
    sigma_all = np.array([
        get_asp(a.atom_name, a.element, a.res_name) for a in atoms])

    # Per-atom solvation class (calibration only): dG_solv is LINEAR in the
    # per-class sigmas, so the buried area summed per class is a sufficient
    # statistic for refitting them.
    class_all = None
    fine_all = None
    res_atoms: List[Dict[tuple, List[int]]] = []
    if collect_calibration:
        from fastpisa.energy.asp_table import atom_class, fine_atom_type
        class_all = [
            "H" if a.element.strip().upper() in ("H", "D")
            else atom_class(a.atom_name, a.element, a.res_name)
            for a in atoms]
        fine_all = [fine_atom_type(a.atom_name, a.element, a.res_name)
                    for a in atoms]
        # heavy atoms of each residue, per molecule (residue-level features)
        for mi in range(n_molecules):
            by_res: Dict[tuple, List[int]] = {}
            for gi in mol_atom_ids[mi]:
                a = atoms[gi]
                by_res.setdefault(
                    (a.auth_asym_id, a.res_seq, a.icode), []).append(gi)
            res_atoms.append(by_res)

    # Per-molecule surface statistics (sigma + isolated ASA of exposed
    # atoms), precomputed once for the P-value model.
    surf_stats = []
    surf_class_asa: List[Dict[str, float]] = []
    surf_type_asa: List[Dict[str, float]] = []
    for mi in range(n_molecules):
        ids = np.array(mol_atom_ids[mi], dtype=int)
        if ids.size == 0:
            surf_stats.append((np.zeros(0), np.zeros(0)))
            if collect_calibration:
                surf_class_asa.append({})
                surf_type_asa.append({})
            continue
        a_iso = np.array([asa_alone.get((mi, gi), 0.0) for gi in ids])
        exposed = a_iso > 0.0
        surf_stats.append((sigma_all[ids[exposed]], a_iso[exposed]))
        if collect_calibration:
            per_class: Dict[str, float] = {}
            per_type: Dict[str, float] = {}
            for gi, a_i in zip(ids[exposed], a_iso[exposed]):
                c = class_all[gi]
                per_class[c] = per_class.get(c, 0.0) + float(a_i)
                t = fine_all[gi]
                per_type[t] = per_type.get(t, 0.0) + float(a_i)
            surf_class_asa.append(per_class)
            surf_type_asa.append(per_type)

    # 7. Detect interfaces between all molecule pairs.
    #
    # PISA semantics: an interface exists between two molecules when placing
    # them together buries surface (pair dASA > 0) -- NOT only when atoms sit
    # within the 5 A contact cutoff. Two atoms can shadow each other's
    # solvent-accessible surface out to r1 + r2 + 2*probe (~6.4 A), so pairs
    # are screened with that geometric bound, then kept if any area is buried.
    interfaces: List[Interface] = []
    interface_id = 0
    shadow_cutoff = 2.0 * all_radii.max() + 2.0 * probe_radius + 0.1

    for mol1 in range(n_molecules):
        for mol2 in range(mol1 + 1, n_molecules):
            mask1, mask2 = masks[mol1], masks[mol2]
            mol1_ids, mol2_ids = mol_atom_ids[mol1], mol_atom_ids[mol2]

            # Cheap screen: any atoms close enough to bury surface at all?
            near1, near2 = find_interface_atoms(atoms, mask1, mask2, shadow_cutoff)
            if len(near1) == 0 or len(near2) == 0:
                continue

            # Pair ASA, computed only where it can differ from the isolated
            # value: an atom's ASA changes on pairing only if a partner atom
            # sits within the shadow cutoff (the ``near`` sets). The ASA of
            # those atoms is evaluated over the changed atoms plus every
            # pair atom that could occlude them (also within the shadow
            # cutoff) -- identical result to a full-pair calculation, at a
            # fraction of the cost on large complexes.
            changed = near1 + near2
            pair_member = np.zeros(n_atoms, dtype=bool)
            pair_member[mol1_ids] = True
            pair_member[mol2_ids] = True
            subset = set(changed)
            for ball in kd_tree.query_ball_point(all_coords[changed], shadow_cutoff):
                for j in ball:
                    if pair_member[j]:
                        subset.add(j)
            subset = sorted(subset)
            asa_12 = calculate_asa(
                atoms=[atoms[i] for i in subset], atom_indices=subset,
                **asa_kwargs)
            bsa_pair: Dict[int, float] = {}
            for mi, ids in ((mol1, near1), (mol2, near2)):
                for gi in ids:
                    b = asa_alone.get((mi, gi), 0.0) - asa_12.get(gi, 0.0)
                    if b > 1e-6:
                        bsa_pair[gi] = b
            interface_area = sum(bsa_pair.values()) / 2.0

            # Atom-atom contacts at the classic 5 A contact cutoff
            idx1, idx2 = find_interface_atoms(atoms, mask1, mask2, interface_cutoff)
            contacts = find_contacts(
                atoms, mask1, mask2, mol1_ids, mol2_ids, idx1, idx2,
            ) if idx1 and idx2 else []

            if interface_area < 0.01 and not contacts:
                continue

            # PISA-grade bond detection (geometric H-bonds; independent
            # predicates -- a charged pair can be both salt bridge and
            # H-bond, exactly as original PISA lists it in both tables).
            bond_flags = detect_bond_flags(contacts, atoms, all_coords, kd_tree)
            for c, f in zip(contacts, bond_flags):
                if "disulfide" in f:
                    c.bond_type = "disulfide"
                elif "salt_bridge" in f:
                    c.bond_type = "salt_bridge"
                elif "hbond" in f:
                    c.bond_type = "hbond"
                else:
                    c.bond_type = "other"
            hbond_pairs = {
                (min(c.atom1_idx, c.atom2_idx), max(c.atom1_idx, c.atom2_idx))
                for c, f in zip(contacts, bond_flags) if "hbond" in f
            }
            # Bond counts are PISA-calibrated in EVERY mode: independent
            # predicates over the atom contacts (PISA counts a charged
            # H-bonded pair in both its h-bond and salt-bridge tables).
            n_hbonds = sum(1 for f in bond_flags if "hbond" in f)
            n_salt = sum(1 for f in bond_flags if "salt_bridge" in f)
            n_ss = sum(1 for f in bond_flags if "disulfide" in f)
            n_other = sum(1 for f in bond_flags if not f)

            # COCOMAPS residue contact map (same cutoff => same interfaces;
            # H-bond classification shares the geometric detector above)
            residue_contacts = None
            if want_cocomaps:
                from fastpisa.cocomaps.contact_map import build_residue_contact_map
                residue_contacts = build_residue_contact_map(
                    atoms, mask1, mask2, mol1_ids, mol2_ids, interaction_cutoff,
                    hbond_pairs=hbond_pairs,
                )

            interface_id += 1

            # Interface atoms/residues, PISA-style: those with pair dASA > 0
            iface_ids1 = [i for i in mol1_ids if i in bsa_pair]
            iface_ids2 = [i for i in mol2_ids if i in bsa_pair]
            if not iface_ids1 and idx1:
                iface_ids1 = idx1
            if not iface_ids2 and idx2:
                iface_ids2 = idx2

            # Energies / scores (identical in every mode). The solvation gain
            # uses the PAIR-specific buried surface -- not the assembly-wide
            # one, which is contaminated by burial against OTHER chains.
            interface_atom_set = set(iface_ids1) | set(iface_ids2)
            solv_energy = calculate_solvation_energy(
                interface_atom_set, bsa_pair, atoms)
            solv_apolar, solv_polar = solvation_energy_components(
                interface_atom_set, bsa_pair, atoms)
            # PISA's stab_en = dG_solv + per-bond contributions (constants
            # recovered exactly from the reference engine; see energy.py).
            stab_energy = solv_energy + bond_energy(n_hbonds, n_salt, n_ss)

            n_res1 = len(set((atoms[i].auth_asym_id, atoms[i].res_seq, atoms[i].icode)
                             for i in iface_ids1))
            n_res2 = len(set((atoms[i].auth_asym_id, atoms[i].res_seq, atoms[i].icode)
                             for i in iface_ids2))

            # P-value: PISA's actual definition -- probability that a random
            # surface patch burying the same areas is at least as hydrophobic.
            surf_sig = np.concatenate([surf_stats[mol1][0], surf_stats[mol2][0]])
            surf_area = np.concatenate([surf_stats[mol1][1], surf_stats[mol2][1]])
            p_value = calculate_p_value_pisa(
                solv_energy, list(bsa_pair.values()), surf_sig, surf_area)
            css = calculate_css_pisa(solv_energy, interface_area)

            iface = Interface(
                interface_id=interface_id,
                molecule1_id=mol1,
                molecule2_id=mol2,
                interface_area=round(interface_area, 2),
                solvation_energy=round(solv_energy, 2),
                solvation_energy_apolar=round(solv_apolar, 2),
                solvation_energy_polar=round(solv_polar, 2),
                stabilization_energy=round(stab_energy, 2),
                p_value=round(p_value, 3),
                css=round(css, 3),
                number_interface_residues=n_res1 + n_res2,
                contacts=contacts,
            )

            iface.number_hydrogen_bonds = n_hbonds
            iface.number_salt_bridges = n_salt
            iface.number_disulfide_bonds = n_ss
            iface.number_covalent_bonds = 0
            iface.number_other_bonds = n_other

            if collect_calibration:
                bsa_by_class: Dict[str, float] = {}
                for gi in interface_atom_set:
                    b = bsa_pair.get(gi, 0.0)
                    if b:
                        c = class_all[gi]
                        bsa_by_class[c] = bsa_by_class.get(c, 0.0) + b
                surf_by_class: Dict[str, float] = {}
                for mi in (mol1, mol2):
                    for c, a_c in surf_class_asa[mi].items():
                        surf_by_class[c] = surf_by_class.get(c, 0.0) + a_c
                surf_by_type: Dict[str, float] = {}
                for mi in (mol1, mol2):
                    for t, a_t in surf_type_asa[mi].items():
                        surf_by_type[t] = surf_by_type.get(t, 0.0) + a_t
                # Residue-level records: every residue of either molecule
                # that buries any area in THIS pair, with its buried area
                # per fine atom type and its isolated-monomer ASA -- the
                # quantities PISA reports per interface residue.
                residues = []
                for mi in (mol1, mol2):
                    for (ch, seq, ic), ids_r in res_atoms[mi].items():
                        b_by_t: Dict[str, float] = {}
                        b_tot = 0.0
                        for gi in ids_r:
                            b = bsa_pair.get(gi, 0.0)
                            if b:
                                t = fine_all[gi]
                                b_by_t[t] = b_by_t.get(t, 0.0) + b
                                b_tot += b
                        if b_tot <= 0.0:
                            continue
                        residues.append({
                            "chain": ch, "seqnum": seq, "icode": ic,
                            "name": atoms[ids_r[0]].res_name.strip(),
                            "asa_iso": float(sum(asa_alone.get((mi, gi), 0.0)
                                                 for gi in ids_r)),
                            "bsa": b_tot,
                            "bsa_by_type": b_by_t,
                        })
                b_vals = np.fromiter(bsa_pair.values(), dtype=float,
                                     count=len(bsa_pair))
                iface.calibration = {
                    "bsa_by_class": bsa_by_class,
                    "surf_asa_by_class": surf_by_class,
                    "surf_asa_by_type": surf_by_type,
                    "residues": residues,
                    "b_sum": float(b_vals.sum()),
                    "b_sq_sum": float((b_vals ** 2).sum()),
                    "n_buried_atoms": int(b_vals.size),
                }

            if want_cocomaps:
                from fastpisa.cocomaps.contact_map import aggregate_residue_pairs
                from collections import Counter
                population = dict(Counter(
                    c.interaction_type for c in residue_contacts))
                contact_map_entries = aggregate_residue_pairs(residue_contacts, atoms)
                iface.cocomaps = {
                    "interaction_population": population,
                    "contact_map": contact_map_entries,
                    "num_residue_pairs": len(contact_map_entries),
                }

            # Per-residue surface data on both molecule entries. PISA
            # convention: residue 'asa' is the ISOLATED-monomer ASA, 'bsa'
            # is the area buried by THIS interface.
            asa_alone_1 = {i: asa_alone.get((mol1, i), 0.0) for i in mol1_ids}
            asa_alone_2 = {i: asa_alone.get((mol2, i), 0.0) for i in mol2_ids}
            mol_info_1 = molecules[mol1].copy()
            mol_info_1["int_natoms"] = len(iface_ids1)
            mol_info_1["int_nres"] = n_res1
            mol_info_1.update(compute_per_residue_surface(
                atoms, asa_alone_1, bsa_pair, set(iface_ids1), mol1_ids))
            mol_info_2 = molecules[mol2].copy()
            mol_info_2["int_natoms"] = len(iface_ids2)
            mol_info_2["int_nres"] = n_res2
            mol_info_2.update(compute_per_residue_surface(
                atoms, asa_alone_2, bsa_pair, set(iface_ids2), mol2_ids))
            iface.molecules = [mol_info_1, mol_info_2]

            # Private per-pair surface data (calibration / downstream tools)
            iface._bsa_pair = bsa_pair
            iface._iface_ids = (iface_ids1, iface_ids2)

            interfaces.append(iface)

    # Optional significance filter (drop weak / artifact interfaces)
    if min_css > 0:
        interfaces = [i for i in interfaces if i.css >= min_css]
        for idx, iface in enumerate(interfaces):
            iface.interface_id = idx + 1

    return CoreState(
        structure=structure,
        atoms=atoms,
        molecules=molecules,
        masks=masks,
        asa_combined=asa_combined,
        asa_alone=asa_alone,
        bsa_combined=bsa_combined,
        assembly_asa=assembly_asa,
        assembly_bsa=assembly_bsa,
        total_asa_alone=total_asa_alone,
        interfaces=interfaces,
    )


def build_documents(
    state: CoreState,
    pdb_id: str = "unknown",
    assembly_id: str = "1",
) -> dict:
    """Assemble the ``interfaces``/``assembly`` JSON documents from a core run.

    The assembly dissociation energy uses the PISA-mode formula
    (sum over interfaces of ``-(dG_solv + dG_contact) + TdS``) in every mode.
    """
    interfaces = state.interfaces
    molecules = state.molecules

    total_interface_area = sum(i.interface_area for i in interfaces)
    total_solv_energy = sum(i.solvation_energy for i in interfaces)

    solv_energies = [i.solvation_energy for i in interfaces]
    contact_energies = [
        bond_energy(i.number_hydrogen_bonds, i.number_salt_bridges,
                    i.number_disulfide_bonds)
        for i in interfaces
    ]
    entropies = [
        calculate_entropy(
            i.interface_area,
            i.number_interface_residues // 2,
            i.number_interface_residues - i.number_interface_residues // 2,
        )
        for i in interfaces
    ]
    diss_energy = calculate_assembly_dissociation_energy(
        [i.interface_area for i in interfaces],
        solv_energies, contact_energies, entropies,
    )

    formula = _build_formula(molecules)
    composition = _build_composition(molecules)
    n_macromolecular = sum(
        1 for m in molecules if m.get("chain_type") == "polymer")

    common = dict(
        pdb_id=pdb_id,
        assembly_id=assembly_id,
        assembly_mmsize=str(n_macromolecular),
        assembly_dissociation_energy=round(diss_energy, 2),
        assembly_asa=round(state.assembly_asa, 2),
        assembly_bsa=round(state.assembly_bsa, 2),
        assembly_entropy=round(sum(entropies), 2),
        assembly_dissociation_area=round(total_interface_area, 2),
        assembly_solvation_energy_gain=round(total_solv_energy, 2),
        assembly_formula=formula,
        assembly_composition=composition,
    )

    interfaces_json = build_interfaces_json(
        interfaces=interfaces,
        total_atoms=len(state.atoms),
        total_asa=state.assembly_asa,
        **common,
    )
    assembly_json = build_assembly_json(
        assembly_size=str(len(molecules)),
        **common,
    )

    return {
        "interfaces": interfaces_json,
        "assembly": assembly_json,
        "interfaces_obj": interfaces,
    }


def analyze(
    input_file: str,
    pdb_id: str = "unknown",
    assembly_id: str = "1",
    probe_radius: float = 1.4,
    point_density: int = 480,
    interface_cutoff: float = 5.0,
    exclude_water: bool = True,
    min_css: float = 0.0,
    mode: str = "combined",
    interaction_cutoff: float = 5.0,
    ligand_mode: str = "separate",
) -> dict:
    """One-call analysis: run the core in the given mode and build the JSON."""
    state = run_core(
        input_file,
        probe_radius=probe_radius,
        point_density=point_density,
        interface_cutoff=interface_cutoff,
        exclude_water=exclude_water,
        min_css=min_css,
        mode=mode,
        interaction_cutoff=interaction_cutoff,
        ligand_mode=ligand_mode,
    )
    return build_documents(state, pdb_id=pdb_id, assembly_id=assembly_id)


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
