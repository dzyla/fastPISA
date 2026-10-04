# Copyright (c) 2026 Dawid Zyla. Part of fastPISA.
# SPDX-License-Identifier: AGPL-3.0-or-later
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

from fastpisa.parser.pdb_parser import (
    parse_pdb, parse_mmcif, is_mmcif_path, PDBStructure,
)
from fastpisa.surface.shrake_rupley import calculate_asa, surface_radius
from fastpisa.interface.contacts import (
    cross_molecule_neighbors, find_contacts, get_molecules, get_molecule_masks,
    filter_water_molecules, Interface,
)
from fastpisa.interface.bonds import detect_bond_flags
from fastpisa.energy.energy import (
    calculate_solvation_energy, solvation_energy_components, bond_energy,
)
from fastpisa.energy.dissociation import (
    DissociationPathway, assembly_dissociation,
)
from fastpisa.energy.entropy import atoms_mass
from fastpisa.scoring.scoring import calculate_p_value_pisa, calculate_css_pisa
from fastpisa.surface.per_residue import (
    compute_per_residue_surface, compute_buried_surface,
)
from fastpisa.output.json_output import build_interfaces_json, build_assembly_json
from fastpisa.assembly.crystal import (
    IDENTITY_COPY, copy_by_label, expand_structure, split_mate_chain_id,
)
from fastpisa.assembly.predict import Assembly, predict_assemblies as _predict

logger = logging.getLogger(__name__)

MODES = ("pisa", "cocomaps", "combined")
SYMMETRY_MODES = ("none", "crystal")


def _shadow_cutoff(atoms, probe_radius: float) -> float:
    """Largest distance at which two atoms can shadow each other's surface."""
    r_max = max((surface_radius(a) for a in atoms), default=2.0)
    return 2.0 * r_max + 2.0 * probe_radius + 0.1


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
    #: Assemblies the crystal can form, most stable first. Populated only
    #: when ``run_core(predict_assemblies=True)``.
    assemblies: List[Assembly] = field(default_factory=list)


@dataclass
class _CoreStateView:
    """The four fields assembly prediction reads, available mid-run.

    ``predict_assemblies`` runs before :class:`CoreState` is constructed, so
    it is handed just the pieces it needs rather than a half-built state.
    """
    atoms: list
    molecules: List[dict]
    masks: List[np.ndarray]
    interfaces: List[Interface]


def _integer_inverse(matrix: np.ndarray) -> np.ndarray:
    """Exact inverse of a unimodular integer matrix, as integers.

    A crystallographic rotation is an integer matrix in FRACTIONAL
    coordinates with determinant +-1, so its inverse is integral too --
    computed here from the adjugate, with no floating point anywhere.

    It is NOT the transpose. A fractional rotation is orthogonal only when
    the cell axes are orthogonal and equal; in hexagonal axes the 3-fold
    ``-y, x-y, z`` has an inverse (the other 3-fold) quite different from
    its transpose. Using the transpose made an interface and its reverse
    view produce different keys, so trigonal and hexagonal crystals reported
    every symmetry interface twice.
    """
    m = np.asarray(matrix, dtype=np.int64)
    det = int(round(float(np.linalg.det(m))))
    if det not in (1, -1):
        raise ValueError(
            f"crystallographic rotation must be unimodular, got det={det}")
    cof = np.empty((3, 3), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            minor = np.delete(np.delete(m, i, axis=0), j, axis=1)
            cof[i, j] = ((-1) ** (i + j)) * (
                int(minor[0, 0]) * int(minor[1, 1])
                - int(minor[0, 1]) * int(minor[1, 0]))
    return (cof.T * det)


def _equivalence_key(mol_a: dict, mol_b: dict) -> tuple:
    """Crystal-symmetry identity of an interface between two molecules.

    EXACT integer arithmetic. A crystallographic placement is an integer
    rotation plus a rational translation in fractional coordinates, so the
    relative placement ``R1^-1 . R2`` is exact and two routes to the same
    placement always produce the same key. Deciding this on rounded
    orthogonal-space floats put real cases on the rounding boundary: 1urn's
    6-fold screw translates by c/6 = 42.55 A, which lands on 42.5 or 42.6
    depending on the last bit, so an interface and its reverse failed to
    collapse and were reported twice.

    The pair is reduced to "molecule A in the asymmetric unit, molecule B
    placed by R" by applying the inverse of A's own placement to both, then
    the smaller of that and the reverse ordering is returned -- so
    A + mate(B, S) and B + mate(A, S^-1), the same physical interface seen
    from its two ends, give one key.
    """
    def side(mol):
        return (
            mol.get("asu_molecule_id",
                    mol.get("chain_id", mol.get("auth_asym_id", ""))),
            np.asarray(mol["frac_rotation"], dtype=np.int64),
            np.asarray(mol["frac_translation"], dtype=np.int64),
        )

    name_a, rot_a, tr_a = side(mol_a)
    name_b, rot_b, tr_b = side(mol_b)

    def relative(r1, t1, r2, t2):
        inv = _integer_inverse(r1)
        return inv @ r2, inv @ (t2 - t1)

    r_ab, t_ab = relative(rot_a, tr_a, rot_b, tr_b)
    r_ba, t_ba = relative(rot_b, tr_b, rot_a, tr_a)
    first = (name_a, name_b, tuple(r_ab.ravel().tolist()), tuple(t_ab.tolist()))
    second = (name_b, name_a, tuple(r_ba.ravel().tolist()), tuple(t_ba.tolist()))
    return min(first, second)


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
    symmetry: str = "none",
    predict_assemblies: bool = False,
) -> CoreState:
    """Run the shared analysis once and return the populated interfaces.

    ``ligand_mode``: ``"separate"`` (classic PISA -- every bound hetero group
    is its own monomer) or ``"merge"`` (jsPISA-on-assembly convention -- a
    chain's bound ligands/cofactors belong to that chain's molecule).

    ``symmetry``: ``"none"`` (default -- analyse the given coordinates) or
    ``"crystal"`` (expand the asymmetric unit by its space group and report
    the crystal's interfaces, including packing contacts with symmetry
    mates, as original PISA does for a deposited entry). ``"crystal"`` needs
    a usable ``CRYST1``/``_cell`` plus space group; without one it is a
    no-op, which is the right answer for a predicted model.

    ``predict_assemblies``: also enumerate the finite assemblies the crystal
    can form and rank them by dissociation energy
    (:mod:`fastpisa.assembly.predict`). Off by default -- it is a search over
    interface subsets and no existing caller should pay for it.

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
    if symmetry not in SYMMETRY_MODES:
        raise ValueError(
            f"Unknown symmetry: {symmetry!r} (expected one of {SYMMETRY_MODES})")
    want_cocomaps = mode in ("cocomaps", "combined")

    # 1. Parse input file
    if is_mmcif_path(input_file):
        structure = parse_mmcif(input_file)
    else:
        structure = parse_pdb(input_file)

    asu_structure = structure
    if not structure.atoms:
        raise ValueError(f"No atoms found in {input_file}")

    # 1b. Crystal expansion. The screen must use the SHADOW cutoff, not the
    # 5 A contact cutoff: PISA defines an interface by buried area, and two
    # atoms shadow each other's accessible surface out to r1 + r2 + 2*probe.
    # (1acb interface 9 is a real 14.2 A^2 interface whose closest atom pair
    # is 5.02 A apart.)
    placements = [IDENTITY_COPY]
    if symmetry == "crystal":
        screen = _shadow_cutoff(structure.atoms, probe_radius)
        structure, placements = expand_structure(
            structure, screen, exclude_water=exclude_water)

    atoms = structure.atoms
    n_atoms = len(atoms)
    logger.info("Parsed %d atoms from %s (%d symmetry copies)",
                n_atoms, input_file, len(placements) - 1)

    # 2. Molecules and masks (exclude ordered water from interface search)
    molecules = get_molecules(structure, merge_ligands=(ligand_mode == "merge"))
    molecules = filter_water_molecules(molecules, exclude_water=exclude_water)
    # The file's own polymer declaration must reach the masks too, or a
    # declared-but-non-standard residue lands in no molecule and its surface
    # silently vanishes from the analysis.
    masks = get_molecule_masks(
        atoms, molecules, polymer_residues=structure.polymer_residues)
    n_molecules = len(molecules)
    logger.info("Found %d molecules", n_molecules)

    # Tag every molecule with the crystal placement it came from, so an
    # interface can name its symmetry operation the way PISA does and so
    # symmetry-equivalent interfaces can be collapsed.
    by_label = copy_by_label(placements)
    is_asu = []
    for mol in molecules:
        asu_chain, label = split_mate_chain_id(mol.get("auth_asym_id", ""))
        placement = by_label.get(label, IDENTITY_COPY) if label else IDENTITY_COPY
        mol["asu_chain_id"] = asu_chain
        # Identity of the asymmetric-unit MOLECULE, not just its chain:
        # "A" for a polymer, "[NAG]A:2" for one sugar of a glycan tree.
        # Keying symmetry equivalence on the chain alone makes every hetero
        # group in a chain indistinguishable, which silently merges real
        # interfaces (1ppf's two eight-sugar glycans collapse to one).
        # Rebuilt from the molecule's OWN fields, never by string surgery on
        # chain_id: a mate ligand's chain id is "[SO4]H~4_555:623", and
        # splitting that on the mate separator from the right swallowed the
        # residue number into the label and left "[SO4]H" -- collapsing every
        # SO4 of chain H onto one molecule. 1a3n held both "[HEM]A" and
        # "[HEM]A:142"; in 1prc three SO4 residues became one node and two
        # different interfaces resolved to it.
        if mol.get("chain_type") == "ligand":
            mol["asu_molecule_id"] = (
                f"[{mol.get('ccd_id', '')}]{asu_chain}"
                f":{mol.get('auth_seq_id', '')}{mol.get('icode', '')}")
        else:
            mol["asu_molecule_id"] = asu_chain
        mol["symop"] = placement.triplet
        mol["symop_no"] = placement.symop_no
        mol["cell"] = placement.cell
        mol["rotation"] = placement.rotation
        mol["translation"] = placement.translation
        # Exact fractional placement: what symmetry equivalence is decided on.
        mol["frac_rotation"] = placement.frac_rotation
        mol["frac_translation"] = placement.frac_translation
        mol["frac_denominator"] = placement.frac_denominator
        if label:
            # Display label matching PISA's convention for a mate.
            mol["chain_id"] = mol["asu_molecule_id"] + f"[{label}]"
        is_asu.append(label is None)

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
    # Two probe spheres overlap out to r_i + r_j + 2*probe; the largest
    # radius present bounds that for every pair.
    neighbor_cutoff = 2.0 * all_radii.max() + 2.0 * probe_radius
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
    heavy_ids = np.flatnonzero(heavy).tolist()
    asa_combined = calculate_asa(
        atoms=[atoms[i] for i in heavy_ids], atom_indices=heavy_ids, **asa_kwargs)

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

    # Which molecule each heavy atom belongs to (-1 = outside the search:
    # hydrogens, excluded water). Both neighbour screens below are single
    # passes over the atoms keyed by this array, instead of an O(N^2) loop
    # over molecule pairs that rebuilt per-molecule arrays and KD-trees each
    # time -- which was 82% of the runtime on a water-inclusive assembly.
    mol_of_atom = np.full(n_atoms, -1, dtype=int)
    for mi, ids in enumerate(mol_atom_ids):
        if ids:
            mol_of_atom[ids] = mi

    shadow_pairs = cross_molecule_neighbors(
        all_coords, mol_of_atom, kd_tree, shadow_cutoff)
    contact_pairs = (
        shadow_pairs if interface_cutoff >= shadow_cutoff
        else cross_molecule_neighbors(
            all_coords, mol_of_atom, kd_tree, interface_cutoff))

    # Every mate-mate contact is symmetry-equivalent to an ASU-mate one (map
    # both molecules through the inverse of the first one's placement), and
    # the equivalent ASU-mate copy is generated because it touches the ASU.
    # So only pairs with at least one asymmetric-unit molecule are evaluated
    # -- which is also what keeps crystal mode affordable.
    candidate_pairs = sorted(
        pair for pair in shadow_pairs
        if is_asu[pair[0]] or is_asu[pair[1]]
    )
    seen_equivalents: Dict[tuple, int] = {}

    for mol1, mol2 in candidate_pairs:
        mol1_ids, mol2_ids = mol_atom_ids[mol1], mol_atom_ids[mol2]
        near1, near2 = shadow_pairs[(mol1, mol2)]

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
        idx1, idx2 = contact_pairs.get((mol1, mol2), ([], []))
        contacts = find_contacts(atoms, idx1, idx2, interface_cutoff)

        if interface_area < 0.01 and not contacts:
            continue

        # PISA-grade bond detection (geometric H-bonds; independent
        # predicates -- a charged pair can be both salt bridge and
        # H-bond, exactly as original PISA lists it in both tables).
        bond_flags = detect_bond_flags(contacts, atoms, all_coords, kd_tree)
        for c, f in zip(contacts, bond_flags):
            c.bond_types = tuple(
                kind for kind in ("hbond", "salt_bridge", "disulfide")
                if kind in f
            )
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
                atoms, mol1_ids, mol2_ids, interaction_cutoff,
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
            atoms, asa_alone_1, bsa_pair, set(iface_ids1), mol1_ids,
            atom_sigma=sigma_all))
        mol_info_2 = molecules[mol2].copy()
        mol_info_2["int_natoms"] = len(iface_ids2)
        mol_info_2["int_nres"] = n_res2
        mol_info_2.update(compute_per_residue_surface(
            atoms, asa_alone_2, bsa_pair, set(iface_ids2), mol2_ids,
            atom_sigma=sigma_all))
        iface.molecules = [mol_info_1, mol_info_2]

        # Private per-pair surface data (calibration / downstream tools)
        iface._bsa_pair = bsa_pair
        iface._iface_ids = (iface_ids1, iface_ids2)

        # Collapse symmetry-equivalent interfaces: A + mate(B, S) and
        # B + mate(A, S^-1) are one interface in the crystal, which PISA
        # reports once (with a multiplicity). Keyed on the MOLECULE pair plus
        # the relative transform, canonicalised over the two orderings.
        #
        # Only in crystal mode. Without symmetry every candidate pair is a
        # distinct pair of molecule indices and nothing can be equivalent,
        # so there is nothing to collapse -- and running the key there risks
        # merging interfaces that are merely hard to tell apart by label.
        if symmetry != "none":
            key = _equivalence_key(mol_info_1, mol_info_2)
            if key in seen_equivalents:
                interface_id -= 1
                interfaces[seen_equivalents[key]].n_occurrences += 1
                continue
            seen_equivalents[key] = len(interfaces)
        iface.n_occurrences = 1

        interfaces.append(iface)

    # Optional significance filter (drop weak / artifact interfaces)
    if min_css > 0:
        interfaces = [i for i in interfaces if i.css >= min_css]
        for idx, iface in enumerate(interfaces):
            iface.interface_id = idx + 1

    assemblies: List[Assembly] = []
    if predict_assemblies:
        assemblies = _predict(_CoreStateView(
            atoms=atoms, molecules=molecules, masks=masks,
            interfaces=interfaces))

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
        assemblies=assemblies,
    )


def dissociation_pathway(state: CoreState) -> DissociationPathway:
    """Cheapest dissociation pathway of the analysed coordinates.

    The pathway is computed over MACROMOLECULAR components, matching PISA's
    assembly analysis: with ``ligand_mode="separate"`` every bound hetero
    group is its own fastPISA molecule, and letting the search peel a single
    ion off a chain would make "dissociation" mean losing a cofactor rather
    than separating the assembly. Each ligand molecule is therefore folded
    into the polymer molecule it buries the most area against -- its host --
    contributing that ligand's mass and re-pointing its interfaces at the
    host. Ligand-to-host interfaces become internal and are not cuttable.
    A structure made only of ligands keeps them as components, so a
    two-ligand input still has a pathway.

    Returns a pathway with zero energy and a single part when there is
    nothing to break (one component, or no interfaces at all).
    """
    molecules = state.molecules
    interfaces = state.interfaces
    n = len(molecules)
    if n == 0:
        return DissociationPathway(0.0, 0.0, [], [])

    is_ligand = [m.get("molecule_class") == "Ligand" for m in molecules]
    polymer_ids = [i for i in range(n) if not is_ligand[i]]

    # host[i] = component that molecule i belongs to
    host = list(range(n))
    if polymer_ids:
        buried_with = {}
        for iface in interfaces:
            a, b = iface.molecule1_id, iface.molecule2_id
            for lig, other in ((a, b), (b, a)):
                if is_ligand[lig] and not is_ligand[other]:
                    key = (lig, other)
                    buried_with[key] = max(buried_with.get(key, 0.0),
                                           iface.interface_area)
        best = {}
        for (lig, other), area in buried_with.items():
            if lig not in best or area > best[lig][1]:
                best[lig] = (other, area)
        for lig, (other, _) in best.items():
            host[lig] = other

    masses = {}
    for i in range(n):
        # state.masks are heavy-atom only (the PISA surface convention), so a
        # hydrogenated model's components come out ~8% light. The entropy term
        # goes as the 3/2 power of a mass RATIO, so that shifts T*dS by
        # ~0.07 kcal/mol -- an order of magnitude below the model's own
        # 0.9 kcal/mol agreement with PISA, and it partly cancels between the
        # parts and the whole. Not worth re-deriving per-molecule atom sets.
        mass = atoms_mass(state.atoms[j] for j in np.flatnonzero(state.masks[i]))
        masses[host[i]] = masses.get(host[i], 0.0) + mass
    # Components with no mass of their own (empty masks) cannot be bodies.
    masses = {k: v for k, v in masses.items() if v > 0.0}

    edges = []
    for iface in interfaces:
        a, b = host[iface.molecule1_id], host[iface.molecule2_id]
        if a == b or a not in masses or b not in masses:
            continue
        edges.append((a, b, iface.stabilization_energy))

    pathway = assembly_dissociation(masses, edges)
    # Label the parts with PISA-style chain ids rather than molecule indices.
    label = {i: molecules[i].get("chain_id", str(i)) for i in range(n)}
    pathway.parts = [[label.get(i, str(i)) for i in part]
                     for part in pathway.parts]
    pathway.cut = [(label.get(a, str(a)), label.get(b, str(b)))
                   for a, b in pathway.cut]
    return pathway


def _symop_of(state, molecule_id: str, placement):
    """Symmetry-operation number of a placement, or ``None`` if unknown.

    Matched on the rotation and the translation MODULO the cell, which is
    what a space-group operation is; the lattice part is reported separately
    as ``cell``. Requiring the full translation to match meant only
    placements in the exact cached shell were ever found, and a composed
    placement fell through to a default of 1 -- on 1urn, 9 of 39 output
    placements claimed the identity operation while their rotation was not
    the identity, so a consumer rebuilding the assembly stacked molecules on
    top of each other.

    Returns ``None`` rather than a plausible default: a miss has to be
    visible. The exact fractional placement is emitted alongside, so the
    document stays reconstructible either way.
    """
    den = placement.denominator
    wanted_rotation = placement.rotation
    wanted_translation = tuple(v % den for v in placement.translation)
    for iface in state.interfaces:
        for mol in iface.molecules:
            if mol.get("asu_molecule_id") != molecule_id:
                continue
            rotation = tuple(tuple(int(v) for v in row)
                             for row in mol.get("frac_rotation", ()))
            if rotation != wanted_rotation:
                continue
            translation = tuple(int(v) % den
                                for v in mol.get("frac_translation", ()))
            if translation == wanted_translation:
                return int(mol.get("symop_no"))
    return None


def build_documents(
    state: CoreState,
    pdb_id: str = "unknown",
    assembly_id: str = "1",
) -> dict:
    """Assemble the ``interfaces``/``assembly`` JSON documents from a core run.

    ``dissociation_energy`` and ``entropy`` follow PISA's own relation,

        dG_diss = -sum(stabilization_energy over the interfaces CUT) - T*dS

    along the cheapest dissociation pathway
    (:func:`fastpisa.energy.dissociation.assembly_dissociation`), with the
    rigid-body entropy of :mod:`fastpisa.energy.entropy`. The pathway itself
    is returned under ``"dissociation_pathway"``.
    """
    interfaces = state.interfaces
    molecules = state.molecules

    total_interface_area = sum(i.interface_area for i in interfaces)
    total_solv_energy = sum(i.solvation_energy for i in interfaces)

    pathway = dissociation_pathway(state)
    diss_energy = pathway.dissociation_energy
    assembly_entropy = pathway.entropy

    predicted = [
        {
            "rank": a.rank,
            "size": a.size,
            "mmsize": a.mmsize,
            "composition": a.composition,
            "formula": a.formula,
            "dissociation_energy": round(a.dissociation_energy, 2),
            "entropy": round(a.entropy, 2),
            "n_interfaces": a.n_interfaces,
            "interface_ids": [int(i) for i in a.interface_ids],
            "molecules": [
                {"asu_molecule_id": molecule,
                 "symop_no": _symop_of(state, molecule, placement),
                 "cell": [int(v) for v in placement.cell()],
                 # The exact placement, so the assembly is reconstructible
                 # without depending on the symop lookup succeeding.
                 "frac_rotation": [[int(v) for v in row]
                                   for row in placement.rotation],
                 "frac_translation": [int(v) for v in placement.translation],
                 "frac_denominator": int(placement.denominator)}
                for molecule, placement in a.nodes
            ],
        }
        for a in getattr(state, "assemblies", [])
    ]

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
        assembly_entropy=round(assembly_entropy, 2),
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
        predicted_assemblies=predicted,
        **common,
    )

    return {
        "interfaces": interfaces_json,
        "assembly": assembly_json,
        "interfaces_obj": interfaces,
        "dissociation_pathway": pathway,
        "assemblies": list(getattr(state, "assemblies", [])),
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
    symmetry: str = "none",
    predict_assemblies: bool = False,
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
        symmetry=symmetry,
        predict_assemblies=predict_assemblies,
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
