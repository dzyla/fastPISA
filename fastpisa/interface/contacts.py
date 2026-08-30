"""
Interface detection and atom-atom contact classification.

PISA identifies interface atoms as those within a distance cutoff
of an atom in a different chain/molecule. The default cutoff is
5.0 A (including the probe radius and a small tolerance).

Atom-atom contacts are classified into:
  - Hydrogen bonds (H-bonds)
  - Salt bridges
  - Disulfide bonds
  - Other bonds (van der Waals contacts)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Standard polymer residue names used for molecule classification and masks.
# Classification is by residue COMPOSITION, not the parser's sticky
# chain.group flag (a protein chain with a bound ligand is mislabeled ligand).
# These are module-level single sources of truth -- edit here, not in each
# function that bins residues (see get_molecules / get_molecule_masks).
# ---------------------------------------------------------------------------
AMINO_ACIDS = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL", "MSE", "SEC", "PYL",
    # common modified residues that are part of the polymer chain (splitting
    # them out as ligands fabricates interfaces PISA does not report)
    "CCS", "CSO", "CSD", "CME", "OCS", "KCX", "LLP", "MLY", "M3L",
    "PTR", "SEP", "TPO", "HYP", "PCA", "CGU", "CSX", "SMC", "NEP",
    "MLZ", "FME", "CRO", "CR2", "CR8", "NH2",
})

# Canonical DNA/RNA plus their common modified forms (PDB CCD codes). Canonical
# mmCIF uses A/G/C/T/U and DA/DG/DC/DT; RA/RG/RC/RT/RU and the modifications are
# non-standard residues that must still be treated as polymer nucleic acids so
# e.g. a methylated rRNA is recognised as one RNA molecule, not split into
# per-residue ligands.
NUCLEIC_ACIDS = frozenset({
    "A", "G", "C", "T", "U",
    "DA", "DG", "DC", "DT", "DU",
    "RA", "RG", "RC", "RT", "RU",
    # common RNA/DNA modifications
    "5MC", "PSU", "7MG", "2MG", "1MA", "H2U", "OMC", "5MU", "M2G", "5HC", "YG",
    "6MA", "OMG", "4SU",
})


@dataclass
class AtomContact:
    """A contact between two atoms across an interface."""
    atom1_idx: int
    atom2_idx: int
    distance: float
    atom1_name: str
    atom2_name: str
    atom1_residue: str
    atom2_residue: str
    atom1_chain: str
    atom2_chain: str
    bond_type: str = "other"  # "hbond", "salt_bridge", "disulfide", "other", "covalent"


@dataclass
class Interface:
    """A detected interface between two molecules."""
    interface_id: int
    molecule1_id: int
    molecule2_id: int
    interface_area: float = 0.0
    solvation_energy: float = 0.0
    stabilization_energy: float = 0.0
    p_value: float = 0.0
    css: float = 0.0
    number_interface_residues: int = 0
    number_hydrogen_bonds: int = 0
    number_covalent_bonds: int = 0
    number_disulfide_bonds: int = 0
    number_salt_bridges: int = 0
    number_other_bonds: int = 0
    contacts: List[AtomContact] = field(default_factory=list)
    molecules: List[dict] = field(default_factory=list)
    # COCOMAPS mode extension: contact map + interaction population
    cocomaps: dict = field(default_factory=dict)

    def to_bond_dict(self, bond_type: str) -> dict:
        """Convert contacts of a given type to a bond dict for output."""
        contacts = [c for c in self.contacts if c.bond_type == bond_type]
        if not contacts:
            return {
                "bond_distances": [],
                "atom_site_1_chains": [],
                "atom_site_1_residues": [],
                "atom_site_1_label_asym_ids": [],
                "atom_site_1_orig_label_asym_ids": [],
                "atom_site_1_unp_accs": [],
                "atom_site_1_unp_nums": [],
                "atom_site_1_seq_nums": [],
                "atom_site_1_label_seq_ids": [],
                "atom_site_1_label_atom_ids": [],
                "atom_site_1_inscodes": [],
                "atom_site_2_chains": [],
                "atom_site_2_residues": [],
                "atom_site_2_label_asym_ids": [],
                "atom_site_2_orig_label_asym_ids": [],
                "atom_site_2_unp_accs": [],
                "atom_site_2_unp_nums": [],
                "atom_site_2_seq_nums": [],
                "atom_site_2_label_seq_ids": [],
                "atom_site_2_label_atom_ids": [],
                "atom_site_2_inscodes": [],
            }
        return {
            "bond_distances": [round(c.distance, 6) for c in contacts],
            "atom_site_1_chains": [c.atom1_chain for c in contacts],
            "atom_site_1_residues": [c.atom1_residue for c in contacts],
            "atom_site_1_label_asym_ids": [c.atom1_chain for c in contacts],
            "atom_site_1_orig_label_asym_ids": [c.atom1_chain for c in contacts],
            "atom_site_1_unp_accs": [None] * len(contacts),
            "atom_site_1_unp_nums": [None] * len(contacts),
            "atom_site_1_seq_nums": [None] * len(contacts),
            "atom_site_1_label_seq_ids": [None] * len(contacts),
            "atom_site_1_label_atom_ids": [c.atom1_name for c in contacts],
            "atom_site_1_inscodes": [None] * len(contacts),
            "atom_site_2_chains": [c.atom2_chain for c in contacts],
            "atom_site_2_residues": [c.atom2_residue for c in contacts],
            "atom_site_2_label_asym_ids": [c.atom2_chain for c in contacts],
            "atom_site_2_orig_label_asym_ids": [c.atom2_chain for c in contacts],
            "atom_site_2_unp_accs": [None] * len(contacts),
            "atom_site_2_unp_nums": [None] * len(contacts),
            "atom_site_2_seq_nums": [None] * len(contacts),
            "atom_site_2_label_seq_ids": [None] * len(contacts),
            "atom_site_2_label_atom_ids": [c.atom2_name for c in contacts],
            "atom_site_2_inscodes": [None] * len(contacts),
        }


# Distance cutoffs for bond classification (A)
# H-bond donor-acceptor cutoff matches original PISA (its reported H-bond
# lists contain donor-acceptor distances up to ~3.89 A; 3.5 undercounted).
HBOND_DISTANCE = 3.89
SALT_BRIDGE_DISTANCE = 4.0
DISULFIDE_DISTANCE = 3.0
OTHER_CONTACT_DISTANCE = 5.0
# NOTE: there is no generic COVALENT_DISTANCE — inter-molecular covalent bonds
# are treated as disulfides only (the old ``d < 2.2`` rule mislabelled crystal
# self-copies and made PISA/COCOMAPS counts diverge).


def is_hydrogen_bond(
    atom1_resname: str,
    atom1_name: str,
    atom1_element: str,
    atom2_resname: str,
    atom2_name: str,
    atom2_element: str,
    distance: float,
) -> bool:
    """Check if a contact is a hydrogen bond (rule-based, no explicit H needed).

    Delegates to the same donor/acceptor chemistry used by COCOMAPS mode
    (``fastpisa.cocomaps.interactions._hbond``). A contact is an H-bond when
    one side provides a donor (N-H / O-H) and the other an acceptor (N/O),
    within HBOND_DISTANCE (3.5 A). Because most modern structures (AlphaFold,
    cryo-EM) contain no explicit hydrogen atoms, this must NOT require an atom
    named ending in 'H'.
    """
    from fastpisa.cocomaps.interactions import _hbond
    if distance >= HBOND_DISTANCE:
        return False
    return _hbond(
        atom1_resname.strip().upper(),
        atom1_name.strip().upper(),
        atom2_resname.strip().upper(),
        atom2_name.strip().upper(),
        atom1_element.upper().strip(),
        atom2_element.upper().strip(),
    )


def is_salt_bridge(
    atom1_resname: str,
    atom1_name: str,
    atom2_resname: str,
    atom2_name: str,
    distance: float,
) -> bool:
    """Check if a contact is a salt bridge (ionic interaction).

    A salt bridge is between a positive side-chain atom (Arg NE/NH1/NH2,
    Lys NZ, His ND1/NE2) and a negative side-chain atom (Asp OD1/OD2,
    Glu OE1/OE2). It uses the SALT_CHARGES table (fastpisa.interface.bonds)
    validated against original PISA's salt-bridge lists -- NOT a generic
    'any N-O pair', which would mis-classify backbone H-bonds as salt
    bridges, and NOT the cation-pi CHARGED_ATOMS table (whose Arg CZ carbon
    is not a salt-bridge partner).
    """
    from fastpisa.interface.bonds import salt_charge
    if distance > SALT_BRIDGE_DISTANCE:
        return False
    c1 = salt_charge(atom1_resname, atom1_name)
    c2 = salt_charge(atom2_resname, atom2_name)
    if c1 is None or c2 is None:
        return False
    return c1 * c2 < 0


def is_disulfide(
    atom1_resname: str,
    atom2_resname: str,
    atom1_element: str,
    atom2_element: str,
    distance: float,
) -> bool:
    """Check if a contact is a disulfide bond (Cys S-gamma ... S-gamma).

    Requires both atoms to be sulfur AND both residues to be CYS. Fixes the
    previous bug where ANY atom pair closer than 3.0 A was counted as a
    disulfide regardless of element/residue.
    """
    return (
        atom1_element.upper().strip() == "S"
        and atom2_element.upper().strip() == "S"
        and atom1_resname.strip().upper() == "CYS"
        and atom2_resname.strip().upper() == "CYS"
        and distance < DISULFIDE_DISTANCE
    )


def _element_from_name(atom_name: str) -> str:
    """Extract element symbol from atom name."""
    name = atom_name.strip()
    el = name[0:2].strip()
    if len(el) == 2 and not el[1].isalpha():
        el = el[0]
    el = el.upper()
    if not el or not el[0].isalpha():
        el = name[0].upper() if name else "C"
    return el


def find_interface_atoms(
    atoms,
    mol1_mask: np.ndarray,
    mol2_mask: np.ndarray,
    cutoff: float = 5.0,
) -> Tuple[List[int], List[int]]:
    """Find interface atoms on each side of an interface.

    Uses KD-tree for efficient spatial search.
    """
    idx1 = [i for i, m in enumerate(mol1_mask) if m]
    idx2 = [i for i, m in enumerate(mol2_mask) if m]

    coords1 = np.array([[atoms[i].x, atoms[i].y, atoms[i].z] for i in idx1])
    coords2 = np.array([[atoms[i].x, atoms[i].y, atoms[i].z] for i in idx2])

    if len(coords1) == 0 or len(coords2) == 0:
        return [], []

    # Build KD-tree for coords2
    tree = cKDTree(coords2)
    dist_sq1 = tree.query_ball_point(coords1, cutoff)  # list of arrays
    interface_idx1 = [idx1[i] for i in range(len(idx1)) if len(dist_sq1[i]) > 0]

    # For mol2 atoms, check against mol1
    tree2 = cKDTree(coords1)
    dist_sq2 = tree2.query_ball_point(coords2, cutoff)
    interface_idx2 = [idx2[j] for j in range(len(idx2)) if len(dist_sq2[j]) > 0]

    return interface_idx1, interface_idx2


def find_contacts(
    atoms,
    mol1_mask: np.ndarray,
    mol2_mask: np.ndarray,
    mol1_ids: list,
    mol2_ids: list,
    interface_atoms1: list,
    interface_atoms2: list,
    contact_cutoff: float = 5.0,
) -> List[AtomContact]:
    """Find all atom-atom contacts across an interface.

    Uses KD-tree for efficient spatial search.
    """
    contacts = []

    coords1 = np.array([[atoms[i].x, atoms[i].y, atoms[i].z] for i in interface_atoms1])
    coords2 = np.array([[atoms[i].x, atoms[i].y, atoms[i].z] for i in interface_atoms2])

    if len(coords1) == 0 or len(coords2) == 0:
        return contacts

    # Build KD-tree for coords2
    tree = cKDTree(coords2)

    # For each atom in mol1 interface, find neighbors in mol2 interface
    dist_pairs = tree.query_ball_point(coords1, contact_cutoff)

    for i1, idx1 in enumerate(interface_atoms1):
        for j2 in dist_pairs[i1]:
            idx2 = interface_atoms2[j2]
            d2 = np.sum((coords1[i1] - coords2[j2]) ** 2)
            if d2 >= contact_cutoff ** 2:
                continue
            d = d2 ** 0.5

            a1 = atoms[idx1]
            a2 = atoms[idx2]

            # Classify contact using the SHARED chemistry (identical disulfide /
            # salt-bridge / H-bond rules as COCOMAPS mode, single source of
            # truth). There is deliberately NO blanket "covalent" class here:
            # genuine inter-molecular covalent bonds are essentially only
            # Cys-Cys disulfides, and a generic ``d < 2.2 A`` rule mislabels
            # crystallographic self-copies (identical atoms ~1.5 A apart) as
            # covalent, which also suppressed their H-bond/salt classification
            # so that PISA counts diverged from COCOMAPS.
            if is_disulfide(a1.res_name, a2.res_name, a1.element, a2.element, d):
                btype = "disulfide"
            elif is_salt_bridge(a1.res_name, a1.atom_name, a2.res_name, a2.atom_name, d):
                btype = "salt_bridge"
            elif is_hydrogen_bond(
                a1.res_name, a1.atom_name, a1.element,
                a2.res_name, a2.atom_name, a2.element, d,
            ):
                btype = "hbond"
            else:
                btype = "other"

            contacts.append(AtomContact(
                atom1_idx=idx1,
                atom2_idx=idx2,
                distance=d,
                atom1_name=a1.atom_name,
                atom2_name=a2.atom_name,
                atom1_residue=a1.res_name,
                atom2_residue=a2.res_name,
                atom1_chain=a1.auth_asym_id,
                atom2_chain=a2.auth_asym_id,
                bond_type=btype,
            ))

    return contacts


def is_water_molecule(mol) -> bool:
    """Whether a molecule dict represents a water/ordered-solvent ligand."""
    return mol.get("ccd_id") in {"HOH", "WAT", "OH2", "DOD", "TP3", "TIP", "SOL"}


def is_water_ligand(ccd_id) -> bool:
    """Whether a CCD code is a water/solvent molecule."""
    return (ccd_id or "").upper() in {"HOH", "WAT", "OH2", "DOD", "TP3", "TIP", "SOL"}


def filter_water_molecules(molecules, exclude_water=True):
    """Drop water/solvent ligand molecules (preserves polymer chains)."""
    if not exclude_water:
        return molecules
    return [m for m in molecules if not is_water_molecule(m)]


def get_molecules(structure, merge_ligands: bool = False):
    """Categorize chains into molecules (polymers and ligands).

    Default (``merge_ligands=False``, classic-PISA convention): a chain that
    contains both polymer (amino acid / nucleotide) residues and bound ligand
    residues (e.g. a heme in a protein chain) is split:
      - one polymer molecule for the chain's standard residues
      - one ligand molecule per ligand residue (per unfolded chain)

    ``merge_ligands=True`` (the jsPISA-on-assembly convention): each chain is
    ONE molecule comprising its polymer residues AND its bound non-water
    hetero groups, so a cofactor at an interface counts toward its parent
    chain's interface. Pure-ligand chains stay single molecules.

    Classification is based on residue composition (standard AA / NA),
    not on the parser's chain.group flag (which is sticky and unreliable
    when a chain mixes polymer and ligand residues).

    Returns a list of molecule dicts with:
      - molecule_id
      - molecule_class: "Protein", "NucleicAcid", "Ligand"
      - chain_id (auth_asym_id)
      - chain_type: "polymer" | "ligand" | "chain" (merged)
    """
    molecules = []
    mol_id = 0

    if merge_ligands:
        for chain in structure.chains:
            atoms_nonwater = [a for a in chain.atoms
                              if not is_water_ligand(a.res_name)]
            if not atoms_nonwater:
                continue
            res_names = set(a.res_name.upper() for a in atoms_nonwater)
            if res_names & AMINO_ACIDS:
                mol_class = "Protein"
            elif res_names & NUCLEIC_ACIDS:
                mol_class = "NucleicAcid"
            else:
                mol_class = "Ligand"
            molecules.append({
                "molecule_id": mol_id,
                "molecule_class": mol_class,
                "chain_id": chain.auth_asym_id,
                "auth_asym_id": chain.auth_asym_id,
                "label_asym_id": chain.label_asym_id,
                "label_comp_id": chain.label_comp_id,
                "chain_type": "chain",
            })
            mol_id += 1
        return molecules

    for chain in structure.chains:
        if not chain.atoms:
            continue

        poly_atoms = []
        ligand_groups = {}  # (res_name) -> {atom}
        for atom in chain.atoms:
            rn = atom.res_name.upper()
            # Standard polymer residues are ATOM records of amino/nucleic acids
            is_poly_res = (rn in AMINO_ACIDS) or (rn in NUCLEIC_ACIDS)
            if is_poly_res:
                poly_atoms.append(atom)
            else:
                ligand_groups.setdefault(rn, []).append(atom)

        # Polymer molecule
        if poly_atoms:
            res_names = set(a.res_name.upper() for a in poly_atoms)
            if res_names & AMINO_ACIDS:
                mol_class = "Protein"
            elif res_names & NUCLEIC_ACIDS:
                mol_class = "NucleicAcid"
            else:
                mol_class = "Other"
            molecules.append({
                "molecule_id": mol_id,
                "molecule_class": mol_class,
                "chain_id": chain.auth_asym_id,
                "auth_asym_id": chain.auth_asym_id,
                "label_asym_id": chain.label_asym_id,
                "label_comp_id": chain.label_comp_id,
                "chain_type": "polymer",
            })
            mol_id += 1

        # Ligand molecules: group atoms into residue units (by CCD + auth seq
        # + insertion code, matching original PISA's monomer naming)
        for ccd, atoms_list in ligand_groups.items():
            by_seq = {}
            for a in atoms_list:
                by_seq.setdefault((a.auth_seq_id, (a.icode or "").strip()), []).append(a)
            for seq_id, icode in sorted(by_seq):
                molecules.append({
                    "molecule_id": mol_id,
                    "molecule_class": "Ligand",
                    "chain_id": f"[{ccd}]{chain.auth_asym_id}:{seq_id}{icode}",
                    "auth_asym_id": chain.auth_asym_id,
                    "label_asym_id": chain.label_asym_id,
                    "label_comp_id": ccd,
                    "ccd_id": ccd,
                    "auth_seq_id": seq_id,
                    "icode": icode,
                    "chain_type": "ligand",
                })
                mol_id += 1

    return molecules


def get_molecule_masks(atoms, molecules):
    """Create boolean masks for each atom indicating which molecule it belongs to.

    Polymer masks include only the chain's standard polymer residues (not
    water / ligand residues that happen to share the same chain ID). Ligand
    masks include only the atoms of the matching ligand residue.
    """
    n = len(atoms)

    # Per-atom attribute arrays built once, so each molecule's mask is a
    # vectorised comparison instead of an O(n_molecules * n_atoms) Python loop.
    chains = np.array([a.auth_asym_id for a in atoms])
    seqs = np.array([a.auth_seq_id for a in atoms])
    icodes = np.array([(a.icode or "").strip() for a in atoms])
    comps = np.array([a.label_comp_id for a in atoms])
    is_poly = np.array([
        a.res_name.upper() in AMINO_ACIDS or a.res_name.upper() in NUCLEIC_ACIDS
        for a in atoms
    ]) if n else np.zeros(0, dtype=bool)

    masks = []
    not_water = None
    for mol in molecules:
        ctype = mol.get("chain_type", "polymer")
        if ctype == "ligand":
            mask = ((chains == mol["auth_asym_id"])
                    & (seqs == mol["auth_seq_id"])
                    & (icodes == mol.get("icode", ""))
                    & (comps == mol.get("ccd_id", None)))
        elif ctype == "chain":
            # merged-ligand molecule: the whole chain minus water
            if not_water is None:
                not_water = np.array(
                    [not is_water_ligand(a.res_name) for a in atoms]
                ) if n else np.zeros(0, dtype=bool)
            mask = (chains == mol["auth_asym_id"]) & not_water
        else:
            # polymer: match chain ID AND a standard polymer residue
            mask = (chains == mol["auth_asym_id"]) & is_poly
        masks.append(mask)
    return masks