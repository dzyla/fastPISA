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
HBOND_DISTANCE = 3.5
SALT_BRIDGE_DISTANCE = 4.0
DISULFIDE_DISTANCE = 3.0
OTHER_CONTACT_DISTANCE = 5.0
COVALENT_DISTANCE = 2.2


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

    A salt bridge is between a positive side-chain atom (Arg NH1/NH2, Lys NZ,
    His ND1/NE2) and a negative side-chain atom (Asp OD1/OD2, Glu OE1/OE2).
    It uses the CHARGED_ATOMS table shared with COCOMAPS mode -- NOT a generic
    'any N-O pair', which would mis-classify backbone H-bonds as salt bridges.
    """
    from fastpisa.cocomaps.interactions import CHARGED_ATOMS
    if distance > SALT_BRIDGE_DISTANCE:
        return False
    r1 = atom1_resname.strip().upper()
    a1 = atom1_name.strip().upper()
    r2 = atom2_resname.strip().upper()
    a2 = atom2_name.strip().upper()
    c1 = CHARGED_ATOMS.get((r1, a1)) if (r1, a1) in CHARGED_ATOMS else None
    c2 = CHARGED_ATOMS.get((r2, a2)) if (r2, a2) in CHARGED_ATOMS else None
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

            # Classify contact (disulfide must be checked before generic
            # covalent so real Cys S-S bonds are not swallowed by 'covalent')
            if is_disulfide(a1.res_name, a2.res_name, a1.element, a2.element, d):
                btype = "disulfide"
            elif d < COVALENT_DISTANCE:
                btype = "covalent"
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


def get_molecules(structure):
    """Categorize chains into molecules (polymers and ligands).

    A chain may contain both polymer (amino acid / nucleotide) residues and
    bound ligand residues (e.g. a heme in a protein chain). These are split:
      - one polymer molecule for the chain's standard residues
      - one ligand molecule per ligand residue (per unfolded chain)

    Classification is based on residue composition (standard AA / NA),
    not on the parser's chain.group flag (which is sticky and unreliable
    when a chain mixes polymer and ligand residues).

    Returns a list of molecule dicts with:
      - molecule_id
      - molecule_class: "Protein", "DNA", "RNA", "Ligand"
      - chain_id (auth_asym_id)
      - chain_type: "polymer" | "ligand"
    """
    molecules = []
    mol_id = 0
    protein_aas = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
                   "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
                   "THR", "TRP", "TYR", "VAL", "MSE", "SEC", "PYL"}
    nuc_acids = {"A", "G", "C", "T", "U", "DA", "DG", "DC", "DT", "DU",
                 "RA", "RG", "RC", "RT", "RU"}

    for chain in structure.chains:
        if not chain.atoms:
            continue

        poly_atoms = []
        ligand_groups = {}  # (res_name) -> {atom}
        for atom in chain.atoms:
            rn = atom.res_name.upper()
            # Standard polymer residues are ATOM records of amino/nucleic acids
            is_poly_res = (rn in protein_aas) or (rn in nuc_acids)
            if is_poly_res:
                poly_atoms.append(atom)
            else:
                ligand_groups.setdefault(rn, []).append(atom)

        # Polymer molecule
        if poly_atoms:
            res_names = set(a.res_name.upper() for a in poly_atoms)
            if res_names & protein_aas:
                mol_class = "Protein"
            elif res_names & nuc_acids:
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

        # Ligand molecules: group atoms into residue units (by CCD + auth seq)
        for ccd, atoms_list in ligand_groups.items():
            # group by auth_seq_id so multiple copies of the same ligand are distinct
            by_seq = {}
            for a in atoms_list:
                by_seq.setdefault(a.auth_seq_id, []).append(a)
            for seq_id in sorted(by_seq):
                molecules.append({
                    "molecule_id": mol_id,
                    "molecule_class": "Ligand",
                    "chain_id": f"[{ccd}]{chain.auth_asym_id}:{seq_id}",
                    "auth_asym_id": chain.auth_asym_id,
                    "label_asym_id": chain.label_asym_id,
                    "label_comp_id": ccd,
                    "ccd_id": ccd,
                    "auth_seq_id": seq_id,
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
    masks = []
    protein_aas = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
                   "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
                   "THR", "TRP", "TYR", "VAL", "MSE", "SEC", "PYL"}
    nuc_acids = {"A", "G", "C", "T", "U", "DA", "DG", "DC", "DT", "DU",
                 "RA", "RG", "RC", "RT", "RU"}

    for mol in molecules:
        mask = np.zeros(n, dtype=bool)
        mol_chain_type = mol.get("chain_type", "polymer")

        if mol_chain_type == "ligand":
            auth_chain = mol["auth_asym_id"]
            auth_seq = mol["auth_seq_id"]
            ccd = mol.get("ccd_id", None)
            for i, atom in enumerate(atoms):
                if (atom.auth_asym_id == auth_chain and
                        atom.auth_seq_id == auth_seq and
                        atom.label_comp_id == ccd):
                    mask[i] = True
        else:
            # polymer: match chain ID AND a standard polymer residue
            auth_chain = mol["auth_asym_id"]
            for i, atom in enumerate(atoms):
                if atom.auth_asym_id != auth_chain:
                    continue
                rn = atom.res_name.upper()
                if rn in protein_aas or rn in nuc_acids:
                    mask[i] = True

        masks.append(mask)
    return masks


# Global reference to the current atoms list (set by the pipeline before analysis)
atoms_global = []