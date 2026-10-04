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
from typing import Optional, List, Tuple, Dict
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
    atom1_seq: int = 0
    atom2_seq: int = 0
    atom1_icode: str = ""
    atom2_icode: str = ""
    # Independent PISA predicates. ``bond_type`` above remains the dominant
    # compatibility label used by existing consumers and visual styling.
    bond_types: tuple = field(default_factory=tuple)

    def has_bond_type(self, kind: str) -> bool:
        """Whether this contact satisfies an independent PISA bond class."""
        if self.bond_types:
            return kind in self.bond_types
        return self.bond_type == kind

    @property
    def label(self) -> str:
        """Human-readable ``A:ARG83.NH1 -- B:GLU76.OE1  2.85 A``."""
        return (f"{self.atom1_chain}:{self.atom1_residue}{self.atom1_seq}"
                f"{self.atom1_icode}.{self.atom1_name.strip()} -- "
                f"{self.atom2_chain}:{self.atom2_residue}{self.atom2_seq}"
                f"{self.atom2_icode}.{self.atom2_name.strip()}  "
                f"{self.distance:.2f} A")

    def as_dict(self) -> dict:
        return {
            "chain_1": self.atom1_chain, "residue_1": self.atom1_residue,
            "seq_1": self.atom1_seq, "icode_1": self.atom1_icode or "",
            "atom_1": self.atom1_name.strip(),
            "chain_2": self.atom2_chain, "residue_2": self.atom2_residue,
            "seq_2": self.atom2_seq, "icode_2": self.atom2_icode or "",
            "atom_2": self.atom2_name.strip(),
            "distance": round(self.distance, 3), "bond_type": self.bond_type,
        }


@dataclass
class Interface:
    """A detected interface between two molecules."""
    interface_id: int
    molecule1_id: int
    molecule2_id: int
    interface_area: float = 0.0
    solvation_energy: float = 0.0
    # hydrophobic (C/S burial) and polar (N/O/ion burial) parts of
    # solvation_energy; they sum to it. PISA's "hydrophobic interactions"
    # live here, not in a contact list.
    solvation_energy_apolar: float = 0.0
    solvation_energy_polar: float = 0.0
    stabilization_energy: float = 0.0
    p_value: float = 0.0
    css: float = 0.0
    number_interface_residues: int = 0
    number_hydrogen_bonds: int = 0
    number_covalent_bonds: int = 0
    number_disulfide_bonds: int = 0
    number_salt_bridges: int = 0
    number_other_bonds: int = 0
    #: How many times this interface occurs per asymmetric unit in the
    #: crystal. Symmetry-equivalent copies are collapsed into one entry
    #: (PISA's ``n_occ``); always 1 outside ``symmetry="crystal"``.
    n_occurrences: int = 1
    contacts: List[AtomContact] = field(default_factory=list)
    molecules: List[dict] = field(default_factory=list)
    # COCOMAPS mode extension: contact map + interaction population
    cocomaps: dict = field(default_factory=dict)
    # Calibration extension (only populated when run_core is called with
    # collect_calibration=True): the sufficient statistics needed to refit
    # the ASP sigmas and the P-value model offline. See
    # fastpisa/reference/calibrate.py.
    calibration: dict = field(default_factory=dict)

    # -- readable accessors -------------------------------------------------
    @property
    def chains(self) -> tuple:
        """``(chain_id_1, chain_id_2)`` as PISA labels them (``[ZN]A:301``
        for a hetero group)."""
        return tuple(m.get("chain_id", "?") for m in self.molecules)

    @property
    def label(self) -> str:
        return " + ".join(self.chains)

    @property
    def hydrogen_bonds(self) -> List[AtomContact]:
        """PISA-style hydrogen bonds (donor/acceptor heavy-atom pairs)."""
        return [c for c in self.contacts if c.has_bond_type("hbond")]

    @property
    def salt_bridges(self) -> List[AtomContact]:
        return [c for c in self.contacts if c.has_bond_type("salt_bridge")]

    @property
    def disulfides(self) -> List[AtomContact]:
        return [c for c in self.contacts if c.has_bond_type("disulfide")]

    @property
    def contact_map(self) -> List[dict]:
        """COCOMAPS residue-residue contact map (``combined``/``cocomaps``
        modes): one dict per residue pair with ``min_distance``,
        ``num_contacts``, ``interaction_counts`` and ``dominant_interaction``.
        Empty in ``pisa`` mode."""
        return list(self.cocomaps.get("contact_map", []))

    @property
    def interaction_population(self) -> dict:
        """COCOMAPS interaction-class counts over all atom pairs."""
        return dict(self.cocomaps.get("interaction_population", {}))

    def residues(self, side: Optional[int] = None) -> List[dict]:
        """Interface residues with ASA / BSA / solvation energy.

        ``side`` 1 or 2 selects a molecule; None returns both, each dict
        carrying ``molecule`` (1 or 2), ``chain``, ``name``, ``seq``,
        ``icode``, ``asa``, ``bsa``, ``solvation_energy``.
        """
        out = []
        for k, m in enumerate(self.molecules, 1):
            if side is not None and k != side:
                continue
            names = m.get("residue_label_comp_ids", [])
            seqs = m.get("residue_seq_ids", [])
            ics = m.get("residue_ins_codes", [])
            asa = m.get("accessible_surface_areas", [])
            bsa = m.get("buried_surface_areas", [])
            solv = m.get("solvation_energies", [])
            for j in range(len(names)):
                out.append({
                    "molecule": k, "chain": m.get("chain_id", ""),
                    "name": names[j], "seq": seqs[j],
                    "icode": (ics[j] or "") if j < len(ics) else "",
                    "asa": asa[j] if j < len(asa) else None,
                    "bsa": bsa[j] if j < len(bsa) else None,
                    "solvation_energy": solv[j] if j < len(solv) else None,
                })
        return out

    def bonds_dataframe(self):
        """All bonded contacts (H-bonds, salt bridges, disulfides) as a
        pandas DataFrame."""
        import pandas as pd
        rows = []
        for kind in ("hbond", "salt_bridge", "disulfide"):
            for contact in self.contacts:
                if contact.has_bond_type(kind):
                    row = contact.as_dict()
                    row["bond_type"] = kind
                    rows.append(row)
        return pd.DataFrame(rows)

    def contact_map_dataframe(self):
        """The residue-residue contact map as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.contact_map)

    def __repr__(self) -> str:
        return (f"<Interface {self.interface_id}: {self.label} | "
                f"area {self.interface_area:.0f} A^2, dG {self.solvation_energy:+.1f} "
                f"(apolar {self.solvation_energy_apolar:+.1f}, polar "
                f"{self.solvation_energy_polar:+.1f}), stab {self.stabilization_energy:+.1f} "
                f"kcal/mol, P {self.p_value:.2f}, CSS {self.css:.2f} | "
                f"{self.number_hydrogen_bonds} H-bonds, {self.number_salt_bridges} "
                f"salt bridges, {self.number_disulfide_bonds} SS>")

    def to_bond_dict(self, bond_type: str) -> dict:
        """Convert contacts of a given type to a bond dict for output."""
        contacts = [c for c in self.contacts if c.has_bond_type(bond_type)]
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
    """CHEMISTRY-ONLY H-bond precondition: donor/acceptor pair within range.

    Delegates to :func:`fastpisa.interface.bonds.hb_roles` -- the one
    calibrated donor/acceptor table in the package. A contact qualifies when
    one side can donate and the other accept, within
    :data:`HBOND_DISTANCE` (3.89 A, PISA's own cutoff). Explicit hydrogens
    are not required: most modern models (AlphaFold, cryo-EM) have none.

    This is NOT the H-bond fastPISA counts. The reported counts additionally
    require the antecedent-angle geometry, metal-coordination exclusion and
    donor/acceptor capacities applied by
    :func:`fastpisa.interface.bonds.detect_bond_flags`, which is the single
    classifier for every output. Use this only as a cheap screen or to ask
    "could these two atoms hydrogen-bond at all?".
    """
    from fastpisa.interface.bonds import hb_roles

    if distance >= HBOND_DISTANCE:
        return False
    roles1 = hb_roles(atom1_resname, atom1_name, atom1_element)
    roles2 = hb_roles(atom2_resname, atom2_name, atom2_element)
    return (("donor" in roles1 and "acceptor" in roles2)
            or ("donor" in roles2 and "acceptor" in roles1))


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


def find_interface_atoms(
    atoms,
    mol1_mask: np.ndarray,
    mol2_mask: np.ndarray,
    cutoff: float = 5.0,
) -> Tuple[List[int], List[int]]:
    """Interface atoms on each side of ONE molecule pair.

    Kept as a standalone utility for ad-hoc queries. Do NOT call it in a loop
    over molecule pairs: it rebuilds both index lists, both coordinate arrays
    and two KD-trees per call, which made interface detection O(N^2) in the
    molecule count. :func:`cross_molecule_neighbors` answers the same question
    for every pair in one pass and is what ``run_core`` uses.
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


def cross_molecule_neighbors(
    coords: np.ndarray,
    mol_of_atom: np.ndarray,
    kd_tree,
    cutoff: float,
) -> Dict[Tuple[int, int], Tuple[List[int], List[int]]]:
    """For every molecule pair in contact, the near atoms on each side.

    One pass over the atoms replaces the former per-pair scan. The old
    :func:`find_interface_atoms` rebuilt a molecule's index list, its
    coordinate array and two KD-trees for each of the O(N^2) molecule pairs,
    which dominated the runtime on assemblies with many molecules even though
    almost every pair was far apart.

    Parameters
    ----------
    coords : (n_atoms, 3) array
        Coordinates of every parsed atom.
    mol_of_atom : (n_atoms,) int array
        Molecule index per atom, or ``-1`` for atoms outside the interface
        search (hydrogens carry no surface; water is excluded by default).
    kd_tree : cKDTree
        Tree over ``coords`` (the one ``run_core`` already builds).
    cutoff : float
        Inclusive distance cutoff.

    Returns
    -------
    dict
        ``{(mol_lo, mol_hi): (near_atoms_in_mol_lo, near_atoms_in_mol_hi)}``,
        each list sorted and duplicate-free. Pairs with no contact are
        absent, so iterating the result skips them for free.
    """
    pairs: Dict[Tuple[int, int], Tuple[set, set]] = {}
    active = np.flatnonzero(mol_of_atom >= 0)
    if active.size == 0:
        return {}

    neighborhoods = kd_tree.query_ball_point(coords[active], cutoff)
    for i, neighbors in zip(active, neighborhoods):
        mi = int(mol_of_atom[i])
        for j in neighbors:
            mj = int(mol_of_atom[j])
            if mj < 0 or mj == mi:
                continue
            if mi < mj:
                key, mine, theirs = (mi, mj), 0, 1
            else:
                key, mine, theirs = (mj, mi), 1, 0
            sides = pairs.get(key)
            if sides is None:
                sides = pairs[key] = (set(), set())
            sides[mine].add(int(i))
            sides[theirs].add(int(j))

    return {key: (sorted(lo), sorted(hi)) for key, (lo, hi) in pairs.items()}


def find_contacts(
    atoms,
    interface_atoms1: list,
    interface_atoms2: list,
    contact_cutoff: float = 5.0,
) -> List[AtomContact]:
    """Atom-atom contacts across an interface, within ``contact_cutoff``.

    GEOMETRY ONLY. Every returned contact carries the default
    ``bond_type="other"`` and an empty ``bond_types``: bond classification is
    the sole responsibility of
    :func:`fastpisa.interface.bonds.detect_bond_flags`, which is the only
    code allowed to decide what an H-bond, salt bridge or disulfide is.

    This function used to classify each pair with a second, looser chemistry
    table whose verdict ``run_core`` then overwrote -- wasted work and a
    dormant source of divergence (see
    ``tests/test_bond_chemistry_single_source.py``).

    ``contact_cutoff`` is honoured exactly: it previously kept its own 5.0 A
    default while the caller's ``interface_cutoff`` reached only the
    interface-atom selection, so the option was half-applied.
    """
    contacts: List[AtomContact] = []
    if not interface_atoms1 or not interface_atoms2:
        return contacts

    coords1 = np.array([[atoms[i].x, atoms[i].y, atoms[i].z]
                        for i in interface_atoms1])
    coords2 = np.array([[atoms[i].x, atoms[i].y, atoms[i].z]
                        for i in interface_atoms2])

    tree = cKDTree(coords2)
    cutoff_sq = contact_cutoff ** 2
    for i1, idx1 in enumerate(interface_atoms1):
        for j2 in tree.query_ball_point(coords1[i1], contact_cutoff):
            d2 = float(np.sum((coords1[i1] - coords2[j2]) ** 2))
            if d2 >= cutoff_sq:
                continue
            idx2 = interface_atoms2[j2]
            a1, a2 = atoms[idx1], atoms[idx2]
            contacts.append(AtomContact(
                atom1_idx=idx1,
                atom2_idx=idx2,
                distance=d2 ** 0.5,
                atom1_name=a1.atom_name,
                atom2_name=a2.atom_name,
                atom1_residue=a1.res_name,
                atom2_residue=a2.res_name,
                atom1_chain=a1.auth_asym_id,
                atom2_chain=a2.auth_asym_id,
                atom1_seq=a1.res_seq, atom2_seq=a2.res_seq,
                atom1_icode=a1.icode or "", atom2_icode=a2.icode or "",
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


def _polymer_predicate(structure=None, polymer_residues=None):
    """Return ``is_polymer(res_name)`` for one structure.

    The hardcoded AMINO_ACIDS / NUCLEIC_ACIDS sets are the floor; a residue
    the FILE declares as part of a polymer (``SEQRES`` /
    ``_pdbx_poly_seq_scheme``) counts too. A list of modified residues can
    never be complete -- 12% of a blind 60-entry draw declared a SEQRES
    residue absent from ours (D-amino acids, sulfotyrosine, acetyl caps) --
    and splitting those off as ligands fabricates interfaces and scatters a
    chain's buried area across fragments. PISA treats them as polymer.
    """
    declared = polymer_residues
    if declared is None:
        declared = getattr(structure, "polymer_residues", None) or set()

    def is_polymer(res_name: str) -> bool:
        upper = res_name.upper()
        if upper in AMINO_ACIDS or upper in NUCLEIC_ACIDS:
            return True
        return upper in declared and not is_water_ligand(upper)

    return is_polymer


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
    is_polymer = _polymer_predicate(structure)

    if merge_ligands:
        for chain in structure.chains:
            atoms_nonwater = [a for a in chain.atoms
                              if not is_water_ligand(a.res_name)]
            if not atoms_nonwater:
                continue
            res_names = set(a.res_name.upper() for a in atoms_nonwater)
            poly_names = {r for r in res_names if is_polymer(r)}
            if poly_names & AMINO_ACIDS or (poly_names - NUCLEIC_ACIDS):
                mol_class = "Protein" if (poly_names & AMINO_ACIDS
                                          or poly_names - NUCLEIC_ACIDS) else "NucleicAcid"
            elif poly_names & NUCLEIC_ACIDS:
                mol_class = "NucleicAcid"
            else:
                mol_class = "Ligand"
            if not poly_names:
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
            # A polymer residue is a standard amino acid / nucleotide, or one
            # the file declares in SEQRES / _pdbx_poly_seq_scheme.
            if is_polymer(rn):
                poly_atoms.append(atom)
            else:
                ligand_groups.setdefault(rn, []).append(atom)

        # Polymer molecule
        if poly_atoms:
            res_names = set(a.res_name.upper() for a in poly_atoms)
            if res_names & NUCLEIC_ACIDS and not (res_names - NUCLEIC_ACIDS):
                mol_class = "NucleicAcid"
            elif res_names & AMINO_ACIDS or res_names - NUCLEIC_ACIDS:
                mol_class = "Protein"
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


def get_molecule_masks(atoms, molecules, polymer_residues=None):
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
    is_polymer = _polymer_predicate(polymer_residues=polymer_residues or set())
    is_poly = np.array([is_polymer(a.res_name) for a in atoms]) \
        if n else np.zeros(0, dtype=bool)

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
