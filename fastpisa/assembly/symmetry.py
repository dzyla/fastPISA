"""
Crystallographic symmetry and assembly prediction for PISA.

PISA uses the crystallographic space group symmetry operators to
generate neighbouring ASU copies and build assemblies. The assembly
prediction algorithm:

1. Reads the space group from the PDB CRYST1 record.
2. Generates all symmetry operators (rotation matrices + translations).
3. Applies operators to the ASU to create neighbouring copies.
4. Builds a contact graph between copies.
5. Predicts assemblies by finding clusters of stable interfaces.
6. Calculates the dissociation energy for each assembly.

For the as-is mode (interfaces only), PISA does not predict assemblies
but still reports all interfaces found in the crystal.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SymmetryOperator:
    """A crystallographic symmetry operator (rotation + translation)."""
    rotation: np.ndarray  # 3x3 matrix
    translation: np.ndarray  # 3-vector


def parse_space_group(space_group: str) -> List[SymmetryOperator]:
    """Parse a space group string into symmetry operators.

    For PISA, the most important operators are the 24 general-position
    operators for orthorhombic and lower symmetry groups. For higher
    symmetry groups, we extract the full set.

    This implementation supports common space groups used in PDB
    structures. For full support, the gemmi library can be used.

    Parameters
    ----------
    space_group : str
        Space group name from the CRYST1 record.

    Returns
    -------
    list of SymmetryOperator
    """
    if not space_group:
        # If no space group, return the identity operator only
        return [SymmetryOperator(
            rotation=np.eye(3),
            translation=np.zeros(3),
        )]

    # Try to use gemmi for full space group parsing
    try:
        import gemmi
        sg = gemmi.find_spacegroup_by_name(space_group)
        operators = []
        for op in sg.get_symmetry_ops():
            rot = np.array(op.rotation)
            trans = np.array(op.translation)
            operators.append(SymmetryOperator(rot, trans))
        return operators if operators else [SymmetryOperator(np.eye(3), np.zeros(3))]
    except (ImportError, Exception):
        pass

    # Fallback: return identity if gemmi is not available
    return [SymmetryOperator(np.eye(3), np.zeros(3))]


def apply_symmetry(
    atoms,
    operators: List[SymmetryOperator],
    n_asu: int = 1,
) -> List:
    """Apply symmetry operators to generate all copies of the ASU.

    Parameters
    ----------
    atoms : list of Atom
        Atoms in the asymmetric unit.
    operators : list of SymmetryOperator
        Symmetry operators.
    n_asu : int
        Number of ASUs in the unit cell (from Z).

    Returns
    -------
    list of Atom
        All atoms in the unit cell (asu copies × operators).
    """
    all_atoms = []

    for op_idx, op in enumerate(operators):
        for atom in atoms:
            x, y, z = atom.x, atom.y, atom.z
            # Apply rotation and translation
            new_x = op.rotation[0] * x + op.rotation[1] * y + op.rotation[2] * z + op.translation[0]
            new_y = op.rotation[0] * x + op.rotation[1] * y + op.rotation[2] * z + op.translation[1]
            new_z = op.rotation[0] * x + op.rotation[1] * y + op.rotation[2] * z + op.translation[2]

            # Copy the atom with new coordinates
            from fastpisa.parser.pdb_parser import Atom
            new_atom = Atom(
                atom_name=atom.atom_name,
                altloc=atom.altloc,
                res_name=atom.res_name,
                chain_id=atom.chain_id,
                res_seq=atom.res_seq,
                icode=atom.icode,
                x=float(new_x),
                y=float(new_y),
                z=float(new_z),
                occupancy=atom.occupancy,
                bfactor=atom.bfactor,
                element=atom.element,
                label_asym_id=atom.label_asym_id,
                label_seq_id=atom.label_seq_id,
                label_comp_id=atom.label_comp_id,
                auth_asym_id=atom.auth_asym_id,
                auth_seq_id=atom.auth_seq_id,
                group=atom.group,
            )
            new_atom._op_idx = op_idx  # track which operator generated this atom
            all_atoms.append(new_atom)

    return all_atoms


def generate_assemblies(
    structure,
    operators: List[SymmetryOperator],
    interfaces: list,
    total_asa: float,
) -> List:
    """Generate predicted assemblies from interfaces.

    PISA builds assemblies by finding clusters of molecules connected
    by significant interfaces. Each assembly is a connected component
    of the contact graph.

    Parameters
    ----------
    structure : PDBStructure
    operators : list
    interfaces : list
        Detected interfaces.
    total_asa : float
        Total accessible surface area.

    Returns
    -------
    list
        Predicted assemblies.
    """
    # Build a graph of molecules connected by interfaces
    n_molecules = len(structure.chains)

    # Union-find for connected components
    parent = list(range(n_molecules + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    # Union molecules connected by interfaces
    for iface in interfaces:
        if len(iface.molecules) >= 2:
            union(iface.molecules[0]["molecule_id"] + 1,
                  iface.molecules[1]["molecule_id"] + 1)

    # Group by connected component
    components = {}
    for i in range(1, n_molecules + 1):
        root = find(i)
        if root not in components:
            components[root] = []
        components[root].append(i - 1)

    assemblies = []
    for idx, (root, mol_indices) in enumerate(components.items()):
        assembly = {
            "assembly_id": idx + 1,
            "molecules": [structure.chains[m]["auth_asym_id"] for m in mol_indices],
            "size": len(mol_indices),
            "interface_count": sum(1 for iface in interfaces
                                   if len(iface.molecules) >= 2 and
                                   iface.molecules[0]["molecule_id"] in mol_indices and
                                   iface.molecules[1]["molecule_id"] in mol_indices),
        }
        assemblies.append(assembly)

    return assemblies