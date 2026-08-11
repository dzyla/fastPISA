"""
Energy calculation for PISA interfaces.

PISA computes the free energy of dissociation (ΔGint) as:
  ΔGint = ΔGsolv + ΔGcont + ΔGes

where:
  ΔGsolv = solvation free energy change = Σ(ASP_k × BSA_k)  (negative = favourable)
  ΔGcont = contact (hydrogen bond + salt bridge) energy
  ΔGes   = electrostatic energy

The total assembly dissociation energy combines all interfaces.
The entropy term (TΔS) is estimated from the buried surface area
and is added to give the total free energy of dissociation.
"""

import numpy as np
from typing import List, Dict
from fastpisa.energy.asp_table import get_asp
from fastpisa.interface.contacts import AtomContact


def calculate_solvation_energy(
    interface_atom_indices: set,
    atom_bsa: dict,
    atoms,
) -> float:
    """Calculate the solvation energy gain for an interface.

    ΔGsolv = Σ(ASP_k × BSA_k) for all interface atoms

    Parameters
    ----------
    interface_atom_indices : set
        Set of atom indices that are at the interface.
    atom_bsa : dict
        Buried surface area per atom (from combined ASA calculation).
    atoms : list of Atom
        All atoms in the structure.

    Returns
    -------
    float
        Solvation energy in kcal/mol.  Negative = favourable.
    """
    solv_energy = 0.0
    for idx in interface_atom_indices:
        atom = atoms[idx]
        asp = get_asp(atom.atom_name, atom.element)
        bsa = atom_bsa.get(idx, 0.0)
        solv_energy += asp * bsa

    return solv_energy


def calculate_contact_energy(
    contacts: List[AtomContact],
) -> tuple:
    """Calculate the contact energy contribution.

    Contact energy comes from hydrogen bonds and salt bridges.
    Each H-bond contributes approximately -0.5 to -1.5 kcal/mol
    (distance-dependent), and each salt bridge contributes
    approximately -0.5 to -2.0 kcal/mol.

    Returns
    -------
    tuple of (contact_energy, hbond_energy, salt_bridge_energy)
    """
    hbond_energy = 0.0
    salt_bridge_energy = 0.0

    hbond_distances = []
    salt_bridge_distances = []

    for contact in contacts:
        if contact.bond_type == "hbond":
            hbond_distances.append(contact.distance)
            if contact.distance < 2.8:
                hbond_energy -= 1.5
            elif contact.distance < 3.2:
                hbond_energy -= 0.8
            else:
                hbond_energy -= 0.3
        elif contact.bond_type == "salt_bridge":
            salt_bridge_distances.append(contact.distance)
            if contact.distance < 3.0:
                salt_bridge_energy -= 2.0
            elif contact.distance < 3.5:
                salt_bridge_energy -= 1.0
            else:
                salt_bridge_energy -= 0.5

    contact_energy = hbond_energy + salt_bridge_energy
    return contact_energy, hbond_energy, salt_bridge_energy, hbond_distances, salt_bridge_distances


def calculate_binding_energy(
    solv_energy: float,
    contacts: List[AtomContact],
) -> float:
    """Calculate the binding energy ΔGint for an interface.

    ΔGint = ΔGsolv + ΔGcont + ΔGes

    Parameters
    ----------
    solv_energy : float
        Solvation energy (kcal/mol).
    contacts : list of AtomContact
        Contacts across the interface.

    Returns
    -------
    float
        Binding energy (kcal/mol).  Negative = favourable.
    """
    contact_energy, _, _, _, _ = calculate_contact_energy(contacts)

    # Note: salt bridges are already included in contact_energy. There is no
    # separate electrostatic term (they were previously counted twice).
    binding_energy = solv_energy + contact_energy
    return binding_energy


def calculate_entropy(
    interface_area: float,
    n_residues_1: int,
    n_residues_2: int,
) -> float:
    """Calculate the entropy change (TΔS) for interface formation.

    PISA estimates the entropy of immobilisation of surface side chains
    and the entropy change due to loss of translational/rotational
    freedom upon assembly.

    The entropy term is estimated from the buried surface area
    and the number of residues involved.

    Parameters
    ----------
    interface_area : float
        Interface area in A^2.
    n_residues_1, n_residues_2 : int
        Number of interface residues on each side.

    Returns
    -------
    float
        Entropy contribution in kcal/mol.
    """
    # Simplified: entropy ~ 0.02 * interface_area + 0.5 * n_residues_total
    # This is a rough approximation calibrated to PISA values
    entropy = 0.02 * interface_area + 0.5 * (n_residues_1 + n_residues_2)
    return entropy


def calculate_dissociation_energy(
    interface_areas: List[float],
    solv_energies: List[float],
    entropies: List[float],
) -> float:
    """Calculate the total dissociation energy for an assembly.

    ΔGdiss = Σ(-ΔGsolv_i + TΔS_i) for all interfaces i

    Parameters
    ----------
    interface_areas : list
    solv_energies : list
    entropies : list

    Returns
    -------
    float
        Total dissociation energy (kcal/mol).  Positive = stable.
    """
    total = 0.0
    for area, solv, entropy in zip(interface_areas, solv_energies, entropies):
        total += -solv + entropy
    return total


def calculate_stabilization_energy(
    solv_energy: float,
    contacts: List[AtomContact],
) -> float:
    """Calculate the stabilization energy for an interface.

    PISA's stabilization energy = ΔGsolv + (H-bond + salt bridge contributions)
    """
    contact_energy, _, _, _, _ = calculate_contact_energy(contacts)
    return solv_energy + contact_energy


def calculate_assembly_dissociation_energy(
    interface_areas: List[float],
    solv_energies: List[float],
    contact_energies: List[float],
    entropies: List[float],
) -> float:
    """Calculate the total dissociation energy for an assembly.

    ΔGdiss = Σ(-ΔGint_i + TΔS_i) for all interfaces

    where ΔGint = ΔGsolv + ΔGcont + ΔGes

    Parameters
    ----------
    interface_areas : list
    solv_energies : list
        Per-interface solvation energies.
    contact_energies : list
        Per-interface contact energies.
    entropies : list
        Per-interface entropy terms.

    Returns
    -------
    float
        Total dissociation energy (kcal/mol).  Positive = stable.
    """
    total = 0.0
    for solv, contact, entropy in zip(solv_energies, contact_energies, entropies):
        dgin = -(solv + contact)  # dissociation energy
        total += dgin + entropy
    return total