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
        asp = get_asp(atom.atom_name, atom.element, atom.res_name)
        bsa = atom_bsa.get(idx, 0.0)
        solv_energy += asp * bsa

    return solv_energy


def solvation_energy_components(
    interface_atom_indices: set,
    atom_bsa: dict,
    atoms,
) -> tuple:
    """(apolar, polar) parts of the interface solvation gain (kcal/mol).

    apolar = sum over carbon / sulfur atoms (the hydrophobic effect), polar =
    everything else (N, O, ions). They sum to
    :func:`calculate_solvation_energy` exactly.
    """
    from fastpisa.energy.asp_table import atom_class, is_apolar_class

    apolar = polar = 0.0
    for idx in interface_atom_indices:
        atom = atoms[idx]
        bsa = atom_bsa.get(idx, 0.0)
        if not bsa:
            continue
        term = get_asp(atom.atom_name, atom.element, atom.res_name) * bsa
        if is_apolar_class(atom_class(atom.atom_name, atom.element, atom.res_name)):
            apolar += term
        else:
            polar += term
    return apolar, polar


# Per-bond free-energy contributions (kcal/mol), recovered EXACTLY from the
# original PISA engine: regressing (stab_en - int_solv_en) on PISA's own
# h-bond / salt-bridge / disulfide counts over 117 EBI reference interfaces
# reproduces stab_en with ZERO residual for these constants.
E_HBOND = -0.444037
E_SALT_BRIDGE = -0.150028
E_DISULFIDE = -4.0


def bond_energy(n_hbonds: int, n_salt_bridges: int, n_disulfides: int) -> float:
    """Contact free energy from bond counts (PISA's stab_en - int_solv_en).

    Counts follow PISA semantics: independent predicates, so a charged pair
    that is also an H-bond contributes BOTH terms.
    """
    return (E_HBOND * n_hbonds + E_SALT_BRIDGE * n_salt_bridges
            + E_DISULFIDE * n_disulfides)


def calculate_contact_energy(
    contacts: List[AtomContact],
) -> tuple:
    """Calculate the contact energy contribution from a contact list.

    Uses the PISA per-bond constants (see :func:`bond_energy`). NOTE: contact
    ``bond_type`` labels are mutually exclusive (a dual salt-bridge/H-bond
    pair is labeled salt_bridge), so prefer :func:`bond_energy` with the
    interface's independent bond COUNTS when they are available.

    Returns
    -------
    tuple of (contact_energy, hbond_energy, salt_bridge_energy,
    hbond_distances, salt_bridge_distances)
    """
    hbond_distances = [c.distance for c in contacts if c.bond_type == "hbond"]
    salt_bridge_distances = [c.distance for c in contacts
                             if c.bond_type == "salt_bridge"]
    n_ss = sum(1 for c in contacts if c.bond_type == "disulfide")

    hbond_energy = E_HBOND * len(hbond_distances)
    salt_bridge_energy = E_SALT_BRIDGE * len(salt_bridge_distances)
    contact_energy = hbond_energy + salt_bridge_energy + E_DISULFIDE * n_ss
    return (contact_energy, hbond_energy, salt_bridge_energy,
            hbond_distances, salt_bridge_distances)


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
    and the entropy change due to loss of translational, rotational, and
    vibrational freedom upon assembly.

    Ideally, these components map to subunit mass, surface area, symmetry 
    numbers, and moments of inertia to calculate the free energy of 
    dissociation effectively. In this fast Python port, the entropy term 
    is estimated as a linear proxy from the buried surface area
    and the number of residues involved, rather than a full rigid-body 
    statistical mechanical computation.

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

    .. warning::
        Rough approximation. PISA uses a statistical-mechanical model
        (translational/rotational entropy of the dissociating molecules via
        Boltzmann statistics, Krissinel & Henrick 2007 Eq. 7-11); the linear
        formula here is a hand-tuned surrogate and will diverge from the CCP4
        binary, which matters most for ``dissociation_energy``. Use for
        relative ranking only.
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