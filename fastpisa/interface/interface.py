"""
Interface area calculation.

The interface area between two molecules is computed as:
  interface_area = (BSA_mol1 + BSA_mol2 - BSA_combined) / 2

where BSA (buried surface area) is the surface area of a molecule
that becomes inaccessible to solvent upon complexation.

For two molecules A and B:
  interface_area = (BSA_A + BSA_B - BSA_AB) / 2

In practice, PISA computes the interface area directly from the
solvent-accessible surface areas:
  interface_area = (ASA_alone_A + ASA_alone_B - ASA_complex) / 2

where ASA_complex is the accessible surface area of the combined
A+B system. This avoids double-counting the buried surfaces.
"""

import numpy as np


def calculate_interface_area(
    asa_alone_1: dict,
    asa_alone_2: dict,
    asa_combined: dict,
    atoms1: list,
    atoms2: list,
) -> float:
    """Calculate the interface area between two molecules.

    interface_area = (BSA_1 + BSA_2 - BSA_combined) / 2
    where BSA = total_surface - ASA.

    Alternatively:
    interface_area = (ASA_alone_1 + ASA_alone_2 - ASA_combined) / 2

    This is because:
      BSA_1 = total_1 - ASA_alone_1
      BSA_2 = total_2 - ASA_alone_2
      BSA_combined = (total_1 + total_2) - ASA_combined

      interface_area = (BSA_1 + BSA_2 - BSA_combined) / 2
                     = (total_1 - ASA_alone_1 + total_2 - ASA_alone_2
                        - (total_1 + total_2 - ASA_combined)) / 2
                     = (ASA_combined - ASA_alone_1 - ASA_alone_2) / 2

    Wait — the correct formula is:
      interface_area = (BSA_1 + BSA_2 - BSA_combined) / 2

    But BSA_combined is NOT simply (total_1+total_2) - ASA_combined
    because ASA_combined already accounts for the combined surface.

    The standard PISA formula:
      interface_area = (ASA_1_alone + ASA_2_alone - ASA_combined) / 2

    This gives the area buried at the interface.
    """
    total_asa_alone_1 = sum(asa_alone_1.values())
    total_asa_alone_2 = sum(asa_alone_2.values())

    # Combined ASA: sum of accessible areas in the combined structure
    # For atoms in molecule 1 and 2, use their ASA from the combined calculation
    total_asa_combined = sum(asa_combined.values())

    # Interface area = (ASA_alone_1 + ASA_alone_2 - ASA_combined) / 2
    # This is the standard PISA formula
    interface_area = (total_asa_alone_1 + total_asa_alone_2 - total_asa_combined) / 2.0

    return max(interface_area, 0.0)


def calculate_bsa_combined(
    total_surface_combined: float,
    asa_combined: dict,
) -> float:
    """Calculate the total buried surface area of a combined structure."""
    total_asa = sum(asa_combined.values())
    return total_surface_combined - total_asa


def calculate_asa_combined(
    atoms,
    asa_combined: dict,
):
    """Get the combined ASA per atom from the combined calculation."""
    return asa_combined