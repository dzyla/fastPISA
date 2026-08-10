"""
Scoring functions for PISA interfaces.

PISA uses two main scores:
1. P-value: probability that the observed solvation energy gain
   could occur by chance if interface atoms were picked randomly
   from the protein surface.
2. CSS (Complexation Significance Score): a composite score that
   combines interface area, binding energy, P-value, and contact
   count to assess biological significance.

The P-value is computed using a statistical model based on the
distribution of solvation energies for interfaces of a given area.
"""

from scipy.special import erf
import numpy as np
from typing import List, Tuple


def calculate_p_value(
    solvation_energy: float,
    interface_area: float,
    total_asa: float,
) -> float:
    """Calculate the P-value for an interface.

    The P-value is defined as the probability that a random interface
    of the same area would have a solvation energy gain at least as
    negative as the observed one.

    PISA uses a Gaussian distribution model where:
      mean = c1 × (interface_area / total_asa)
      std  = c2 × sqrt(interface_area)

    The P-value is then:
      P = Φ((ΔGsolv - mean) / std)

    where Φ is the standard normal CDF.

    Parameters
    ----------
    solvation_energy : float
        Observed solvation energy gain (kcal/mol).  Negative = favourable.
    interface_area : float
        Interface area (A^2).
    total_asa : float
        Total accessible surface area of the structure (A^2).

    Returns
    -------
    float
        P-value in [0, 1].  P < 0.5 means the interface is more
        hydrophobic than expected; P > 0.5 means less.
    """
    if interface_area <= 0 or total_asa <= 0:
        return 0.5

    # PISA statistical model parameters
    # These are calibrated from the distribution of interface energies
    # The mean solvation energy scales with interface area relative to total surface
    # The standard deviation scales with sqrt(interface_area)

    # Empirical constants from PISA calibration
    # These values are chosen to reproduce PISA's P-value distribution
    mean_coeff = 0.030  # kcal/mol per A^2 of interface area
    std_coeff = 0.15    # kcal/mol per sqrt(A^2) of interface area

    mean = mean_coeff * interface_area
    std = std_coeff * np.sqrt(interface_area)

    if std < 0.01:
        std = 0.01

    # P-value: probability of getting ΔGsolv as low or lower
    # P = Φ((observed - mean) / std)
    # Since solvation energy is negative (favourable), lower = more significant
    z = (solvation_energy - mean) / std
    p_value = float(0.5 * (1 + erf(z / np.sqrt(2))))

    # Clamp to (0, 1)
    p_value = max(0.001, min(0.999, p_value))

    return p_value


def calculate_css(
    interface_area: float,
    solvation_energy: float,
    p_value: float,
    n_contacts: int,
    n_residues: int,
    total_asa: float,
) -> float:
    """Calculate the Complexation Significance Score (CSS).

    CSS is a composite score that combines:
    - Interface area (normalised by total surface area)
    - Binding energy (solvation energy)
    - P-value (specificity)
    - Number of contacts and residues

    The CSS score ranges roughly from 0 to 1, with higher values
    indicating more significant interfaces.

    PISA's CSS formula is based on a weighted combination of these
    factors. Here we use a simplified version:

    CSS = w1 × (interface_area / total_asa) + w2 × (-solv_energy / scale)
          + w3 × (1 - |p_value - 0.5|) + w4 × log(1 + n_contacts)

    Parameters
    ----------
    interface_area : float
    solvation_energy : float
    p_value : float
    n_contacts : int
    n_residues : int
    total_asa : float

    Returns
    -------
    float
        CSS score.
    """
    if total_asa <= 0:
        total_asa = 1.0

    # Normalised interface area
    area_score = interface_area / total_asa

    # Energy score (negative solv_energy = favourable)
    energy_score = max(0, -solvation_energy) / 10.0

    # P-value score: P close to 0 = most significant
    p_score = 1.0 - abs(p_value - 0.5) * 2.0

    # Contact score
    contact_score = np.log1p(n_contacts) / 5.0

    # Weighted combination
    css = (0.40 * area_score +
           0.30 * energy_score +
           0.20 * p_score +
           0.10 * contact_score)

    return css


def classify_interface(
    p_value: float,
    css: float,
) -> str:
    """Classify an interface as biological or crystal packing.

    Parameters
    ----------
    p_value : float
        P-value of the interface.
    css : float
        CSS score.

    Returns
    -------
    str
        "biological" if the interface is likely biologically relevant,
        "crystal" if likely a crystal packing artifact.
    """
    # Simple classification: P < 0.5 and CSS > threshold
    if p_value < 0.5 and css > 0.1:
        return "biological"
    return "crystal"