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


def calculate_p_value_pisa(
    solvation_energy: float,
    buried_areas,
    surface_sigmas,
    surface_areas,
) -> float:
    """PISA's hydrophobicity P-value, computed from its actual definition.

    P is the probability that a randomly drawn surface patch burying the
    same per-atom areas would yield a solvation gain at least as negative as
    the observed one (Krissinel & Henrick 2007). Under the random model each
    buried patch ``b_j`` lands on a random surface atom whose ASP sigma is
    drawn from the parent molecules' surface distribution (weighted by
    exposed area), so

        mean = sum(b_j) * E[sigma],  var = sum(b_j^2) * Var[sigma],
        P = Phi((dG_obs - mean) / sqrt(var)).

    Low P (< 0.5): the interface is more hydrophobic than a random surface
    patch -- interaction-specific. High P: unremarkable, packing-like.

    Parameters
    ----------
    solvation_energy : float
        Observed interface solvation gain (kcal/mol).
    buried_areas : array-like
        Per-atom buried areas of the interface (A^2).
    surface_sigmas : array-like
        ASP sigma of each SURFACE atom of the two parent molecules.
    surface_areas : array-like
        Isolated-monomer ASA of those surface atoms (weights).
    """
    b = np.asarray(buried_areas, dtype=float)
    s = np.asarray(surface_sigmas, dtype=float)
    w = np.asarray(surface_areas, dtype=float)
    if b.size == 0 or s.size == 0 or w.sum() <= 0:
        return 0.5
    mean_sigma = float(np.average(s, weights=w))
    var_sigma = float(np.average((s - mean_sigma) ** 2, weights=w))
    mean = b.sum() * mean_sigma
    var = float((b ** 2).sum()) * var_sigma
    if var <= 1e-12:
        return 0.5
    z = (solvation_energy - mean) / np.sqrt(var)
    # The independent-atom variance underestimates the true patch variance
    # (buried areas are correlated within residues), which over-spreads z.
    # A single deflation constant fitted against EBI PISA p-values over the
    # polymer-polymer interfaces of the 36-entry benchmark corrects for it
    # (median |p error| 0.11, Spearman 0.73 on those interfaces). PISA's
    # p-values for small-ligand/ion interfaces follow different statistics
    # and are NOT reproduced by this model -- treat ligand-interface
    # p-values as indicative only.
    z *= P_VALUE_Z_SCALE
    p = float(0.5 * (1 + erf(z / np.sqrt(2))))
    return min(max(p, 0.0), 1.0)


# Effective-z deflation for correlated buried patches (see above).
P_VALUE_Z_SCALE = 0.219


def calculate_css_pisa(solvation_energy: float, interface_area: float) -> float:
    """CSS surrogate calibrated against original PISA's CSS.

    PISA's true CSS comes from its crystal-wide assembly analysis (which
    needs symmetry-mate enumeration -- out of fastPISA's scope). This
    logistic surrogate was fitted to EBI PISA CSS values over the 36-entry
    reference benchmark (262 interfaces):

        css = sigmoid(-6.9088 - 0.1699 * dG_solv + 0.8485 * ln(1 + area))

    Leave-one-PDB-out performance: Spearman 0.68 vs PISA's CSS, mean
    absolute error 0.22, and 79% agreement on the css >= 0.5
    biological-vs-packing call. Treat it as a well-behaved [0, 1]
    significance score, not an exact reproduction.
    """
    x = (-6.9088
         - 0.1699 * solvation_energy
         + 0.8485 * np.log1p(max(interface_area, 0.0)))
    return float(1.0 / (1.0 + np.exp(-x)))


def calculate_p_value(
    solvation_energy: float,
    interface_area: float,
    total_asa: float,
) -> float:
    """LEGACY P-value model (superseded by :func:`calculate_p_value_pisa`).

    The pipelines use :func:`calculate_p_value_pisa`, which implements
    PISA's actual random-surface-patch definition and is validated against
    the EBI PISA reference. This simpler area-scaled model is kept only for
    backward compatibility with external callers.

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

    .. warning::
        This is an UNCALIBRATED approximation. The original PISA fitted its
        P-value model to a database of >10,000 known interfaces (Krissinel &
        Henrick 2007); the constants here are ad-hoc defaults so the absolute
        P-values do NOT match PDBe PISA output. Use them for relative ranking
        of interfaces within one structure, not for cross-tool comparison, and
        treat assembly predictions based on them as preliminary.
    """
    if interface_area <= 0 or total_asa <= 0:
        return 0.5

    # PISA statistical model parameters
    # These are calibrated from the distribution of interface energies
    # The mean solvation energy scales with interface area relative to total surface
    # The standard deviation scales with sqrt(interface_area)

    # Empirical constants. BOTH terms must scale the same way in interface area or the
    # z-score diverges and the P-value saturates at its clamp for every real interface.
    #
    # The previous form used mean = 0.030 * area with std = 0.15 * sqrt(area): the mean
    # grows LINEARLY while the spread grows as sqrt(area), so z ~ -0.2 * sqrt(area) and
    # runs away with size. Measured on 63 interfaces from 7 antibody-antigen references,
    # EVERY p-value came back as exactly 0.001 -- the clamp floor -- including a 23.6 A^2
    # contact (z = -4.89) and a 3692 A^2 interface (z = -31.5). A field with zero variance
    # carries no information, and because CSS takes P as one of its four terms, CSS's
    # specificity contribution was silently inert.
    #
    # Both moments are now per unit area, so z is scale-free in interface size and the
    # P-value again discriminates hydrophobic from polar interfaces of any size. The
    # constants remain UNCALIBRATED against Krissinel's fitted model (see the warning
    # above); the fix restores variance, it does not claim PDBe-matching absolute values.
    mean_coeff = -0.025  # kcal/mol per A^2: a typical interface buries favourably
    std_coeff = 0.020    # kcal/mol per A^2 of spread about that expectation

    mean = mean_coeff * interface_area
    std = std_coeff * interface_area

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
    """LEGACY CSS composite (superseded by :func:`calculate_css_pisa`).

    The pipelines use :func:`calculate_css_pisa`, calibrated against original
    PISA's CSS. Kept for backward compatibility with external callers.

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

    .. warning::
        Simplified hand-weighted composite, not the Krissinel statistical CSS.
        Values differ from PDBe PISA and should be treated as a relative
        significance heuristic, not an absolute calibrated score.
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
    interface_area: float = 0.0,
) -> str:
    """Classify an interface as biological or crystal packing.

    Uses the PISA-style multi-criteria thresholds instead of P-value alone:
    an interface is called ``"biological"`` only when it is large
    (>= 800 A^2), scores highly on CSS (>= 0.5) and is more hydrophobic than
    expected (p_value < 0.5). Small / weak interfaces are ``"crystal"``.

    Parameters
    ----------
    p_value : float
        P-value of the interface.
    css : float
        CSS score.
    interface_area : float
        Interface area in A^2 (default 0 -> treated as non-biological unless
        thresholds are otherwise met).

    Returns
    -------
    str
        ``"biological"`` if the interface is likely biologically relevant,
        ``"crystal"`` otherwise.
    """
    if (p_value < 0.5 and css >= 0.5 and interface_area >= 800.0):
        return "biological"
    return "crystal"