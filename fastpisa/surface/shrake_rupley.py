"""
Solvent-accessible surface area (SASA) calculation using the
Shrake-Rupley algorithm, vectorised with numpy and optimised with
KD-tree neighbour lookup.

The algorithm places a spherical probe of radius r_probe (default 1.4 A)
on each atom's van der Waals surface and computes the fraction of the
probe sphere that is not buried by neighbouring atoms.

Key optimisations:
  1. KD-tree for spatial neighbour lookup (avoids O(n^2) full distance matrices)
  2. Combined-structure ASA cached and reused across interface computations
  3. Batch processing of probe sphere points
  4. Vectorised numpy operations throughout
"""

import os

import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, List, Optional

#: Name of the reference surface algorithm. The pure-Python implementation in
#: this module IS that reference: it defines the ``point_density`` quadrature
#: and is the engine the solvation parameters are calibrated against.
PYTHON_ALGORITHM = "Shrake-Rupley"

#: Environment variable pinning the ASA engine, for reproducibility and for
#: backend comparisons: ``auto`` (default), ``python`` or ``freesasa``.
BACKEND_ENV_VAR = "FASTPISA_SASA_BACKEND"
_BACKENDS = ("auto", "python", "freesasa")


def active_backend() -> str:
    """Which ASA engine will run: ``"python"`` or ``"freesasa"``.

    ``auto`` (the default) prefers FreeSASA when it is importable. Setting
    ``FASTPISA_SASA_BACKEND=python`` forces the reference engine even when
    FreeSASA is installed, which is what the backend comparison and the
    calibration re-validation use. ``freesasa`` demands the C backend and
    fails loudly rather than silently degrading to a different quadrature.
    """
    from fastpisa.surface.freesasa_backend import available

    choice = os.environ.get(BACKEND_ENV_VAR, "auto").strip().lower()
    if choice not in _BACKENDS:
        raise ValueError(
            f"{BACKEND_ENV_VAR}={choice!r} is not one of {_BACKENDS}")
    if choice == "python":
        return "python"
    if choice == "freesasa":
        if not available():
            raise RuntimeError(
                f"{BACKEND_ENV_VAR}=freesasa but freesasa is not installed")
        return "freesasa"
    return "freesasa" if available() else "python"


# van der Waals radii (A) — PISA convention
VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
    "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98, "FE": 1.80, "ZN": 1.80,
    "MG": 1.70, "CA": 1.70, "CU": 1.80, "NA": 1.80, "PB": 1.80,
    "CO": 1.80, "NI": 1.80, "MO": 1.80, "W": 1.80,
}


# ---------------------------------------------------------------------------
# Surface (ASA) radii: the NACCESS / Chothia (1976) set original PISA uses.
#
# Recovered empirically (2026-09-01) by fitting per-element radii to PISA's
# per-residue isolated-monomer ASA over ~6000 interface residues: the fit
# lands within 0.01-0.03 A of the published NACCESS values, and switching to
# them drops the per-residue ASA error against PISA from 3.3% (median,
# +1.3% bias; Bondi-like table above) to 0.9% with no bias. NACCESS's
# convention is that trigonal (sp2) carbons -- carbonyl / carboxylate /
# guanidinium / aromatic ring -- take 1.76 A and tetrahedral carbons 1.87 A.
#
# These are for SURFACE calculations only. The contact-classification radii
# in VDW_RADII / get_vdw_radius (COCOMAPS-validated) are a different
# convention and must stay as they are.
# ---------------------------------------------------------------------------
SURFACE_RADII = {
    "C": 1.87, "N": 1.65, "O": 1.40, "S": 1.85, "P": 1.90, "SE": 1.90,
    # Ions, recovered EXACTLY from PISA: a lone ion's isolated ASA is
    # 4*pi*(r + 1.4)^2, and PISA's per-residue ASA for each monatomic ion is
    # constant to 0.0% across the benchmark, so r = sqrt(ASA/4pi) - 1.4 to
    # two decimals (2026-09-01, 674 entries). Note Ca at 1.20 A (PISA's own
    # value, adopted as-is: "follow the same rules") and K at 2.75 A.
    "MG": 1.39, "ZN": 1.39, "CA": 1.20, "NA": 1.75, "K": 2.75, "CL": 1.75,
    "BR": 1.85, "I": 1.98, "MN": 1.73, "FE": 1.90, "CO": 1.90, "NI": 1.63,
    "CU": 1.75, "CD": 1.58, "HG": 1.90, "GD": 2.75,
}
_SP2_CARBON = frozenset({"C"})  # backbone carbonyl C (every residue)
_SP2_SIDECHAIN = frozenset({
    ("ASP", "CG"), ("GLU", "CD"), ("ASN", "CG"), ("GLN", "CD"), ("ARG", "CZ"),
    ("PHE", "CG"), ("PHE", "CD1"), ("PHE", "CD2"), ("PHE", "CE1"), ("PHE", "CE2"), ("PHE", "CZ"),
    ("TYR", "CG"), ("TYR", "CD1"), ("TYR", "CD2"), ("TYR", "CE1"), ("TYR", "CE2"), ("TYR", "CZ"),
    ("TRP", "CG"), ("TRP", "CD1"), ("TRP", "CD2"), ("TRP", "CE2"), ("TRP", "CE3"),
    ("TRP", "CZ2"), ("TRP", "CZ3"), ("TRP", "CH2"),
    ("HIS", "CG"), ("HIS", "CD2"), ("HIS", "CE1"),
})
_NUCLEOTIDES = frozenset({"A", "G", "C", "U", "DA", "DG", "DC", "DT", "DU",
                          "RA", "RG", "RC", "RT", "RU"})
SURFACE_RADIUS_SP2_C = 1.76


def surface_radius(atom) -> float:
    """Radius used for solvent-accessible-surface calculations (NACCESS set).

    Falls back to :func:`get_vdw_radius` for elements outside the set
    (metals, halogens), so hetero groups keep the radii they had.
    """
    el = atom.element.strip().upper()
    if el == "C":
        name = atom.atom_name.strip().upper()
        res = atom.res_name.strip().upper()
        if (name in _SP2_CARBON or (res, name) in _SP2_SIDECHAIN
                or (res in _NUCLEOTIDES and not name.endswith("'"))):
            return SURFACE_RADIUS_SP2_C
        return SURFACE_RADII["C"]
    r = SURFACE_RADII.get(el)
    return r if r is not None else get_vdw_radius(el)


def get_vdw_radius(element: str) -> float:
    """Van der Waals radius for CONTACT classification (COCOMAPS convention).

    Not the surface radius -- see :func:`surface_radius`."""
    el = element.upper().strip()
    if el in VDW_RADII:
        return VDW_RADII[el]
    for key, val in VDW_RADII.items():
        if el == key or (el and el[0] == key[0]):
            return val
    return 1.70  # default carbon-like


def _prepare_atom_data(atoms) -> tuple:
    """Extract coordinates, radii, and indices from a list of atoms."""
    n = len(atoms)
    coords = np.zeros((n, 3))
    radii = np.zeros(n)
    for i, atom in enumerate(atoms):
        coords[i] = [atom.x, atom.y, atom.z]
        radii[i] = surface_radius(atom)
    return coords, radii


def _generate_fibonacci_sphere(n_points: int) -> np.ndarray:
    """Generate n_points on a unit sphere using Fibonacci spiral."""
    phi = np.arccos(1 - 2 * np.arange(n_points) / (n_points - 0.5))
    theta = np.pi * (1 + 5 ** 0.5) * np.arange(n_points)
    return np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ])


def calculate_asa(
    atoms,
    probe_radius: float = 1.4,
    point_density: int = 480,
    atom_radii: Optional[Dict] = None,
    atom_indices: Optional[List[int]] = None,
    combined_coords: Optional[np.ndarray] = None,
    combined_radii: Optional[np.ndarray] = None,
    kd_tree: Optional[object] = None,
    neighbor_cutoff: Optional[float] = None,
) -> Dict[int, float]:
    """Calculate solvent-accessible surface area per atom.

    Dispatches to the C-accelerated FreeSASA backend when installed, otherwise
    uses the pure-Python Shrake-Rupley implementation. The choice can be
    pinned with the ``FASTPISA_SASA_BACKEND`` environment variable
    (``auto`` | ``python`` | ``freesasa``) -- see :func:`active_backend`.
    Callers that want the reference engine unconditionally can call
    :func:`calculate_asa_python` directly.

    See :func:`calculate_asa_python` for the Shrake-Rupley algorithm.
    """
    from fastpisa.surface.freesasa_backend import calculate_asa_freesasa

    if active_backend() == "freesasa":
        return calculate_asa_freesasa(
            atoms=atoms,
            probe_radius=probe_radius,
            point_density=point_density,
            atom_radii=atom_radii,
            atom_indices=atom_indices,
            combined_coords=combined_coords,
            combined_radii=combined_radii,
            kd_tree=kd_tree,
            neighbor_cutoff=neighbor_cutoff,
        )
    return calculate_asa_python(
        atoms=atoms,
        probe_radius=probe_radius,
        point_density=point_density,
        atom_radii=atom_radii,
        atom_indices=atom_indices,
        combined_coords=combined_coords,
        combined_radii=combined_radii,
        kd_tree=kd_tree,
        neighbor_cutoff=neighbor_cutoff,
    )


def calculate_asa_python(
    atoms,
    probe_radius: float = 1.4,
    point_density: int = 480,
    atom_radii: Optional[Dict] = None,
    atom_indices: Optional[List[int]] = None,
    combined_coords: Optional[np.ndarray] = None,
    combined_radii: Optional[np.ndarray] = None,
    kd_tree: Optional[object] = None,
    neighbor_cutoff: Optional[float] = None,
) -> Dict[int, float]:
    """Pure-Python Shrake-Rupley per-atom ASA (kept for reference/testing).

    Uses the Shrake-Rupley algorithm: for each atom, generate points on
    a sphere of radius (r_vdw + r_probe), test whether each point is
    accessible (not inside any other atom's van der Waals sphere), and
    weight by the accessible fraction.

    Parameters
    ----------
    atoms : list of Atom
        Atoms to process.
    probe_radius : float
        Probe sphere radius (default 1.4 A).
    point_density : int
        Number of points on the sphere (default 480).
    atom_radii : dict, optional
        Custom vdw radii mapping (atom_idx -> radius).
    atom_indices : list, optional
        Indices of atoms into the coords array. If None, uses all atoms.
    combined_coords : np.ndarray, optional
        Coordinates of all atoms in the combined structure (for neighbor lookup).
    combined_radii : np.ndarray, optional
        Vdw radii of all atoms in the combined structure.
    kd_tree : scipy.spatial.cKDTree, optional
        Pre-computed KD-tree of the combined structure.
    neighbor_cutoff : float, optional
        Maximum distance for the neighbour search. This is a PERFORMANCE hint
        only: a value smaller than the geometric requirement
        ``2 * r_max + 2 * probe_radius`` would silently drop real occluders,
        so it is clamped up to that bound. If None, that bound is used.

    Returns
    -------
    dict
        Mapping atom index -> accessible surface area (A^2).
    """
    n = len(atoms)
    if n == 0:
        return {}

    # Prepare atom data
    coords, radii = _prepare_atom_data(atoms)

    if atom_radii:
        for i, atom in enumerate(atoms):
            if i in atom_radii:
                radii[i] = atom_radii[i]

    total_r = radii + probe_radius
    full_sa = 4.0 * np.pi * total_r ** 2

    # Generate probe sphere points (unit sphere, will be scaled)
    unit_pts = _generate_fibonacci_sphere(point_density)

    accessible_area = np.zeros(n)

    # Neighbour cutoff. Atom j can hide part of atom i's probe sphere only
    # while the two probe spheres overlap, i.e. out to r_i + r_j + 2*probe.
    # The largest radius present bounds that for every pair, so anything
    # shorter (a caller's stale hint included) is clamped UP: a short cutoff
    # drops real occluders and inflates ASA, which is a physics error, not a
    # speed/accuracy trade.
    r_max = float(radii.max())
    if combined_radii is not None and len(combined_radii):
        r_max = max(r_max, float(np.max(combined_radii)))
    required_cutoff = 2.0 * r_max + 2.0 * probe_radius
    if neighbor_cutoff is None or neighbor_cutoff < required_cutoff:
        neighbor_cutoff = required_cutoff

    # Build KD-tree for neighbor lookup if not provided
    if kd_tree is None and combined_coords is not None:
        kd_tree = cKDTree(combined_coords)

    # For each atom, find neighbors and compute accessibility
    # Precompute the global index of each local atom (for self-exclusion)
    if atom_indices is not None:
        global_idx = atom_indices
    else:
        global_idx = np.arange(n)

    # Build a lookup set of global indices belonging to the *current* selection
    # so we do NOT branch on coordinates (which is extremely slow).
    in_selection = np.zeros(len(combined_coords) if combined_coords is not None else n, dtype=bool)
    in_selection[global_idx] = True

    for i in range(n):
        gi = global_idx[i]
        # Find candidate neighbors within cutoff (global indices)
        if kd_tree is not None and combined_coords is not None:
            neighbor_idx = kd_tree.query_ball_point(coords[i], neighbor_cutoff)
            # Exclude self and any atom NOT in the current selection (so when
            # analysing a single molecule we don't count neighbors from others)
            neighbor_idx = [j for j in neighbor_idx if j != gi and in_selection[j]]
            if not neighbor_idx:
                accessible = np.ones(point_density, dtype=bool)
                frac = 1.0
            else:
                neighbor_idx = np.asarray(neighbor_idx, dtype=np.intp)
                neighbor_coords = combined_coords[neighbor_idx]
                neighbor_rads = (combined_radii[neighbor_idx]
                                 if combined_radii is not None
                                 else radii[neighbor_idx])

                # Nearest neighbour first, then drop points as they are
                # buried and stop once none survive. The full
                # (points x neighbours) distance matrix costs the same for a
                # deeply buried atom as for an exposed one, while in practice
                # the closest few neighbours account for almost all occlusion:
                # ~2.2x faster on a protein, and bit-identical (asserted by
                # test_python_asa_is_independent_of_atom_order and the
                # closed-form geometry tests).
                offsets = neighbor_coords - coords[i]
                order = np.argsort(np.einsum("ij,ij->i", offsets, offsets))
                neighbor_coords = neighbor_coords[order]
                neighbor_rads = neighbor_rads[order]

                # Points on this atom's probe sphere
                pts = coords[i] + total_r[i] * unit_pts  # (n_points, 3)
                live = np.ones(point_density, dtype=bool)
                survivors = pts
                for centre, radius in zip(neighbor_coords, neighbor_rads):
                    # A test point is the CENTRE of a probe sphere, so it is
                    # inaccessible when it lies within (r_j + probe) of
                    # neighbour j -- not within r_j. Dropping the probe radius
                    # here leaves the probe-centre sphere occluded by bare
                    # van-der-Waals spheres only and overstates ASA ~6x.
                    delta = survivors - centre
                    hit = np.einsum("ij,ij->i", delta, delta) < (
                        radius + probe_radius) ** 2
                    if not hit.any():
                        continue
                    live[np.flatnonzero(live)[hit]] = False
                    if not live.any():
                        break
                    survivors = pts[live]
                frac = live.sum() / point_density
        else:
            # Fallback: O(n^2) against all atoms
            pts = coords[i] + total_r[i] * unit_pts
            dist_sq = np.sum((pts[:, None, :] - coords[None, :, :]) ** 2, axis=2)
            buried = dist_sq < ((radii[None, :] + probe_radius) ** 2)
            buried[:, i] = False  # exclude self (column i = the atom itself)
            accessible = ~buried.any(axis=1)
            frac = accessible.sum() / point_density

        accessible_area[i] = frac * full_sa[i]

    # Map back to original atom indices
    result = {}
    if atom_indices is not None:
        for local_i, global_i in enumerate(atom_indices):
            result[global_i] = accessible_area[local_i]
    else:
        for i in range(n):
            result[i] = accessible_area[i]

    return result


def calculate_bsa(
    total_asa: Dict[int, float],
    atoms,
    atom_radii: Optional[Dict] = None,
) -> Dict[int, float]:
    """Deprecated: per-atom 'buried' surface using the old 4*pi*r_vdw^2 - ASA convention.

    .. deprecated:: 0.2.0
        This convention treats every atom's full sphere area as 'surface' and
        calls the difference from probe-ASA 'buried', which vastly overstates
        BSA (assembly BSA could exceed assembly ASA). Use
        ``fastpisa.surface.per_residue.compute_buried_surface`` instead, which
        computes the physically meaningful buried area = isolated-molecule ASA
        - combined ASA (the convention both analysis pipelines use).

    Retained only for backward compatibility; emits a :class:`DeprecationWarning`.
    """
    import warnings

    warnings.warn(
        "calculate_bsa uses the obsolete 4*pi*r_vdw^2 - ASA BSA convention and "
        "overstates buried surface. Use "
        "fastpisa.surface.per_residue.compute_buried_surface instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    bsa = {}
    for i, atom in enumerate(atoms):
        r = get_vdw_radius(atom.element)
        if atom_radii and i in atom_radii:
            r = atom_radii[i]
        total = 4.0 * np.pi * r ** 2
        acc = total_asa.get(i, 0.0)
        bsa[i] = total - acc
    return bsa