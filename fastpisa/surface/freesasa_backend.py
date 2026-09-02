"""
Optional FreeSASA (C-accelerated) backend for solvent-accessible surface area.

FreeSASA (github.com/mittinatten/freesasa) is a C library implementing the
Shrake-Rupley and Lee-Richards algorithms. When installed, this module provides
a drop-in replacement for :func:`fastpisa.surface.shrake_rupley.calculate_asa`
that runs in C and is ~100x faster on the SASA hot path.

If freesasa is not available, imports/uses fall back to the pure-Python
implementation automatically (see :func:`calculate_asa`).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from fastpisa.surface.shrake_rupley import surface_radius

try:  # fmt: off
    import freesasa
    _HAVE_FREESASA = True
except Exception:  # pragma: no cover - fallback path
    _HAVE_FREESASA = False


def available() -> bool:
    """Whether the C-accelerated FreeSASA backend is usable."""
    return _HAVE_FREESASA


def calculate_asa_freesasa(
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
    """Calculate per-atom ASA using the FreeSASA C library.

    The signature mirrors :func:`fastpisa.surface.shrake_rupley.calculate_asa`
    so it can be used as a drop-in replacement.

    ``atom_indices`` selects a subset of ``atoms`` (global->local mapping).
    Because FreeSASA computes SASA over exactly the atoms it is given, passing
    ``atoms`` = the target subset yields the correct "isolated" ASA for that
    subset (no cross-molecule contributions), which is exactly the semantics
    needed for combined / molecule / interface ASA.
    """
    # ``atoms`` is already the target subset (matching _prepare_atom_data in the
    # pure-Python implementation). ``atom_indices`` maps local position -> global
    # index for the OUTPUT dict, it does NOT index into ``atoms``.
    target = atoms

    n = len(target)
    if n == 0:
        return {}

    coords = np.zeros(3 * n, dtype=float)
    radii = np.zeros(n, dtype=float)
    for i, a in enumerate(target):
        coords[3 * i] = a.x
        coords[3 * i + 1] = a.y
        coords[3 * i + 2] = a.z
        radii[i] = surface_radius(a)
        if atom_radii:
            gi = atom_indices[i] if atom_indices is not None else i
            if gi in atom_radii:
                radii[i] = atom_radii[gi]

    params = freesasa.Parameters()
    params.setProbeRadius(probe_radius)
    params.setNPoints(point_density)
    # NOTE: do NOT call setAlgorithm here. Calling setAlgorithm("ShrakeRupley")
    # triggers a segfault in the freesasa C library for single-atom inputs
    # (e.g. a lone metal ion). Shrake-Rupley is already the default algorithm.

    result = freesasa.calcCoord(coords, radii, params)

    out = {}
    for local_i in range(n):
        global_i = atom_indices[local_i] if atom_indices is not None else local_i
        out[global_i] = float(result.atomArea(local_i))
    return out


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
    """Drop-in ASA calculator.

    Uses the FreeSASA C backend when available, otherwise falls back to the
    pure-Python Shrake-Rupley implementation.
    """
    if _HAVE_FREESASA:
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
    # Fallback
    from fastpisa.surface.shrake_rupley import calculate_asa as _py_calculate_asa
    return _py_calculate_asa(
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