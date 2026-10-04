"""
Optional FreeSASA (C-accelerated) backend for solvent-accessible surface area.

FreeSASA (github.com/mittinatten/freesasa) is a C library implementing both
the Shrake-Rupley and the Lee-Richards algorithms. When installed, this module
provides a drop-in accelerator for
:func:`fastpisa.surface.shrake_rupley.calculate_asa_python`.

WHICH ALGORITHM RUNS, AND WHY
-----------------------------
fastPISA's reference surface engine is the pure-Python Shrake-Rupley
implementation in :mod:`fastpisa.surface.shrake_rupley`; it defines the
``point_density`` quadrature and is the one the solvation parameters are
calibrated against.

This backend is pinned to **Lee-Richards with a fixed slice count**, set
EXPLICITLY rather than inherited:

* FreeSASA's library default has been Lee-Richards (20 slices), not
  Shrake-Rupley, so code that only called ``setNPoints`` was silently
  configuring a parameter the active algorithm ignores. Never rely on the
  default -- it is not ours to depend on.
* FreeSASA 2.2.1's Shrake-Rupley kernel **segfaults on spatially sparse
  inputs** (reproduced with two atoms 20 A apart, at every point count, with
  both list and ndarray input). fastPISA's per-molecule and per-pair ASA
  calls generate exactly such subsets, and a segfault cannot be caught --
  it would abort a user's analysis. ``tests/test_sasa_backends.py::
  test_freesasa_survives_spatially_sparse_subsets`` crashes the suite if
  anyone switches this back.

The two quadratures are not identical, so they are held to a measured
tolerance by ``test_python_and_freesasa_agree_on_a_real_structure``: on 1ktz
they agree to 0.01% of total ASA, with a median per-atom difference of
0.1 A^2. See ``docs/surface_backends.md`` for the full comparison and for
the benchmark re-validation of the fitted sigmas under each backend.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from fastpisa.surface.shrake_rupley import PYTHON_ALGORITHM, surface_radius

try:  # fmt: off
    import freesasa
    _HAVE_FREESASA = True
except Exception:  # pragma: no cover - fallback path
    _HAVE_FREESASA = False

#: Algorithm this backend is pinned to (see the module docstring). Set
#: explicitly on every call; never inherited from the library default.
FREESASA_ALGORITHM = "LeeRichards"

#: Lee-Richards slice count. 20 is FreeSASA's own default and is already
#: converged to ~0.03% of total ASA against a 5000-point Shrake-Rupley
#: reference, so it is pinned rather than exposed as a tuning knob.
LEE_RICHARDS_SLICES = 20


def available() -> bool:
    """Whether the C-accelerated FreeSASA backend is usable."""
    return _HAVE_FREESASA


def surface_backend_info(point_density: int = 480) -> dict:
    """Describe the active ASA backend without changing its parameters.

    ``algorithm`` and ``quadrature`` report what will ACTUALLY run, so a
    methods section can quote them. ``point_density`` is only echoed back for
    the Python backend, which is the one that uses it.
    """
    # Ask the dispatcher, not "is freesasa importable": the engine can be
    # pinned with FASTPISA_SASA_BACKEND, and provenance that disagrees with
    # what ran is worse than no provenance.
    from fastpisa.surface.shrake_rupley import active_backend

    if active_backend() == "python":
        return {
            "backend": "python",
            "algorithm": PYTHON_ALGORITHM,
            "quadrature": f"{point_density} sphere points",
            "version": None,
        }
    try:
        from importlib.metadata import version
        backend_version = version("freesasa")
    except Exception:  # pragma: no cover - package metadata is optional
        backend_version = None
    return {
        "backend": "FreeSASA",
        "algorithm": FREESASA_ALGORITHM,
        "quadrature": f"{LEE_RICHARDS_SLICES} slices",
        "version": backend_version,
    }


def lee_richards_parameters(n_slices: int = LEE_RICHARDS_SLICES,
                            probe_radius: float = 1.4):
    """FreeSASA parameters, explicitly pinned (see the module docstring)."""
    params = freesasa.Parameters()
    params.setProbeRadius(probe_radius)
    params.setAlgorithm(FREESASA_ALGORITHM)
    params.setNSlices(n_slices)
    return params


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

    # Fast path: the caller already holds the whole structure's coordinates
    # and surface radii (run_core does), so the subset is a gather rather
    # than n Python attribute reads plus n surface_radius() calls. This was
    # ~50% of the ASA hot path on multi-chain inputs.
    gathered = False
    if (atom_indices is not None and combined_coords is not None
            and combined_radii is not None and not atom_radii
            and len(atom_indices) == n):
        idx = np.asarray(atom_indices, dtype=np.intp)
        coords = np.ascontiguousarray(
            np.asarray(combined_coords, dtype=float)[idx]).reshape(-1)
        radii = np.asarray(combined_radii, dtype=float)[idx]
        gathered = True
    if not gathered:
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

    # A single atom has no occluders at all, so its ASA is the closed-form
    # probe sphere. Short-circuiting it is exact and faster, and a lone metal
    # ion ([ZN]A:301) is a routine molecule here.
    if n == 1:
        global_i = atom_indices[0] if atom_indices is not None else 0
        return {global_i: 4.0 * math.pi * (radii[0] + probe_radius) ** 2}

    result = freesasa.calcCoord(
        coords, radii, lee_richards_parameters(LEE_RICHARDS_SLICES, probe_radius))

    out = {}
    for local_i in range(n):
        global_i = atom_indices[local_i] if atom_indices is not None else local_i
        out[global_i] = float(result.atomArea(local_i))
    return out


# NOTE: this module deliberately exposes NO ``calculate_asa`` dispatcher.
# Dispatch lives in exactly one place --
# :func:`fastpisa.surface.shrake_rupley.calculate_asa`, which honours
# ``FASTPISA_SASA_BACKEND``. A second dispatcher here used to shadow that
# switch and could route a run to a different engine than the one reported
# by :func:`surface_backend_info`.
