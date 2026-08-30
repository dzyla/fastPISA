"""
PISA-mode analysis pipeline for fastPISA.

Since the unified-core refactor, all physics lives in :mod:`fastpisa.core`,
which every mode (``pisa``, ``cocomaps``, ``combined``) shares. This module
keeps the historical PISA-mode entry point :func:`analyze_structure`.

The full analysis:
1. Parse PDB/mmCIF input
2. Calculate solvent-accessible surface areas (ASA) for isolated molecules
3. Calculate ASA for the combined structure
4. Detect interfaces between all molecule pairs
5. Find atom-atom contacts (H-bonds, salt bridges, disulfides, other bonds)
6. Calculate solvation/binding energy, entropy, P-value, CSS per interface
7. Generate assembly.json and interfaces.json output (PDBe PISA JSON schema)
"""

import logging

from fastpisa.core import analyze, _build_formula, _build_composition  # noqa: F401 (re-export)

logger = logging.getLogger(__name__)


def analyze_structure(
    input_file: str,
    pdb_id: str = "unknown",
    assembly_id: str = "1",
    probe_radius: float = 1.4,
    point_density: int = 480,
    interface_cutoff: float = 5.0,
    asis: bool = False,
    extended_data: bool = False,
    exclude_water: bool = True,
    min_css: float = 0.0,
) -> dict:
    """Run the full PISA analysis on a structure.

    Parameters
    ----------
    input_file : str
        Path to PDB or mmCIF file.
    pdb_id : str
        PDB identifier (e.g. "6nxr").
    assembly_id : str
        Assembly ID for output.
    probe_radius : float
        Probe sphere radius for ASA calculation (default 1.4 A).
    point_density : int
        Number of points on the probe sphere (default 480).
    interface_cutoff : float
        Distance cutoff for interface atom detection (default 5.0 A).
    asis : bool
        If True, only calculate interfaces (no assembly prediction).
    extended_data : bool
        If True, include extended -list data.
    min_css : float
        If > 0, only interfaces with CSS >= min_css are kept (a
        significance filter that drops weak/artifact crystal-packing
        contacts). Default 0.0 keeps everything.

    Returns
    -------
    dict
        interfaces.json and assembly.json data.
    """
    return analyze(
        input_file,
        pdb_id=pdb_id,
        assembly_id=assembly_id,
        probe_radius=probe_radius,
        point_density=point_density,
        interface_cutoff=interface_cutoff,
        exclude_water=exclude_water,
        min_css=min_css,
        mode="pisa",
    )
