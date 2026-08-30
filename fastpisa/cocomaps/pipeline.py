"""
COCOMAPS analysis pipeline for fastPISA.

Runs the same shared core as the PISA pipeline (:mod:`fastpisa.core`: parse
structure, detect molecules, compute ASA/BSA, find interface atom pairs at
the same 5 A cutoff) but reports the interface in COCOMAPS style: a
residue-residue contact map with per-contact interaction-type classification.

The output is a dict identical in shape to the PISA pipeline's result
(interfaces / assembly JSON) so that the two modes are interchangeable, with
the COCOMAPS contact-map and interaction-count fields added.

Note: since the unified-core refactor the assembly ``dissociation_energy``
uses the same formula in every mode (it was previously computed with an
inconsistent sign convention in this mode).
"""

from __future__ import annotations

import logging

from fastpisa.core import analyze

logger = logging.getLogger(__name__)


def analyze_structure_cocomaps(
    input_file: str,
    pdb_id: str = "unknown",
    assembly_id: str = "1",
    probe_radius: float = 1.4,
    point_density: int = 480,
    interface_cutoff: float = 5.0,
    asis: bool = False,
    extended_data: bool = False,
    interaction_cutoff: float = 5.0,
    exclude_water: bool = True,
    min_css: float = 0.0,
) -> dict:
    """Run COCOMAPS analysis on a structure.

    Returns a dict with keys:
      - "interfaces": JSON-compatible doc, same schema as PISA mode plus
        COCOMAPS-specific "interface_contact_map" per interface.
      - "assembly": assembly JSON doc.
      - "interfaces_obj": list of Interface objects extended with COCOMAPS
        attributes (contact_map, interaction_population).

    min_css : float
        If > 0, only interfaces with CSS >= min_css are kept (a
        significance filter that drops weak/artifact crystal-packing
        contacts). Default 0.0 keeps everything (PISA-compatible).
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
        mode="cocomaps",
        interaction_cutoff=interaction_cutoff,
    )
