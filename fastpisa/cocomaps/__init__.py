"""COCOMAPS mode for fastPISA.

Implements the COCOMAPS 2.0 analysis approach (intermolecular contact maps
plus atomic interaction classification) on top of the shared fastPISA
surface/interface machinery, so that PISA and COCOMAPS modes identify
the same interfaces.

The output is JSON-compatible with the PISA schema while additionally
carrying the COCOMAPS-specific contact-map and interaction-type fields.
"""

from fastpisa.cocomaps.interactions import (
    classify_atom_pair,
    INTERACTION_TYPES,
)
from fastpisa.cocomaps.contact_map import (
    ResidueContact,
    build_residue_contact_map,
    aggregate_residue_pairs,
    build_contact_matrix,
)
from fastpisa.cocomaps.pipeline import analyze_structure_cocomaps

__all__ = [
    "classify_atom_pair",
    "INTERACTION_TYPES",
    "ResidueContact",
    "build_residue_contact_map",
    "aggregate_residue_pairs",
    "build_contact_matrix",
    "analyze_structure_cocomaps",
]