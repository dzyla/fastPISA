"""AlphaFold confidence filtering: Predicted Aligned Error (PAE) and ipTM.

AlphaFold 2/3 and most AlphaFold-derivative predictors emit a
``*_predicted_aligned_error.json`` alongside each model, containing a
residue x residue pairwise ``predicted_aligned_error`` matrix (in Angstrom,
lower = more confident) plus the global ``iptm`` / ``ptm`` scores. This module
lets fastPISA rank/filter interfaces by how confidently they are predicted,
addressing item 4.4 of fastpisa_improvements.md.

Conventions
-----------
The PAE matrix is indexed by SEQUENTIAL residue number over the whole model
(row i, column j for residues i and j), where residues are numbered in the
order the chains/residues appear in the model file. To score an interface we
look up the PAE for every CONTACTING inter-molecule residue pair and take the
mean -- a standard "interface confidence" surrogate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PAEData:
    """Parsed AlphaFold confidence JSON."""

    matrix: Optional[List[List[float]]] = None   # NxN (Angstrom)
    iptm: Optional[float] = None
    ptm: Optional[float] = None
    max_pae: Optional[float] = None

    @property
    def n_residues(self) -> int:
        return len(self.matrix) if self.matrix else 0

    @property
    def has_pae(self) -> bool:
        return bool(self.matrix)


def load_pae(json_path: str) -> PAEData:
    """Load an AlphaFold ``*_predicted_aligned_error.json`` file."""
    with open(json_path) as fh:
        data = json.load(fh)
    return PAEData(
        matrix=data.get("predicted_aligned_error"),
        iptm=data.get("iptm"),
        ptm=data.get("ptm"),
        max_pae=data.get("max_predicted_aligned_error"),
    )


def build_pae_index_map(structure) -> Dict[Tuple[str, int, str], int]:
    """Map ``(auth_asym_id, res_seq, icode)`` -> sequential PAE-matrix index.

    PAE rows/columns are numbered over the whole model in the order the
    residues appear; we walk the parsed chains/atoms in order and assign
    indices on first encounter of each residue, which reproduces that order.
    """
    mapping: Dict[Tuple[str, int, str], int] = {}
    seq = 0
    for chain in structure.chains:
        for atom in chain.atoms:
            key = (chain.auth_asym_id, atom.res_seq, atom.icode or "")
            if key not in mapping:
                mapping[key] = seq
                seq += 1
    return mapping


def interface_pae_score(interface, atoms, pae_index: Dict[Tuple[str, int, str], int],
                        pae: PAEData) -> Optional[float]:
    """Mean PAE (Angstrom) over the interface's contacting residue pairs.

    Returns ``None`` when no PAE matrix is available or no residue pair can be
    mapped. Lower value = more confidently predicted interface.
    """
    if not pae.has_pae or len(atoms) == 0:
        return None
    n = pae.n_residues
    if n == 0:
        return None
    pairs = set()
    for ct in interface.contacts:
        a1 = atoms[ct.atom1_idx] if ct.atom1_idx < len(atoms) else None
        a2 = atoms[ct.atom2_idx] if ct.atom2_idx < len(atoms) else None
        if a1 is None or a2 is None:
            continue
        k1 = pae_index.get((a1.auth_asym_id, a1.res_seq, a1.icode or ""))
        k2 = pae_index.get((a2.auth_asym_id, a2.res_seq, a2.icode or ""))
        if k1 is None or k2 is None or k1 == k2:
            continue
        if k1 >= n or k2 >= n:
            continue
        pairs.add((min(k1, k2), max(k1, k2)))
    if not pairs:
        return None
    matrix = pae.matrix or []
    vals = [matrix[i][j] for (i, j) in pairs]
    return float(sum(vals) / len(vals))


# ---------------------------------------------------------------------------
# B-factor / pLDDT confidence (portable across predictors)
# ---------------------------------------------------------------------------
# The AF *predicted_aligned_error.json* is emitted only by some pipelines and useless
# for most other methods. The portable confidence signal is the per-residue
# pLDDT, which AlphaFold, ColabFold and Protenix all write into the **B-factor
# column** of the output PDB/mmCIF (0-100, higher = more confident). A b-factor
# / pLDDT path therefore works for any predictor without extra files.
PLDDT_MIN = 0.0
PLDDT_MAX = 100.0


def build_plddt_map(structure) -> Dict[Tuple[str, int, str], float]:
    """Per-residue pLDDT map from the B-factor column.

    ``(auth_asym_id, res_seq, icode) -> mean B-factor`` (usually 0-100).
    """
    acc: Dict[Tuple[str, int, str], List[float]] = {}
    for chain in structure.chains:
        for atom in chain.atoms:
            key = (chain.auth_asym_id, atom.res_seq, atom.icode or "")
            acc.setdefault(key, []).append(atom.bfactor)
    return {k: float(sum(v) / len(v)) for k, v in acc.items()}


def model_plddt(plddt_map: Dict[Tuple[str, int, str], float]) -> Optional[float]:
    """Overall mean pLDDT over all residues (a global confidence proxy)."""
    if not plddt_map:
        return None
    return float(sum(plddt_map.values()) / len(plddt_map))


def interface_plddt(interface, atoms,
                    plddt_map: Dict[Tuple[str, int, str], float]) -> Optional[float]:
    """Mean pLDDT over the interface's residues (both molecules).

    Higher = more confidently predicted interface. Returns ``None`` if no
    residue can be mapped (e.g. no per-residue confidence loaded).
    """
    if not plddt_map or len(atoms) == 0:
        return None
    seen = set()
    for ct in interface.contacts:
        a1 = atoms[ct.atom1_idx] if ct.atom1_idx < len(atoms) else None
        a2 = atoms[ct.atom2_idx] if ct.atom2_idx < len(atoms) else None
        if a1 is not None:
            seen.add((a1.auth_asym_id, a1.res_seq, a1.icode or ""))
        if a2 is not None:
            seen.add((a2.auth_asym_id, a2.res_seq, a2.icode or ""))
    vals = [v for k, v in plddt_map.items() if k in seen and v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))
