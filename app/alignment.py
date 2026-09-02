"""Chain detection and superposition across complexes, via ``pdb_align``.

Used by the comparison mode: given a reference complex whose antigen
chains the user has named, find the corresponding chains in every other
complex (sequence identity, optimal 1:1 assignment) and superpose each
complex onto the reference on those chains so all binders can be shown on
one antigen in Mol*.

``pdb_align`` (https://github.com/dzyla/pdb_align) is an app-only
dependency; the core fastPISA package does not import it.
"""
from __future__ import annotations

import gzip
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _plain_path(path: str) -> str:
    """pdb_align reads plain files; decompress .gz into a temp file once."""
    if not path.endswith(".gz"):
        return path
    out = os.path.join(tempfile.gettempdir(), "fastpisa_align_" + os.path.basename(path)[:-3])
    if not os.path.exists(out):
        with gzip.open(path, "rt", errors="replace") as fh, open(out, "w") as oh:
            oh.write(fh.read())
    return out


@dataclass
class ChainMatch:
    ref_chain: str
    mob_chain: str
    identity: float          # percent


@dataclass
class Superposition:
    matches: List[ChainMatch]
    rmsd: float
    n_aligned: int
    tm_score: Optional[float]
    aligned_text: str        # mobile structure moved onto the reference, PDB format
    per_chain: list = field(default_factory=list)


def detect_shared_chains(ref_path: str, mob_path: str, ref_chains: Sequence[str],
                         mob_candidates: Optional[Sequence[str]] = None,
                         min_identity: float = 30.0) -> List[ChainMatch]:
    """Which chains of ``mob`` correspond to ``ref_chains`` of ``ref``?

    Optimal 1:1 assignment on percent sequence identity (pdb_align's
    ``match_chains``); pairs below ``min_identity`` are dropped.
    """
    from pdb_align import PDBAligner, match_chains

    al = PDBAligner()
    al.add_reference(_plain_path(ref_path))
    al.add_mobile(_plain_path(mob_path))
    mob_ids = list(mob_candidates) if mob_candidates else list(al.mob_seqs.keys())
    cm = match_chains(al.ref_seqs, al.mob_seqs, al.ref_struct, al.mob_struct,
                      list(ref_chains), mob_ids)
    return [ChainMatch(r, m, float(ident)) for r, m, ident, _ in cm.pairs if ident >= min_identity]


def superpose(ref_path: str, mob_path: str, ref_chains: Sequence[str],
              mob_chains: Sequence[str]) -> Superposition:
    """Superpose ``mob`` onto ``ref`` using the given chain pairs (CA atoms).

    Returns the whole mobile structure transformed into the reference frame
    as PDB text, so it can be shown together with the reference in Mol*.
    """
    import gemmi
    from pdb_align import PDBAligner

    al = PDBAligner()
    al.add_reference(_plain_path(ref_path), chains=list(ref_chains))
    al.add_mobile(_plain_path(mob_path), chains=list(mob_chains))
    res = al.align()
    st = res.aligned_structure(color_by="bfactor")
    st.setup_entities()
    text = st.make_pdb_string()
    stats = res.summary_stats()
    pc = getattr(res, "per_chain", None)
    per_chain = pc.to_dict("records") if hasattr(pc, "to_dict") else []
    matches = [ChainMatch(r, m, float("nan")) for r, m in zip(ref_chains, mob_chains)]
    return Superposition(matches=matches, rmsd=float(stats.get("rmsd", float("nan"))),
                         n_aligned=int(stats.get("n_aligned", 0) or 0),
                         tm_score=stats.get("tm_score"), aligned_text=text, per_chain=per_chain)


def structure_text(path: str) -> Tuple[str, str]:
    """(text, format) of a structure file for the viewer ('pdb' | 'mmcif')."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as fh:
        txt = fh.read()
    fmt = "mmcif" if any(path.endswith(s) for s in (".cif", ".cif.gz", ".mmcif", ".mmcif.gz")) else "pdb"
    return txt, fmt
