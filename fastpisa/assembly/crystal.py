"""Crystallographic symmetry-mate generation.

For a deposited crystal entry, original PISA analyses the CRYSTAL, not the
deposited coordinates: it expands the asymmetric unit by the space group's
operations and the neighbouring unit cells, and reports every interface a
molecule makes -- including packing contacts with symmetry mates. Across the
37 cached reference entries, 480 of PISA's 802 interfaces (60%) involve at
least one mate.

This module produces the mates. It generates, for each space-group operation
and each neighbouring cell, the rigid motion that places a copy of the
asymmetric unit, and keeps the copies that come close enough to bury surface
against it.

Conventions that matter
-----------------------
* **The contact screen is the SHADOW cutoff, not 5 A.** PISA defines an
  interface by buried area (pair dASA > 0), and two atoms shadow each other's
  solvent-accessible surface out to ``r1 + r2 + 2 * probe`` (~6.6 A for
  protein atoms). 1acb's interface 9 is a genuine 14.2 A^2 interface whose
  closest atom pair is 5.02 A apart -- a 5 A screen drops it.
* **The cell range is derived, not fixed at +-1.** A molecule whose bounding
  box spans more than one cell edge has contacting images two cells away.
  :func:`cell_search_range` computes the range from the cell, the coordinate
  extent and the cutoff.
* **Transforms are returned in orthogonal (Cartesian) coordinates**, matching
  what PISA reports per molecule (``rxx``...``rzz``, ``tx/ty/tz``), so a
  generated copy can be compared to PISA's own placement without reconciling
  cell conventions. ``tests/test_crystal.py`` asserts that every transform
  PISA used is generated.

``gemmi`` supplies the space-group operations and the cell matrices; it is
already the optional mmCIF dependency.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Cells to search in each direction, whatever the geometry suggests. One
#: cell each way is the minimum that can ever produce a contact.
MIN_CELL_RANGE = 1

#: Hard ceiling on the per-axis cell range, so a pathological cell cannot
#: make the expansion unbounded.
MAX_CELL_RANGE = 4


#: Denominator of the exact fractional translations (gemmi's ``Op.DEN``).
FRAC_DENOMINATOR = 24


@dataclass(frozen=True)
class CrystalOperator:
    """One space-group operation, in FRACTIONAL coordinates.

    ``rotation``/``translation`` are floats for geometry; ``int_rotation``
    and ``int_translation`` are the EXACT integer form (the translation in
    units of ``1 / FRAC_DENOMINATOR``). A crystallographic rotation is always
    an integer matrix in fractional coordinates, so composing placements is
    exact integer arithmetic -- which is what symmetry equivalence must be
    decided on, never on rounded floats.
    """

    symop_no: int
    triplet: str
    rotation: tuple          # 3x3, row-major
    translation: tuple       # 3
    int_rotation: tuple = ()      # 3x3 integers
    int_translation: tuple = ()   # 3 integers, units of 1/FRAC_DENOMINATOR


@dataclass(frozen=True)
class SymmetryCopy:
    """A placement of the asymmetric unit, in ORTHOGONAL coordinates.

    ``x_copy = x_asu @ rotation.T + translation``, the same convention PISA
    reports per interface molecule.
    """

    symop_no: int
    triplet: str
    cell: Tuple[int, int, int]
    rotation: tuple          # 3x3, row-major, orthogonal
    translation: tuple       # 3
    #: Exact fractional placement, cell shift INCLUDED: an integer rotation
    #: and a translation in units of 1/:data:`frac_denominator`. Symmetry
    #: equivalence is decided from these, so it is exact.
    frac_rotation: tuple = ()
    frac_translation: tuple = ()
    frac_denominator: int = FRAC_DENOMINATOR

    @property
    def label(self) -> str:
        """``"2_556"``-style tag: operation number and cell offsets +5."""
        i, j, k = self.cell
        return f"{self.symop_no}_{i + 5}{j + 5}{k + 5}"


def crystal_operators(space_group: str) -> List[CrystalOperator]:
    """All general-position operations of ``space_group``, centring included.

    Raises ``ValueError`` when the symbol is not recognised rather than
    falling back to P1: silently analysing only the asymmetric unit of a
    crystal entry would look like a clean result.
    """
    import gemmi

    sg = gemmi.find_spacegroup_by_name((space_group or "").strip())
    if sg is None:
        raise ValueError(f"unrecognised space group: {space_group!r}")
    out = []
    for n, op in enumerate(sg.operations(), start=1):
        int_rot = np.array(op.rot, dtype=np.int64) // op.DEN
        int_tran = np.array(op.tran, dtype=np.int64)
        # Plain Python ints, not numpy scalars: these travel into the output
        # JSON on every interface molecule, and json.dump refuses int64.
        if op.DEN != FRAC_DENOMINATOR:  # pragma: no cover - gemmi pins DEN=24
            int_tran = int_tran * FRAC_DENOMINATOR // op.DEN
        rot = np.array(op.rot, dtype=float) / op.DEN
        tran = np.array(op.tran, dtype=float) / op.DEN
        out.append(CrystalOperator(
            symop_no=n,
            triplet=op.triplet(),
            rotation=tuple(map(tuple, rot)),
            translation=tuple(tran),
            int_rotation=tuple(tuple(int(v) for v in row) for row in int_rot),
            int_translation=tuple(int(v) for v in int_tran),
        ))
    return out


def cell_search_range(cell, coords: np.ndarray,
                      cutoff: float) -> Tuple[int, int, int]:
    """WIDTH of the cell window to search, per axis -- not its position.

    An image shifted by ``n`` cells can still reach the original when the
    coordinate extent plus the cutoff exceeds ``n - 1`` cell widths. The
    perpendicular cell widths (volume / face area) rather than the edge
    lengths keep this right for oblique cells, and the extent is taken as the
    bounding-box diagonal because a symmetry operation may rotate the copy.

    This is only the HALF-WIDTH of the window. Where the window sits is set
    per operator by :func:`operator_cell_offset`: deposited coordinates are
    not centred on the cell origin, so an operation can place a copy many
    cells away and the shift that brings it back has nothing to do with the
    molecule's size.
    """
    extent = (coords.max(axis=0) - coords.min(axis=0)) if len(coords) else np.zeros(3)
    reach = float(np.linalg.norm(extent)) + cutoff
    widths = _perpendicular_widths(cell)
    out = []
    for w in widths:
        n = MIN_CELL_RANGE if w <= 0 else int(np.ceil(reach / w))
        out.append(int(min(max(n, MIN_CELL_RANGE), MAX_CELL_RANGE)))
    return tuple(out)  # type: ignore[return-value]


def _perpendicular_widths(cell) -> np.ndarray:
    """Distance between opposite faces of the cell, per axis."""
    orth = np.array(cell.orth.mat.tolist(), dtype=float)
    a, b, c = orth[:, 0], orth[:, 1], orth[:, 2]
    volume = abs(float(np.dot(a, np.cross(b, c))))
    if volume <= 0:
        return np.array([0.0, 0.0, 0.0])
    return np.array([
        volume / np.linalg.norm(np.cross(b, c)),
        volume / np.linalg.norm(np.cross(c, a)),
        volume / np.linalg.norm(np.cross(a, b)),
    ])


def operator_cell_offset(frac_coords: np.ndarray, rot: np.ndarray,
                         tran: np.ndarray) -> np.ndarray:
    """Integer cell shift that brings an operator's copy back onto the molecule.

    In P 1 21 1 the 2-fold maps fractional ``x`` to ``-x``, so a molecule
    sitting five cells from the origin lands ten cells away; the contacting
    image is at a shift of ten, which no window centred on the origin finds
    at any sane width. Centring the window on the offset that aligns the two
    centroids makes the search independent of where the depositor put the
    coordinates.
    """
    centre = frac_coords.mean(axis=0)
    moved = (frac_coords @ np.asarray(rot, dtype=float).T
             + np.asarray(tran, dtype=float)).mean(axis=0)
    return np.rint(centre - moved).astype(int)


def _unit_cell(crystal_info: Dict):
    import gemmi

    needed = ("a", "b", "c", "alpha", "beta", "gamma")
    if not all(k in crystal_info for k in needed):
        return None
    cell = gemmi.UnitCell(*(float(crystal_info[k]) for k in needed))
    return cell if cell.volume > 0 else None


def symmetry_copies(
    coords: np.ndarray,
    crystal_info: Dict,
    cutoff: float,
    operators: Optional[Sequence[CrystalOperator]] = None,
) -> List[SymmetryCopy]:
    """Symmetry mates of ``coords`` that come within ``cutoff`` of it.

    Parameters
    ----------
    coords : (n, 3) array
        Asymmetric-unit coordinates (heavy atoms, water excluded -- water
        takes no part in the interface search and only slows the screen).
    crystal_info : dict
        As parsed from ``CRYST1`` / ``_cell``: ``a b c alpha beta gamma`` and
        ``space_group``. Missing or degenerate entries yield no copies, which
        is the correct answer for a predicted model.
    cutoff : float
        Contact cutoff -- pass the SHADOW cutoff, see the module docstring.
    operators : sequence, optional
        Pre-computed operators, to avoid re-deriving them per call.

    Returns
    -------
    list of SymmetryCopy
        Never includes the identity placement.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or len(coords) == 0:
        return []
    cell = _unit_cell(crystal_info)
    if cell is None:
        return []
    space_group = crystal_info.get("space_group") or ""
    if not space_group.strip():
        return []
    if operators is None:
        operators = crystal_operators(space_group)

    import gemmi
    from scipy.spatial import cKDTree

    frac_mat = np.array(cell.frac.mat.tolist(), dtype=float)
    orth_mat = np.array(cell.orth.mat.tolist(), dtype=float)
    frac = coords @ frac_mat.T
    tree = cKDTree(coords)
    ranges = cell_search_range(cell, coords, cutoff)

    out: List[SymmetryCopy] = []
    seen = set()
    for op in operators:
        rot_f = np.asarray(op.rotation, dtype=float)
        tr_f = np.asarray(op.translation, dtype=float)
        base = frac @ rot_f.T + tr_f
        # Orthogonal-space rotation of this operation (a rigid motion).
        rot_o = orth_mat @ rot_f @ frac_mat
        # Centre the cell window on THIS operator's displacement.
        centre_shift = operator_cell_offset(frac, rot_f, tr_f)
        for window in itertools.product(
                *(range(-r, r + 1) for r in ranges)):
            shift = tuple(int(c + w) for c, w in zip(centre_shift, window))
            key = (op.symop_no, shift)
            if key in seen:
                continue
            offset = np.asarray(shift, dtype=float)
            moved = (base + offset) @ orth_mat.T
            if op.symop_no == 1 and not any(shift):
                continue  # the asymmetric unit itself
            if not tree.query_ball_point(
                    moved, cutoff, return_length=True).any():
                continue
            # Translation that maps the ASU onto this copy, orthogonal space.
            tr_o = orth_mat @ (tr_f + offset)
            seen.add(key)
            int_tran = (np.asarray(op.int_translation, dtype=np.int64)
                        + np.asarray(shift, dtype=np.int64) * FRAC_DENOMINATOR)
            rot_o_py = tuple(tuple(float(v) for v in row) for row in rot_o)
            out.append(SymmetryCopy(
                symop_no=op.symop_no,
                triplet=op.triplet,
                cell=tuple(int(s) for s in shift),
                rotation=rot_o_py,
                translation=tuple(float(v) for v in tr_o),
                frac_rotation=op.int_rotation,
                frac_translation=tuple(int(v) for v in int_tran),
                frac_denominator=FRAC_DENOMINATOR,
            ))
    return out


# ---------------------------------------------------------------------------
# Expanding a parsed structure
# ---------------------------------------------------------------------------
#: Separator between an asymmetric-unit chain id and its symmetry label in
#: the chain id of a generated mate ("A" -> "A~2_556"). Chosen because it
#: cannot occur in a PDB or mmCIF chain identifier, so a mate id can always
#: be split back and never collides with a deposited chain.
MATE_SEPARATOR = "~"

#: Placement of the asymmetric unit itself, as a :class:`SymmetryCopy`.
IDENTITY_COPY = SymmetryCopy(
    symop_no=1,
    triplet="x,y,z",
    cell=(0, 0, 0),
    rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    translation=(0.0, 0.0, 0.0),
    frac_rotation=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    frac_translation=(0, 0, 0),
    frac_denominator=FRAC_DENOMINATOR,
)


def expand_structure(structure, cutoff: float,
                     exclude_water: bool = True):
    """Return ``(expanded_structure, copies)`` for a crystal entry.

    The expanded structure holds the asymmetric unit followed by one chain
    per (asymmetric-unit chain, symmetry copy) that comes within ``cutoff``
    of it. Mate chains are renamed ``<chain><sep><symop>_<cell>`` so the
    existing molecule/mask machinery treats them as separate molecules and
    so intra-molecule logic keyed on chain id (covalent antecedents, ring
    geometry) stays correct.

    ``copies`` is the list of placements, the asymmetric unit's identity
    placement first, aligned with the ``copy_index`` recorded on each
    returned atom's chain id.

    When the structure carries no usable cell, the structure is returned
    unchanged with only the identity placement -- the right answer for a
    predicted model.
    """
    import copy as _copy

    from fastpisa.interface.contacts import is_water_ligand

    coords = np.array(
        [[a.x, a.y, a.z] for chain in structure.chains for a in chain.atoms
         if a.element.strip().upper() not in ("H", "D")
         and not (exclude_water and is_water_ligand(a.res_name))],
        dtype=float,
    )
    if len(coords) == 0:
        return structure, [IDENTITY_COPY]
    try:
        copies = symmetry_copies(coords, structure.crystal_info or {}, cutoff)
    except ValueError:
        raise
    if not copies:
        return structure, [IDENTITY_COPY]

    expanded = _copy.copy(structure)
    expanded.chains = list(structure.chains)
    for index, placement in enumerate(copies, start=1):
        rot = np.asarray(placement.rotation, dtype=float)
        tr = np.asarray(placement.translation, dtype=float)
        suffix = f"{MATE_SEPARATOR}{placement.symop_no}_" \
                 f"{placement.cell[0] + 5}{placement.cell[1] + 5}" \
                 f"{placement.cell[2] + 5}"
        for chain in structure.chains:
            if not chain.atoms:
                continue
            mate = _copy.copy(chain)
            mate.auth_asym_id = chain.auth_asym_id + suffix
            mate.chain_id = chain.chain_id + suffix
            mate.label_asym_id = chain.label_asym_id + suffix
            mate.atoms = []
            xyz = np.array([[a.x, a.y, a.z] for a in chain.atoms], dtype=float)
            moved = xyz @ rot.T + tr
            for atom, pos in zip(chain.atoms, moved):
                new = _copy.copy(atom)
                new.x, new.y, new.z = float(pos[0]), float(pos[1]), float(pos[2])
                new.auth_asym_id = mate.auth_asym_id
                new.chain_id = mate.chain_id
                new.label_asym_id = mate.label_asym_id
                mate.atoms.append(new)
            expanded.chains.append(mate)
    return expanded, [IDENTITY_COPY] + list(copies)


def split_mate_chain_id(chain_id: str):
    """``("A~2_556")`` -> ``("A", "2_556")``; a plain id -> ``(id, None)``."""
    if MATE_SEPARATOR not in chain_id:
        return chain_id, None
    base, _, label = chain_id.rpartition(MATE_SEPARATOR)
    return base, label


def copy_by_label(copies: Sequence[SymmetryCopy]):
    """``{label: copy}`` for the placements of :func:`expand_structure`."""
    return {c.label: c for c in copies}
