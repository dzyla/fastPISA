"""The crystal contact graph: placed molecules joined by interfaces.

A node is a molecule of the asymmetric unit plus the crystallographic
placement that puts it somewhere in the crystal. An edge is one interface,
carrying the RELATIVE placement between its two molecules -- so the same edge
can be applied at any node to reach that node's partner, which is what lets a
finite assembly be grown outwards from a seed (see
:mod:`fastpisa.assembly.predict`).

The graph is INFINITE. Nothing here materialises it; :func:`neighbours`
expands one node at a time.

All placement arithmetic is exact integer work on fractional coordinates.
Two traps, both of which were live bugs in the interface layer:

* a fractional rotation is **not** orthogonal, so its inverse is not its
  transpose -- in hexagonal axes the 3-fold's inverse is the *other* 3-fold;
* rounding orthogonal-space translations to decimals puts real cases on the
  boundary (a 6-fold screw translates by c/6 = 42.55 A).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

#: Denominator of the exact fractional translations, matching
#: :data:`fastpisa.assembly.crystal.FRAC_DENOMINATOR`.
FRAC_DENOMINATOR = 24


@dataclass(frozen=True)
class Placement:
    """An exact crystallographic placement in fractional coordinates."""

    rotation: tuple          # 3x3 integers, row-major
    translation: tuple       # 3 integers, units of 1/denominator
    denominator: int = FRAC_DENOMINATOR

    def compose(self, other: "Placement") -> "Placement":
        """``self`` then ``other``: the placement ``self . other``."""
        rot_a = np.asarray(self.rotation, dtype=np.int64)
        rot_b = np.asarray(other.rotation, dtype=np.int64)
        tran_a = np.asarray(self.translation, dtype=np.int64)
        tran_b = np.asarray(other.translation, dtype=np.int64)
        return Placement(
            rotation=tuple(tuple(int(v) for v in row) for row in rot_a @ rot_b),
            translation=tuple(int(v) for v in (rot_a @ tran_b + tran_a)),
            denominator=self.denominator,
        )

    def inverse(self) -> "Placement":
        from fastpisa.core import _integer_inverse

        inv = _integer_inverse(np.asarray(self.rotation, dtype=np.int64))
        tran = -inv @ np.asarray(self.translation, dtype=np.int64)
        return Placement(
            rotation=tuple(tuple(int(v) for v in row) for row in inv),
            translation=tuple(int(v) for v in tran),
            denominator=self.denominator,
        )

    def cell(self) -> Tuple[int, int, int]:
        """Integer lattice part of the translation."""
        den = self.denominator
        return tuple(int(v // den) for v in self.translation)  # type: ignore

    def key(self) -> tuple:
        """Hashable identity of the FULL placement, lattice part included."""
        return (self.rotation, self.translation)

    def rotation_key(self) -> tuple:
        """Identity of the rotation alone -- the 'same operation' test."""
        return self.rotation


IDENTITY_PLACEMENT = Placement(
    rotation=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    translation=(0, 0, 0),
    denominator=FRAC_DENOMINATOR,
)

#: A placed molecule: ``(asu_molecule_id, placement)``.
Node = Tuple[str, Placement]


@dataclass(frozen=True)
class ContactEdge:
    """One crystal interface, as a relative placement between two molecules."""

    molecule_a: str
    molecule_b: str
    placement: Placement     # places B relative to A
    stabilization: float     # kcal/mol, negative = favourable
    area: float
    interface_id: int


def _placement_of(molecule: dict) -> Placement:
    return Placement(
        rotation=tuple(tuple(int(v) for v in row)
                       for row in molecule["frac_rotation"]),
        translation=tuple(int(v) for v in molecule["frac_translation"]),
        denominator=int(molecule.get("frac_denominator", FRAC_DENOMINATOR)),
    )


def contact_edges(state) -> List[ContactEdge]:
    """One edge per crystal interface, most stabilising first.

    The ordering is the dissociation order the assembly search walks: PISA
    takes interfaces apart weakest-first, so growing an assembly adds them
    strongest-first.
    """
    edges = []
    for iface in state.interfaces:
        mol_a, mol_b = iface.molecules
        place_a = _placement_of(mol_a)
        place_b = _placement_of(mol_b)
        edges.append(ContactEdge(
            molecule_a=mol_a["asu_molecule_id"],
            molecule_b=mol_b["asu_molecule_id"],
            placement=place_a.inverse().compose(place_b),
            stabilization=float(iface.stabilization_energy),
            area=float(iface.interface_area),
            interface_id=int(iface.interface_id),
        ))
    edges.sort(key=lambda e: (e.stabilization, -e.area, e.interface_id))
    return edges


def neighbours(node: Node, edges: Sequence[ContactEdge]):
    """Every ``(partner_node, edge)`` reachable from ``node``.

    Each edge is used in both directions: an interface between A and B placed
    by P also joins B to A placed by P inverse.
    """
    molecule, placement = node
    out = []
    for edge in edges:
        if edge.molecule_a == molecule:
            out.append(((edge.molecule_b,
                         placement.compose(edge.placement)), edge))
        if edge.molecule_b == molecule:
            out.append(((edge.molecule_a,
                         placement.compose(edge.placement.inverse())), edge))
    return out
