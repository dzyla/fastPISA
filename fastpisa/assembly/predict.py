"""Enumerating, scoring and ranking the assemblies a crystal can form.

A chosen set of interfaces either closes into a finite assembly or tiles the
crystal. :func:`grow` decides which, by expanding a connected component from
a seed molecule and watching for translational repetition: reaching the same
molecule under the same ROTATION at a different lattice cell means the
component repeats forever -- a lattice, a sheet or a fibre, not an assembly.

That test, not the node cap, is what must catch an infinite component. The cap
exists so a bug cannot hang an analysis, and the result says which guard fired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from fastpisa.assembly.graph import ContactEdge, Node, neighbours

#: Hard ceiling on a grown component. Only a backstop: a genuine crystal
#: assembly is far smaller, and an infinite one is caught by the
#: translational test long before this.
MAX_ASSEMBLY_NODES = 512


@dataclass(frozen=True)
class GrowthResult:
    """What growing a component from one seed produced."""

    nodes: Tuple[Node, ...]
    edges: Tuple[ContactEdge, ...]
    finite: bool
    reason: str


def grow(seed: Node, edges: Sequence[ContactEdge],
         max_nodes: int = MAX_ASSEMBLY_NODES) -> GrowthResult:
    """Grow the connected component of ``seed`` over ``edges``.

    Returns ``finite=False`` with a ``reason`` when the component repeats
    translationally or exceeds ``max_nodes``. The node list is deterministic:
    breadth-first in the given edge order.
    """
    seed_molecule, seed_placement = seed
    # (molecule, rotation) -> the lattice cell it was first reached at. A
    # second cell for the same pair is translational repetition.
    cell_of: Dict[Tuple[str, tuple], Tuple[int, int, int]] = {
        (seed_molecule, seed_placement.rotation_key()): seed_placement.cell()}
    visited = {(seed_molecule, seed_placement.key()): seed_placement}
    order: List[Node] = [seed]
    used: Dict[int, ContactEdge] = {}
    queue: List[Node] = [seed]

    while queue:
        node = queue.pop(0)
        for partner, edge in neighbours(node, edges):
            molecule, placement = partner
            identity = (molecule, placement.key())
            if identity in visited:
                # The edge closes the component rather than extending it;
                # it is still part of the assembly's internal contacts.
                used.setdefault(id(edge), edge)
                continue
            rotation_identity = (molecule, placement.rotation_key())
            previous = cell_of.get(rotation_identity)
            if previous is not None and previous != placement.cell():
                return GrowthResult(
                    nodes=tuple(order), edges=tuple(used.values()),
                    finite=False,
                    reason=(f"translational repetition: {molecule} under the "
                            f"same rotation at cells {previous} and "
                            f"{placement.cell()}"))
            used.setdefault(id(edge), edge)
            visited[identity] = placement
            cell_of.setdefault(rotation_identity, placement.cell())
            order.append(partner)
            if len(order) > max_nodes:
                return GrowthResult(
                    nodes=tuple(order), edges=tuple(used.values()),
                    finite=False,
                    reason=f"node cap {max_nodes} exceeded")
            queue.append(partner)

    return GrowthResult(nodes=tuple(order), edges=tuple(used.values()),
                        finite=True, reason="closed")
