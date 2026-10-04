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


@dataclass
class Assembly:
    """One finite assembly the crystal can form."""

    rank: int
    nodes: tuple
    size: int
    mmsize: int
    composition: str
    formula: str
    dissociation_energy: float
    entropy: float
    interface_ids: tuple
    n_interfaces: int


def _molecule_index(state) -> Dict[str, dict]:
    """``asu_molecule_id`` -> its molecule dict (asymmetric unit copies)."""
    index: Dict[str, dict] = {}
    for iface in state.interfaces:
        for mol in iface.molecules:
            index.setdefault(mol["asu_molecule_id"], mol)
    for mol in state.molecules:
        key = mol.get("asu_molecule_id", mol.get("chain_id"))
        if key is not None:
            index.setdefault(key, mol)
    return index


def _molecule_masses(state) -> Dict[str, float]:
    """Mass in daltons of each asymmetric-unit molecule."""
    import numpy as np

    from fastpisa.energy.entropy import atoms_mass

    masses: Dict[str, float] = {}
    for mol, mask in zip(state.molecules, state.masks):
        key = mol.get("asu_molecule_id", mol.get("chain_id"))
        if key is None or key in masses:
            continue
        masses[key] = atoms_mass(state.atoms[i]
                                 for i in np.flatnonzero(mask))
    return masses


def _composition(nodes, index) -> str:
    """PISA-shaped composition string, e.g. ``E[4]I[4][CA][4]``."""
    from collections import Counter

    counts = Counter(molecule for molecule, _ in nodes)
    parts = []
    for molecule in sorted(counts, key=lambda m: (m.startswith("["), m)):
        n = counts[molecule]
        parts.append(molecule if n == 1 else f"{molecule}[{n}]")
    return "".join(parts)


def _formula(nodes, index) -> str:
    """PISA-shaped formula: upper case per distinct polymer, lower for ligands."""
    from collections import Counter

    counts = Counter(molecule for molecule, _ in nodes)
    polymers = [m for m in sorted(counts)
                if index.get(m, {}).get("molecule_class") != "Ligand"]
    ligands = [m for m in sorted(counts)
               if index.get(m, {}).get("molecule_class") == "Ligand"]
    out = []
    for letter, molecule in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", polymers):
        n = counts[molecule]
        out.append(letter if n == 1 else f"{letter}{n}")
    for letter, molecule in zip("abcdefghijklmnopqrstuvwxyz", ligands):
        n = counts[molecule]
        out.append(letter if n == 1 else f"{letter}{n}")
    return "".join(out)


def internal_contacts(nodes, edges):
    """``{(low, high): edge}`` for every contact inside this node set.

    Each physical contact appears ONCE. A homomolecular edge whose placement
    is self-inverse -- any pure 2-fold, inversion centre or mirror, which is
    the commonest crystallographic dimer -- is reached from both directions
    and lands on the SAME partner node, so without this it entered the
    dissociation cut twice: ``assembly_dissociation`` sums repeated pairs, and
    1ktz's A[2] scored dG_diss 20.87 against PISA's 5.63.

    The de-duplication is per (node pair, interface), NOT per direction: in a
    3-fold trimer the forward and reverse branches reach *different* nodes and
    both are genuine contacts, so dropping the reverse branch would be wrong.
    """
    node_of = {(molecule, placement.key()): position
               for position, (molecule, placement) in enumerate(nodes)}
    found = {}
    for position, (molecule, placement) in enumerate(nodes):
        for (partner, partner_placement), edge in neighbours(
                (molecule, placement), edges):
            other = node_of.get((partner, partner_placement.key()))
            if other is None or other == position:
                continue
            key = (min(position, other), max(position, other),
                   edge.interface_id)
            found.setdefault(key, edge)
    return found


def _score(nodes, edges, index, masses):
    """(dG_diss, T dS) of this assembly, via the shared dissociation search.

    ``edges`` must be the FULL crystal edge list, not the subset that closed
    the component: an assembly's dissociation cut is over its own internal
    interface graph (spec R4), and scoring over the sparser closing subset
    systematically understates dG_diss -- 1a3n's ABCD tetramer closes over
    two interfaces while four are internal to it.
    """
    from fastpisa.energy.dissociation import assembly_dissociation

    node_mass = {position: masses.get(molecule, 0.0)
                 for position, (molecule, _) in enumerate(nodes)}
    internal = [(low, high, edge.stabilization)
                for (low, high, _), edge
                in internal_contacts(nodes, edges).items()]
    pathway = assembly_dissociation(node_mass, internal)
    return pathway.dissociation_energy, pathway.entropy


def predict_assemblies(state,
                       max_nodes: int = MAX_ASSEMBLY_NODES) -> List[Assembly]:
    """Finite assemblies this crystal can form, most stable first.

    Candidates come from NESTED interface subsets: with the crystal's
    interfaces sorted most-stabilising first, grow components from every
    molecule using only the top k, for k = 1..n. Bounded at n iterations,
    and it is the order PISA takes an assembly apart in. An exhaustive 2^n
    subset search is deliberately not attempted.

    Components that repeat translationally are dropped (they are lattices,
    not assemblies), as are components made only of hetero groups -- with
    ``ligand_mode="separate"`` a lone ion is its own molecule and "an
    assembly of one ion" is not a useful answer.

    A structure with no symmetry and no interfaces yields its monomers, which
    is the right answer for a predicted single chain.
    """
    from collections import Counter

    from fastpisa.assembly.graph import IDENTITY_PLACEMENT, contact_edges

    index = _molecule_index(state)
    masses = _molecule_masses(state)
    all_edges = contact_edges(state)
    seeds = [(key, IDENTITY_PLACEMENT) for key in sorted(masses)]

    found: Dict[tuple, Assembly] = {}
    subsets = range(1, len(all_edges) + 1) if all_edges else [0]
    for k in subsets:
        edges = all_edges[:k]
        for seed in seeds:
            result = grow(seed, edges, max_nodes=max_nodes)
            if not result.finite:
                continue
            molecules = [m for m, _ in result.nodes]
            if all(index.get(m, {}).get("molecule_class") == "Ligand"
                   for m in molecules):
                continue
            signature = tuple(sorted(
                (molecule, placement.key())
                for molecule, placement in result.nodes))
            if signature in found:
                continue
            # Scored over ALL crystal edges, not the top-k that closed the
            # component: whichever k first reaches an assembly, its
            # dissociation cut is over its own internal interface graph.
            diss, entropy = _score(result.nodes, all_edges, index, masses)
            internal = internal_contacts(result.nodes, all_edges)
            found[signature] = Assembly(
                rank=0,
                nodes=result.nodes,
                size=len(result.nodes),
                mmsize=sum(
                    1 for m in molecules
                    if index.get(m, {}).get("molecule_class") != "Ligand"),
                composition=_composition(result.nodes, index),
                formula=_formula(result.nodes, index),
                dissociation_energy=diss,
                entropy=entropy,
                interface_ids=tuple(sorted(
                    {edge.interface_id for edge in internal.values()})),
                n_interfaces=len(internal),
            )

    # Collapse assemblies with identical molecule content, keeping the most
    # stable: two seeds in the same orbit give the same assembly placed
    # differently.
    by_content: Dict[tuple, Assembly] = {}
    for assembly in found.values():
        key = (tuple(sorted(Counter(m for m, _ in assembly.nodes).items())),
               assembly.size)
        previous = by_content.get(key)
        if (previous is None
                or assembly.dissociation_energy > previous.dissociation_energy):
            by_content[key] = assembly

    # The primary assembly is the largest STABLE one, not the most strongly
    # bound. dG_diss measures an assembly's weakest link, so it is not
    # monotonic in size: a tight dimer can out-score the tetramer containing
    # it, and a chain plus a bound ion scores high because pulling the ion
    # off costs area for almost no entropy gain. PISA's own output settles
    # the convention -- its first assembly has the largest mmsize in 31 of 34
    # cached entries and the highest dG_diss in only 25 of 34 -- so stable
    # assemblies order by macromolecular size first and dG_diss second.
    # dG_diss is still reported per assembly for callers who want it.
    ranked = sorted(
        by_content.values(),
        key=lambda a: (a.dissociation_energy <= 0,          # stable first
                       -a.mmsize,
                       -a.dissociation_energy,
                       -a.size,
                       a.composition))
    for position, assembly in enumerate(ranked, start=1):
        assembly.rank = position
    return ranked
