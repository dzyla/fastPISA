# Copyright (c) 2026 Dawid Zyla. Part of fastPISA.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Assembly dissociation energy along the cheapest pathway (PISA semantics).

PISA does not sum every interface to judge an assembly: it reports the free
energy of the *easiest* way to take the assembly apart. Recovered from the
PDBe PISA 2.0 JSON of 20 reference assemblies,

    dG_diss = -sum(stabilization_energy over the interfaces CUT) - T*dS

where the cut is a set of interfaces whose removal separates the assembly
into independent bodies, and ``T*dS`` is the rigid-body entropy of releasing
them (:mod:`fastpisa.energy.entropy`). Positive ``dG_diss`` means the
assembly is stable against that pathway.

fastPISA previously summed -(dG_solv + dG_contact) over ALL interfaces and
ADDED a hand-tuned pseudo-entropy, which gets the sign of the entropy term,
the set of interfaces and the entropy itself wrong at once.

Why a search, and why it is exact
---------------------------------
Because the entropy term is additive over a splitting tree (see
:func:`fastpisa.energy.entropy.dissociation_entropy`), the cheapest pathway
obeys

    D(S) = min over bipartitions (A, B) of S of
             [ -sum(stab over cut(A, B)) - T*dS(m_A, m_B)
               + min(0, D(A)) + min(0, D(B)) ]

The ``min(0, ...)`` terms are what let a pathway release more than two
bodies at once when that is cheaper -- PISA does exactly this (9eyh releases
all four chains). Memoised over molecule subsets this is exact for a bound
group of up to :data:`MAX_EXACT_MOLECULES` components. Larger groups fall
back to a flat single-peel search (:func:`_restricted_pathway`), whose result
is a real pathway and therefore an upper bound, flagged ``exact=False``.

Ligands are folded into their host chain before any of this runs (see
``fastpisa.core.dissociation_pathway``), so the component count is the number
of macromolecules even when a structure carries 500 ordered waters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, List, Sequence, Tuple

from fastpisa.energy.entropy import dissociation_entropy

#: Above this many bound components the exact O(3^n) subset search is
#: replaced by the flat single-peel search of :func:`_restricted_pathway`.
#: Every assembly in the PISA reference set is far below it (the largest is
#: 8 macromolecules), and ligand folding keeps the component count at the
#: number of macromolecules.
MAX_EXACT_MOLECULES = 14


@dataclass
class DissociationPathway:
    """The cheapest way found to take an assembly apart.

    Attributes
    ----------
    dissociation_energy : float
        ``-sum(cut stab) - T*dS``, kcal/mol. Positive = stable against this
        pathway. ``0.0`` when the assembly has nothing to break.
    entropy : float
        ``T*dS`` of the released bodies, kcal/mol (positive).
    parts : list of list
        The molecule keys in each released body.
    cut : list of tuple
        The ``(key_a, key_b)`` interfaces the pathway breaks, sorted.
    exact : bool
        Whether the whole pathway space was searched (see
        :data:`MAX_EXACT_MOLECULES`).
    """

    dissociation_energy: float
    entropy: float
    parts: List[List[Hashable]]
    cut: List[Tuple[Hashable, Hashable]]
    exact: bool = True


def _components(keys, adjacency) -> List[List[Hashable]]:
    """Connected components of ``keys`` under ``adjacency``."""
    remaining = set(keys)
    out = []
    while remaining:
        seed = remaining.pop()
        stack, block = [seed], {seed}
        while stack:
            node = stack.pop()
            for nb in adjacency.get(node, ()):  # pragma: no branch
                if nb in remaining:
                    remaining.discard(nb)
                    block.add(nb)
                    stack.append(nb)
        out.append(sorted(block, key=str))
    return out


def assembly_dissociation(
    masses: Dict[Hashable, float],
    interfaces: Sequence[Tuple[Hashable, Hashable, float]],
) -> DissociationPathway:
    """Cheapest dissociation pathway for one assembly.

    Parameters
    ----------
    masses : dict
        Mass in daltons of every component, keyed however the caller likes
        (chain id, molecule index...).
    interfaces : sequence of (key_a, key_b, stabilization_energy)
        One entry per interface between two components. ``stabilization``
        follows PISA's sign convention: negative = favourable. Several
        entries for the same pair are summed.

    Returns
    -------
    DissociationPathway

    Notes
    -----
    Components that share no interface are already separate bodies: their
    separation is not a dissociation pathway and contributes no entropy
    term, so each connected group of the interface graph is treated on its
    own and the weakest one is reported.
    """
    keys = sorted(masses, key=str)
    if not keys:
        return DissociationPathway(0.0, 0.0, [], [])

    weight: Dict[Tuple[Hashable, Hashable], float] = {}
    adjacency: Dict[Hashable, set] = {k: set() for k in keys}
    for a, b, stab in interfaces:
        if a == b or a not in masses or b not in masses:
            continue
        pair = (a, b) if str(a) <= str(b) else (b, a)
        weight[pair] = weight.get(pair, 0.0) + float(stab)
        adjacency[a].add(b)
        adjacency[b].add(a)

    if not weight:
        return DissociationPathway(0.0, 0.0, [[k] for k in keys], [])

    # Each connected group dissociates independently; report the weakest.
    best: DissociationPathway | None = None
    for group in _components(keys, adjacency):
        if len(group) < 2:
            continue
        path = _cheapest_pathway(group, masses, weight, adjacency)
        if best is None or path.dissociation_energy < best.dissociation_energy:
            best = path
    if best is None:  # pragma: no cover - guarded by `if not weight`
        return DissociationPathway(0.0, 0.0, [[k] for k in keys], [])
    return best


def _cut_weight(block_a, block_b, weight) -> float:
    """Sum of stabilization energies of interfaces crossing the split."""
    total = 0.0
    for a in block_a:
        for b in block_b:
            pair = (a, b) if str(a) <= str(b) else (b, a)
            total += weight.get(pair, 0.0)
    return total


def _cut_pairs(block_a, block_b, weight):
    out = []
    for a in block_a:
        for b in block_b:
            pair = (a, b) if str(a) <= str(b) else (b, a)
            if pair in weight:
                out.append(pair)
    return sorted(out, key=lambda p: (str(p[0]), str(p[1])))


def _restricted_pathway(group, masses, weight) -> DissociationPathway:
    """Flat fallback for groups too large for the exact subset search.

    Evaluates only the bipartitions that peel ONE component off the rest --
    O(n) candidates, no recursion. The exact search recurses into subsets, so
    even a "peel one at a time" split rule inside it would still reach every
    subset of a large group and never return; a long chain of bound ordered
    waters, or a filament, would hang the analysis. The pathway returned here
    is a real one, so its energy is a valid UPPER BOUND on the dissociation
    energy, and ``exact=False`` says so.
    """
    best = None
    for key in group:
        rest = [k for k in group if k != key]
        cut = _cut_weight([key], rest, weight)
        if cut == 0.0:
            continue
        tds = dissociation_entropy(
            [masses[key], sum(masses[k] for k in rest)])
        value = -cut - tds
        if best is None or value < best[0]:
            best = (value, key, rest, tds)
    if best is None:  # pragma: no cover - a connected group always splits
        return DissociationPathway(0.0, 0.0, [sorted(group, key=str)], [],
                                   exact=False)
    value, key, rest, tds = best
    return DissociationPathway(
        dissociation_energy=value,
        entropy=tds,
        parts=[sorted(rest, key=str), [key]],
        cut=_cut_pairs([key], rest, weight),
        exact=False,
    )


def _cheapest_pathway(group, masses, weight, adjacency) -> DissociationPathway:
    if len(group) > MAX_EXACT_MOLECULES:
        return _restricted_pathway(group, masses, weight)
    exact = True
    full = (1 << len(group)) - 1
    mass_of = [masses[k] for k in group]

    def subset_mass(bits: int) -> float:
        return sum(m for i, m in enumerate(mass_of) if bits >> i & 1)

    def members(bits: int):
        return [group[i] for i in range(len(group)) if bits >> i & 1]

    def splits(bits: int):
        """Every bipartition of ``bits`` into two non-empty halves, once.

        Enumerates the proper non-empty subsets containing the lowest set
        bit, so (A, B) and (B, A) are not both visited.
        """
        low = bits & -bits
        sub = bits
        while sub:
            sub = (sub - 1) & bits
            if sub and (sub & low):
                yield sub, bits ^ sub

    memo: Dict[int, Tuple[float, object]] = {}

    def solve(bits: int):
        """(best dG_diss, pathway tree) for dissociating this subset."""
        if bits in memo:
            return memo[bits]
        if bin(bits).count("1") < 2:
            memo[bits] = (0.0, None)
            return memo[bits]
        best_val = None
        best_tree = None
        for left, right in splits(bits):
            cut = _cut_weight(members(left), members(right), weight)
            if cut == 0.0:
                # Nothing to break between these halves: not a pathway step
                # (they are already separate bodies within this subset).
                continue
            tds = dissociation_entropy([subset_mass(left), subset_mass(right)])
            value = -cut - tds
            left_val, left_tree = solve(left)
            right_val, right_tree = solve(right)
            if left_val < 0.0:
                value += left_val
            else:
                left_tree = None
            if right_val < 0.0:
                value += right_val
            else:
                right_tree = None
            if best_val is None or value < best_val:
                best_val = value
                best_tree = (left, right, left_tree, right_tree)
        if best_val is None:
            memo[bits] = (0.0, None)
        else:
            memo[bits] = (best_val, best_tree)
        return memo[bits]

    value, tree = solve(full)
    if tree is None:  # pragma: no cover - a connected group always splits
        return DissociationPathway(0.0, 0.0, [sorted(group, key=str)], [],
                                   exact=exact)

    parts_bits: List[int] = []
    cut_pairs: List[Tuple[Hashable, Hashable]] = []

    def walk(bits, node):
        if node is None:
            parts_bits.append(bits)
            return
        left, right, left_tree, right_tree = node
        cut_pairs.extend(_cut_pairs(members(left), members(right), weight))
        walk(left, left_tree)
        walk(right, right_tree)

    walk(full, tree)
    parts = [sorted(members(b), key=str) for b in parts_bits]
    entropy = dissociation_entropy([subset_mass(b) for b in parts_bits])
    broken = sorted(set(cut_pairs), key=lambda p: (str(p[0]), str(p[1])))
    cut_total = sum(weight[p] for p in broken)
    return DissociationPathway(
        dissociation_energy=-cut_total - entropy,
        entropy=entropy,
        parts=sorted(parts, key=lambda p: (-len(p), str(p))),
        cut=broken,
        exact=exact,
    )
