#!/usr/bin/env python
"""Audit / refit the dissociation-entropy constant against PISA. Offline.

    python examples/calibrate_entropy.py

Reads the cached PDBe PISA 2.0 assembly documents in
``tests/data/reference/json/`` and the matching assembly mmCIFs, identifies
the dissociation cut PISA chose from PISA's OWN numbers, and refits the
single constant ``STANDARD_STATE_LN_TERM`` of
:mod:`fastpisa.energy.entropy`.

Identifying the cut without using the entropy model keeps the fit
non-circular. For each assembly, PISA satisfies

    dG_diss = -sum(stabilization_energy over the cut) - T*dS

so the cut must be an interface subset whose stabilization energies sum to
``-dG_diss - entropy`` AND whose removal disconnects the chain graph.
Assemblies whose JSON omits interfaces cannot close that sum and are
skipped, with a note.

Add more entries by fetching them first (network):

    python -c "from fastpisa.reference.ebi_pisa import \\
        fetch_pisa_assembly_json, fetch_assembly_cif; \\
        fetch_pisa_assembly_json('8abc'); fetch_assembly_cif('8abc')"
"""

from __future__ import annotations

import argparse
import glob
import gzip
import itertools
import json
import math
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from fastpisa.energy.entropy import (  # noqa: E402
    MASS_EXPONENT, RT_KCAL, STANDARD_STATE_LN_TERM, element_mass,
)

JSON_DIR = os.path.join(REPO, "tests", "data", "reference", "json")
CIF_DIR = os.path.join(REPO, "tests", "data", "reference", "assemblies")

# Ideal-gas value of the constant at a 1 M standard state, masses in daltons.
_AMU_KG = 1.66053906660e-27
_KB = 1.380649e-23
_H = 6.62607015e-34
_NA = 6.02214076e23
_T = 298.15
IDEAL_1M = (math.log(1e-3 / _NA)
            + 1.5 * math.log((2 * math.pi * _KB * _T) / (_H * _H))
            + 1.5 * math.log(_AMU_KG)
            + 2.5)


def _assemblies():
    for path in sorted(glob.glob(os.path.join(JSON_DIR, "*.json.gz"))):
        with gzip.open(path, "rt") as fh:
            yield os.path.basename(path)[:4], json.load(fh)["assembly"]


def _chain_masses(pdb_id):
    from fastpisa.parser.pdb_parser import parse_mmcif

    cif = os.path.join(CIF_DIR, f"{pdb_id}-assembly1.cif.gz")
    if not os.path.exists(cif):
        return None
    masses = {}
    for atom in parse_mmcif(cif).atoms:
        try:
            m = element_mass(atom.element)
        except KeyError:
            m = 12.011
        masses[atom.auth_asym_id] = masses.get(atom.auth_asym_id, 0.0) + m
    return masses


def _identify_cut(assembly):
    """The partition PISA dissociated into, or None if not identifiable."""
    edges = []
    for iface in assembly["interfaces"]:
        mols = iface.get("molecules") or []
        if len(mols) != 2:
            return None
        edges.append((mols[0]["chain_id"], mols[1]["chain_id"],
                      iface["stabilization_energy"]))
    if not edges or len(edges) > 14:
        return None
    nodes = sorted({c for e in edges for c in e[:2]})
    target = -assembly["dissociation_energy"] - assembly["entropy"]

    for r in range(1, len(edges) + 1):
        for comb in itertools.combinations(range(len(edges)), r):
            # PISA reports 2 decimals, so allow per-interface rounding.
            if abs(sum(edges[i][2] for i in comb) - target) > 0.015 * r + 0.005:
                continue
            adjacency = {n: set() for n in nodes}
            for k, (u, v, _) in enumerate(edges):
                if k in comb:
                    continue
                adjacency[u].add(v)
                adjacency[v].add(u)
            seen, parts = set(), []
            for n in nodes:
                if n in seen:
                    continue
                stack, block = [n], set()
                while stack:
                    x = stack.pop()
                    if x in block:
                        continue
                    block.add(x)
                    stack.extend(adjacency[x] - block)
                seen |= block
                parts.append(sorted(block))
            if len(parts) > 1:
                return parts
    return None


def collect():
    """[(pdb_id, n_released, log-mass term, PISA entropy)] for the fit."""
    rows, skipped = [], []
    for pdb_id, assembly in _assemblies():
        parts = _identify_cut(assembly)
        if parts is None:
            skipped.append((pdb_id, "cut not identifiable (incomplete "
                                    "interface list in the JSON)"))
            continue
        masses = _chain_masses(pdb_id)
        if masses is None:
            skipped.append((pdb_id, "no cached assembly mmCIF"))
            continue
        try:
            blocks = [sum(masses[c] for c in block) for block in parts]
        except KeyError as exc:
            skipped.append((pdb_id, f"chain {exc} absent from the mmCIF"))
            continue
        term = (math.fsum(MASS_EXPONENT * math.log(m) for m in blocks)
                - MASS_EXPONENT * math.log(math.fsum(blocks)))
        rows.append((pdb_id, len(blocks) - 1, term, assembly["entropy"]))
    return rows, skipped


def fit(rows):
    """Weighted least squares for the one constant."""
    num = math.fsum(k * ((y / RT_KCAL) - term) for _, k, term, y in rows)
    den = math.fsum(k * k for _, k, _, _ in rows)
    return num / den


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emit", action="store_true",
                    help="print a paste-ready constant assignment")
    args = ap.parse_args()

    rows, skipped = collect()
    if not rows:
        raise SystemExit("no usable reference assemblies found")
    constant = fit(rows)

    print(f"reference assemblies usable: {len(rows)}")
    for pdb_id, reason in skipped:
        print(f"  skipped {pdb_id}: {reason}")
    print()
    print(f"{'entry':6} {'k':>2} {'PISA':>7} {'shipped':>8} {'refit':>7} "
          f"{'d_ship':>7}")
    shipped_err, refit_err = [], []
    for pdb_id, k, term, y in rows:
        shipped = RT_KCAL * (term + k * STANDARD_STATE_LN_TERM)
        refitted = RT_KCAL * (term + k * constant)
        shipped_err.append(shipped - y)
        refit_err.append(refitted - y)
        print(f"{pdb_id:6} {k + 1:>2} {y:7.2f} {shipped:8.2f} {refitted:7.2f} "
              f"{shipped - y:+7.2f}")

    def report(label, errors):
        print(f"{label}: median |err| {statistics.median(map(abs, errors)):.2f}  "
              f"bias {statistics.mean(errors):+.2f}  "
              f"max {max(map(abs, errors)):.2f} kcal/mol")

    print()
    report("shipped", shipped_err)
    report("refit  ", refit_err)
    print(f"\nshipped STANDARD_STATE_LN_TERM = {STANDARD_STATE_LN_TERM:.4f}")
    print(f"refit                          = {constant:.4f}")
    print(f"ideal gas, 1 M standard state  = {IDEAL_1M:.4f} "
          f"({IDEAL_1M - constant:.2f} RT above the fit; free-volume factor "
          f"{math.exp(constant - IDEAL_1M):.3g})")
    if args.emit:
        print(f"\nSTANDARD_STATE_LN_TERM = {constant:.4f}")


if __name__ == "__main__":
    main()
