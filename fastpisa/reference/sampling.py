"""Reproducible, non-redundant sampling of PDB entries for calibration.

The original 36-entry benchmark was hand-picked (protease-inhibitor,
antibody-antigen, ... ). Hand-picked sets are fine for *illustrating* coverage
but they cannot support a claim about accuracy on "the PDB", because the
selection is correlated with the thing being measured. This module draws the
validation/calibration entries from an explicit, stated sampling frame with a
fixed seed, so anyone can regenerate the identical entry list.

Sampling frame
--------------
RCSB search over ``experimental`` entries with

* ``exptl.method == X-RAY DIFFRACTION`` -- the classic EBI PISA CGI (our
  ground truth) holds the PISA run for deposited crystal entries;
* resolution <= :data:`MAX_RESOLUTION` -- below that, side-chain placement is
  unreliable enough that PISA's own H-bond list is not a stable target;
* >= 2 deposited polymer instances -- an interface must exist in the ASU;
* deposited atom count <= :data:`MAX_ATOMS` -- keeps a full-benchmark run to
  minutes rather than hours (a size cut, stated, not a quality cut);
* released on or before :data:`MAX_RELEASE_DATE` -- the classic CGI database
  is frozen; entries released after it return "Entry not found".

Redundancy control
------------------
The PDB is massively redundant (lysozyme, trypsin, ...). Sampling entries
uniformly would weight the fit toward whatever protein happens to be
over-deposited. We therefore sample over RCSB's **30% sequence-identity
clusters**, taking one representative entity per cluster, and then one entry
per sampled cluster. Interfaces within an entry remain correlated, which is
why every cross-validation in :mod:`fastpisa.reference.calibrate` groups by
entry.
"""

from __future__ import annotations

import json
import random
import urllib.request
from typing import List

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"

#: Frozen-CGI cutoff: entries released after this are not in the classic
#: EBI PISA database (probed 2026-09-01: 6cvm/2018 present, 7nhm/2021 absent).
MAX_RELEASE_DATE = "2018-06-30"
MAX_RESOLUTION = 3.0
MAX_ATOMS = 12000
SEQUENCE_IDENTITY_CUTOFF = 30

#: Seed for the reproducible draw. Changing it changes the benchmark.
SAMPLING_SEED = 20260901


def _frame_query() -> dict:
    return {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "exptl.method", "operator": "exact_match",
                "value": "X-RAY DIFFRACTION"}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal", "value": MAX_RESOLUTION}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.deposited_polymer_entity_instance_count",
                "operator": "greater_or_equal", "value": 2}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.deposited_atom_count",
                "operator": "less_or_equal", "value": MAX_ATOMS}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_accession_info.initial_release_date",
                "operator": "less_or_equal", "value": MAX_RELEASE_DATE}},
        ],
    }


def fetch_cluster_representatives(timeout: int = 300) -> List[str]:
    """Return one representative PDB entry ID per 30%-identity cluster.

    Network call to the RCSB search API. The returned order is RCSB's, which
    is deterministic for a fixed database snapshot but NOT stable across RCSB
    releases -- always pair it with :func:`sample_entries`, which sorts before
    sampling so the draw depends only on the *set* of representatives.
    """
    query = {
        "query": _frame_query(),
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": 0, "rows": 10000},
            "group_by": {"aggregation_method": "sequence_identity",
                         "similarity_cutoff": SEQUENCE_IDENTITY_CUTOFF},
            "group_by_return_type": "representatives",
            "results_content_type": ["experimental"],
        },
    }
    req = urllib.request.Request(
        RCSB_SEARCH_URL, data=json.dumps(query).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        doc = json.loads(resp.read().decode("utf-8"))
    # identifiers look like "1ABC_1" (entry_entity)
    seen, out = set(), []
    for r in doc.get("result_set", []):
        entry = r["identifier"].split("_")[0].lower()
        if entry not in seen:
            seen.add(entry)
            out.append(entry)
    return out


def sample_entries(representatives: List[str], n: int,
                   exclude: List[str] = (),
                   seed: int = SAMPLING_SEED) -> List[str]:
    """Draw ``n`` entries without replacement from ``representatives``.

    Sorting first makes the draw a pure function of the representative *set*
    and the seed, independent of RCSB's result ordering.
    """
    excl = {e.lower() for e in exclude}
    pool = sorted({r.lower() for r in representatives} - excl)
    rng = random.Random(seed)
    if n >= len(pool):
        return pool
    return sorted(rng.sample(pool, n))
