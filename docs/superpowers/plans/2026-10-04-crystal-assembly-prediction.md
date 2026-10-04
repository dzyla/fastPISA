# Crystal Assembly Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Predict which assembly a crystal forms in solution, by enumerating finite assemblies over the crystal contact graph and scoring them with the existing dissociation machinery, validated against PISA's own published assembly predictions.

**Architecture:** `symmetry="crystal"` already yields every crystal interface with its exact fractional placement. Task 2 turns those into a lazily-grown contact graph whose nodes are `(molecule, symop, cell)`. Task 3 grows connected components over nested subsets of interface types, rejecting any component that recurs at a shifted cell (infinite lattice). Task 4 dedupes and scores survivors with `energy.dissociation.assembly_dissociation`. Tasks 5–7 expose, validate and document.

**Tech Stack:** Python 3.10+, numpy, scipy; `gemmi` (already optional, needed only for space groups); pytest.

**Spec:** `docs/superpowers/specs/2026-10-04-crystal-assembly-prediction.md`

## Global Constraints

- Hard dependencies stay `numpy` + `scipy`. `gemmi` stays optional; guard its use with `pytest.importorskip("gemmi")` in tests.
- `requires-python = ">=3.10"`.
- No new fitted constant without a refit script and a drift-guard test.
- All reference data committed gzipped; the full validation runs offline.
- Default behaviour unchanged — assembly prediction is opt-in (`predict_assemblies=False`).
- Symmetry equivalence is decided on **exact integer** fractional placements (`frac_rotation`, `frac_translation`, denominator 24). Never on rounded floats, and never use a transpose as the inverse of a fractional rotation — use `fastpisa.core._integer_inverse`.
- Every new numeric claim in a docstring must be a measured value, with the measurement recorded in `tests/data/reference/assembly_validation.json`.
- Run `pytest tests/ -q` before each commit; it must stay at 275 passed / 12 skipped or better.

## Review Focus

Input classes the spec implies but no task's happy path exercises. Each has its test added to the owning task.

1. **A structure with no cell / no space group** (predicted model) reaches assembly prediction — must return an empty list, not raise. *(Task 4)*
2. **`total_asm = 0`** — PISA legitimately predicts no stable assembly (1brs). The reference parser must round-trip zero rather than treating it as a fetch error. *(Task 1)*
3. **A translationally infinite component** (a crystal where every molecule is bridged into one lattice) must be rejected without an unbounded loop, and the node cap must be provably reached rather than silently truncating a finite assembly. *(Task 3)*
4. **A single-molecule asymmetric unit with no interfaces at all** — enumeration must yield the trivial monomer assembly, not crash on an empty edge list. *(Task 3)*
5. **Ligand-only components.** With `ligand_mode="separate"` a lone ion is its own molecule; it must not be grown into an "assembly" of one ion ranked above the protein assembly. *(Task 4)*

---

### Task 1: PISA assembly reference — fetch, parse, cache

**Files:**
- Modify: `fastpisa/reference/ebi_pisa.py` (add after `fetch_pisa_xml`)
- Test: `tests/test_reference_multimers.py` (create)
- Data: `tests/data/reference/<pdbid>.multimers.xml.gz` (37 files, generated in Step 6)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `PISA_MULTIMERS_URL: str`
  - `fetch_pisa_multimers(pdb_id: str, cache_dir: str = REFERENCE_DIR, timeout: int = 120) -> str`
  - `parse_pisa_multimers(text: str) -> dict` returning
    `{"pdb_id": str, "total_asm": int, "assemblies": [ {...} ]}` where each
    assembly dict has keys `set_no:int, id:int, size:int, mmsize:int,
    formula:str, composition:str, diss_energy:float|None,
    entropy:float|None, diss_area:float|None, int_energy:float|None,
    symmetry_number:int, n_diss:int, r350:int, score:str,
    molecules:[{"chain_id":str,"rotation":[[float]*3]*3,"translation":[float]*3}]`
  - `load_cached_multimers(pdb_id: str, cache_dir: str = REFERENCE_DIR) -> dict | None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_reference_multimers.py`:

```python
"""PISA's own assembly predictions, the ground truth for Task 6.

Published per entry at multimers.pisa (verified 2026-10-04). Two fields make
this more than a self-comparison: ``r350`` is non-zero when PISA's assembly
matches the author-deposited REMARK-350 assembly, and ``score`` is PISA's
text verdict. ``total_asm == 0`` is a legitimate answer -- PISA finds no
stable assembly for barnase-barstar (1brs) -- and must not look like a
fetch failure.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastpisa.reference.ebi_pisa import load_cached_multimers, parse_pisa_multimers

MINIMAL = """<pisa_multimers>
  <status>Ok</status>
  <pdb_entry>
    <pdb_code>1acb</pdb_code>
    <status>Ok</status>
    <total_asm>1</total_asm>
    <asm_set>
      <ser_no>1</ser_no>
      <assembly>
        <id>1</id><size>2</size><mmsize>2</mmsize>
        <score>This assembly appears to be stable in solution.</score>
        <diss_energy>5.5696952172</diss_energy>
        <entropy>10.679620819</entropy>
        <diss_area>770.846279</diss_area>
        <int_energy>-12.252981565</int_energy>
        <n_diss>2</n_diss><symNumber>1</symNumber><R350>1</R350>
        <formula>AB</formula><composition>EI</composition>
        <molecule>
          <chain_id>E</chain_id>
          <rxx>1</rxx><rxy>0</rxy><rxz>0</rxz><tx>0</tx>
          <ryx>0</ryx><ryy>1</ryy><ryz>0</ryz><ty>0</ty>
          <rzx>0</rzx><rzy>0</rzy><rzz>1</rzz><tz>0</tz>
        </molecule>
      </assembly>
    </asm_set>
  </pdb_entry>
</pisa_multimers>"""

EMPTY = """<pisa_multimers>
  <status>Ok</status>
  <pdb_entry>
    <pdb_code>1brs</pdb_code><status>Ok</status><total_asm>0</total_asm>
  </pdb_entry>
</pisa_multimers>"""


def test_parses_one_assembly():
    doc = parse_pisa_multimers(MINIMAL)
    assert doc["pdb_id"] == "1acb"
    assert doc["total_asm"] == 1
    assert len(doc["assemblies"]) == 1
    asm = doc["assemblies"][0]
    assert asm["set_no"] == 1 and asm["id"] == 1
    assert asm["size"] == 2 and asm["mmsize"] == 2
    assert asm["formula"] == "AB" and asm["composition"] == "EI"
    assert asm["diss_energy"] == pytest.approx(5.5696952172)
    assert asm["entropy"] == pytest.approx(10.679620819)
    assert asm["symmetry_number"] == 1
    assert asm["r350"] == 1
    assert "stable in solution" in asm["score"]
    assert np.allclose(asm["molecules"][0]["rotation"], np.eye(3))
    assert asm["molecules"][0]["chain_id"] == "E"


def test_no_predicted_assembly_is_a_valid_answer():
    """1brs: PISA predicts nothing stable. Zero must round-trip."""
    doc = parse_pisa_multimers(EMPTY)
    assert doc["total_asm"] == 0
    assert doc["assemblies"] == []


def test_cached_reference_is_available_offline():
    doc = load_cached_multimers("1acb")
    assert doc is not None, "commit tests/data/reference/1acb.multimers.xml.gz"
    assert doc["total_asm"] >= 1
    assert doc["assemblies"][0]["composition"]


def test_a_missing_entry_returns_none():
    assert load_cached_multimers("zzzz") is None


def test_every_cached_entry_parses():
    import glob
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    paths = glob.glob(os.path.join(here, "data", "reference", "*.multimers.xml.gz"))
    assert len(paths) >= 30, f"only {len(paths)} cached multimer references"
    for path in paths:
        pdb_id = os.path.basename(path)[:4]
        doc = load_cached_multimers(pdb_id)
        assert doc is not None and doc["pdb_id"] == pdb_id
        assert len(doc["assemblies"]) >= 0
        for asm in doc["assemblies"]:
            assert asm["size"] >= 1
            assert asm["mmsize"] <= asm["size"]
            assert asm["molecules"], f"{pdb_id} assembly {asm['id']} has no molecules"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reference_multimers.py -q`
Expected: FAIL at collection — `ImportError: cannot import name 'load_cached_multimers'`.

- [ ] **Step 3: Implement fetch, parse and cache**

In `fastpisa/reference/ebi_pisa.py`, add next to `EBI_PISA_URL`:

```python
#: PISA's own assembly predictions for a deposited entry. Same frozen
#: classic-CGI database as the interface list, so the two agree by
#: construction for any entry both cover.
PISA_MULTIMERS_URL = (
    "https://www.ebi.ac.uk/pdbe/pisa/cgi-bin/multimers.pisa?{pdbid}")
```

and, after `fetch_pisa_xml`:

```python
def fetch_pisa_multimers(pdb_id: str, cache_dir: str = REFERENCE_DIR,
                         timeout: int = 120) -> str:
    """Return PISA's predicted-assembly XML for ``pdb_id`` (cached, gzipped).

    ``total_asm == 0`` is cached like any other answer: PISA genuinely
    predicts no stable assembly for some entries (1brs), and treating that
    as a failure would re-fetch it forever.
    """
    pdb_id = pdb_id.lower()
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{pdb_id}.multimers.xml.gz")
    if os.path.exists(path):
        with gzip.open(path, "rt") as fh:
            return fh.read()
    url = PISA_MULTIMERS_URL.format(pdbid=pdb_id)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    if "<pisa_multimers>" not in text:
        raise RuntimeError(f"EBI PISA returned no multimer XML for {pdb_id}")
    root = ET.fromstring(text)
    status = root.findtext("status")
    if status and status.strip().lower() != "ok":
        raise RuntimeError(f"EBI PISA status {status!r} for {pdb_id}")
    entry_status = root.findtext("pdb_entry/status")
    if entry_status and entry_status.strip().lower() != "ok":
        raise RuntimeError(
            f"EBI PISA (classic CGI) has no assembly data for {pdb_id}: "
            f"{entry_status}")
    with gzip.open(path, "wt") as fh:
        fh.write(text)
    return text


def _multimer_molecule(el) -> dict:
    rotation = [[_f(el, f"r{row}{col}", 0.0) or 0.0 for col in "xyz"]
                for row in "xyz"]
    translation = [_f(el, t, 0.0) or 0.0 for t in ("tx", "ty", "tz")]
    return {"chain_id": _s(el, "chain_id"),
            "rotation": rotation,
            "translation": translation}


def parse_pisa_multimers(text: str) -> dict:
    """Parse PISA's multimer XML into ``{pdb_id, total_asm, assemblies}``."""
    root = ET.fromstring(text)
    out = {
        "pdb_id": (root.findtext("pdb_entry/pdb_code") or "").lower(),
        "total_asm": int(float(root.findtext("pdb_entry/total_asm") or 0)),
        "assemblies": [],
    }
    for asm_set in root.iter("asm_set"):
        set_no = int(float(asm_set.findtext("ser_no") or 0))
        for asm in asm_set.findall("assembly"):
            out["assemblies"].append({
                "set_no": set_no,
                "id": int(float(_f(asm, "id", 0) or 0)),
                "size": int(float(_f(asm, "size", 0) or 0)),
                "mmsize": int(float(_f(asm, "mmsize", 0) or 0)),
                "formula": _s(asm, "formula"),
                "composition": _s(asm, "composition"),
                "diss_energy": _f(asm, "diss_energy"),
                "entropy": _f(asm, "entropy"),
                "diss_area": _f(asm, "diss_area"),
                "int_energy": _f(asm, "int_energy"),
                "symmetry_number": int(float(_f(asm, "symNumber", 1) or 1)),
                "n_diss": int(float(_f(asm, "n_diss", 0) or 0)),
                "r350": int(float(_f(asm, "R350", 0) or 0)),
                "score": " ".join((_s(asm, "score") or "").split()),
                "molecules": [_multimer_molecule(m)
                              for m in asm.findall("molecule")],
            })
    return out


def load_cached_multimers(pdb_id: str,
                          cache_dir: str = REFERENCE_DIR) -> Optional[dict]:
    """Parsed multimer reference from the cache, or None if not cached."""
    path = os.path.join(cache_dir, f"{pdb_id.lower()}.multimers.xml.gz")
    if not os.path.exists(path):
        return None
    with gzip.open(path, "rt") as fh:
        return parse_pisa_multimers(fh.read())
```

- [ ] **Step 4: Run the two parser tests to verify they pass**

Run: `pytest tests/test_reference_multimers.py -q -k "parses_one or valid_answer or missing_entry"`
Expected: PASS (3 tests).

- [ ] **Step 5: Verify against the live service for one entry**

Run:
```bash
python -c "
from fastpisa.reference.ebi_pisa import fetch_pisa_multimers, parse_pisa_multimers
d = parse_pisa_multimers(fetch_pisa_multimers('2ptc'))
print(d['total_asm'], [(a['id'], a['composition'], a['mmsize'], a['r350']) for a in d['assemblies']])"
```
Expected: `2 [(1, 'E[4]I[4][CA][4]', 8, 2), (2, 'EI[CA]', 2, 1)]`.
If the composition strings differ, the parser is wrong — fix it rather than editing the expectation.

- [ ] **Step 6: Populate the offline cache for every reference entry**

Run:
```bash
python - <<'EOF'
import glob, os
from fastpisa.reference.ebi_pisa import fetch_pisa_multimers
ids = sorted({os.path.basename(p)[:4]
              for p in glob.glob("tests/data/reference/*.pisa.xml.gz")})
for pdb_id in ids:
    try:
        fetch_pisa_multimers(pdb_id)
        print("ok", pdb_id)
    except Exception as exc:
        print("FAILED", pdb_id, type(exc).__name__, exc)
EOF
du -sh tests/data/reference/*.multimers.xml.gz | tail -1
ls tests/data/reference/*.multimers.xml.gz | wc -l
```
Expected: 37 files, a few hundred KB in total. Any entry that fails is recorded in the Task 6 notes, not silently skipped.

- [ ] **Step 7: Run the whole new test file, then the suite**

Run: `pytest tests/test_reference_multimers.py -q && pytest tests/ -q`
Expected: 5 passed in the new file; suite at 280 passed / 12 skipped.

- [ ] **Step 8: Commit**

```bash
git add fastpisa/reference/ebi_pisa.py tests/test_reference_multimers.py \
        tests/data/reference/*.multimers.xml.gz \
        docs/superpowers/specs/2026-10-04-crystal-assembly-prediction.md \
        docs/superpowers/plans/2026-10-04-crystal-assembly-prediction.md
git commit -m "ref: cache PISA's own assembly predictions as ground truth

multimers.pisa gives PISA's predicted assemblies with composition, mmsize,
dissociation energy and R350 (author-assembly agreement). Cached for the 37
reference entries so assembly validation runs offline. total_asm=0 is a real
answer (1brs) and round-trips.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Crystal contact graph

**Files:**
- Create: `fastpisa/assembly/graph.py`
- Test: `tests/test_assembly_graph.py` (create)

**Interfaces:**
- Consumes: `fastpisa.core.run_core(..., symmetry="crystal")` → `CoreState`, whose `interfaces[i].molecules[j]` carry `asu_molecule_id`, `symop_no`, `cell`, `frac_rotation`, `frac_translation`, `frac_denominator`; and `fastpisa.core._integer_inverse`.
- Produces:
  - `Placement` — frozen dataclass `(rotation: tuple, translation: tuple, denominator: int)` with `compose(other) -> Placement`, `inverse() -> Placement`, `key() -> tuple`, and `IDENTITY_PLACEMENT`.
  - `Node = Tuple[str, Placement]` — a molecule id plus where it sits.
  - `ContactEdge` — frozen dataclass `(molecule_a: str, molecule_b: str, placement: Placement, stabilization: float, area: float, interface_id: int)`.
  - `contact_edges(state) -> List[ContactEdge]` — one edge per distinct crystal interface, sorted most-stabilising first.
  - `neighbours(node, edges) -> List[Tuple[Node, ContactEdge]]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_assembly_graph.py`:

```python
"""The crystal contact graph: nodes are placed molecules, edges are interfaces.

An edge records the RELATIVE placement between its two molecules, so it can be
applied at any node to reach that node's partner -- which is what lets a finite
assembly be grown from a seed. Placement arithmetic is exact integer work on
fractional coordinates: a fractional rotation is not orthogonal, so its
inverse is not its transpose (in hexagonal axes the 3-fold's inverse is the
other 3-fold), and rounding orthogonal translations puts a 6-fold screw's
c/6 = 42.55 A on a rounding boundary.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

gemmi = pytest.importorskip("gemmi")

from fastpisa.assembly.graph import (  # noqa: E402
    IDENTITY_PLACEMENT, ContactEdge, Placement, contact_edges, neighbours,
)
from fastpisa.core import run_core  # noqa: E402

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data", "reference")

THREE_FOLD = Placement(rotation=((0, -1, 0), (1, -1, 0), (0, 0, 1)),
                       translation=(0, 0, 8), denominator=24)


def test_identity_placement_is_neutral():
    assert THREE_FOLD.compose(IDENTITY_PLACEMENT) == THREE_FOLD
    assert IDENTITY_PLACEMENT.compose(THREE_FOLD) == THREE_FOLD


def test_inverse_uses_the_integer_inverse_not_the_transpose():
    """R^-1 of the hexagonal 3-fold is the other 3-fold, not R^T."""
    inverse = THREE_FOLD.inverse()
    rot = np.asarray(THREE_FOLD.rotation)
    assert not np.array_equal(np.asarray(inverse.rotation), rot.T)
    assert THREE_FOLD.compose(inverse) == IDENTITY_PLACEMENT
    assert inverse.compose(THREE_FOLD) == IDENTITY_PLACEMENT


def test_composition_is_exact_and_associative():
    squared = THREE_FOLD.compose(THREE_FOLD)
    cubed = squared.compose(THREE_FOLD)
    assert cubed.rotation == IDENTITY_PLACEMENT.rotation
    # Three applications of a +1/3 screw advance exactly one cell.
    assert cubed.translation == (0, 0, 24)
    assert all(isinstance(v, int) for v in cubed.translation)


def test_key_separates_the_rotation_from_the_lattice_part():
    """Growth needs 'same operation, different cell' to be detectable."""
    shifted = Placement(rotation=THREE_FOLD.rotation,
                        translation=(24, 0, 8), denominator=24)
    assert shifted.key() != THREE_FOLD.key()
    assert shifted.rotation == THREE_FOLD.rotation
    assert shifted.cell() != THREE_FOLD.cell()
    assert shifted.cell() == (1, 0, 0)


def test_edges_come_from_crystal_interfaces_sorted_by_stability():
    state = run_core(os.path.join(REF, "pdb", "1acb.pdb.gz"),
                     mode="pisa", symmetry="crystal")
    edges = contact_edges(state)
    assert len(edges) == len(state.interfaces)
    assert all(isinstance(e, ContactEdge) for e in edges)
    stabilities = [e.stabilization for e in edges]
    assert stabilities == sorted(stabilities)
    assert all(e.placement.denominator == 24 for e in edges)
    assert {e.molecule_a for e in edges} <= {"E", "I"}


def test_an_edge_applied_at_a_node_reaches_its_partner():
    state = run_core(os.path.join(REF, "pdb", "1acb.pdb.gz"),
                     mode="pisa", symmetry="crystal")
    edges = contact_edges(state)
    seed = ("E", IDENTITY_PLACEMENT)
    found = neighbours(seed, edges)
    assert found, "chain E must have crystal neighbours"
    for (molecule, placement), edge in found:
        assert molecule in ("E", "I")
        assert isinstance(placement, Placement)
    # Applying an edge then its inverse returns to the seed.
    (molecule, placement), edge = found[0]
    back = neighbours((molecule, placement), edges)
    assert any(node == seed for node, _ in back)


def test_a_structure_without_symmetry_yields_only_asu_edges():
    state = run_core(os.path.join(os.path.dirname(REF), "1ktz.pdb"),
                     mode="pisa")
    edges = contact_edges(state)
    assert edges
    assert all(e.placement == IDENTITY_PLACEMENT for e in edges)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_assembly_graph.py -q`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'fastpisa.assembly.graph'`.

- [ ] **Step 3: Implement the graph**

Create `fastpisa/assembly/graph.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_assembly_graph.py -q`
Expected: PASS (7 tests). If `test_an_edge_applied_at_a_node_reaches_its_partner` fails, the composition order in `contact_edges` is reversed — check that `place_a.inverse().compose(place_b)` is what maps A's frame to B's.

- [ ] **Step 5: Run the suite**

Run: `pytest tests/ -q`
Expected: 287 passed / 12 skipped.

- [ ] **Step 6: Commit**

```bash
git add fastpisa/assembly/graph.py tests/test_assembly_graph.py
git commit -m "feat: crystal contact graph of placed molecules

Nodes are (molecule, exact fractional placement); edges are crystal
interfaces carrying the relative placement, so an edge applies at any node.
Infinite and grown lazily. Placement arithmetic is exact integer work --
a fractional rotation's inverse is not its transpose.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Finite-assembly growth

**Files:**
- Create: `fastpisa/assembly/predict.py`
- Test: `tests/test_assembly_growth.py` (create)

**Interfaces:**
- Consumes: `Placement`, `IDENTITY_PLACEMENT`, `Node`, `ContactEdge`, `neighbours` from Task 2.
- Produces:
  - `MAX_ASSEMBLY_NODES: int = 512`
  - `GrowthResult` — frozen dataclass `(nodes: tuple[Node, ...], edges: tuple[ContactEdge, ...], finite: bool, reason: str)`
  - `grow(seed: Node, edges: Sequence[ContactEdge], max_nodes: int = MAX_ASSEMBLY_NODES) -> GrowthResult`

- [ ] **Step 1: Write the failing test**

Create `tests/test_assembly_growth.py`:

```python
"""Growing a connected component, and knowing when it never stops.

A set of interfaces either closes into a finite assembly or tiles the crystal.
The discriminator: if growth reaches the same molecule under the same
ROTATION but at a different lattice cell, the component repeats
translationally -- it is a lattice, a sheet or a fibre, not an assembly. A
missed infinite case is an unbounded loop, so a hard node cap backs the test
up and says which guard stopped it.
"""

from __future__ import annotations

from fastpisa.assembly.graph import (
    IDENTITY_PLACEMENT, ContactEdge, Placement,
)
from fastpisa.assembly.predict import MAX_ASSEMBLY_NODES, grow

DEN = 24


def _placement(rotation, translation):
    return Placement(rotation=rotation, translation=translation,
                     denominator=DEN)


#: A 2-fold: -x, y, -z. Self-inverse, so A + mate(A) closes into a dimer.
TWO_FOLD = _placement(((-1, 0, 0), (0, 1, 0), (0, 0, -1)), (0, 0, 0))

#: A pure lattice translation along a: tiles the crystal forever.
TRANSLATION = _placement(((1, 0, 0), (0, 1, 0), (0, 0, 1)), (DEN, 0, 0))

#: A 3-fold: closes into a trimer after three applications.
THREE_FOLD = _placement(((0, -1, 0), (1, -1, 0), (0, 0, 1)), (0, 0, 0))


def _edge(a, b, placement, stab=-10.0):
    return ContactEdge(molecule_a=a, molecule_b=b, placement=placement,
                       stabilization=stab, area=500.0, interface_id=1)


def test_no_edges_gives_the_lone_seed():
    result = grow(("A", IDENTITY_PLACEMENT), [])
    assert result.finite is True
    assert len(result.nodes) == 1
    assert result.nodes[0] == ("A", IDENTITY_PLACEMENT)
    assert result.edges == ()


def test_a_two_fold_closes_into_a_dimer():
    result = grow(("A", IDENTITY_PLACEMENT), [_edge("A", "A", TWO_FOLD)])
    assert result.finite is True
    assert len(result.nodes) == 2
    assert {m for m, _ in result.nodes} == {"A"}


def test_a_three_fold_closes_into_a_trimer():
    result = grow(("A", IDENTITY_PLACEMENT), [_edge("A", "A", THREE_FOLD)])
    assert result.finite is True
    assert len(result.nodes) == 3


def test_a_heterodimer_interface_closes():
    result = grow(("A", IDENTITY_PLACEMENT),
                  [_edge("A", "B", IDENTITY_PLACEMENT)])
    assert result.finite is True
    assert {m for m, _ in result.nodes} == {"A", "B"}
    assert len(result.nodes) == 2


def test_a_pure_lattice_translation_is_infinite():
    """The core discrimination: same molecule, same rotation, other cell."""
    result = grow(("A", IDENTITY_PLACEMENT), [_edge("A", "A", TRANSLATION)])
    assert result.finite is False
    assert "translation" in result.reason
    assert len(result.nodes) <= MAX_ASSEMBLY_NODES


def test_a_screw_axis_is_infinite():
    """A 2-fold screw advances along its axis forever."""
    screw = _placement(((-1, 0, 0), (0, 1, 0), (0, 0, -1)), (0, DEN // 2, 0))
    result = grow(("A", IDENTITY_PLACEMENT), [_edge("A", "A", screw)])
    assert result.finite is False


def test_a_finite_assembly_plus_a_lattice_edge_is_infinite():
    """One bridging contact turns a closed dimer into a sheet."""
    result = grow(("A", IDENTITY_PLACEMENT),
                  [_edge("A", "A", TWO_FOLD), _edge("A", "A", TRANSLATION)])
    assert result.finite is False


def test_the_node_cap_is_the_backstop_not_the_primary_guard():
    """An infinite case must be caught by the rotation test, not the cap."""
    result = grow(("A", IDENTITY_PLACEMENT), [_edge("A", "A", TRANSLATION)],
                  max_nodes=MAX_ASSEMBLY_NODES)
    assert result.finite is False
    assert "translation" in result.reason
    assert len(result.nodes) < 10, (
        "the cap stopped growth instead of the translational test")


def test_the_cap_still_rejects_an_unexpectedly_large_component():
    """Defence in depth: a huge finite component is reported, not returned."""
    edges = [_edge("A", "A", THREE_FOLD)]
    result = grow(("A", IDENTITY_PLACEMENT), edges, max_nodes=2)
    assert result.finite is False
    assert "cap" in result.reason


def test_growth_is_deterministic():
    edges = [_edge("A", "B", IDENTITY_PLACEMENT), _edge("A", "A", TWO_FOLD)]
    first = grow(("A", IDENTITY_PLACEMENT), edges)
    second = grow(("A", IDENTITY_PLACEMENT), edges)
    assert first.nodes == second.nodes
    assert first.finite == second.finite


def test_the_edges_used_are_reported():
    edges = [_edge("A", "B", IDENTITY_PLACEMENT, stab=-20.0),
             _edge("A", "A", TWO_FOLD, stab=-5.0)]
    result = grow(("A", IDENTITY_PLACEMENT), edges)
    assert result.finite is True
    assert len(result.edges) >= 1
    assert all(isinstance(e, ContactEdge) for e in result.edges)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_assembly_growth.py -q`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'fastpisa.assembly.predict'`.

- [ ] **Step 3: Implement growth**

Create `fastpisa/assembly/predict.py`:

```python
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
            used.setdefault(id(edge), edge)
            identity = (molecule, placement.key())
            if identity in visited:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_assembly_growth.py -q`
Expected: PASS (11 tests).

If `test_the_edges_used_are_reported` fails because `used` collects edges that
were examined rather than traversed, move the `used.setdefault(...)` line to
after the `if identity in visited: continue` guard — an edge counts as used
only when it is actually followed to a new node *or* closes the component.
Then re-run; `test_a_two_fold_closes_into_a_dimer` must still pass.

- [ ] **Step 5: Run the suite**

Run: `pytest tests/ -q`
Expected: 298 passed / 12 skipped.

- [ ] **Step 6: Commit**

```bash
git add fastpisa/assembly/predict.py tests/test_assembly_growth.py
git commit -m "feat: grow crystal components, rejecting infinite ones

A set of interfaces either closes into a finite assembly or tiles the
crystal. Reaching the same molecule under the same rotation at a different
cell means translational repetition -- a lattice, not an assembly. The node
cap is a backstop and the result says which guard fired.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Enumerate, dedupe, score and rank assemblies

**Files:**
- Modify: `fastpisa/assembly/predict.py` (append)
- Test: `tests/test_assembly_predict.py` (create)

**Interfaces:**
- Consumes: `grow`, `GrowthResult`, `MAX_ASSEMBLY_NODES` (Task 3); `contact_edges` (Task 2); `fastpisa.energy.dissociation.assembly_dissociation`, `fastpisa.energy.entropy.atoms_mass`.
- Produces:
  - `Assembly` — dataclass `(rank: int, nodes: tuple, size: int, mmsize: int, composition: str, formula: str, dissociation_energy: float, entropy: float, interface_ids: tuple, n_interfaces: int)`
  - `predict_assemblies(state, max_nodes: int = MAX_ASSEMBLY_NODES) -> List[Assembly]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_assembly_predict.py`:

```python
"""Enumerating the assemblies a crystal can form, then ranking them.

Candidates come from NESTED interface subsets: sort the crystal's interfaces
most-stabilising first and, for k = 1..n, grow components using only the top
k. That is PISA's dissociation ordering, it is bounded at n iterations, and it
avoids the 2^n subset search. Each survivor is scored with the same
minimum-cut dissociation machinery the assembly document already uses.
"""

from __future__ import annotations

import os

import pytest

gemmi = pytest.importorskip("gemmi")

from fastpisa.assembly.predict import Assembly, predict_assemblies  # noqa: E402
from fastpisa.core import run_core  # noqa: E402

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data", "reference")


def _crystal(pdb_id):
    return run_core(os.path.join(REF, "pdb", f"{pdb_id}.pdb.gz"),
                    mode="pisa", symmetry="crystal")


def test_1acb_predicts_the_heterodimer_pisa_predicts():
    """PISA: one assembly, composition EI, mmsize 2, matching REMARK 350."""
    assemblies = predict_assemblies(_crystal("1acb"))
    assert assemblies
    top = assemblies[0]
    assert isinstance(top, Assembly)
    assert top.mmsize == 2
    assert set(top.composition.replace("[", "").replace("]", "")) >= {"E", "I"}
    assert top.dissociation_energy > 0


def test_assemblies_are_ranked_by_dissociation_energy():
    assemblies = predict_assemblies(_crystal("1a3n"))
    energies = [a.dissociation_energy for a in assemblies]
    assert energies == sorted(energies, reverse=True)
    assert [a.rank for a in assemblies] == list(range(1, len(assemblies) + 1))


def test_duplicate_assemblies_are_collapsed():
    assemblies = predict_assemblies(_crystal("1acb"))
    seen = {tuple(sorted(a.composition)) + (a.size,) for a in assemblies}
    assert len(seen) == len(assemblies)


def test_no_assembly_is_infinite_or_oversized():
    for pdb_id in ("1acb", "1ktz", "2ptc"):
        for assembly in predict_assemblies(_crystal(pdb_id)):
            assert 1 <= assembly.size <= 512
            assert assembly.size == len(assembly.nodes)
            assert assembly.mmsize <= assembly.size


def test_a_model_without_a_cell_predicts_from_its_own_coordinates():
    """Review Focus 1: a predicted model has no symmetry; do not raise."""
    state = run_core(os.path.join(os.path.dirname(REF), "1ktz.pdb"),
                     mode="pisa")
    assemblies = predict_assemblies(state)
    assert isinstance(assemblies, list)
    for assembly in assemblies:
        assert assembly.size >= 1


def test_a_lone_molecule_yields_the_monomer(tmp_path):
    """Review Focus 4: no interfaces at all must not crash enumeration."""
    path = tmp_path / "single.pdb"
    path.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00"
        "           C\nEND\n")
    assemblies = predict_assemblies(run_core(str(path), mode="pisa"))
    assert len(assemblies) == 1
    assert assemblies[0].size == 1
    assert assemblies[0].mmsize == 0 or assemblies[0].mmsize == 1


def test_a_ligand_only_component_does_not_outrank_a_polymer_assembly():
    """Review Focus 5: a lone ion is not an assembly worth reporting."""
    assemblies = predict_assemblies(_crystal("2ptc"))
    assert assemblies
    top = assemblies[0]
    assert top.mmsize >= 2, (
        f"top assembly is {top.composition} with mmsize {top.mmsize}")
    assert all(a.mmsize >= 1 for a in assemblies), (
        "a component of ligands only must not be reported as an assembly")


def test_composition_and_formula_follow_pisa_shape():
    """PISA writes 'E[4]I[4][CA][4]' and formula 'A4B4a4'."""
    assemblies = predict_assemblies(_crystal("2ptc"))
    top = assemblies[0]
    assert top.composition
    assert top.formula
    assert top.formula[0].isalpha()
    # A homo-oligomer repeats a chain; the count appears in the composition.
    if top.size > top.mmsize:
        assert "[" in top.composition


def test_interface_ids_point_back_at_the_interfaces_used():
    state = _crystal("1acb")
    valid = {i.interface_id for i in state.interfaces}
    for assembly in predict_assemblies(state):
        assert set(assembly.interface_ids) <= valid
        assert assembly.n_interfaces == len(assembly.interface_ids)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_assembly_predict.py -q`
Expected: FAIL at collection — `ImportError: cannot import name 'Assembly'`.

- [ ] **Step 3: Implement enumeration, dedupe, scoring and ranking**

Append to `fastpisa/assembly/predict.py`:

```python
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
    index = {}
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

    masses = {}
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
    for letter, molecule in zip(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ", polymers):
        n = counts[molecule]
        out.append(letter if n == 1 else f"{letter}{n}")
    for letter, molecule in zip("abcdefghijklmnopqrstuvwxyz", ligands):
        n = counts[molecule]
        out.append(letter if n == 1 else f"{letter}{n}")
    return "".join(out)


def _score(nodes, edges, index, masses):
    """(dG_diss, T dS) of this assembly, via the shared dissociation search."""
    from fastpisa.energy.dissociation import assembly_dissociation

    node_mass = {}
    for position, (molecule, placement) in enumerate(nodes):
        node_mass[position] = masses.get(molecule, 0.0)
    node_of = {(molecule, placement.key()): position
               for position, (molecule, placement) in enumerate(nodes)}
    internal = []
    for position, (molecule, placement) in enumerate(nodes):
        for (partner, partner_placement), edge in neighbours(
                (molecule, placement), edges):
            other = node_of.get((partner, partner_placement.key()))
            if other is None or other <= position:
                continue
            internal.append((position, other, edge.stabilization))
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
            diss, entropy = _score(result.nodes, edges, index, masses)
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
                    e.interface_id for e in result.edges)),
                n_interfaces=len(result.edges),
            )

    # Collapse assemblies with identical molecule content, keeping the most
    # stable: two seeds in the same orbit give the same assembly placed
    # differently.
    by_content: Dict[tuple, Assembly] = {}
    for assembly in found.values():
        from collections import Counter

        key = (tuple(sorted(Counter(m for m, _ in assembly.nodes).items())),
               assembly.size)
        previous = by_content.get(key)
        if (previous is None
                or assembly.dissociation_energy > previous.dissociation_energy):
            by_content[key] = assembly

    ranked = sorted(by_content.values(),
                    key=lambda a: (-a.dissociation_energy, -a.size,
                                   a.composition))
    for position, assembly in enumerate(ranked, start=1):
        assembly.rank = position
    return ranked
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_assembly_predict.py -q`
Expected: PASS (9 tests).

If `test_a_lone_molecule_yields_the_monomer` fails with an empty list, the
`subsets = [0]` branch is producing `all_edges[:0]` for a structure that has
no interfaces — confirm `grow` is still called once per seed with an empty
edge list.

- [ ] **Step 5: Run the suite**

Run: `pytest tests/ -q`
Expected: 307 passed / 12 skipped.

- [ ] **Step 6: Commit**

```bash
git add fastpisa/assembly/predict.py tests/test_assembly_predict.py
git commit -m "feat: enumerate, dedupe, score and rank crystal assemblies

Candidates from nested interface subsets in PISA's dissociation order,
bounded at n iterations rather than 2^n. Survivors scored with the shared
minimum-cut dissociation machinery and ranked by dG_diss. Lattices and
ligand-only components are dropped.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Expose it — core, API, CLI, JSON

**Files:**
- Modify: `fastpisa/core.py` (`CoreState`, `run_core`, `build_documents`, `analyze`)
- Modify: `fastpisa/api.py` (`PISAInterfaceAnalyzer.__init__`, `analyze`, `analysis_provenance`)
- Modify: `fastpisa/cli.py` (argument + summary print)
- Modify: `fastpisa/output/json_output.py` (`build_assembly_json`)
- Test: `tests/test_assembly_output.py` (create)

**Interfaces:**
- Consumes: `predict_assemblies`, `Assembly` (Task 4).
- Produces:
  - `CoreState.assemblies: List[Assembly]` (default `[]`)
  - `run_core(..., predict_assemblies: bool = False)`
  - `analyze(..., predict_assemblies: bool = False)`
  - `PISAInterfaceAnalyzer(..., predict_assemblies: bool = False)` with `.assemblies`
  - assembly JSON key `"predicted_assemblies"`: list of
    `{rank, size, mmsize, composition, formula, dissociation_energy, entropy, n_interfaces, interface_ids, molecules:[{asu_molecule_id, symop_no, cell}]}`
  - CLI flag `--predict-assemblies`

- [ ] **Step 1: Write the failing test**

Create `tests/test_assembly_output.py`:

```python
"""Predicted assemblies must be reachable and serialisable.

Opt-in: the default run is unchanged, so no existing caller pays for the
search. Every number that leaves the package has to survive json.dump --
numpy scalars in the crystal placements have broken that before.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

gemmi = pytest.importorskip("gemmi")

import fastpisa  # noqa: E402
from fastpisa.api import PISAInterfaceAnalyzer  # noqa: E402
from fastpisa.core import run_core  # noqa: E402

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data", "reference")
ACB = os.path.join(REF, "pdb", "1acb.pdb.gz")


def test_prediction_is_off_by_default():
    state = run_core(ACB, mode="pisa", symmetry="crystal")
    assert state.assemblies == []


def test_prediction_populates_core_state():
    state = run_core(ACB, mode="pisa", symmetry="crystal",
                     predict_assemblies=True)
    assert state.assemblies
    assert state.assemblies[0].rank == 1


def test_assembly_json_carries_the_predictions():
    result = fastpisa.analyze(ACB, pdb_id="1acb", mode="pisa",
                              symmetry="crystal", predict_assemblies=True)
    doc = result.assembly_json["assembly"]
    assert "predicted_assemblies" in doc
    entries = doc["predicted_assemblies"]
    assert entries
    first = entries[0]
    assert set(first) >= {"rank", "size", "mmsize", "composition", "formula",
                          "dissociation_energy", "entropy", "n_interfaces",
                          "interface_ids", "molecules"}
    assert first["rank"] == 1
    for molecule in first["molecules"]:
        assert set(molecule) >= {"asu_molecule_id", "symop_no", "cell"}
    # Must survive serialisation: numpy scalars have broken this before.
    json.dumps(result.assembly_json)


def test_the_key_is_absent_when_prediction_is_off():
    result = fastpisa.analyze(ACB, pdb_id="1acb", mode="pisa",
                              symmetry="crystal")
    assert "predicted_assemblies" not in result.assembly_json["assembly"]


def test_analyzer_exposes_assemblies_and_provenance():
    analyzer = PISAInterfaceAnalyzer(ACB, pdb_id="1acb", mode="pisa",
                                     symmetry="crystal",
                                     predict_assemblies=True)
    analyzer.analyze()
    assert analyzer.assemblies
    assert analyzer.assemblies[0].mmsize == 2
    assert analyzer.analysis_provenance()["predict_assemblies"] is True


def test_cli_predicts_assemblies(tmp_path):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run(
        [sys.executable, "-m", "fastpisa.cli", ACB, "--pdb_id", "1acb",
         "--mode", "pisa", "--symmetry", "crystal", "--predict-assemblies",
         "-o", str(tmp_path)],
        capture_output=True, text=True, cwd=repo, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "assembl" in out.stdout.lower()
    doc = json.load(open(tmp_path / "1acb-assembly1.json"))
    assert doc["assembly"]["predicted_assemblies"][0]["mmsize"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_assembly_output.py -q`
Expected: FAIL — `AttributeError: 'CoreState' object has no attribute 'assemblies'`.

- [ ] **Step 3: Wire it through core**

In `fastpisa/core.py`, add to `CoreState`:

```python
    assemblies: List["Assembly"] = field(default_factory=list)
```

add the import beside the other assembly imports:

```python
from fastpisa.assembly.predict import Assembly, predict_assemblies as _predict
```

add the parameter to `run_core` (after `symmetry`):

```python
    predict_assemblies: bool = False,
```

document it in the `run_core` docstring, after the ``symmetry`` paragraph:

```
    ``predict_assemblies``: also enumerate the finite assemblies the crystal
    can form and rank them by dissociation energy
    (:mod:`fastpisa.assembly.predict`). Off by default -- it is a search over
    interface subsets and no existing caller should pay for it.
```

and populate it immediately before the `return CoreState(...)`:

```python
    assemblies: List[Assembly] = []
    if predict_assemblies:
        assemblies = _predict(_CoreStateView(
            atoms=atoms, molecules=molecules, masks=masks,
            interfaces=interfaces))
```

That view exists because `predict_assemblies` only needs four fields and
`CoreState` is built after it. Add it above `run_core`:

```python
@dataclass
class _CoreStateView:
    """The four fields assembly prediction reads, available mid-run."""
    atoms: list
    molecules: List[dict]
    masks: List[np.ndarray]
    interfaces: List[Interface]
```

then pass `assemblies=assemblies` in the `CoreState(...)` call.

In `build_documents`, after `pathway = dissociation_pathway(state)`:

```python
    predicted = [
        {
            "rank": a.rank,
            "size": a.size,
            "mmsize": a.mmsize,
            "composition": a.composition,
            "formula": a.formula,
            "dissociation_energy": round(a.dissociation_energy, 2),
            "entropy": round(a.entropy, 2),
            "n_interfaces": a.n_interfaces,
            "interface_ids": [int(i) for i in a.interface_ids],
            "molecules": [
                {"asu_molecule_id": molecule,
                 "symop_no": _symop_of(state, molecule, placement),
                 "cell": list(placement.cell())}
                for molecule, placement in a.nodes
            ],
        }
        for a in getattr(state, "assemblies", [])
    ]
```

and pass `predicted_assemblies=predicted` into `build_assembly_json`. Add the
helper beside `dissociation_pathway`:

```python
def _symop_of(state, molecule_id: str, placement) -> int:
    """Symmetry-operation number of a placement, for the output document."""
    for iface in state.interfaces:
        for mol in iface.molecules:
            if (mol.get("asu_molecule_id") == molecule_id
                    and tuple(mol.get("frac_rotation", ())) == placement.rotation
                    and tuple(mol.get("frac_translation", ())) == placement.translation):
                return int(mol.get("symop_no", 1))
    return 1
```

Add `predict_assemblies: bool = False` to `analyze` and forward it to
`run_core`.

In `fastpisa/output/json_output.py`, add the parameter
`predicted_assemblies: Optional[List[dict]] = None` to
`build_assembly_json` and, before the return:

```python
    # Omitted entirely rather than emitted as null, so a consumer can test
    # membership -- the same convention the COCOMAPS contact map uses.
    if predicted_assemblies:
        document["assembly"]["predicted_assemblies"] = predicted_assemblies
```

(adjusting to whatever local name the assembled dict has).

In `fastpisa/api.py`: add `predict_assemblies: bool = False` to
`__init__`, store `self.predict_assemblies`, add
`self.assemblies: List = []` beside `self.interfaces`, forward it in the
`kwargs` dict in `analyze`, set
`self.assemblies = result.get("assemblies", [])`, and add
`"predict_assemblies": self.predict_assemblies` to
`analysis_provenance`. Have `core.analyze` include
`"assemblies": state.assemblies` in its returned dict.

In `fastpisa/cli.py`, beside `--symmetry`:

```python
    parser.add_argument(
        "--predict-assemblies", action="store_true",
        help="enumerate the assemblies the crystal can form and rank them by "
             "dissociation energy (implies a search over interface subsets; "
             "most useful with --symmetry crystal)",
    )
```

forward `predict_assemblies=args.predict_assemblies`, and in the summary:

```python
    if analyzer.assemblies:
        print("\nPredicted assemblies (most stable first):")
        for assembly in analyzer.assemblies[:5]:
            print(f"  {assembly.rank}. {assembly.composition} "
                  f"(size {assembly.size}, mm {assembly.mmsize})  "
                  f"dG_diss {assembly.dissociation_energy:+.2f} kcal/mol")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_assembly_output.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Run the suite**

Run: `pytest tests/ -q`
Expected: 313 passed / 12 skipped.

- [ ] **Step 6: Commit**

```bash
git add fastpisa/core.py fastpisa/api.py fastpisa/cli.py \
        fastpisa/output/json_output.py tests/test_assembly_output.py
git commit -m "feat: expose predicted assemblies via core, API, CLI and JSON

Opt-in (predict_assemblies=False by default), so no existing caller pays for
the subset search. The JSON key is omitted rather than null when off.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Validate against PISA, and record what it actually scores

**Files:**
- Modify: `fastpisa/reference/compare.py` (append)
- Create: `examples/validate_assemblies.py`
- Create: `tests/data/reference/assembly_validation.json`
- Test: `tests/test_assembly_vs_pisa.py` (create)

**Interfaces:**
- Consumes: `load_cached_multimers` (Task 1), `predict_assemblies` (Task 4).
- Produces:
  - `compare_assembly_entry(pdb_id: str, allow_fetch: bool = True) -> dict | None` returning
    `{"pdb_id", "reference_total", "predicted_total", "top_reference", "top_predicted", "top_mmsize_match": bool, "top_composition_match": bool, "author_assembly_match": bool|None, "recall": float, "rows": [...]}`
  - `summarize_assembly_predictions(results) -> dict`

- [ ] **Step 1: Write the failing test**

Create `tests/test_assembly_vs_pisa.py`:

```python
"""Assembly prediction against PISA's own published predictions.

The accuracy was unknown before it was measured, so this test asserts the
RECORDED number in tests/data/reference/assembly_validation.json rather than
an aspiration. If a change moves it, the recorded value is what has to be
re-measured and justified -- not the threshold quietly relaxed.

Two axes: agreement with PISA's top assembly, and agreement with the
author-deposited assembly (PISA's R350 marks which of its assemblies the
depositor asserted). The second is a claim about biology rather than about
reproducing PISA.
"""

from __future__ import annotations

import json
import os

import pytest

gemmi = pytest.importorskip("gemmi")

from fastpisa.reference.compare import (  # noqa: E402
    compare_assembly_entry, summarize_assembly_predictions,
)

RECORD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                      "reference", "assembly_validation.json")


def _recorded():
    with open(RECORD) as fh:
        return json.load(fh)


def test_a_single_entry_compares():
    result = compare_assembly_entry("1acb", allow_fetch=False)
    assert result is not None
    assert result["reference_total"] == 1
    assert result["top_reference"]["composition"] == "EI"
    assert result["predicted_total"] >= 1
    assert isinstance(result["top_mmsize_match"], bool)


def test_an_entry_with_no_predicted_assembly_is_handled():
    """1brs: PISA predicts nothing stable. Must compare, not crash."""
    result = compare_assembly_entry("1brs", allow_fetch=False)
    assert result is not None
    assert result["reference_total"] == 0
    assert result["top_reference"] is None


def test_an_uncached_entry_returns_none():
    assert compare_assembly_entry("zzzz", allow_fetch=False) is None


def test_measured_accuracy_matches_the_recorded_value():
    recorded = _recorded()
    results = [compare_assembly_entry(p, allow_fetch=False)
               for p in recorded["entries"]]
    results = [r for r in results if r is not None]
    assert len(results) == len(recorded["entries"])

    stats = summarize_assembly_predictions(results)
    expected = recorded["measured"]
    for key in ("top_mmsize_match_rate", "top_composition_match_rate",
                "author_assembly_match_rate", "mean_recall"):
        assert stats[key] == pytest.approx(expected[key], abs=0.02), (
            f"{key}: measured {stats[key]:.3f}, recorded "
            f"{expected[key]:.3f} -- re-measure and update the record "
            f"with the reason, do not relax this")


def test_the_record_states_an_honest_number():
    """The record must not claim perfection it has not earned."""
    recorded = _recorded()
    measured = recorded["measured"]
    assert 0.0 <= measured["top_mmsize_match_rate"] <= 1.0
    assert len(recorded["entries"]) >= 30
    assert recorded["note"], "the record must say what the number means"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_assembly_vs_pisa.py -q`
Expected: FAIL at collection — `ImportError: cannot import name 'compare_assembly_entry'`.

- [ ] **Step 3: Implement the comparison**

Append to `fastpisa/reference/compare.py`:

```python
def _composition_counts(composition: str) -> dict:
    """``'E[4]I[4][CA][4]'`` -> ``{'E': 4, 'I': 4, '[CA]': 4}``.

    PISA writes a bracketed CCD code for a hetero group and a bracketed
    integer for a repeat count, so the two uses of brackets have to be told
    apart: a bracket group whose contents are digits is a count for the token
    before it.
    """
    import re

    counts: dict = {}
    tokens = re.findall(r"\[[^\]]*\]|[A-Za-z0-9]", composition or "")
    current = None
    for token in tokens:
        if token.startswith("[") and token[1:-1].isdigit():
            if current is not None:
                counts[current] = counts.get(current, 0) + int(token[1:-1]) - 1
            continue
        current = token
        counts[current] = counts.get(current, 0) + 1
    return counts


def compare_assembly_entry(pdb_id: str,
                           allow_fetch: bool = True) -> Optional[dict]:
    """Compare predicted assemblies against PISA's own for one entry.

    Returns ``None`` when the multimer reference or the coordinates are not
    available. ``reference_total == 0`` is a real answer, not an error: PISA
    predicts no stable assembly for some entries.
    """
    from fastpisa.core import run_core
    from fastpisa.reference.ebi_pisa import (
        cached_pdb_path, fetch_pdb_file, fetch_pisa_multimers,
        load_cached_multimers, parse_pisa_multimers,
    )

    reference = load_cached_multimers(pdb_id)
    if reference is None:
        if not allow_fetch:
            return None
        try:
            reference = parse_pisa_multimers(fetch_pisa_multimers(pdb_id))
        except Exception:
            return None
    path = cached_pdb_path(pdb_id)
    if path is None:
        if not allow_fetch:
            return None
        try:
            path = fetch_pdb_file(pdb_id)
        except Exception:
            return None

    state = run_core(path, mode="pisa", symmetry="crystal",
                     predict_assemblies=True)
    predicted = state.assemblies

    # PISA's primary prediction is the first assembly of the first set.
    reference_assemblies = sorted(
        reference["assemblies"], key=lambda a: (a["set_no"], a["id"]))
    top_reference = reference_assemblies[0] if reference_assemblies else None
    top_predicted = predicted[0] if predicted else None

    mmsize_match = bool(
        top_reference and top_predicted
        and top_reference["mmsize"] == top_predicted.mmsize)
    composition_match = bool(
        top_reference and top_predicted
        and _composition_counts(top_reference["composition"])
        == _composition_counts(top_predicted.composition))

    # Did we reproduce PISA's assembly SET, by mmsize?
    reference_sizes = {a["mmsize"] for a in reference_assemblies}
    predicted_sizes = {a.mmsize for a in predicted}
    recall = (len(reference_sizes & predicted_sizes) / len(reference_sizes)
              if reference_sizes else 1.0)

    # The author-deposited assembly: PISA's R350 marks which of its own
    # assemblies the depositor asserted.
    author = next((a for a in reference_assemblies if a["r350"]), None)
    author_match = (None if author is None
                    else bool(top_predicted
                              and top_predicted.mmsize == author["mmsize"]))

    return {
        "pdb_id": pdb_id,
        "reference_total": reference["total_asm"],
        "predicted_total": len(predicted),
        "top_reference": top_reference,
        "top_predicted": (None if top_predicted is None else {
            "composition": top_predicted.composition,
            "formula": top_predicted.formula,
            "size": top_predicted.size,
            "mmsize": top_predicted.mmsize,
            "dissociation_energy": top_predicted.dissociation_energy,
        }),
        "top_mmsize_match": mmsize_match,
        "top_composition_match": composition_match,
        "author_assembly_match": author_match,
        "recall": recall,
        "rows": [
            {"pdb_id": pdb_id,
             "diss_ref": a["diss_energy"],
             "mmsize_ref": a["mmsize"]}
            for a in reference_assemblies
        ],
    }


def summarize_assembly_predictions(results: List[dict]) -> Dict[str, float]:
    """Match rates over several :func:`compare_assembly_entry` results."""
    comparable = [r for r in results if r["top_reference"] is not None]
    with_author = [r for r in results if r["author_assembly_match"] is not None]
    stats = {
        "n_entries": len(results),
        "n_comparable": len(comparable),
        "n_with_author_assembly": len(with_author),
    }
    stats["top_mmsize_match_rate"] = (
        sum(r["top_mmsize_match"] for r in comparable) / len(comparable)
        if comparable else float("nan"))
    stats["top_composition_match_rate"] = (
        sum(r["top_composition_match"] for r in comparable) / len(comparable)
        if comparable else float("nan"))
    stats["author_assembly_match_rate"] = (
        sum(r["author_assembly_match"] for r in with_author) / len(with_author)
        if with_author else float("nan"))
    stats["mean_recall"] = (
        float(np.mean([r["recall"] for r in comparable]))
        if comparable else float("nan"))
    return stats
```

- [ ] **Step 4: Verify the two unit-level tests pass**

Run: `pytest tests/test_assembly_vs_pisa.py -q -k "single_entry or no_predicted or uncached"`
Expected: PASS (3 tests). The two record-based tests still fail — the record does not exist yet.

- [ ] **Step 5: Write the validation script**

Create `examples/validate_assemblies.py`:

```python
#!/usr/bin/env python
"""Validate assembly prediction against PISA's own predictions.

    python examples/validate_assemblies.py              # cached entries, offline
    python examples/validate_assemblies.py --record     # write the record
    python examples/validate_assemblies.py --entries 1acb 4ins

Two axes are reported. Agreement with PISA's top assembly says whether we
reproduce PISA. Agreement with the author-deposited assembly (PISA's R350
flag marks which of its assemblies the depositor asserted) is a claim about
biology, and is the number worth improving.

``--record`` rewrites tests/data/reference/assembly_validation.json with the
measured values, which tests/test_assembly_vs_pisa.py then asserts. Run it
when the algorithm changes AND you can explain the movement.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from fastpisa.reference.compare import (  # noqa: E402
    compare_assembly_entry, summarize_assembly_predictions,
)

REFERENCE_DIR = os.path.join(REPO, "tests", "data", "reference")
RECORD = os.path.join(REFERENCE_DIR, "assembly_validation.json")


def cached_entries():
    out = []
    for path in sorted(glob.glob(os.path.join(REFERENCE_DIR,
                                              "*.multimers.xml.gz"))):
        pdb_id = os.path.basename(path)[:4]
        if os.path.exists(os.path.join(REFERENCE_DIR, "pdb",
                                       f"{pdb_id}.pdb.gz")):
            out.append(pdb_id)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entries", nargs="*", default=None)
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()

    entries = ([e.lower() for e in args.entries] if args.entries
               else cached_entries())
    print(f"{len(entries)} entries\n")
    results = []
    started = time.time()
    for pdb_id in entries:
        begin = time.time()
        result = compare_assembly_entry(pdb_id, allow_fetch=False)
        if result is None:
            print(f"{pdb_id}: no reference, skipped", flush=True)
            continue
        results.append(result)
        reference = result["top_reference"]
        predicted = result["top_predicted"]
        marks = ("mm" if result["top_mmsize_match"] else "  ") + \
                ("+comp" if result["top_composition_match"] else "     ")
        print(f"{pdb_id}: PISA {reference['composition'] if reference else '-':<18}"
              f"-> ours {predicted['composition'] if predicted else '-':<18}"
              f"{marks}  recall {result['recall']:.2f}  "
              f"{time.time() - begin:5.1f}s", flush=True)

    stats = summarize_assembly_predictions(results)
    print(f"\n=== {stats['n_entries']} entries in {time.time() - started:.0f}s ===")
    print(f"top assembly, mmsize match      : "
          f"{100 * stats['top_mmsize_match_rate']:.1f}%  "
          f"({stats['n_comparable']} comparable)")
    print(f"top assembly, composition match : "
          f"{100 * stats['top_composition_match_rate']:.1f}%")
    print(f"author-deposited assembly match : "
          f"{100 * stats['author_assembly_match_rate']:.1f}%  "
          f"({stats['n_with_author_assembly']} entries)")
    print(f"recall of PISA's assembly sizes  : "
          f"{100 * stats['mean_recall']:.1f}%")

    if args.record:
        with open(RECORD, "w") as fh:
            json.dump({
                "purpose": ("Measured agreement of assembly prediction with "
                            "PISA's own predictions (multimers.pisa) and with "
                            "the author-deposited assembly (PISA's R350)."),
                "entries": [r["pdb_id"] for r in results],
                "measured": {
                    "top_mmsize_match_rate": stats["top_mmsize_match_rate"],
                    "top_composition_match_rate":
                        stats["top_composition_match_rate"],
                    "author_assembly_match_rate":
                        stats["author_assembly_match_rate"],
                    "mean_recall": stats["mean_recall"],
                },
                "note": ("Nested-interface-subset search in PISA's "
                         "dissociation order, not PISA's exhaustive search, "
                         "so exact agreement is not expected. Re-measure with "
                         "examples/validate_assemblies.py --record and explain "
                         "any movement."),
            }, fh, indent=1)
            fh.write("\n")
        print(f"\nrecorded -> {RECORD}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Measure and record**

Run:
```bash
python examples/validate_assemblies.py --record
```
Expected: a per-entry table and four percentages, then the record written.

**Read the numbers before continuing.** If `top_mmsize_match_rate` is below
about 0.5, do not tune the ranking against this set. Instead inspect three
disagreeing entries, write down the cause in the record's `note`, and report
it — the honest number is the deliverable. Likely causes, in order: PISA's
exhaustive subset search finding an assembly our nested ordering misses; our
ΔG_diss ranking preferring a larger assembly; and ligand handling changing
`mmsize`.

- [ ] **Step 7: Run the full new test file, then the suite**

Run: `pytest tests/test_assembly_vs_pisa.py -q && pytest tests/ -q`
Expected: 5 passed in the new file; suite at 318 passed / 12 skipped.

- [ ] **Step 8: Commit**

```bash
git add fastpisa/reference/compare.py examples/validate_assemblies.py \
        tests/test_assembly_vs_pisa.py \
        tests/data/reference/assembly_validation.json
git commit -m "test: measure assembly prediction against PISA and the depositor

Two axes: PISA's top assembly (reproduction) and the author-deposited
assembly via PISA's R350 flag (biology). The measured rates are recorded and
asserted, so a change has to re-measure and justify rather than relax a
threshold.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Document it

**Files:**
- Modify: `CLAUDE.md` (new section before "Removed on purpose")
- Modify: `README.md` (after the crystal-symmetry section)
- Modify: `fastpisa/assembly/__init__.py` (the subpackage docstring says prediction is absent)

- [ ] **Step 1: Update the subpackage docstring**

In `fastpisa/assembly/__init__.py`, replace the paragraph beginning
"What IS here, in :mod:`fastpisa.energy.dissociation`" with:

```
What IS here: :mod:`fastpisa.assembly.crystal` generates the symmetry mates,
:mod:`fastpisa.assembly.graph` turns the crystal's interfaces into a contact
graph of placed molecules, and :mod:`fastpisa.assembly.predict` enumerates the
finite assemblies that graph admits and ranks them by the dissociation energy
of :mod:`fastpisa.energy.dissociation`. What is still absent is a biological-
versus-crystal verdict calibrated against an independent dataset; the
predicted assemblies are its input, not a substitute for it.
```

- [ ] **Step 2: Add the CLAUDE.md section**

Insert before `## Removed on purpose`, filling the bracketed numbers from
`tests/data/reference/assembly_validation.json`:

```markdown
## Assembly prediction (`predict_assemblies=True`)

Enumerates the finite assemblies a crystal admits and ranks them by
`ΔG_diss`. Opt-in; needs `symmetry="crystal"` to be useful.

- **Candidates come from NESTED interface subsets** in PISA's dissociation
  order (interfaces sorted most-stabilising first; grow with the top k for
  k = 1..n). Bounded at n iterations. An exhaustive 2^n search is a non-goal
  — `fastpisa/assembly/predict.py` says so.
- **Finite vs infinite is the hard part.** A component that reaches the same
  molecule under the same ROTATION at a different cell repeats
  translationally: a lattice, sheet or fibre, not an assembly. The node cap
  (`MAX_ASSEMBLY_NODES = 512`) is a backstop only, and `GrowthResult.reason`
  says which guard fired — `tests/test_assembly_growth.py` asserts the
  rotation test fires first.
- **Ligand-only components are dropped**: with `ligand_mode="separate"` a
  lone ion is its own molecule and "an assembly of one ion" is noise.
- Measured vs PISA's own predictions (`multimers.pisa`, cached for the 37
  reference entries): top-assembly mmsize match [X]%, composition match
  [Y]%, author-deposited assembly match [Z]%, recall of PISA's assembly
  sizes [W]%. Recorded in `tests/data/reference/assembly_validation.json`
  and asserted by `tests/test_assembly_vs_pisa.py`. Re-measure with
  `python examples/validate_assemblies.py --record`; do NOT relax the
  assertion instead.
- `total_asm = 0` is a real PISA answer (1brs — no stable assembly for
  barnase–barstar) and round-trips through the reference parser.
```

- [ ] **Step 3: Add the README section**

After the crystal-symmetry table, insert (numbers from the record):

```markdown
**Assembly prediction.** `--predict-assemblies` enumerates the finite
assemblies the crystal admits and ranks them by dissociation energy:

```bash
python -m fastpisa.cli 1acb.pdb --pdb_id 1acb --symmetry crystal \
    --predict-assemblies -o out/
```

Candidates come from nested interface subsets in PISA's dissociation order,
not an exhaustive search, so exact agreement with PISA is not expected.
Measured against PISA's own predictions over the 37 cached entries:
top-assembly stoichiometry match [X]%, composition match [Y]%, and agreement
with the **author-deposited** assembly [Z]%. Reproduce with
`python examples/validate_assemblies.py`.
```

- [ ] **Step 4: Verify the docs match the record**

Run:
```bash
python -c "
import json
rec = json.load(open('tests/data/reference/assembly_validation.json'))
print({k: round(100*v, 1) for k, v in rec['measured'].items()})"
grep -n "mmsize match" CLAUDE.md README.md
```
Expected: the printed percentages appear verbatim in both files, with no `[X]`
placeholders left.

- [ ] **Step 5: Run the suite**

Run: `pytest tests/ -q`
Expected: 318 passed / 12 skipped.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md README.md fastpisa/assembly/__init__.py
git commit -m "docs: assembly prediction, with the measured agreement

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## What follows this plan

Not in scope here, in the order they should be taken:

1. **QSbio calibration** — a biological-vs-crystal probability trained on
   QSbio's confidence-tiered labels, using these predicted assemblies as
   features. Needs an external dataset version-pinned into `tests/data/`.
   This is the step that would make fastPISA *better* than PISA rather than
   equal to it.
2. **Revisit CSS.** With real assemblies available, the fitted logistic
   surrogate in `scoring/scoring.py` can be replaced by a computed score.
   Do not touch it before step 1 exists to judge the replacement.
3. **Biological-assembly generation** from `_pdbx_struct_assembly_gen`, which
   would also stop `--assembly_id` being a label that builds nothing.
