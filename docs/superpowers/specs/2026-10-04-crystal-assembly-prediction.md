# Crystal assembly prediction — design spec

**Date:** 2026-10-04
**Status:** proposed

## Problem

fastPISA now reproduces PISA's *interface* list for a deposited crystal entry
(`symmetry="crystal"`: 566/566 cached and 732/732 blind polymer–polymer
interfaces, 1.2% median area error, ΔG Pearson 0.995). It does not answer the
question PISA is actually used for:

> Given this crystal, which assembly exists in solution?

That gap is why two numbers in the package are still surrogates:

* `scoring.calculate_css_pisa` is a logistic fit to PISA's CSS
  (Spearman 0.73) because real CSS comes from PISA's assembly analysis.
* `scoring.classify_interface` is three hand-picked thresholds, deprecated,
  used by nothing.

The machinery to close it already exists and is validated:
`energy/dissociation.py` recovered PISA's exact relation
`ΔG_diss = −Σ(stab over the cut) − TΔS` (±0.01 kcal/mol) with a minimum-cut
search, and `assembly/crystal.py` supplies the lattice. What is missing is the
enumeration of candidate assemblies in the crystal.

## Ground truth

PISA's own assembly prediction is published per entry at

    https://www.ebi.ac.uk/pdbe/pisa/cgi-bin/multimers.pisa?<pdbid>

Verified 2026-10-04. Schema: `pisa_multimers/pdb_entry` with `total_asm` and
`asm_set[]`, each `asm_set` holding `assembly[]`, each assembly carrying

| field | meaning |
|---|---|
| `id`, `size`, `mmsize` | assembly rank, total molecules, macromolecules |
| `formula`, `composition` | e.g. `A2B2a4`, `ACBD[HEM][4]` |
| `diss_energy`, `entropy`, `diss_area`, `int_energy` | energetics |
| `symNumber`, `n_diss`, `n_uc` | symmetry number, dissociation count |
| `R350` | non-zero when it matches a REMARK-350 (author) assembly |
| `score` | PISA's text verdict ("stable in solution", "grey region", ...) |
| `molecule[]` | chain id + the transform that places it |

This gives two validation axes, one of them only weakly dependent on PISA's
energetics:

1. **vs PISA's prediction** — does our top assembly have PISA's composition
   and `mmsize`; do we recover PISA's assembly set; does ΔG_diss agree.
2. **vs the author-deposited assembly** — `R350` marks which of PISA's
   assemblies the depositor asserted. Agreement with that is a claim about
   biology, not about reproducing PISA.

Observed range in the reference set: 1acb → 1 assembly (`AB`, R350=1);
1a3n → 2 (`A2B2a4` R350=1, and the `ABa2` half); 4ins → 7 sets;
1ktz → 5, none matching R350; **1brs → `total_asm=0`** (PISA finds no stable
assembly for barnase–barstar). Zero is a legitimate answer and must round-trip.

## Requirements

### R1 — Crystal contact graph
Nodes are `(asu_molecule_id, symop_no, cell)`. Edges come from crystal-mode
interfaces, each carrying the *relative* fractional placement between its two
molecules, so an edge can be applied to any node to reach its partner. The
graph is infinite and must be generated lazily by growth, never materialised.

### R2 — Finite/infinite discrimination
An assembly grown from a seed over a set of interface types is finite only if
growth terminates. If growth ever reaches the same `(molecule, symop_no)` at a
*different* cell, the component is translationally infinite — a lattice or a
fibre, not an assembly — and must be rejected. This is the single hardest
correctness requirement; a missed infinite case produces an unbounded loop.
Growth must also carry a hard node cap as a backstop.

### R3 — Candidate enumeration
Enumerate assemblies over *nested* interface subsets: sort the distinct
interface types by stabilisation energy (most stabilising first) and, for
k = 1..n, grow components using only the top-k types. This is bounded at n
iterations and mirrors PISA's dissociation ordering. Exhaustive 2^n
enumeration is explicitly out of scope.

### R4 — Deduplication and scoring
Assemblies that are symmetry-equivalent, or that have identical molecule
content, collapse to one entry. Each surviving assembly is scored with
`energy.dissociation.assembly_dissociation` over its internal interface graph,
giving ΔG_diss and TΔS. Results rank by ΔG_diss descending.

### R5 — Output
`CoreState` gains the predicted assemblies; they appear in the assembly JSON
document as a list with composition, formula, `size`, `mmsize`, ΔG_diss,
entropy and the per-molecule placements. Reachable via
`PISAInterfaceAnalyzer.assemblies` and `--predict-assemblies`.

### R6 — Honest accuracy
The agreement with PISA is unknown before measurement. The deliverable is the
*measured* number, recorded in `tests/data/reference/assembly_validation.json`
and asserted by a test, in the same style as the entropy constant. If top-
assembly agreement comes out low, that is the finding and it gets reported —
the ranking must not be tuned against the reference set to flatter it.

## Non-goals

* Exhaustive interface-subset search (R3 fixes the ordering).
* QSbio / biological-vs-crystal classification — that is the next step and
  needs an external dataset; this spec only produces its input features.
* Replacing the fitted CSS. Once assemblies are predicted, CSS can be
  revisited; not here.
* Biological-assembly *generation* from `_pdbx_struct_assembly_gen` (separate,
  smaller piece of work).

## Global constraints

* Hard dependencies stay `numpy` + `scipy`; `gemmi` remains optional and is
  only needed where symmetry already needs it.
* Python >= 3.10.
* No new fitted constant without a refit script and a drift-guard test.
* Reference data committed gzipped; the full validation must run offline.
* Default behaviour unchanged: assembly prediction is opt-in.
