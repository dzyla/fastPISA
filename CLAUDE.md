# CLAUDE.md

Context for Claude Code (and similar coding agents) working in this repository.

## Project summary

`fastPISA` is a dependency-light Python package that reproduces the CCP4/PDBe
**PISA** interface analysis locally — numerically calibrated against the
original engine — plus a **COCOMAPS 2.0** contact-map mode. Input: PDB(.gz) or
mmCIF (AlphaFold-predicted complexes supported). Output: PDBe-PISA-shaped JSON;
the default `combined` mode carries both PISA thermodynamics and the COCOMAPS
contact map on every interface.

## Build / run

```bash
pip install -e .
pip install freesasa gemmi   # freesasa strongly recommended (15x ASA speed)

# CLI (combined mode is the default)
python -m fastpisa.cli in.pdb --pdb_id X -o out/
python -m fastpisa.cli complex.cif --mode pisa --time --json-summary -o out/

# Python
from fastpisa.api import PISAInterfaceAnalyzer
ana = PISAInterfaceAnalyzer("in.pdb", pdb_id="X"); ana.analyze()
ana.interfaces        # list of fastpisa.interface.contacts.Interface
ana.summary(), ana.write_json("out/")
```

## Commands you'll most often run

- `pytest tests/ -q` — full suite incl. the offline accuracy regressions vs
  original PISA (`tests/test_vs_pdbe_pisa.py`) and vs COCOMAPS 2.0
  (`tests/test_vs_cocomaps2.py`), both from cached reference data
- `python examples/calibrate.py` — audit/refit the fitted constants (offline,
  seconds); `--emit-sigma` prints a paste-ready SIGMA block
- `python examples/compare_vs_pisa.py` — head-to-head vs original PISA
  (add entries with `--fetch <pdbid>`; `--assembly-entries <ids>` compares
  recent entries via the PDBe PISA 2.0 JSON API; network only for fetching)

## Validation status (2026-09-01; don't regress these)

- vs original PISA, **6881 identity interfaces from 674 entries** (400 a
  seeded random draw from a stated sampling frame, de-duplicated at 30%
  sequence identity; 36 legacy hand-picked). All figures are **grouped
  10-fold CV**, folds never splitting a PDB entry:
  polymer-polymer (n=2303) area 1.8% median, dG r 0.971 / R^2(1:1) 0.940 /
  median |err| 0.74 kcal/mol, stab r 0.990, P-value median |err| 0.067
  (Spearman 0.85), CSS Spearman 0.73; ligand-involving (n=4578) is much
  weaker -- dG r 0.81 overall, area 12% median -- and is documented as
  indicative, not calibrated. H-bonds 91% within +-1; salt 0.08 mean diff;
  disulfides exact. Blind on 20 recent (2023-24) assemblies: dG r 0.978,
  stab 0.986.
- `tests/test_vs_pdbe_pisa.py` (legacy 36 entries) is IN-SAMPLE -- a
  breakage regression, not a generalisation measure. The out-of-sample
  guard is `tests/test_calibration_benchmark.py`, which asserts the grouped
  CV numbers above and runs offline in ~3 s from the committed feature
  table.
- vs COCOMAPS 2.0 standalone (same inputs, REDUCE hydrogens): contact maps
  IDENTICAL (1ktz 30/30, 1vfb 28/28, 1aay protein-DNA 57/57 residue pairs;
  interface residue sets identical); COCOMAPS-convention salt bridges match
  per residue pair.

## Architecture in one line

ALL modes (`combined`/`pisa`/`cocomaps`) run `fastpisa/core.py` exactly once —
identical interfaces by construction; `pipeline.py` and `cocomaps/pipeline.py`
are thin wrappers.

## Conventions & non-obvious traps (READ before editing)

- **Calibration is load-bearing, and now reproducible.** `energy/asp_table.py`
  (sigma per atom class) and `scoring/scoring.py` (P_VALUE_Z_SCALE, CSS
  logistic) are FITTED; `energy/energy.py` (E_HBOND/E_SALT_BRIDGE/
  E_DISULFIDE) was recovered EXACTLY from PISA and is not a fit. Refit and
  audit with `python examples/calibrate.py` (offline, from
  `tests/data/calibration/features.json.gz`);
  `tests/test_calibration_benchmark.py::test_shipped_constants_match_a_full_refit`
  fails if the shipped sigmas ever drift from what the data refits to. Don't
  tweak constants by hand.
- **The feature table is sufficient statistics, not coordinates.** dG_solv is
  linear in the sigmas, so per-class buried areas + the buried-patch moments
  reproduce the pipeline's dG and P-value EXACTLY. That is why a 674-entry
  benchmark fits in 746 kB and its regression test runs in 3 s. Regenerate
  with `examples/extract_calibration_features.py` after any change to
  `atom_class`, the surface code, or interface detection.
- **CSS was re-examined and deliberately NOT refitted**: a refit lowers MAE
  but degrades rank agreement, which is what CSS is used for. Don't "improve"
  it on log-loss.
- **Interfaces are defined by buried area (pair dASA > 0), not by 5 A atom
  contacts** — PISA semantics. The shadow cutoff (2*r_max + 2*probe) screens
  candidate pairs; per-pair ASA is evaluated only near the interface.
- **Bond classes are independent predicates** (a charged pair can be both salt
  bridge and H-bond — PISA lists it in both tables). Counting lives in
  `interface/bonds.py` (geometric H-bonds: 3.89 A + antecedent angles +
  capacities; explicit-H criteria when the model has hydrogens).
- **Two salt-bridge conventions on purpose**: PISA-schema `number_salt_bridges`
  uses PISA's rule (`interface/bonds.py`, no phosphates, 4.0 A); the COCOMAPS
  contact-map classes use COCOMAPS 2.0's rule (Lys/Arg vs carboxylate or DNA
  phosphate, 4.5 A) in `cocomaps/interactions.py`. Don't "unify" them.
- **Pi classes need ring geometry** (`cocomaps/rings.py` centroids/normals);
  proximity-only rules over-count CH-pi ~20x. Contact-map vdW classes require
  r1+r2+0.5 A; farther pairs inside 5 A are "proximal" (COCOMAPS vocabulary).
- **`ligand_mode`**: `"separate"` (default, classic PISA: each hetero group
  its own monomer) vs `"merge"` (jsPISA-on-assembly: a chain's cofactors
  belong to the chain).
- **Hydrogens carry no surface**: masks are heavy-atom-only; H atoms stay in
  the parsed list for H-bond geometry.
- **Molecules by residue composition, not `chain.group`.** The parser flag is
  sticky. Standard AA/NA sets live in `interface/contacts.py` (incl. modified
  residues like MSE/CCS/PTR — splitting them out fabricates interfaces).
- **Water is excluded from interface search** by default.
- **Element = PDB columns 77–78.** Never derive from atom-name prefix.
- **Solvation classes must not silently swallow chemistry.** `atom_class`
  maps every heavy atom to a key of `SIGMA`; the catch-all `X` has sigma 0,
  so anything routed there contributes NOTHING to dG. Halogens used to land
  there (a buried chloride scored 0.0 against PISA's -12 kcal/mol) and now
  have their own fitted `HAL` class. `P` exists as a class but is pinned to
  0: phosphorus buries a median 1.5 A^2 (the OP oxygens shield it) and its
  fitted value is indistinguishable from zero. Before adding an element to
  `X`, check whether the benchmark can fit it instead.
- **FreeSASA:** do NOT call `Parameters.setAlgorithm("ShrakeRupley")`
  (segfaults on single-atom inputs). `calcCoord` wants a flat 3N coord array;
  `atom_indices` maps output indices only — callers pre-subset `atoms`.
- **gemmi 0.7.x:** `Block` has no `.tags` / `find_tags_loop`; use
  `block.find("_atom_site.", [tag, ...])`.

## Verify after changes

```bash
pytest tests/ -q                          # must stay green (esp. test_vs_pdbe_pisa)
python examples/compare_vs_pisa.py        # accuracy table vs original PISA
```

## Linkage

- PISA paper: Krissinel & Henrick, JMB 372:774–797 (2007). Schema: PDBe-KB/pdbe-pisa-json.
- Original PISA ground truth: EBI service `https://www.ebi.ac.uk/pdbe/pisa/cgi-bin/interfaces.pisa?<pdbid>`
  (fetch/parse via `fastpisa/reference/`); optional CCP4-binary tests are
  enabled via FASTPISA_PISA_BIN / FASTPISA_EXTERNAL_MODELS_GLOB /
  FASTPISA_EXTERNAL_CIF (skip when unset).
- COCOMAPS 2.0: Chawla et al., Bioinformatics (2025), PMC12684709; standalone
  code Zenodo 10.5281/zenodo.17390665 (reference outputs cached in
  tests/data/reference/cocomaps2/). Its HBPLUS/NACCESS steps are
  license-walled — H-bond deliverables are validated against PISA instead.
- PDBe PISA 2.0 JSON API (recent entries, biological assemblies):
  `https://www.ebi.ac.uk/pdbe/api/pisa/interfaces/{pdbid}/{assembly}`.
- Sampling frame + seed for the 400-entry random draw:
  `fastpisa/reference/sampling.py`; the drawn list and frame parameters are
  recorded in `tests/data/calibration/entries.json`. The RCSB search API is
  used for 30%-identity cluster representatives.
- Design spec: docs/superpowers/specs/2026-08-29-pisa-parity-combined-mode-design.md
