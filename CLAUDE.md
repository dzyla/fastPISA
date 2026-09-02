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
- `python examples/make_figures.py` — regenerate the README comparison
  figures (docs/figures/) from the committed tables
- `python examples/compare_vs_pisa.py` — head-to-head vs original PISA
  (add entries with `--fetch <pdbid>`; `--assembly-entries <ids>` compares
  recent entries via the PDBe PISA 2.0 JSON API; network only for fetching)

## Validation status (2026-09-01; don't regress these)

- vs original PISA, **6904 identity interfaces / 119k interface residues
  from 674 entries** (400 a seeded random draw from a stated sampling frame,
  de-duplicated at 30% sequence identity; 36 legacy hand-picked). All
  figures are **grouped 10-fold CV**, folds never splitting a PDB entry:
  polymer-polymer (n=2314) area 1.8% median (per-residue BSA 1.75%), dG r
  0.987 / R^2(1:1) 0.975 / median |err| 0.33 kcal/mol (legacy 36 in-sample:
  r 0.997, 0.24), stab r 0.996, P-value median |err| 0.060 (Spearman 0.88),
  CSS Spearman 0.75;
  protein-NA (n=241) R^2 0.96. H-bond ATOM PAIRS vs PISA's list: precision
  0.958 / recall 0.952; salt bridges 0.985 / 0.979; disulfides exact.
  All interfaces incl. ligands (n=6915): dG r 0.956 / R^2 0.914 / median
  0.76; ligand-pair area 6.0% median (oxo-anions, halides, Na/Ca, organics
  within a few %; short-bond transition metals Mg/Mn/Fe/Cu still 12-35%
  over-buried vs PISA -- documented limit). Blind on 20 recent (2023-24)
  assemblies (earlier constants): dG r 0.978, stab 0.986.
- `tests/test_vs_pdbe_pisa.py` (legacy 36 entries) is IN-SAMPLE -- a
  breakage regression, not a generalisation measure. The out-of-sample
  guard is `tests/test_calibration_benchmark.py`, which asserts the grouped
  CV numbers above and runs offline in ~3 s from the committed feature
  table.
- vs COCOMAPS 2.0 standalone (same inputs, REDUCE hydrogens): contact maps
  IDENTICAL (1ktz 30/30, 1vfb 28/28, 1aay protein-DNA 57/57 residue pairs;
  interface residue sets identical); COCOMAPS-convention salt bridges match
  per residue pair.

## Interface Explorer app (`app/`)

`app/streamlit_app.py` (presentation only) + `app/app_helpers.py` (figures,
py3Dmol HTML, Excel; no Streamlit calls, unit-testable) over
`fastpisa/report.py::group_interface` -- the manuscript digest of the
interface BETWEEN two chain groups (buried surface per side, energies,
bond/contact counts, epitope/paratope residues, prose, viewer commands).
Numbers are sums over the chain-pair interfaces spanning the groups;
P-value/CSS are per pair only (not additive). Deploy on Streamlit Cloud
with main file `app/streamlit_app.py`; `app/requirements.txt` installs the
package with `-e .`. Headless smoke: `streamlit.testing.v1.AppTest` (the
chain-group data_editor cannot be driven by AppTest; test `report.py` and
`app_helpers.py` directly instead).

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
- **The solvation model is two-level and fitted at RESIDUE level.**
  `sigma(atom) = SIGMA[class] + DELTA[fine type]` (32 chemical classes +
  167 shrunk per-atom-type deviations, ridge 1000). It is fitted to PISA's
  per-residue solvation energies (`tests/data/calibration/residue_fit.json.gz`,
  2.4 MB, committed) because interface sums cannot separate the atoms they
  add (interface-level fit is 60% worse out of fold). Judged at interface
  level by grouped CV on `features.json.gz` (4.9 MB, committed; buried area
  per FINE type + surface composition per fine type -> exact dG and P-value).
  Regenerate both with `examples/extract_calibration_features.py`
  (`--residues` for the local 10 MB audit table) after any change to atom
  typing, the surface code, or interface detection.
- **Surface radii are the NACCESS/Chothia set** (`surface_radius()` in
  `surface/shrake_rupley.py`: sp3 C 1.87, sp2/aromatic C 1.76, N 1.65,
  O 1.40, S 1.85) -- recovered empirically as what PISA uses -- plus
  per-element ION radii read off PISA's own lone-ion ASA (Zn/Mg 1.39,
  Ca 1.20, K 2.75, Fe/Hg 1.90, ...; `SURFACE_RADII`). They are for ASA
  ONLY; `get_vdw_radius()` (COCOMAPS contact classification, validated
  identical to COCOMAPS 2.0) is a different convention. Don't merge them.
  Metal fine types are element-resolved (`het:MET:MG`) so each ion has its
  own shrunk sigma deviation. What we could NOT reproduce: PISA buries
  short-bond transition metals (Mg/Mn/Fe/Cu, 1.8-2.2 A coordination) and
  their coordinating residue 12-35% LESS than probe-rolling ASA does; no
  radius, point density, altloc or neighbour-exclusion variant matches it.
- **Residue numbers can be negative.** The PDB parser used `isdigit()` and
  collapsed "-4" onto 0, silently merging residues (DNA numbered about a
  centre, expression tags). Fixed; `tests/test_pisa_fidelity.py` guards it.
- **H-bond criteria were audited pair-by-pair against PISA's lists (30k
  candidate pairs) and left alone**: the 90 degree antecedent-angle cutoff
  sits exactly on PISA's step (5% listed below, 95% above). Known residual:
  PISA lists only ~half of Watson-Crick N1-N3 pairs in DNA duplexes with no
  discernible rule; we list them all. `fastpisa/reference/bonds_audit.py`
  reproduces the audit.
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
  fitted value is indistinguishable from zero (same for `NA_P`). Nucleotide
  classes fit with INVERTED signs vs protein (positive C, negative O); that
  is well-determined (|z| > 7, block condition 24) and is PISA's behaviour,
  not a bug. Before adding an element to `X`, check whether the benchmark
  can fit it instead.
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
