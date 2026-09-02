# PISA fidelity: residue-level calibration of the solvation model, bond audit, hydrophobic/polar breakdown

Date: 2026-09-01. Status: implemented (see the Outcome section at the end).

## Goal

Make fastPISA's per-chain-pair energetics (`solvation_energy`,
`stabilization_energy`), interface areas and bond lists agree more closely
with the original PISA engine on protein / nucleic-acid chain pairs, and
expose the hydrophobic-vs-polar split of the solvation energy. Ligand
geometry, a vdW energy term and the dissociation entropy model are
explicitly out of scope.

## Why now, and why this shape

After the 2026-09-01 recalibration (674 entries / 6881 interfaces, grouped
CV), the residual polymer-polymer dG error is **chemistry, not geometry**:
rescaling our dG by PISA's own interface area moves the median error from
0.725 to 0.717 kcal/mol; the relative error is a roughly constant 8-18% of
|dG| across size bins; bond-count contributions to the stab-energy error
are nil. An 11-class solvation scheme (C, N, N+, O, O-, ...) is too coarse
against PISA's per-atom-type parameters.

The cached EBI XML carries, for every interface residue, PISA's own ASA,
BSA and solvation energy. The parser reads it (`include_residues=True`);
nothing consumes it. That is ~1e5 residue-level observations against 6881
interface sums -- the signal needed to determine finer atom types, and a
per-residue-type geometry audit interface totals cannot give.

## Design

### 1. Residue-level feature extraction (new; local artifact)

`fastpisa/reference/calibrate.py` gains `extract_entry_residues(pdb_id)`.
For each matched identity interface it matches our interface residues to
PISA's by `(chain, seq_num, ins_code)` and records per residue:

- `pdb_id`, `pair`, `is_polymer_pair`, residue name, chain, seqnum
- `bsa_by_type`: our pair-specific BSA summed per **fine atom type**, keyed
  `"RES:ATOM"` for standard residues (element-tagged `"el:X"` for
  everything else) -- the finest key any scheme could want
- `asa_fp`, `bsa_fp` (ours), `asa_ref`, `bsa_ref`, `solv_ref` (PISA)

Written by `examples/extract_calibration_features.py --residues` to
`tests/data/calibration/residues.json.gz`, **gitignored** (~10 MB). The
committed artifact stays the class-level `features.json.gz`. A class scheme
is a mapping `fine type -> class`, so comparing schemes is a groupby over
the fine table and takes seconds; no re-extraction.

`run_core(collect_calibration=True)` additionally stores, per interface,
`residue_bsa_by_type` (residue key -> {fine type: BSA}) and per-residue
isolated ASA, computed from the same `bsa_pair` / `asa_alone` the pipeline
already has. Off by default; zero cost otherwise.

### 2. Geometry audit before chemistry

From the residue table: per-residue-type median relative BSA error and
signed bias vs PISA, and the same for isolated ASA. Any residue type that
is systematically off by more than the population (~2%) is investigated
and fixed *before* any sigma is fitted to it -- fitting chemistry to a
geometric bias bakes the bias in. Expected candidates: residues with
unusual atom radii (Met SD, His ring N), terminal residues (OXT),
alternate conformations, and insertion-code handling.

### 3. Solvation class scheme

Two candidates, both fitted at residue level (target: PISA's per-residue
`solv_en`, design row: that residue's `bsa_by_type`) and **judged at
interface level by the existing grouped 10-fold CV** on `features.json.gz`
regenerated under the candidate scheme:

- **B. Chemically motivated classes (~30).** Carbon: aliphatic, aromatic,
  backbone CA, carbonyl/carboxylate C, guanidinium/amide C; nitrogen:
  backbone N, side-chain amide, aromatic ring N, Trp NE1, Arg NE/NH, Lys
  NZ, Pro N; oxygen: backbone O, hydroxyl, carboxylate, amide O; sulfur:
  Met SD vs Cys SG; nucleic acid: base C/N/O, sugar C/O, phosphate OP/P;
  existing hetero classes (OI, HAL, MET, ZN, X) unchanged.
- **C. Hierarchical (recommended).** Scheme B as the base level; per fine
  atom type a deviation `delta_t` shrunk toward zero by an L2 penalty with
  `lambda` chosen by grouped CV. Types with ample buried area earn their
  own value; sparse types fall back to their class.

Shipping rule: a scheme replaces the incumbent only if out-of-fold
polymer-polymer dG median |error| and R^2(1:1) both improve, and no
polymer metric in `tests/test_calibration_benchmark.py` regresses. If C
does not beat B out of fold, B ships (simpler). If neither beats the
11-class incumbent, nothing ships and the negative result is recorded.

Implementation: `atom_class()` returns the fine key; a module-level
`CLASS_OF: Dict[fine, class]` and `SIGMA: Dict[class, float]` (plus
`DELTA: Dict[fine, float]` if C ships) live in `asp_table.py`; `get_asp`
sums them. The calibration `predict_dg` uses the same mapping so the
"shipped constants match a refit" test keeps working.

### 4. Hydrogen-bond / salt-bridge atom-level audit

PISA's XML lists each bond's atoms. New `fastpisa/reference/bonds_audit.py`
matches our `Interface.contacts` flagged `hbond` / `salt_bridge` to
PISA's pairs (chain, seqnum, atom name, both orders) and reports
precision / recall overall and broken down by donor element / acceptor
element / residue pair / distance bin. Criteria in `interface/bonds.py`
are adjusted only where a clear systematic pattern appears (e.g. a
distance or angle cutoff) and only if interface-level counts do not
regress. A pair-level regression test (precision and recall floors, from
the committed `features.json.gz` extended with per-interface matched /
missed / extra counts) is added.

### 5. Hydrophobic / polar decomposition (output)

`Interface` gains `solvation_energy_apolar` (sum over C and S classes)
and `solvation_energy_polar` (the rest); both are written to the
interfaces JSON as additional keys. They sum to `solvation_energy` to
rounding. Existing schema tests must stay green -- extra keys only.

## Testing

- `tests/test_calibration_benchmark.py` remains the out-of-sample gate;
  thresholds move only upward.
- New: residue-level agreement test (median |BSA rel err| per common
  residue type below a floor) from a small committed residue sample.
- New: bond pair-level precision/recall floors.
- New: apolar + polar == solvation_energy on a cached structure.
- `pytest tests/ -q` green; `examples/compare_vs_pisa.py` re-run and
  README / CLAUDE.md numbers updated.

## Non-goals

Ligand / ion interface geometry; any vdW or electrostatic energy term
beyond PISA's; `calculate_entropy` / dissociation energy; PISA 2.0 H-bond
criteria (the target is the classic engine the reference data comes from).

## Outcome (2026-09-01)

* **Geometry audit found the radius set.** Per-residue isolated ASA was 3.3%
  off PISA with residue-type-specific bias; fitting per-element radii landed
  on the published NACCESS/Chothia values (sp3 C 1.87, sp2 C 1.76, N 1.65,
  O 1.40, S 1.85). Adopted verbatim: residue BSA error 6.1% -> 1.75%,
  isolated ASA 3.3% -> 0.87%, ligand interface area 12% -> 9%.
* **Parser bug:** negative residue numbers collapsed onto 0. Fixed.
* **Scheme C shipped** (B + shrunk fine types, ridge 1000; residue-level
  fit on 119,078 residues): polymer-polymer out-of-fold median |dG error|
  0.71 -> 0.32 kcal/mol, R^2 0.951 -> 0.987. Residue-level fitting is
  essential (interface-level hierarchical fit: 0.52).
* **Bond audit:** H-bond pairs precision 0.958 / recall 0.952, salt
  bridges 0.985 / 0.979 after the parser fix. The 90-degree antecedent cutoff
  is exactly PISA's step; no criteria changed. Residual: PISA lists ~53% of
  DNA N1-N3 Watson-Crick pairs with no discernible rule.
* Asn/Gln amide carbon (z = 1.9) and Arg CZ (z = 2.8) were not determinable
  as their own classes -- each is shielded by its heteroatoms -- and were
  merged with the carboxylate carbon into `C_sp2_polar` (32 classes).
* **Hydrophobic/polar split** exposed as `solvation_energy_apolar/_polar`.
* Committed tables: `features.json.gz` (4.9 MB, fine-typed) and
  `residue_fit.json.gz` (2.4 MB); the 10 MB audit table stays local.
