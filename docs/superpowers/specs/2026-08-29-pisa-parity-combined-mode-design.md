# fastPISA: PISA parity, combined mode, performance, and reliability

Date: 2026-08-29
Status: approved (all 4 phases; crystal-symmetry enumeration deferred; `combined` becomes the default mode)

## Goal

Make fastPISA reproduce original PISA's interface numbers (not just interface
detection), merge PISA + COCOMAPS results into one unified per-interface
report, make it faster, and harden it into reliable software with CI and an
offline validation suite.

Ground truth on this machine: the EBI PISA web service
(`https://www.ebi.ac.uk/pdbe/pisa/cgi-bin/interfaces.pisa?<pdbid>`) returns
original-PISA XML per entry: per-interface `int_area`, `int_solv_en`,
`pvalue`, `stab_en`, `css`, and full H-bond / salt-bridge / disulfide atom
lists. The CCP4 binary tests remain and keep working on the lab machine.

## Findings driving the design (review of 2026-08-29)

1. Interface *detection* and *area* are sound (1ktz: 496.0 vs PISA ~505 Å²).
2. Energies are wrong in structure and calibration: ΔG_solv −29.2 vs PISA
   −8.9 on 1ktz. `calculate_solvation_energy` uses assembly-wide per-atom BSA
   (contaminated for >2 chains) and an ad-hoc ASP table.
3. P-value (0.045 vs 0.220) and CSS (1.009 vs 1.0, exceeding its own 0–1
   range) are ad-hoc composites, documented as uncalibrated.
4. `pipeline.py` and `cocomaps/pipeline.py` duplicate ~80% of their logic;
   the COCOMAPS-mode assembly dissociation energy formula is inconsistent
   with PISA mode (sign convention wrong, solvation double-counted).
5. No crystal-symmetry interfaces (deferred by decision).
6. Performance: pure-Python ASA fallback; per-pair ASA recomputed over whole
   molecule pairs; masks built in Python loops.
7. Reliability: no CI; lab-machine paths hardcoded in `tests/conftest.py`;
   16 tests skip on this machine.

## Phase 1 — Unified core + `combined` mode

- New `fastpisa/core.py`: one function computes the shared state ONCE
  (parse → molecules/masks → combined ASA → per-molecule isolated ASA →
  per-atom pair ΔASA → interface detection → atom contacts). Both existing
  pipelines become thin decorators over it; `analyze_structure` /
  `analyze_structure_cocomaps` keep their signatures (back-compat).
- New mode `"combined"`: one pass emits interfaces carrying BOTH PISA
  energetics (area, ΔG, P-value, CSS, bond counts) AND COCOMAPS contact map +
  interaction populations. It becomes the default in `PISAInterfaceAnalyzer`
  and the CLI (`--mode pisa|cocomaps|combined`).
- The "identical interfaces" invariant becomes true by construction; a test
  asserts all three modes give the same interface IDs and areas.
- Fixes the COCOMAPS ΔG_diss inconsistency (single energy path).

## Phase 2 — Numerical parity with PISA

- Solvation energy uses PAIR-specific ΔASA (`asa_isolated − asa_pair`), not
  assembly-wide BSA.
- Benchmark set: ~15–30 diverse deposited entries (1ktz, antibody–antigen,
  barnase–barstar, obligate dimers, protein–DNA, disulfide-linked, ligand
  interfaces). For each: PDB file + EBI PISA XML cached under
  `tests/data/reference/`. Comparisons filter to identity-symop interfaces.
- Calibrate ASP by least squares: ΔG_pisa ≈ Σ_k σ_k · ΔASA_k with k = atom
  classes (start element-level C/N/O/S/P; refine to polar/charged subclasses
  by atom name if residuals demand). Calibrate per-H-bond / per-salt-bridge
  energies against `stab_en − int_solv_en` regression.
- P-value: implement PISA's definition — probability that a random surface
  patch of the same size is at least as hydrophobic — analytically from the
  surface-atom distribution of σ·ASA (z-score over N-atom sample sums), then
  validate against XML `pvalue` (rank correlation; scale fit allowed).
- H-bond/salt/disulfide criteria tuned against the XML bond lists.
- CSS: closest reproducible surrogate mapped to [0,1], validated against XML
  `css`; measured agreement reported honestly in docs. (Exact CSS requires
  assembly enumeration — deferred with symmetry.)
- Acceptance on held-out entries: interface area <2%; ΔG_solv within
  ~15–20%; bond counts within ±2; P-value rank-correlated.

## Phase 3 — Performance

- FreeSASA becomes the expected backend (documented; pure-Python stays as
  fallback).
- Per-pair ASA: only recompute atoms within reach of the partner molecule
  (an atom's ASA can change only if a partner atom is within
  2·(r_max + probe) of it); all other atoms keep their isolated ASA.
- Vectorized mask construction (numpy, not per-atom Python loops).
- `examples/benchmark.py` with before/after timings; combined mode must be
  faster than running the two old modes sequentially (shared ASA).

## Phase 4 — Reliability / productization

- `tests/test_vs_pdbe_pisa.py`: offline regression against the cached
  reference fixtures (never needs network in CI).
- `examples/compare_vs_pisa.py`: fetches/refreshes EBI PISA reference data,
  runs fastPISA, and emits a comparison table (area/ΔG/bonds/P-value/CSS
  deltas + summary statistics).
- GitHub Actions CI: pytest on 3.10–3.12, with and without freesasa.
- `tests/conftest.py` machine paths become environment variables
  (`FASTPISA_PISA_BIN`, `FASTPISA_CASP17_DIR`, `FASTPISA_OPENDDE_AB`) with
  the current lab paths as defaults.
- Uncommitted working-tree changes: keep `weight_energies_by_confidence` and
  the entropy docstring edit (committed as-is); drop the always-skipped
  Ponstingl placeholder test.
- README: combined mode, calibration status/results, validation story.
  Version bump to 0.3.0.

## Out of scope (deferred)

- Crystal-symmetry interface enumeration and assembly (multimeric state)
  prediction, and therefore exact CSS. Comparisons filter to identity-op
  interfaces, as the existing binary tests already do.
