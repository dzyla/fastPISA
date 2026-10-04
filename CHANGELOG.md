# Changelog

All notable changes to fastPISA. Versions follow [semantic versioning](https://semver.org/);
a change that moves computed numbers is called out explicitly under
**Results changed**, because a published figure may depend on it.

## 0.5.0 — 2026-10-04

### Results changed — read this if you used an earlier version

- **The pure-Python surface engine computed ASA ~6x too large.** Its burial
  test compared a probe-*centre* point against the bare van der Waals sphere
  (`r_j`) instead of `r_j + probe`, and its neighbour cutoff fell short of
  `2*r_max + 2*probe`. Measured on 400 atoms of 1ktz: 27037 Å² instead of
  4508 Å². Every ΔG, P-value, CSS and buried-surface value derived from it
  was wrong by roughly that factor.
  **Who is affected:** anyone who ran fastPISA **without `freesasa`
  installed**. The FreeSASA path was unaffected. If you are unsure which ran,
  `PISAInterfaceAnalyzer.analysis_provenance()["surface_backend"]` now
  reports it; re-run any affected analysis.
- **`--interface_cutoff` was only half-applied.** `find_contacts` kept its own
  5.0 Å default, so a cutoff below 5 Å still reported longer contacts and one
  above 5 Å did nothing at all. Contacts are now bounded by the requested
  cutoff.
- **`dissociation_energy` and `entropy` were wrong in sign, scope and model.**
  They now follow PISA's own relation, `ΔG_diss = −Σ(stabilisation over the
  interfaces cut) − TΔS`, along the cheapest dissociation pathway. The
  entropy was previously added rather than subtracted, summed over every
  interface rather than one dissociation, and computed as
  `0.02*area + 0.5*n_residues` — a surrogate with a 468 kcal/mol median error
  against PISA, anticorrelated (r = −0.11).
- **Alternate conformers are selected per residue as a consistent set.**
  Selection keyed on the residue name, so microheterogeneity (altloc A = SER,
  B = ALA) survived as *two* residues with atoms at identical coordinates,
  corrupting that residue's ASA, BSA and ΔG. Per-atom selection could also
  mix CA from one conformer with CB from another.
- **Polymer membership now comes from the file** (`SEQRES`,
  `_pdbx_poly_seq_scheme`) and not only from a hardcoded residue list. 12% of
  a blind 60-entry draw declares a residue the list did not know; 2izq's
  D-peptide chains were being split into 715 interfaces instead of 201.
- **Interfaces between distinct hetero groups of one chain were being merged.**
  A glycan tree's sugars share a chain, and the symmetry-equivalence key
  identified a molecule by its chain letter, so all but the first collapsed.

### Added

- **Crystal symmetry mode** (`symmetry="crystal"`, `--symmetry crystal`):
  expands the asymmetric unit by its space group and reports the crystal's
  interfaces, including the packing contacts PISA reports for a deposited
  entry — 60% of its output. Validated interface-by-interface against PISA on
  chain pair *and* relative crystal transform: 566/566 on the cached
  reference set and 732/732 on a blind 60-entry random draw, 1.2% median area
  error, ΔG Pearson 0.995/0.991.
- **Assembly prediction** (`predict_assemblies=True`, `--predict-assemblies`):
  enumerates the finite assemblies a crystal admits and ranks them, largest
  stable first. Measured against PISA's own published predictions
  (`multimers.pisa`, cached for all 37 reference entries): top-assembly
  stoichiometry **79.4%**, composition 41.2%, **author-deposited assembly
  73.1%**, recall 88.8% with precision 85.8%; on the three entries where PISA
  predicts nothing stable we agree on 2 of 3. Denominators and both
  superseded measurements are recorded in
  `tests/data/reference/assembly_validation.json`.
- `FASTPISA_SASA_BACKEND` (`auto` | `python` | `freesasa`) pins the surface
  engine for reproducibility; provenance reports the algorithm and quadrature
  that actually ran.
- Rigid-body dissociation entropy (`fastpisa.energy.entropy`) with one
  constant fitted to PISA's own assembly entropies: median error
  0.91 kcal/mol, r = 0.983.
- PDBe-shaped bond tables (`hydrogen_bonds`, `salt_bridges`,
  `disulfide_bonds`, `covalent_bonds`) in the interfaces JSON.
- `examples/validate_crystal.py`, `examples/validate_assemblies.py` and
  `examples/calibrate_entropy.py` — all offline against committed references.
- `docs/surface_backends.md`: the measured comparison of the two surface
  engines and why FreeSASA is pinned to Lee-Richards.

- **`fastpisa.interfaces(path)`** — the one-call entry point, returning a
  plain list of `Interface` objects sorted largest-first. `Interface`,
  `AtomContact` and `PISAInterfaceAnalyzer` are now exported at package
  level (lazily, so `import fastpisa` stays cheap), with `__all__`.
- A **release workflow** publishing to PyPI on a version tag via Trusted
  Publishing, gated on tag/package/changelog version agreement, so
  `pip install fastpisa` works and the tag can be minted as a Zenodo DOI.
- CI now also runs on **macOS and Windows**, not ubuntu only.
- `examples/benchmark_vs_pisa.py` — a large blind benchmark against original
  PISA over a fresh-seed draw from the calibration sampling frame, excluding
  every entry that informed a fitted constant. Resumable, offline
  `--report`, caches outside the repository via `FASTPISA_BENCHMARK_CACHE`.
- `FASTPISA_REFERENCE_DIR` and a `cache_dir` argument on
  `compare_crystal_entry`, so a multi-GB benchmark cache stays out of
  `tests/data/reference`.

### Changed

- The pure-Python surface engine is **~2.4x faster** (2.17 s → 0.91 s on
  1a3n) and bit-identical: neighbours are visited nearest-first and test
  points dropped as they are buried, stopping once none survive, instead of
  always forming the full (points × neighbours) distance matrix. Order
  independence and exact zero for a fully buried atom are asserted.

### Fixed

- `CRYST1` was parsed at 5-character field widths instead of 9/9/9/7/7/7,
  turning 1acb's cell into a = 55.0, b = 0.3, c = 59.0, α = 400°. Nothing
  consumed it, so nothing looked wrong — but `PDBStructure.space_group` is
  public and was silently garbage.
- FreeSASA's library default is Lee-Richards, so setting only `nPoints`
  configured a parameter the active algorithm ignores: `point_density` was a
  silent no-op while the documentation claimed Shrake-Rupley. The algorithm is
  now set explicitly on every call.
- `PISAInterfaceAnalyzer.structure` was documented but never assigned.
- Molecule-pair screening is a single pass over the atoms instead of a
  per-pair rebuild: 31.5 s → 2.4 s on 1brs with ordered water (519
  molecules), where it was 82% of the runtime.
- A **LICENSE** file is present: **AGPL-3.0-or-later** with an attribution
  term under section 7(b). Earlier releases declared MIT in metadata while
  granting nothing, leaving the code all-rights-reserved. The AGPL choice is
  deliberate — a modified version stays open, section 13 means hosting it as
  a service also triggers source disclosure, and the 7(b) term keeps the
  citation notice attached. Nothing had been released or pushed under the
  earlier MIT declaration.
- `pyproject.toml` names a citable author and carries OSI classifiers; it
  previously credited the username `dzyla`.
- `CITATION.cff` had a broken author record and claimed version 0.2.0.

### Changed

- `requires-python` raised to `>=3.10`, matching what CI tests.
- The uncalibrated legacy helpers (`scoring.calculate_p_value`,
  `calculate_css`, `classify_interface`, `energy.calculate_entropy`,
  `calculate_dissociation_energy`, `calculate_stabilization_energy`) now emit
  `DeprecationWarning`. They were importable from a package whose claim is
  calibration; a full analysis raises none of them.
- `fastpisa/assembly/symmetry.py` removed: never imported or tested, and its
  docstring advertised space-group generation the tool did not do.
- CI exercises both surface engines on the same tests. The previous
  "pure-python" leg uninstalled freesasa, which made every accuracy test skip
  — which is how the 6x ASA error survived.

## 0.4.0 and earlier

Not separately documented. 0.4.0 added the unified `combined` mode, the
COCOMAPS 2.0 contact-map mode, the 674-entry calibration of the solvation
parameters, and the Interface Explorer app.
