# AGENTS.md

Guidance for AI agents working in the fastPISA repository.

## What this is

fastPISA is a local Python reproduction of the CCP4/PDBe **PISA**
interface-analysis engine — numerically calibrated against the original — with
a **COCOMAPS 2.0** contact-map mode. The default `combined` mode emits one
unified report per interface (PISA thermodynamics + COCOMAPS contact map).

## Important invariants — do not break these

1. **All modes find identical interfaces by construction.** `combined`, `pisa`
   and `cocomaps` are decorations over one core (`fastpisa/core.py`) that runs
   the physics exactly once. Never fork mode-specific detection/energy logic.
2. **Interfaces follow PISA semantics: pair dASA > 0**, not "atoms within
   5 A". Candidate pairs are screened at the shadow cutoff
   (2*r_max + 2*probe); per-pair ASA is evaluated only near the interface
   (identical numbers to a full-pair computation — keep it that way).
3. **The calibration is load-bearing.** ASP sigmas (`energy/asp_table.py`),
   the exact per-bond constants (`energy/energy.py`: -0.444037 / -0.150028 /
   -4.0 kcal/mol), the P-value z-scale and the CSS logistic
   (`scoring/scoring.py`) are fitted against the EBI PDBe PISA service over
   21 entries. `tests/test_vs_pdbe_pisa.py` pins the resulting accuracy;
   `examples/compare_vs_pisa.py` prints the full table. Re-run both after
   touching any of these.
4. **Bond classes are independent predicates** (PISA lists a charged H-bonded
   pair in BOTH its h-bond and salt-bridge tables). Detection lives in
   `fastpisa/interface/bonds.py`: donor/acceptor tables + 3.89 A +
   >=90 deg antecedent angles + metal-coordination exclusion + capacity-limited
   greedy assignment; explicit-H (HBPLUS-style) criteria when the model has H.
5. **Hydrogens carry no surface** (heavy-atom masks) but stay in the atom list
   for H-bond geometry.
6. **Molecule classification is by residue composition, not `chain.group`.**
   Standard AA/NA sets (incl. modified residues MSE/CCS/PTR/SEP/...) live in
   `interface/contacts.py`. Splitting a modified residue out as a ligand
   fabricates interfaces PISA does not report.
7. **Exclude water from the interface search by default.**
8. **Element symbol comes from PDB columns 77–78**, never from the atom name.

## ASA backend is auto-selected

- `fastpisa/surface/shrake_rupley.calculate_asa` dispatches to the **FreeSASA C
  backend** (`freesasa_backend.py`) when installed (~15x faster), else the
  pure-Python implementation.
- **Do NOT call `Parameters.setAlgorithm("ShrakeRupley")`** — it segfaults the
  FreeSASA C library on single-atom inputs. Shrake-Rupley is already the
  default. `setProbeRadius`/`setNPoints` are safe.
- `freesasa.calcCoord` takes a **flat 1D array of 3N floats** plus radii;
  per-atom areas come from `result.atomArea(i)`. Callers pass `atoms` already
  subsetted; `atom_indices` only maps output local->global indices.

## Tests / verification

```bash
pytest tests/ -q                    # includes offline accuracy regression vs
                                    # original PISA (cached EBI reference data)
python examples/compare_vs_pisa.py  # human-readable accuracy table
```

`tests/data/1ktz.pdb` (chains A/B) is the canonical small test case. The EBI
reference cache lives in `tests/data/reference/` (21 entries; extend with
`examples/compare_vs_pisa.py --fetch <pdbid>`).

Optional integration tests (original CCP4 binary, external model sets) are
enabled via `FASTPISA_PISA_BIN`, `FASTPISA_EXTERNAL_MODELS_GLOB` and
`FASTPISA_EXTERNAL_CIF`, and skip when unset.

## Environment

- Core: `numpy`, `scipy`. mmCIF: `gemmi` (0.7.x: `Block` has no `.tags` /
  `find_tags_loop`; use `block.find("_atom_site.", [tags...])`).
- Fast ASA: `pip install freesasa` (compiles the C library with gcc).

## Reference

- PISA algorithm: Krissinel & Henrick, J. Mol. Biol. **372**, 774–797 (2007).
- PISA JSON schema + examples: github.com/PDBe-KB/pdbe-pisa-json.
- Original PISA ground truth: EBI service
  `https://www.ebi.ac.uk/pdbe/pisa/cgi-bin/interfaces.pisa?<pdbid>`
  (NOTE: its per-residue `solv_en` values are sign-inverted relative to the
  interface `int_solv_en`).
- COCOMAPS 2.0: Chawla et al., Bioinformatics (2025), PMC12684709.
- Design spec: `docs/superpowers/specs/2026-08-29-pisa-parity-combined-mode-design.md`.
