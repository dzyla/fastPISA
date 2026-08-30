# fastPISA tests

`pytest` test suite for the fastPISA PISA + COCOMAPS interface-analysis package.

## Running

```bash
python -m pytest             # full suite
python -m pytest -q          # quiet
python -m pytest tests/test_vs_pdbe_pisa.py   # just the accuracy regression
```

pytest is configured via `[tool.pytest.ini_options]` in `pyproject.toml`
(`testpaths = ["tests"]`).

## What the tests cover

### Unit / integration (run everywhere, always)
- `test_contact_classification.py` — atom chemistry: disulfides are real
  Cys-Sg..Cys-Sg pairs only; salt bridges are charged side-chain pairs only;
  H-bonds work without explicit H atoms.
- `test_combined_mode.py` — all three modes (`combined`/`pisa`/`cocomaps`)
  find identical interfaces (single shared core) and the combined mode carries
  both the PISA energetics and the COCOMAPS contact map.
- `test_pipeline.py` — end-to-end invariants on `tests/data/1ktz.pdb` plus,
  when `FASTPISA_EXTERNAL_CIF` points at an AlphaFold-style complex, H-free
  chemistry checks on it.
- `test_pisa_calibration_regressions.py` — pins historical defects
  (P-value degeneracy, ASP sign convention, package importability).

### Accuracy regression vs ORIGINAL PISA (`test_vs_pdbe_pisa.py`)
Runs fully offline against reference data cached in `tests/data/reference/`
(EBI PDBe PISA service XML + RCSB PDB files, 21 entries, 117 matched identity
interfaces). Pins interface-area, ΔG, stab-energy, P-value, CSS and bond-count
agreement with the original engine. Requires the FreeSASA backend
(`pip install freesasa`); refresh the human-readable picture with
`python examples/compare_vs_pisa.py`.

### Reproducibility vs a local CCP4 PISA binary (`test_reproduce_pisa.py`)
Optional: compares interface counts and per-pair areas against an original
CCP4 `pisa` binary on the same inputs. Enabled by environment variables and
skipped otherwise:

| Variable | Meaning |
|---|---|
| `FASTPISA_PISA_BIN` | path to the CCP4 `pisa` binary |
| `FASTPISA_EXTERNAL_MODELS_GLOB` | glob of predicted-model CIFs (no crystal data) to compare |
| `FASTPISA_EXTERNAL_CIF` | one AlphaFold-style complex CIF for the H-free integration tests |

Acceptance thresholds encoded in the tests: interface count must match
exactly; per-pair area must agree within 10% (the engines use slightly
different surface conventions).

## Known scope gaps (documented, not bugs)
- fastPISA does not apply crystallographic symmetry, so for structures WITH
  crystal data the original PISA also reports symmetry-copy interfaces that
  fastPISA does not generate. Comparisons match on the identity (X,Y,Z)
  interfaces.
- CSS is a calibrated surrogate (Spearman 0.80 vs PISA); exact CSS requires
  PISA's crystal-wide assembly analysis.
