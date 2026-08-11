# fastPISA tests

`pytest` test suite for the fastPISA PISA + COCOMAPS interface-analysis package.

## Running

```bash
cd /home/dzyla/Code/fastPISA
python3 -m pytest            # full suite
python3 -m pytest -q         # quiet
python3 -m pytest tests/test_contact_classification.py   # just the chemistry unit tests
```

pytest is configured via `[tool.pytest.ini_options]` in `pyproject.toml`
(`testpaths = ["tests"]`).

## What the tests cover

### Unit / integration (run everywhere, always)
- `test_contact_classification.py` — the fixed atom-chemistry classifier:
  - Disulfides are real Cys-Sg..Cys-Sg pairs only (guards the old bug where ANY
    pair < 3.0 A was called a disulfide).
  - Salt bridges are charged side-chain pairs only (no more backbone N-O pairs).
  - H-bonds use a donor/acceptor rule that works WITHOUT explicit H atoms (so
    they work on AlphaFold / cryo-EM / most X-ray structures).
- `test_pipeline.py` — end-to-end invariants on `tests/data/1ktz.pdb`:
  - PISA and COCOMAPS modes report **identical interface IDs** (the core
    invariant in AGENTS.md).
  - No bogus disulfides; H-free structures still report H-bonds.
  - Binding energy = solv + contact (no salt-bridge double counting).
  - Assembly BSA < ASA (correct convention).
  - `min_css` significance filter works and is consistent across modes.
  - Python API (`summary`, `write_json`) and CLI (both modes).

### Reproducibility vs the original CCP4 PISA binary (`tests/test_reproduce_pisa.py`)
These are the publishability checks: they run the **original PISA v2.2.0
binary** (`/programs/xtal/ccp4-9/bin/pisa`) on the same input and compare
interface **count** and per-pair **area**.

They are **skipped automatically** when the original binary, the CCP4 setup,
or the OpenDDE structure directories are not present, so the suite still runs
green on machines without them.

- `test_1ktz_matches_original_pisa` — 1ktz asymmetric-unit (X,Y,Z) interface
  area vs PISA.
- `test_casp17_matches_original_pisa[case]` (H1443, H1400, H2343, H1346) —
  CASP17 AlphaFold antibody models: interface count + per-pair area vs PISA
  (these have no crystal data, so the original reports exactly the ASU
  interfaces — a direct comparison).

## Reproducibility results (Aug 2026)

Run with the original CCP4 PISA v2.2.0 binary:

| Structure | Interfaces (ours/PISA) | Per-pair area agreement |
|-----------|------------------------|--------------------------|
| 1ktz (X,Y,Z ASU) | 1/1 | 483.5 vs 493.4 Å² (2.0%) |
| H1443 | 3/3 | 0–2% |
| H1400 | 3/3 | 1–5% |
| H2343 | 3/3 | 1–2% |
| H1346 | 5/5 | 0–5% |

Acceptance thresholds encoded in the tests: count must match exactly; area
must agree within 10% (the engines use slightly different surface conventions).

## Known scope gaps (documented, not bugs)
- fastPISA does not yet apply crystallographic symmetry to build the biological
  assembly, so for structures WITH crystal data (e.g. 1ktz) the original PISA
  reports additional symmetry-copy interfaces that fastPISA does not generate.
  The reproducibility tests therefore compare the asymmetric-unit (X,Y,Z)
  interface for those cases. Adding assembly prediction is a separate task.
- H-bond / salt-bridge counts are a rule-based subset (no external H-add /
  HBPLUS backend), so absolute counts are fewer than the original binary's.
  Interface detection and area are the quantities that agree with PISA.
