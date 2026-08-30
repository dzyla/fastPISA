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

- `pytest tests/ -q` — full suite incl. the offline accuracy regression vs
  original PISA (`tests/test_vs_pdbe_pisa.py`, cached EBI reference data)
- `python examples/compare_vs_pisa.py` — head-to-head vs original PISA
  (add entries with `--fetch <pdbid>`; needs network only for fetching)

## Architecture in one line

ALL modes (`combined`/`pisa`/`cocomaps`) run `fastpisa/core.py` exactly once —
identical interfaces by construction; `pipeline.py` and `cocomaps/pipeline.py`
are thin wrappers.

## Conventions & non-obvious traps (READ before editing)

- **Calibration is load-bearing.** `energy/asp_table.py` (sigma per atom
  class), `energy/energy.py` (E_HBOND/E_SALT_BRIDGE/E_DISULFIDE — recovered
  EXACTLY from PISA), `scoring/scoring.py` (P_VALUE_Z_SCALE, CSS logistic) are
  fitted against the EBI reference; `tests/test_vs_pdbe_pisa.py` pins the
  accuracy. Don't tweak constants without re-running
  `examples/compare_vs_pisa.py`.
- **Interfaces are defined by buried area (pair dASA > 0), not by 5 A atom
  contacts** — PISA semantics. The shadow cutoff (2*r_max + 2*probe) screens
  candidate pairs; per-pair ASA is evaluated only near the interface.
- **Bond classes are independent predicates** (a charged pair can be both salt
  bridge and H-bond — PISA lists it in both tables). Counting lives in
  `interface/bonds.py` (geometric H-bonds: 3.89 A + antecedent angles +
  capacities; explicit-H criteria when the model has hydrogens).
- **Hydrogens carry no surface**: masks are heavy-atom-only; H atoms stay in
  the parsed list for H-bond geometry.
- **Molecules by residue composition, not `chain.group`.** The parser flag is
  sticky. Standard AA/NA sets live in `interface/contacts.py` (incl. modified
  residues like MSE/CCS/PTR — splitting them out fabricates interfaces).
- **Water is excluded from interface search** by default.
- **Element = PDB columns 77–78.** Never derive from atom-name prefix.
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
- COCOMAPS 2.0: Chawla et al., Bioinformatics (2025), PMC12684709.
- Design spec: docs/superpowers/specs/2026-08-29-pisa-parity-combined-mode-design.md
