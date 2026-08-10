# CLAUDE.md

Context for Claude Code (and similar coding agents) working in this repository.

## Project summary

`fastPISA` is a dependency-light Python package that reproduces the CCP4/PDBe **PISA**
interface analysis locally, with a second **COCOMAPS 2.0** mode. Input: PDB or mmCIF
(AlphaFold-predicted complexes supported). Output: PDBe-PISA-shaped JSON plus, in
COCOMAPS mode, residue-residue contact maps with atomic interaction-type labels.

## Build / run

```bash
pip install -e .
pip install freesasa gemmi   # optional speed + mmCIF

# CLI
python -m fastpisa.cli in.pdb --pdb_id X --mode pisa -o out/
python -m fastpisa.cli complex.cif --mode cocomaps --time --json-summary -o out/

# Python
from fastpisa.api import PISAInterfaceAnalyzer
ana = PISAInterfaceAnalyzer("in.pdb", pdb_id="X", mode="pisa"); ana.analyze()
ana.interfaces        # list of fastpisa.interface.contacts.Interface
ana.summary(), ana.write_json("out/")
```

## Commands you'll most often run

- `python -m fastpisa.cli <file> --mode pisa --json-summary --time -o <dir>` — one run
- `pip install freesasa` / `pip install gemmi` — enable fast ASA / mmCIF support

## Conventions & non-obvious traps (READ before editing)

- **Never break the "two modes find identical interfaces" invariant.** PISA and
  COCOMAPS share `fastpisa/interface/contacts.py` + the ASA code. Verify both modes
  give the same interface IDs after any shared change.
- **Molecules by residue composition, not `chain.group`.** The parser flag is sticky:
  a protein chain carrying a ligand gets mislabeled `ligand`. Use standard AA/NA sets.
- **Water is excluded from interface search** by default; polymer masks must require a
  standard polymer residue, not just a matching chain ID.
- **Element = PDB columns 77–78.** Never derive from atom-name prefix.
- **FreeSASA:** do NOT call `Parameters.setAlgorithm("ShrakeRupley")` (segfaults on
  single-atom inputs). `calcCoord` wants a flat 3N coord array; `atom_indices` maps
  output indices only — callers pre-subset `atoms`.
- **gemmi 0.7.x:** `Block` has no `.tags` / `find_tags_loop`; use
  `block.find("_atom_site.", [tag, ...])`.

## Verify after changes

```python
from fastpisa.api import PISAInterfaceAnalyzer
PISAInterfaceAnalyzer("1ktz.pdb", mode="pisa").analyze()
PISAInterfaceAnalyzer("1ktz.pdb", mode="cocomaps").analyze()
```
Both must agree on interface IDs. `1ktz.pdb` (chains A/B) is the canonical small test.

## Linkage

- Full pitfalls + skill: `fastpisa-cocomaps` (Hermes skill). Read AGENTS.md for the
  same invariants in more detail.
- PISA paper: Krissinel & Henrick, JMB 372:774–797 (2007). Schema: PDBe-KB/pdbe-pisa-json.
- COCOMAPS 2.0: Chawla et al., Bioinformatics (2025), PMC12684709.