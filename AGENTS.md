# AGENTS.md

Guidance for AI agents working in the fastPISA repository.

## What this is

fastPISA is a local Python reproduction of the CCP4/PDBe **PISA** interface-analysis
schema, with a second **COCOMAPS 2.0** mode. The package parses PDB and mmCIF,
finds biomolecular interfaces, and emits PDBe-PISA-shaped JSON plus (in COCOMAPS
mode) residue-residue contact maps with atomic interaction-type labels.

## Important invariants — do not break these

1. **Both modes must find identical interfaces.** PISA and COCOMAPS pipelines share
   `fastpisa/interface/contacts.py` (molecule detection, masks, contacts) and the
   ASA machinery. If you change shared logic, always verify both modes still report
   the same interface IDs for the same input.
2. **Molecule classification is by residue composition, not `chain.group`.** The PDB
   parser's `chain.group` flag is sticky and unreliable (a protein chain with a
   bound ligand is mislabeled ligand). Use the standard AA/nucleotide residue sets.
3. **Exclude water from the interface search by default.** `get_molecule_masks` for a
   polymer must require a standard polymer residue name (chain ID alone leaks water
   into protein-protein contact maps).
4. **Element symbol comes from PDB columns 77–78**, never from the first chars of
   the atom name (`CA` is carbon, not calcium).

## ASA backend is auto-selected

- `fastpisa/surface/shrake_rupley.calculate_asa` dispatches to the **FreeSASA C
  backend** (`freesasa_backend.py`) when installed, else falls back to the pure-Python
  implementation (`calculate_asa_python`).
- **Do NOT call `Parameters.setAlgorithm("ShrakeRupley")`** — it segfaults the FreeSASA
  C library on single-atom inputs (a lone metal ion). Shrake-Rupley is already the
  default. `setProbeRadius`/`setNPoints` are safe.
- `freesasa.calcCoord` takes a **flat 1D array of 3N floats** plus radii; per-atom
  areas come from `result.atomArea(i)`. Callers pass `atoms` already subsetted;
  `atom_indices` only maps output local->global indices (do not re-index `atoms` with it).

## Tests / verification

There is no test framework configured. A quick correctness check after any change:

```python
from fastpisa.api import PISAInterfaceAnalyzer
p = PISAInterfaceAnalyzer("1ktz.pdb", mode="pisa").analyze()["interfaces"]
c = PISAInterfaceAnalyzer("1ktz.pdb", mode="cocomaps").analyze()["interfaces"]
assert [i["interface_id"] for i in p["assembly"]["interfaces"]] == \
       [i["interface_id"] for i in c["assembly"]["interfaces"]]
```

`1ktz.pdb` (chains A/B) is a good small test case.

## Environment

- Pure-Python core: `numpy`, `scipy`.
- mmCIF: `gemmi` (`parse_mmcif` uses `gemmi.read_structure`; `gemmi.cif.Block` has no
  `.tags`/`find_tags_loop` in 0.7.x — use `block.find("_atom_site.", [tags...])`).
- Fast ASA (optional): `pip install freesasa` (compiles the C library with gcc).

## Reference

- PISA algorithm: Krissinel & Henrick, J. Mol. Biol. **372**, 774–797 (2007).
- PISA JSON schema + examples: github.com/PDBe-KB/pdbe-pisa-json.
- Original CCP4 binary (this machine): `/programs/xtal/ccp4-9/bin/pisa` v2.2.0.
- COCOMAPS 2.0: Chawla et al., Bioinformatics (2025), PMC12684709.
- Full implementation notes and pitfalls: the `fastpisa-cocomaps` skill
  (`/home/dzyla/.hermes/skills/cryo-structural-proteomics/fastpisa-cocomaps/SKILL.md`).