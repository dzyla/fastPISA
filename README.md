# fastPISA

Local, fast analysis of biomolecular interfaces — a Python reproduction of the
[PDBe PISA](https://www.ebi.ac.uk/pdbe/api/pisa/) interface/assembly schema, with
an optional **COCOMAPS 2.0** contact-map mode. Reads PDB **and mmCIF** files
(including AlphaFold predicted complexes).

Both modes share the same interface-detection and surface machinery, so they
**always identify exactly the same interfaces** for a structure.

| Mode | What it reports | Output |
|------|-----------------|--------|
| `pisa` | Thermo/surface analysis: ASA/BSA, interface areas, ΔG, H-bonds / salt bridges / disulfides | PDBe PISA `assembly.json` + `interfaces.json` |
| `cocomaps` | Residue–residue contact map with atomic interaction-type classification (H-bond, salt bridge, pi-pi, cation-pi, ch-pi, …) | Superset of the PISA schema + `interface_contact_map` per interface |

---

## Install

```bash
cd fastPISA
pip install -e .

# Optional but recommended: C-accelerated ASA backend (~15x faster)
pip install freesasa numpy scipy
#   - numpy, scipy: always required
#   - gemmi: required for mmCIF parsing
#   - freesasa: C library; Shrake-Rupley SASA in C. Installed wheels use
#     gcc to compile the C library with Python bindings.
```

Dependencies: `numpy`, `scipy` (core); `gemmi` (mmCIF); `freesasa` (fast ASA, optional).

If `freesasa` is not installed, fastPISA automatically falls back to the pure-Python
Shrake-Rupley implementation.

---

## Quick start (CLI)

```bash
python -m fastpisa.cli 6nxr.pdb --pdb_id 6nxr --mode pisa --output_dir out
python -m fastpisa.cli complex.cif --mode cocomaps --pdb_id my --time --json-summary -o out
```

Common options:

| Flag | Meaning |
|------|---------|
| `--mode {pisa,cocomaps}` | Analysis mode (default `pisa`) |
| `--pdb_id` | PDB id used in output filenames |
| `--probe_radius`, `--point_density`, `--interface_cutoff` | Core geometry knobs |
| `--no-water` / `--with-water` | Exclude (default) or include ordered water in the interface search |
| `--time` | Print wall-clock analysis time |
| `--json-summary` | Print a compact JSON summary instead of the text report |
| `-o`, `--output_dir` | Where to write the two JSON documents |

Output files per run:

```
{pdb_id}-assembly{assembly_id}-interfaces.json   # per-interface detail
{pdb_id}-assembly{assembly_id}.json              # assembly-level stats
```

---

## Python API

Use fastPISA directly from Python — introspect `Interface` objects, not just JSON.

```python
from fastpisa.api import PISAInterfaceAnalyzer

# PISA mode
ana = PISAInterfaceAnalyzer("6nxr.pdb", pdb_id="6nxr", mode="pisa")
ana.analyze()

print(ana.summary())                       # human-readable report
for iface in ana.interfaces:               # list of Interface objects
    print(iface.interface_id,
          round(iface.interface_area, 1),
          iface.number_interface_residues,
          len(iface.contacts))

# Switch to COCOMAPS on the same instance
ana.mode = "cocomaps"
ana.analyze(recompute=True)
cm = ana.get_interface(1).cocomaps         # contact map + interaction population
print(cm["interaction_population"])

# Write the JSON documents
ana.write_json("out/")
```

One-shot function:

```python
from fastpisa.api import analyze_interface
result = analyze_interface("complex.cif", pdb_id="c1", mode="cocomaps")
# result = {"interfaces": {...}, "assembly": {...}, "interfaces_obj": [...]}
```

Key accessors on `PISAInterfaceAnalyzer`:

- `.interfaces` — list of `Interface` dataclass objects
- `.interfaces_json` / `.assembly_json` — the two JSON documents as dicts
- `.summary()` — human-readable report
- `.write_json(dir)` — write the two JSON files, returns their paths
- `.get_interface(id)` — fetch one interface object
- `.mode` — set to `"pisa"` or `"cocomaps"`; `.analyze(recompute=True)` re-runs

---

## Processing many antibody complexes (batch example)

The `fastpisa.api` class is designed to drive batch pipelines. A worked example that
analyzed 107 AlphaFold antibody–antigen complexes (classifying antibody vs antigen
chains from the companion iptm matrix) lives in `/tmp/ab_batch/run_batch.py`.
Its output CSVs/JSON are a ready template for your own batch runs.

---

## Benchmark vs original CCP4 PISA

On 7 structures (1.4k–10.6k atoms, total wall time for a single run of each):

| | original PISA v2.2.0 (C++) | fastPISA pure-Python | fastPISA + FreeSASA |
|---|---|---|---|
| total | 18.2 s | 78.4 s | **5.05 s** |

With the FreeSASA C backend fastPISA is ~**3.6x faster than the original CCP4
binary** and ~**15x faster** than our pure-Python implementation.

Note on interface *counts*: the original binary reports *all* crystal-packing
contacts (including symmetry copies); fastPISA currently reports the interfaces
present in the given coordinate set without applying crystallographic symmetry.
So raw count comparison with PDBe differs for high-symmetry entries — this is a
known, documented scope difference, not a per-mode bug.

---

## Layout

```
fastpisa/
├── api.py                 # PISAInterfaceAnalyzer class + analyze_interface()
├── cli.py                 # command-line interface
├── pipeline.py            # PISA analysis pipeline
├── cocomaps/              # COCOMAPS 2.0 mode
│   ├── interactions.py    # atomic interaction-type classifier
│   ├── contact_map.py     # residue-residue contact map + matrix
│   └── pipeline.py        # COCOMAPS analysis pipeline
├── interface/contacts.py  # molecule detection, masks, contacts (shared)
├── surface/
│   ├── shrake_rupley.py   # pure-Python Shrake-Rupley ASA
│   └── freesasa_backend.py# optional C-accelerated ASA (auto-dispatched)
├── energy/                # ΔGsolv, ΔGint, entropy
├── scoring/               # P-value, CSS
├── output/                # PDBe PISA JSON builders
└── parser/pdb_parser.py   # PDB + mmCIF parsing (gemmi)
```

---

## Notes / caveats

- **ASA/BSA convention**: with the FreeSASA backend, absolute ASA/BSA values use
  FreeSASA's parameters; with the pure-Python backend they use our 480-point
  convention. Values differ in magnitude between the two backends, but interface
  *detection* (which interfaces, residues and H-bonds exist) is identical.
- **Symmetry**: biological-assembly prediction from crystallographic symmetry is
  not yet implemented (see benchmark note).
- **COCOMAPS classifier** is a rule-based subset of COCOMAPS 2.0's 16 interaction
  classes; it does not run HBPLUS or add hydrogens.

## License

Original PISA is (C) Eugene Krissinel (CCP4) — see the CCP4 license. This
reimplementation is provided for research use.