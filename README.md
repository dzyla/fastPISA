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

## Worked example — analyze a PDB and visualize an interface

`tests/data/1ktz.pdb` is a small two-chain (A/B) complex — a good first run.

CLI:

```bash
# PISA mode -> PDBe-schema JSON
python -m fastpisa.cli tests/data/1ktz.pdb --pdb_id 1ktz --mode pisa -o out

# COCOMAPS mode -> adds a residue-residue contact map per interface
python -m fastpisa.cli tests/data/1ktz.pdb --pdb_id 1ktz --mode cocomaps -o out
```

```text
=== Summary ===
Mode: cocomaps
Interfaces found: 1
Assembly dissociation energy: -97.06
Total ASA: 11576.73
Total BSA: 541.13

Top 5 hotspot residues (by buried area):
  A94 (ARG) BSA=175.1 A^2  interfaces=[1]
  A31 (LYS) BSA=125.2 A^2  interfaces=[1]
  A91 (TYR) BSA=97.1 A^2   interfaces=[1]
  B49 (SER) BSA=92.0 A^2   interfaces=[1]
  B53 (ILE) BSA=79.0 A^2   interfaces=[1]
```

Python — introspect the interfaces and write visualizations in one go:

```python
from fastpisa.api import PISAInterfaceAnalyzer

ana = PISAInterfaceAnalyzer("tests/data/1ktz.pdb", pdb_id="1ktz", mode="cocomaps")
ana.analyze()

for iface in ana.interfaces:
    m1, m2 = iface.molecules
    print(f"interface {iface.interface_id}: {m1['molecule_class']} {m1['auth_asym_id']} "
          f"<-> {m2['molecule_class']} {m2['auth_asym_id']}")
    print(f"  area={iface.interface_area:.1f} A^2  hbonds={iface.number_hydrogen_bonds} "
          f"salt={iface.number_salt_bridges}")

# --- visualization ---
ana.write_pymol_script("1ktz_iface.pml")    # color interface residues by BSA (blue->red)
ana.write_molstar_html("1ktz_iface.html")   # self-contained 3D Mol* viewer (open in a browser)
ana.plot_contact_heatmap(1, out_path="1ktz_cmap.png")  # residue contact heatmap (needs matplotlib)

print(ana.hot_spot_residues(top_n=5))       # top buried residues across interfaces
```

- `1ktz_iface.pml`: `pymol 1ktz_iface.pml` opens the model with the interface
  residues coloured by buried surface area.
- `1ktz_iface.html`: a standalone Mol* viewer (loads Molecule from CDN on first
  open) with interface residues as ball-and-stick.
- `1ktz_cmap.png`: a residue-residue contact-count heatmap (requires
  `pip install fastpisa[viz]`).

Confidence from existing B-factors (works for any AlphaFold/ColabFold/Protenix
model, no JSON needed):

```python
ana = PISAInterfaceAnalyzer("tests/data/1ktz.pdb", pdb_id="1ktz", mode="pisa")
ana.analyze()
ana.load_plddt()                    # read pLDDT from the B-factor column
print(ana.model_plddt())            # overall model confidence
print(ana.plddt_scores())           # mean interface pLDDT
ana.filter_by_plddt(min_plddt=70.0) # keep confident interfaces only
```

---

## Batch analysis (`fastpisa.batch`)

Analyse many structures (e.g. AlphaFold antibody–antigen complexes) in one call,
optionally in parallel — no extra dependency (`concurrent.futures`).

```python
from fastpisa.batch import analyze_many, expand_inputs

files = expand_inputs("models/*.cif", "/path/to/a_database")
for r in analyze_many(files, mode="pisa", n_jobs=4):
    print(r["path"], r["ok"], r["n_interfaces"], r["error"])
```

`expand_inputs` expands globs / scans directories for `.pdb`/`.cif`/`.cif.gz`.
`analyze_many` returns one dict per input `{path, ok, result, n_interfaces, error}`
in the same order; a single bad file never aborts the batch. Use `n_jobs=1`
(serial) for < ~10 files and `n_jobs>1` (process pool) for larger batches.

A complete worked example lives at `examples/batch_analyze.py`:

```bash
python examples/batch_analyze.py "results/**/*.cif" -o out.jsonl --n_jobs 4
```

---

## AlphaFold confidence filtering (PAE / ipTM)

AlphaFold models ship a `*_predicted_aligned_error.json` (a residue×residue PAE
matrix plus the global `iptm`/`ptm`). fastPISA can rank and filter interfaces by
how confidently they are predicted — the standard check for whether an AlphaFold
interface is real.

```python
from fastpisa.api import PISAInterfaceAnalyzer

ana = PISAInterfaceAnalyzer("complex.cif", mode="pisa")
ana.analyze()
ana.load_pae("complex_predicted_aligned_error.json")

print(ana.pae_scores())          # mean PAE (A) per interface, lower = more confident
ana.filter_by_pae(max_pae=5.0)   # keep only interfaces with mean PAE <= 5 A
ana.filter_by_iptm(min_iptm=0.8) # drop all interfaces if model ipTM < 0.8
```

CLI: `python -m fastpisa.cli complex.cif --pae complex_..._error.json --min-pae 5.0 --min-iptm 0.8`.

### Portable confidence from B-factors (pLDDT)

The PAE JSON above is only emitted by Protenix / OpenDDE; most predictors do not
produce it. The broadly-applicable confidence signal is the **per-residue pLDDT in the
B-factor column**, which AlphaFold, ColabFold and Protenix all write into the model
(0-100, higher = more confident). No extra file is needed.

```python
ana.load_plddt()                    # read pLDDT from B-factors
print(ana.model_plddt())            # overall model confidence
print(ana.plddt_scores())           # mean interface pLDDT, higher = more confident
ana.filter_by_plddt(min_plddt=70.0) # keep interfaces whose mean pLDDT >= 70
```

CLI: `python -m fastpisa.cli complex.cif --min-plddt 70.0`. Raises a clear `ValueError`
if the model's B-factors are constant (no confidence signal).

---

## Visualisation (`fastpisa.viz`)

Three ways to look at an interface (item 4.3):

```python
from fastpisa.api import PISAInterfaceAnalyzer
ana = PISAInterfaceAnalyzer("6nxr.pdb", mode="cocomaps")
ana.analyze()
iface = ana.interfaces[0]

ana.write_pymol_script("iface.pml")             # colour interface residues by BSA
ana.write_molstar_html("iface.html")            # self-contained Mol* 3D viewer
ana.plot_contact_heatmap(1, out_path="cmap.png")# matplotlib residue-contact heatmap
```

CLI equivalents: `--pymol-script out.pml`, `--molstar out.html`, `--heatmap cmap.png`,
and `--hotspots N` prints the top-N buried residues. The matplotlib heatmap needs
`pip install fastpisa[viz]`; PyMOL-script and Mol* HTML need no extra deps.

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
├── pae.py                 # AlphaFold PAE / ipTM reading + interface filtering
├── viz.py                 # PyMOL script / matplotlib heatmap / Mol* HTML
├── batch.py               # parallel batch analysis (analyze_many)
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