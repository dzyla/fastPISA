# fastPISA

Local, fast analysis of biomolecular interfaces — a Python reproduction of
[PISA](https://www.ebi.ac.uk/pdbe/pisa/) (Krissinel & Henrick 2007) calibrated
against the original engine, with a **COCOMAPS 2.0** contact-map mode. Reads
PDB **and mmCIF** files (including AlphaFold predicted complexes).

All modes run one shared analysis core, so they **always identify exactly the
same interfaces** for a structure.

| Mode | What it reports | Output |
|------|-----------------|--------|
| `combined` *(default)* | One unified report per interface: PISA thermodynamics **and** the COCOMAPS contact map | PISA schema + `interface_contact_map` per interface |
| `pisa` | Thermo/surface analysis: ASA/BSA, interface areas, ΔG, P-value, CSS, H-bonds / salt bridges / disulfides | PDBe PISA `assembly.json` + `interfaces.json` |
| `cocomaps` | Residue–residue contact map with atomic interaction-type classification (H-bond, salt bridge, pi-pi, cation-pi, ch-pi, …) | Superset of the PISA schema + `interface_contact_map` per interface |

**Validated against original PISA** (EBI PDBe PISA service, 262 identity
interfaces from 36 diverse PDB entries — proteases, antibodies, receptors,
hemoglobin, protein–DNA/RNA, glycans, cofactors, ions; reproduce with
`python examples/compare_vs_pisa.py`):

| Quantity | All interfaces (n=262) | Polymer–polymer only (n=153)* |
|---|---|---|
| Interface area | median rel. error 2.2% (1.3% > 300 Å²) | 1.5% |
| Solvation ΔG | Pearson 0.956, median error 1.05 kcal/mol | **Pearson 0.980** |
| Stabilization energy | Pearson 0.977 (per-bond constants recovered exactly) | **Pearson 0.988** |
| P-value | median error 0.12 | Spearman 0.72 |
| CSS | Spearman 0.71 (calibrated surrogate) | Spearman 0.75 |
| H-bond counts | 91% within ±1 | — |
| Salt bridges | mean diff 0.08 per interface | — |
| Disulfides | 100% exact | — |

\* the cryo-EM / AlphaFold-model regime (protein/nucleic-acid chain pairs, no
small-molecule ligand side). Verified out-of-sample twice: on 15 classic
entries the fit had never seen (polymer–polymer ΔG Pearson 0.977), and on
**20 recent (2023–2024) depositions** compared blind against the modern PDBe
PISA 2.0 JSON API on biological-assembly coordinates — 87 interfaces, ΔG
Pearson **0.978**, stab **0.986**, area 3.0% median, P-value error 0.11
(`python examples/compare_vs_pisa.py --assembly-entries <ids>`). H-bond
counts differ more vs PISA *2.0* (64% within ±1) than vs classic PISA (91%)
— the two PISA versions themselves disagree on H-bond criteria; fastPISA is
calibrated to the classic engine.

**Validated against COCOMAPS 2.0** (the actual standalone tool, Zenodo
`10.5281/zenodo.17390665`, run on the same inputs with REDUCE-added
hydrogens): the residue–residue **contact map is identical** on all tested
complexes — protein–protein (1ktz, 30/30 pairs), antibody–antigen (1vfb
VH–lysozyme, 28/28) and protein–DNA (1aay zinc-finger, 57/57) — with
identical interface residue sets, and COCOMAPS-convention salt bridges
(including Lys/Arg–DNA-phosphate) matching per residue pair. Interaction
classes follow COCOMAPS 2.0 conventions (vdW contacts within r₁+r₂+0.5 Å,
"proximal" beyond, ring-geometry-validated π classes); the residual
differences are H-dependent classes (their H-bonds come from HBPLUS, their
weak C–H bonds use explicit-H angles), pinned in
`tests/test_vs_cocomaps2.py`.

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
python -m fastpisa.cli 6nxr.pdb --pdb_id 6nxr --output_dir out            # combined (default)
python -m fastpisa.cli complex.cif --mode pisa --pdb_id my --time --json-summary -o out
```

Common options:

| Flag | Meaning |
|------|---------|
| `--mode {combined,pisa,cocomaps}` | Analysis mode (default `combined`) |
| `--pdb_id` | PDB id used in output filenames |
| `--probe_radius`, `--point_density`, `--interface_cutoff` | Core geometry knobs |
| `--no-water` / `--with-water` | Exclude (default) or include ordered water in the interface search |
| `--ligand-mode {separate,merge}` | `separate` (default): each bound hetero group is its own monomer, classic PISA. `merge`: a chain's ligands/cofactors count toward that chain's interfaces (jsPISA assembly convention) |
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
- `.mode` — `"combined"` (default), `"pisa"` or `"cocomaps"`; `.analyze(recompute=True)` re-runs

---

## Worked example — analyze a PDB and visualize an interface

`tests/data/1ktz.pdb` is a small two-chain (A/B) complex — a good first run.

CLI:

```bash
# combined mode (default) -> PISA thermodynamics + COCOMAPS contact map
python -m fastpisa.cli tests/data/1ktz.pdb --pdb_id 1ktz -o out --hotspots 5
```

```text
=== Summary ===
Mode: combined
Interfaces found: 1
Assembly dissociation energy: 30.92
Total ASA: 11576.73
Total BSA: 541.13

Top 5 hotspot residues (by buried area):
  A94 (ARG) BSA=83.4 A^2  interfaces=[1]
  B53 (ILE) BSA=81.9 A^2  interfaces=[1]
  A91 (TYR) BSA=75.6 A^2  interfaces=[1]
  A31 (LYS) BSA=70.2 A^2  interfaces=[1]
  A93 (GLY) BSA=66.5 A^2  interfaces=[1]

  Interface 1: 30 residue pairs
    Interaction population: {'hydrogen_bond': 6, 'salt_bridge': 10,
    'weak_hbond': 22, 'polar_vdw': 101, 'ch_pi': 73, 'apolar_vdw': 58, 'cation_pi': 6}
```

For 1ktz's A–B interface original PISA reports area 493.4 Å², ΔG −4.3
kcal/mol, P-value 0.50, 9 H-bonds, 8 salt bridges; fastPISA gives 483.5 Å²,
−3.3 kcal/mol, P-value 0.52, 9 H-bonds, 8 salt bridges on the same input.

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

The PAE JSON above is only emitted by Protenix-style pipelines; most predictors do not
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

## Validation & benchmark vs original PISA

Accuracy: `python examples/compare_vs_pisa.py` runs the full head-to-head
against the original PISA engine (EBI PDBe PISA service; XML + PDB files
cached under `tests/data/reference/`, so it works offline) and prints the
agreement table shown at the top of this README. The same numbers are pinned
as a regression test in `tests/test_vs_pdbe_pisa.py`. `--fetch <pdbid> ...`
extends the benchmark with new entries (network required once).

Speed (with the FreeSASA C backend): the original 21-entry benchmark runs in ~7 s
total; GroEL/GroES (1aon: 58k atoms, 21 chains, 70 interfaces) takes 7.4 s in
combined mode. fastPISA is ~3–4x faster than the original CCP4 binary on
comparable inputs and ~15x faster than its own pure-Python fallback. Per-pair
surfaces are computed only near each interface, so runtime scales with
interface count and local size, not with (chains)² × structure size.

Note on interface *counts*: original PISA run on a crystal entry also reports
symmetry-mate (crystal packing) interfaces; fastPISA reports the interfaces
present in the given coordinate set (the identity/ASU interfaces — everything
an AlphaFold/cryo-EM model has). Comparisons therefore match on identity
interfaces, which is a documented scope decision, not a bug.

---

## Layout

```
fastpisa/
├── core.py                # THE shared analysis core (all modes run this once)
├── api.py                 # PISAInterfaceAnalyzer class + analyze_interface()
├── cli.py                 # command-line interface
├── pipeline.py            # PISA-mode entry point (thin wrapper over core)
├── pae.py                 # AlphaFold PAE / ipTM reading + interface filtering
├── viz.py                 # PyMOL script / matplotlib heatmap / Mol* HTML
├── batch.py               # parallel batch analysis (analyze_many)
├── cocomaps/              # COCOMAPS 2.0 mode
│   ├── interactions.py    # atomic interaction-type classifier
│   ├── contact_map.py     # residue-residue contact map + matrix
│   └── pipeline.py        # COCOMAPS-mode entry point (thin wrapper over core)
├── interface/
│   ├── contacts.py        # molecule detection, masks, contacts (shared)
│   └── bonds.py           # geometric H-bond / salt-bridge / disulfide detection
├── surface/
│   ├── shrake_rupley.py   # pure-Python Shrake-Rupley ASA
│   └── freesasa_backend.py# optional C-accelerated ASA (auto-dispatched)
├── energy/                # PISA-calibrated ASP table, ΔGsolv, bond energies
├── scoring/               # PISA-definition P-value, calibrated CSS
├── reference/             # EBI PISA reference fetching + comparison harness
├── output/                # PDBe PISA JSON builders
└── parser/pdb_parser.py   # PDB(.gz) + mmCIF parsing (gemmi)
```

---

## Notes / caveats

- **Calibration scope**: the ΔG/P-value/CSS calibration was fitted on the
  21-entry EBI benchmark (leave-one-PDB-out validated). Single-ion interfaces
  (a lone Zn²⁺/Ca²⁺) carry the largest relative ΔG errors — PISA uses
  ion-specific desolvation terms that fastPISA approximates with one metal
  class. CSS is a calibrated surrogate: exact CSS requires PISA's crystal-wide
  assembly analysis.
- **ASA/BSA convention**: with the FreeSASA backend, absolute ASA/BSA values use
  FreeSASA's parameters; with the pure-Python backend they use our 480-point
  convention. Values differ slightly between the two backends, but interface
  *detection* is identical. Explicit hydrogens are excluded from all surfaces
  (the PISA convention) but are used for H-bond geometry when present.
- **Symmetry**: crystal-symmetry (packing-mate) interface enumeration and
  biological-assembly prediction are not implemented (see benchmark note).
- **COCOMAPS classifier** is a rule-based subset of COCOMAPS 2.0's 16 interaction
  classes; H-bonds share fastPISA's geometric detector, but it does not run
  HBPLUS or add hydrogens.

## License

Original PISA is (C) Eugene Krissinel (CCP4) — see the CCP4 license. This
reimplementation is provided for research use.