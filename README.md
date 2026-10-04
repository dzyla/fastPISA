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

## How it compares to PISA

fastPISA is calibrated against the original PISA engine on **674 PDB entries /
6,915 interfaces / 119k interface residues** (400 entries a seeded random draw
from a stated sampling frame, de-duplicated at 30% sequence identity; 36 legacy
hand-picked). It uses PISA's own conventions where they could be recovered —
the NACCESS/Chothia surface radii, per-element ion radii read off PISA's own
lone-ion surfaces, a per-atom-type solvation model fitted to PISA's
per-residue solvation energies — and reports every number below by **grouped
10-fold cross-validation** (no fold is scored on an entry it was fitted on).

![fastPISA vs PISA: area, solvation energy, P-value](docs/figures/energetics_vs_pisa.png)

| Quantity (vs original PISA) | Polymer–polymer (n=2,314) | All interfaces incl. ligands/ions (n=6,915) |
|---|---|---|
| Interface area | median rel. error **1.8%** (1.5% > 300 Å²) | 3.4% (ligand pairs 6.0%) |
| Per-residue buried area | median rel. error **1.75%** (119k residues) | — |
| Solvation ΔG | **r 0.987**, R² about 1:1 **0.975**, median error **0.33 kcal/mol**, slope 0.98 | r 0.956, R² 0.914, median 0.76 |
| Stabilization energy | r 0.996 (per-bond constants recovered exactly) | — |
| Hydrophobicity P-value | median error **0.060**, Spearman **0.88** | Spearman 0.38 |
| CSS | Spearman 0.75 (calibrated surrogate) | Spearman 0.64 |
| H-bond atom pairs | precision **0.958** / recall **0.952** against PISA's own bond list | — |
| Salt-bridge atom pairs | precision 0.985 / recall 0.979 | — |
| Disulfides | exact | — |

![Per-residue buried area and solvation energy](docs/figures/residues_vs_pisa.png)

Every interface also carries the **hydrophobic / polar split** of ΔG
(`solvation_energy_apolar` = carbon + sulfur burial, `solvation_energy_polar` =
the rest; they sum to `solvation_energy`). PISA has no separate hydrophobic
contact list — its hydrophobic term *is* this favourable burial.

**Where it is weaker, stated plainly.** Ligand / ion interfaces are less
accurate than chain pairs. Oxo-anions, halides, Na⁺/Ca²⁺ and organic ligands
now agree to within a few percent in area, but transition metals with short
coordination bonds (Mg, Mn, Fe, Cu, and to a lesser extent Zn) are still
buried 12–35% more than PISA buries them, and no radius or neighbour rule we
tested reproduces PISA there. Treat those energies as indicative.

![Ligand interface area by ligand type](docs/figures/ligand_area_by_type.png)

Contact maps and bond lists come out of the same run. Below: the barnase–barstar
interface (1brs A+D), COCOMAPS-classified residue contacts on the left, the
H-bond / salt-bridge atom pairs on the right, cross-checked against PISA's
list for the same interface.

![1brs contact map and bonds](docs/figures/contact_map_1brs.png)

Regenerate every figure and number offline with `python examples/make_figures.py`
and `python examples/calibrate.py` from the committed tables in
`tests/data/calibration/`.

**Speed** (single core, FreeSASA backend, `combined` mode = PISA energetics
*and* the contact map):

| atoms | molecules | interfaces | time |
|---|---|---|---|
| 1,784 | 4 | 5 | 0.4 s |
| 4,062 | 16 | 27 | 0.7 s |
| 11,550 | 6 | 7 | 1.0 s |
| 58,000 (GroEL/ES, 21 chains) | 21 | 70 | ~7 s |

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

## Getting interfaces, in three lines

```python
import fastpisa

for interface in fastpisa.interfaces("complex.pdb"):      # largest first
    print(interface.label, round(interface.interface_area), interface.solvation_energy)
    for bond in interface.hydrogen_bonds:
        print("   ", bond.label)
```

`fastpisa.interfaces()` returns a plain list of `Interface` objects sorted by
buried area, so the interface you care about is `[0]`. Each one carries its
area, solvation and stabilisation energies, P-value, CSS, bond counts and the
bonds themselves; `interface.residues()` gives the per-residue ASA/BSA table
and `interface.bonds_dataframe()` a pandas frame. Pass any
`PISAInterfaceAnalyzer` option through — `mode`, `ligand_mode`, `symmetry`,
`predict_assemblies`, `interface_cutoff`.

For the full PDBe-shaped JSON documents, assembly totals and file output use
`fastpisa.analyze(...)`; to re-run with different options or load AlphaFold
confidence, hold a `PISAInterfaceAnalyzer`.

```bash
cd fastPISA
pip install -e .

# Optional but recommended: C-accelerated ASA backend (~13x faster end to end)
pip install freesasa numpy scipy
#   - numpy, scipy: always required
#   - gemmi: required for mmCIF parsing
#   - freesasa: C library, Lee-Richards SASA in C. Installed wheels use
#     gcc to compile the C library with Python bindings.
```

Dependencies: `numpy`, `scipy` (core); `gemmi` (mmCIF); `freesasa` (fast ASA, optional).

**Crystal symmetry.** For a deposited crystal entry, original PISA analyses
the whole crystal, so **60% of the interfaces it reports involve a symmetry
mate** (322 of 802 across the cached reference set are interfaces of the
deposited coordinates alone). `--symmetry crystal` reproduces them: it
expands the asymmetric unit by its space group, keeps the copies that can
bury surface against it, and reports one entry per distinct crystal contact
with the operator that generates it.

```bash
python -m fastpisa.cli 1acb.pdb --pdb_id 1acb --symmetry crystal -o out/
```

Validated against PISA interface-by-interface, matched on chain pair *and*
relative crystal transform:

| | entries | PISA polymer–polymer interfaces found | area median err | ΔG Pearson |
|---|---|---|---|---|
| cached reference set | 37 | **566 / 566 (100%)** | 1.20% | 0.995 |
| blind random draw | 60 | **732 / 732 (100%)** | 1.16% | 0.991 |

The blind draw is a fresh seed over the stated sampling frame
(`fastpisa/reference/sampling.py`), disjoint from the 400 calibration and 36
legacy entries, so no entry in it informed any constant. Median 3.4 s per
entry, 8 s worst case (114k atoms); 0 failures. Needs a usable cell and
space group — without one the option is a no-op, which is what a predicted
model wants.

**Assembly prediction.** `--predict-assemblies` enumerates the finite
assemblies the crystal admits and ranks them, most stable first:

```bash
python -m fastpisa.cli 1acb.pdb --pdb_id 1acb --symmetry crystal \
    --predict-assemblies -o out/
```

Candidates come from nested interface subsets in PISA's dissociation order,
not an exhaustive search, so exact agreement with PISA is not expected.
Measured against PISA's own published predictions (`multimers.pisa`) over the
37 cached entries:

| | agreement | n |
|---|---|---|
| top assembly, stoichiometry (`mmsize`) | **79.4%** | 34 |
| top assembly, full composition | 41.2% | 34 |
| **author-deposited assembly** (PISA's `R350`) | **73.1%** | 26 |
| recall of PISA's assembly sizes | 88.8% | 34 |
| precision (of the sizes we emit) | 85.8% | 34 |
| PISA predicts nothing — so do we | 2 of 3 | 3 |

The denominators differ on purpose. PISA predicts no stable assembly at all
for 1ay7, 1brs and 1gpw, so those have no top assembly to compare — they are
scored on their own terms instead, because predicting a stable assembly for
barnase–barstar must cost something. Recall is reported with precision
because recall alone rises mechanically with how many assemblies are emitted,
and it excludes PISA's ligand-only assemblies, which our ligand-only filter
can never match by design.

Composition (41.2%) lags stoichiometry (79.4%) mostly because PISA's
composition string includes chain-bound hetero groups that
`ligand_mode="separate"` keeps as separate molecules (PISA `EI[SO4][ACE]`
against our `EI`) — a ligand-convention difference rather than a wrong
assembly. Reproduce with `python examples/validate_assemblies.py`.

**Surface backends.** The reference engine is the pure-Python Shrake-Rupley
implementation in `fastpisa/surface/shrake_rupley.py`; it defines the
`--point_density` quadrature and the fitted solvation parameters are validated
against it. When `freesasa` is installed it is used instead, pinned to
Lee-Richards (FreeSASA's Shrake-Rupley kernel segfaults on sparse inputs). The
two agree to **0.01% of total ASA** and the full PISA accuracy benchmark passes
on either, so results are backend-independent; pin one with
`FASTPISA_SASA_BACKEND=python|freesasa|auto` and see
[docs/surface_backends.md](docs/surface_backends.md) for the measured
comparison.

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

```python
import fastpisa

res = fastpisa.analyze("complex.pdb")        # PDB or mmCIF; AlphaFold models fine
res                                          # readable summary of every interface
# <fastPISA complex: 3 interfaces (combined mode)>
#   <Interface 1: A + B | area 779 A^2, dG -1.9 (apolar -13.3, polar +11.4), stab -10.3 kcal/mol,
#    P 0.55, CSS 0.28 | 15 H-bonds, 12 salt bridges, 0 SS>
#   ...

iface = res.interface_between("A", "B")     # or res[0], or `for iface in res:`
iface.interface_area, iface.solvation_energy, iface.stabilization_energy
iface.solvation_energy_apolar, iface.solvation_energy_polar   # hydrophobic / polar parts
iface.p_value, iface.css

for hb in iface.hydrogen_bonds:              # AtomContact objects; also .salt_bridges, .disulfides
    print(hb.label)                          #  A:ARG83.O -- D:TYR29.OH  2.65 A
iface.bonds_dataframe()                      # pandas: chain/residue/seq/atom x2, distance, type

iface.contact_map                            # COCOMAPS residue-residue map (list of dicts)
iface.contact_map_dataframe()                # ... as pandas
iface.interaction_population                 # {'hydrogen_bond': 15, 'salt_bridge': 12, 'ch_pi': ...}
iface.residues(side=1)                       # interface residues with ASA / BSA / dG each

res.to_dataframe()                           # one row per interface
res.hot_spot_residues(top_n=10)              # residues burying the most area
res.write_json("out/")                       # PDBe-PISA-shaped JSON (+ contact maps)
```

Options go through the same call: `fastpisa.analyze(path, mode="pisa")`
(energetics only, fastest), `mode="cocomaps"`, `ligand_mode="merge"` (cofactors
belong to their chain, the jsPISA-on-assembly convention), `min_css=0.5`.
The full class is `fastpisa.api.PISAInterfaceAnalyzer` (same object
`analyze()` returns) and the legacy one-shot `analyze_interface()` still works.

Visualisation helpers in `fastpisa.viz`: `plot_contact_heatmap`,
`write_pymol_script`, `write_molstar_html`; a contact-map figure like the one
above is `examples/make_figures.py::fig_contact_map`.

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

## Interface Explorer app (Streamlit) — manuscript digests

`app/streamlit_app.py` turns one or more complexes plus a **chain-group
selection** (e.g. antigen vs. antibody H+L) into what a paper needs for the
interface *between the groups*:

- buried surface **per side** and in total ("buries 820 Å² on the antigen"),
  interface area in the PISA convention; ΔG solvation with its hydrophobic /
  polar split; stabilisation energy; hydrogen bonds, salt bridges,
  disulfides (PISA rules) and COCOMAPS contact classes;
- automatic **interpretation** (packing-contact warning, hydrophobicity
  P-value reading, asymmetric burial, hot-spot candidates, polar-rich
  interfaces) and a **Guide** explaining every quantity, typical values and
  the caveats to check before quoting a number;
- **publication figures** (PNG/SVG, 300 dpi, colour-blind-safe): interface
  footprint along the sequence, per-residue buried area with hot spots,
  residue-class composition and energy decomposition, bond network, contact
  map;
- a **Mol\*** 3D view (both sides coloured, interface residues as
  ball-and-stick, bonds drawn with their distances, optional surfaces and
  labels) plus ChimeraX / PyMOL selections;
- **comparison mode** for two or more complexes on the same antigen with
  different binders: the shared chains are detected automatically by
  sequence identity (`pdb_align`), footprints are aligned residue-by-residue
  (by number, or by sequence when numbering differs), with side-by-side
  numbers, overlap (shared / unique residues, Jaccard), footprint tracks and
  a heatmap, prose, and all binders superposed on one antigen in Mol\*;
- Excel / CSV / JSON export and Results / Methods text to paste.

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

Deploy on Streamlit Community Cloud with main file `app/streamlit_app.py`.
The same digest is a plain Python API:

```python
from fastpisa.report import group_interface, interpret, compare, ComplexEntry
gi = group_interface(res, ["A"], ["H", "L"], "antigen", "Fab")
gi.buried_side1, gi.n_hbonds, gi.residue_string(1)     # epitope as R59, H102, ...
gi.results_paragraph(); gi.bonds_table(); gi.chimerax_command(); interpret(gi)
cmp = compare([ComplexEntry("Fab1", gi, res), ComplexEntry("Fab2", gi2, res2)])
cmp.summary_table(); cmp.overlap_table(); cmp.residue_matrix(); cmp.prose()
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

## What's new in 0.4.0

- **Calibrated on 674 sampled PDB entries, at residue level.** The 36-entry
  hand-picked benchmark is replaced by a seeded random draw from a stated
  sampling frame (30% identity de-duplicated); every accuracy figure is
  grouped cross-validated. PISA's per-residue solvation energies (119k
  residues) fit a 32-class + per-atom-type solvation model; polymer ΔG
  median error 1.0 → 0.33 kcal/mol.
- **PISA's own surface conventions recovered**: NACCESS/Chothia radii
  (sp2/sp3 carbon distinguished), per-element ion radii read off PISA's
  lone-ion surfaces. Per-residue buried area error 6.1% → 1.75%; ligand
  interface area 12% → 6%.
- **Atom-level bond audit** against PISA's H-bond / salt-bridge lists
  (`fastpisa/reference/bonds_audit.py`): precision/recall 0.96/0.95 and
  0.985/0.98. Fixed a parser bug that collapsed negative residue numbers
  onto 0.
- **Hydrophobic / polar split** of the solvation energy on every interface.
- **Pythonic API**: `fastpisa.analyze()`, iterable results, readable
  `repr`, `iface.hydrogen_bonds` / `.salt_bridges` / `.contact_map` /
  `.residues()` / DataFrame helpers, `res.interface_between("A", "B")`.
- **Reproducible calibration**: `examples/calibrate.py` refits every
  constant from committed tables; a test fails if the shipped constants
  drift from the data. README figures from `examples/make_figures.py`.

## What's new in 0.3.0

- **One shared analysis core** (`fastpisa/core.py`): all modes run the same
  physics once; `combined` mode (new default) delivers PISA thermodynamics
  *and* the COCOMAPS contact map in a single report.
- **Numerical parity with original PISA**: PISA interface semantics
  (buried-area-based detection), pair-specific buried surfaces, heavy-atom
  surfaces, geometric H-bond detection, ASP table + P-value + CSS calibrated
  against the EBI PISA engine (262 interfaces, 36 entries), PISA's per-bond
  energy constants recovered exactly (−0.444/−0.150/−4.0 kcal/mol).
- **COCOMAPS 2.0-faithful contact maps**: identical residue-pair maps
  (validated against the actual COCOMAPS 2.0 tool), ring-geometry-validated
  π classes, COCOMAPS conventions for salt bridges (incl. DNA phosphates),
  vdW/proximal/clash classes, full per-pair class breakdowns.
- **Faster**: local-delta per-pair surfaces, vectorised masks; GroEL/GroES
  (58k atoms, 70 interfaces) in ~7 s with FreeSASA.
- New: `ligand_mode="merge"`, `.pdb.gz` input, PDBe PISA 2.0 assembly
  comparison (`--assembly-entries`), offline accuracy regression tests, CI.

## Validation & benchmark vs original PISA

**Ground truths used** (all comparisons reproducible from this repo):

1. **Classic EBI PISA engine** — XML from
   `https://www.ebi.ac.uk/pdbe/pisa/cgi-bin/interfaces.pisa?<id>` for 674
   entries (identity/ASU interfaces; per-interface energetics, per-residue
   ASA/BSA/ΔG and the atom-level bond lists). The 36 legacy entries are
   cached in `tests/data/reference/`; the 400 sampled entries are distilled
   into `tests/data/calibration/` (sampling frame and seed:
   `fastpisa/reference/sampling.py`, entry list `entries.json`). Drives the
   calibration, `tests/test_calibration_benchmark.py` (out-of-sample) and
   `tests/test_vs_pdbe_pisa.py` (in-sample regression).
2. **PDBe PISA 2.0 JSON API** (biological assemblies; covers recent
   entries) — blind test on 20 depositions from 2023–2024, fastPISA run on
   the same assembly coordinates.
3. **COCOMAPS 2.0 standalone code** (Zenodo `10.5281/zenodo.17390665`) run
   locally on the same inputs; its residue-pair tables are cached in
   `tests/data/reference/cocomaps2/` and pinned by
   `tests/test_vs_cocomaps2.py`.

Accuracy: `python examples/compare_vs_pisa.py` runs the full head-to-head
against the original PISA engine (EBI PDBe PISA service; XML + PDB files
cached under `tests/data/reference/`, so it works offline) and prints the
agreement table shown at the top of this README. The same numbers are pinned
as a regression test in `tests/test_vs_pdbe_pisa.py`. `--fetch <pdbid> ...`
extends the benchmark with new entries (network required once).

Speed (with the FreeSASA C backend): typical complexes take 0.2–1 s (table
above); GroEL/GroES (1aon: 58k atoms, 21 chains, 70 interfaces) takes ~7 s in
combined mode. fastPISA is ~3–4x faster than the original CCP4 binary on
comparable inputs and ~13x faster than its own pure-Python fallback. Per-pair
surfaces are computed only near each interface, and candidate molecule pairs
come from a single pass over the atoms rather than a scan per pair, so runtime
scales with interface count and local size, not with (molecules)² × structure
size. (That screening pass is what makes many-molecule inputs tractable: 1brs
with ordered water included — 519 molecules, 134k candidate pairs — went from
31.5 s to 2.4 s.)

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
│   ├── rings.py           # ring centroids/normals for pi-class geometry
│   ├── contact_map.py     # residue-residue contact map + matrix
│   └── pipeline.py        # COCOMAPS-mode entry point (thin wrapper over core)
├── interface/
│   ├── contacts.py        # molecule detection, masks, contacts (shared)
│   └── bonds.py           # geometric H-bond / salt-bridge / disulfide detection
├── surface/
│   ├── shrake_rupley.py   # reference Shrake-Rupley ASA + backend switch
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
  674-entry benchmark described under *Validation status* in `CLAUDE.md` and
  asserted by `tests/test_calibration_benchmark.py` (grouped 10-fold CV, folds
  never splitting a PDB entry). Single-ion interfaces
  (a lone Zn²⁺/Ca²⁺) carry the largest relative ΔG errors — PISA uses
  ion-specific desolvation terms that fastPISA approximates with one metal
  class. CSS is a calibrated surrogate: exact CSS requires PISA's crystal-wide
  assembly analysis.
- **ASA/BSA convention**: the two backends run different quadratures of the
  same surface (Shrake-Rupley with `--point_density` points in Python;
  Lee-Richards with 20 slices in FreeSASA) and agree to 0.01% of total ASA,
  median 0.09 Å² per atom, max 1.6 Å² — see
  [docs/surface_backends.md](docs/surface_backends.md). Interface *detection* is
  identical. Explicit hydrogens are excluded from all surfaces (the PISA
  convention) but are used for H-bond geometry when present.
- **Assembly dissociation**: `dissociation_energy` and `entropy` follow PISA's
  own relation, `ΔG_diss = −Σ(stabilization energy over the interfaces cut) −
  TΔS`, along the cheapest dissociation pathway. `TΔS` is the rigid-body
  translational entropy of the released bodies with one constant fitted to
  PISA. Against PISA's own assembly values: entropy median |err| 0.90 kcal/mol
  (r 0.90), dissociation energy median |err| 1.9 kcal/mol, Spearman 0.98
  (n = 20). Validated by `tests/test_dissociation.py`.
- **Symmetry**: `--symmetry crystal` enumerates crystal packing interfaces
  with symmetry mates (see below). Biological-assembly *prediction* (searching
  the crystal for the most stable assembly) is still not implemented; the
  dissociation machinery it would need is in `fastpisa/energy/dissociation.py`.
- **Partial occupancy**: interface areas are unreliable when many atoms carry
  occupancy < 1. On a blind 60-entry draw, the 2 entries with >10%
  partial-occupancy heavy atoms had a 33% median area error against PISA
  versus 0.91% for the other 58. Investigated without resolution — see
  *Known limit* in `CLAUDE.md`.
- **COCOMAPS classifier** is a rule-based subset of COCOMAPS 2.0's 16 interaction
  classes; H-bonds share fastPISA's geometric detector, but it does not run
  HBPLUS or add hydrogens.

## Licence and citation

**AGPL-3.0-or-later**, with an attribution term under section 7(b). In plain
terms:

- Use it, run it, study it and share it freely — including inside a company
  and including commercially.
- Distribute it or a modified version, and it must go out under this same
  licence with source available.
- **Run a modified version as a network service and section 13 applies:** you
  must offer that service's users the source of your modification. Hosting
  fastPISA as a closed service is not permitted. That clause is the reason
  for AGPL over plain GPL.
- Charging money is not restricted, and could not be without making this
  non-free software. Keeping a distributed or hosted modification *closed* is
  what is restricted.
- It cannot be combined into proprietary software, or into a project whose
  licence will not carry these terms. If you need that, ask.

Dependencies are compatible: numpy/scipy BSD, freesasa MIT, gemmi MPL-2.0.

**If you publish results from fastPISA, please cite it** — `CITATION.cff` has
the machine-readable record, and every release carries its own version and
DOI. Cite the version you ran: the numbers have changed between versions, and
`CHANGELOG.md` says how. `analysis_provenance()` reports the version, surface
backend and options of a given run, which is what belongs in a methods
section.

Original PISA is © Eugene Krissinel (CCP4) and is a separate work under the
CCP4 licence; fastPISA is an independent reimplementation containing no CCP4
code. Please cite the PISA paper (Krissinel & Henrick, *J. Mol. Biol.*
**372**:774–797, 2007) alongside this software, and COCOMAPS 2.0 (Chawla
*et al.*, *Bioinformatics*, 2025) if you use the contact-map mode.