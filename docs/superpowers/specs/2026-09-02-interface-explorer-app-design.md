# Interface Explorer: a Streamlit app for manuscript-ready interface digests

Date: 2026-09-02. Status: approved, implementing.

## Goal

Replace the PISA-XML-based `pisa_interface_plotter` with an app that runs
fastPISA on an uploaded structure (or a PDB ID) and turns a **chain-group
selection** (e.g. antigen vs. antibody H+L) into the numbers, tables,
figures and sentences a manuscript needs. fastPISA is the only engine; no
PISA XML input. Deployable on Streamlit Cloud from this repository.

## Non-negotiables from the user

- Select chains into two groups and get the interface *between the groups*
  in digested form: buried surface (per side and total), energies, how many
  interactions of each kind, which residues.
- Output usable in a manuscript immediately: copy-pasteable numbers and
  sentences, exportable tables and figures.
- Keep everything in the fastPISA repo.

## Design

### 1. `fastpisa/report.py` -- the digest, as a library (tested, no Streamlit)

`group_interface(result, group1, group2, label1="Group 1", label2="Group 2")`
-> `GroupInterface` dataclass. The group interface is the set of chain-pair
interfaces with one chain in each group (hetero groups belonging to a chain
count with that chain when `ligand_mode="merge"`; in the default mode they
are separate molecules and appear only if selected explicitly).

Fields (every quantity states its convention in the docstring):

- `pairs`: the contributing `Interface` objects
- `interface_area` (PISA convention: half the total buried area, summed over
  pairs); `buried_total` (= 2x); `buried_side1`, `buried_side2` (sum of each
  side's BSA -- "buries N A^2 on the antigen")
- `dg_solv`, `dg_apolar`, `dg_polar`, `stab_energy` (sums over pairs; these
  are additive by construction)
- `n_hbonds`, `n_salt_bridges`, `n_disulfides` (PISA rules), plus the
  COCOMAPS interaction population summed over pairs (`n_apolar_contacts`,
  `n_pi_contacts`, ...) and `n_residue_pairs`
- `residues_side1`, `residues_side2`: interface residues with chain, name,
  one-letter code, seq, icode, `bsa`, `asa`, `dg`, `fraction_buried`,
  bonds count -- the epitope / paratope
- per-pair rows (`pair_table()`), bonds (`bonds_table()`), residues
  (`residue_table(side)`), contact map (`contact_map_table()`), all pandas
- `results_sentence()` / `results_paragraph()`: the numbers as prose with
  the group labels; `methods_paragraph()`: one paragraph citing PISA
  (Krissinel & Henrick 2007), COCOMAPS 2.0 and fastPISA with the conventions
  used (probe 1.4 A, NACCESS radii, PISA H-bond criteria, buried-area
  interface definition)
- `chimerax_command()` / `pymol_command()`: select interface residues per
  side, coloured by side
- `to_dict()` for JSON export

P-values and CSS are not additive; they are reported per pair only.

### 2. `app/streamlit_app.py` -- presentation only

Sidebar: upload PDB/mmCIF/.gz or PDB ID (RCSB fetch, cached), analysis
options (ligand mode, exclude water, min CSS), then chain assignment: each
chain has a checkbox for Group 1 / Group 2 and an editable group label.
Chains are listed with residue count and type (protein / DNA / RNA / ligand)
so antibody chains are easy to spot.

Tabs:
1. **Summary** -- metric tiles (buried per side, total, dG with the
   apolar/polar split, stab energy, H-bonds, salt bridges), the results
   paragraph in a copy box, per-pair table.
2. **Residues** -- epitope / paratope tables and the one-letter strings,
   BSA barplot per side (matplotlib, PNG/SVG download).
3. **Bonds & contacts** -- bonds table with the old app's grouping options
   (by side 1 / by side 2 / all pairs, merge symmetric copies), COCOMAPS
   contact map figure (PNG/SVG), interaction class counts.
4. **3D** -- py3Dmol view, interface residues coloured by side, chains as
   cartoon; ChimeraX / PyMOL commands in code boxes.
5. **Export** -- CSV / Excel of every table, JSON of the digest, the
   methods paragraph, the raw fastPISA JSON.
6. **All interfaces** -- the plain per-interface table for the whole
   structure (the fastPISA `to_dataframe()`), for when no grouping is wanted.

Requirements (`app/requirements.txt`): fastpisa (from the repo root via
`-e .`), streamlit, pandas, matplotlib, py3Dmol, freesasa, gemmi, openpyxl.
Streamlit Cloud: main file `app/streamlit_app.py`.

### 3. Tests

`tests/test_report.py`: on 1brs (barnase A vs barstar D; and the multi-chain
case A+B vs D+E+F) assert additivity (sums equal per-pair sums), side BSA
convention, residue lists non-empty and sorted, sentences contain the
numbers, ChimeraX command syntax. The app itself is exercised by importing
its pure helpers only.

## Out of scope

PISA XML import; assembly / symmetry-mate generation; editing structures.
