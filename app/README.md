# fastPISA Interface Explorer (Streamlit)

Upload a PDB/mmCIF (or type a PDB ID), assign chains to two groups (e.g.
*antigen* vs *antibody* H+L) and get, for the interface **between the
groups**: buried surface per side and in total, solvation / stabilisation
energies with the hydrophobic-polar split, hydrogen bonds, salt bridges,
disulfides and COCOMAPS contact classes, epitope / paratope residue lists,
publication figures (footprint, residue bars, composition, bond network,
contact map), a Mol* 3D view, ChimeraX / PyMOL selections, Excel/CSV/JSON
export, a Results paragraph plus Methods text, automatic interpretation and
a guide -- and a comparison mode for several complexes sharing an antigen
(chains matched by sequence with pdb_align, footprints aligned, binders
superposed in one Mol* scene).

Run locally from the repository root:

    pip install -e . && pip install -r app/requirements.txt
    streamlit run app/streamlit_app.py

Deploy on Streamlit Community Cloud: point the app at this repository with
**main file path** `app/streamlit_app.py`; `app/requirements.txt` installs
fastPISA from the repository root. In *Advanced settings* choose
**Python 3.12** (the repository's `.python-version` says the same): the
comparison mode's dependency `pdb_align` pins `numpy<2`, which has no wheels
for Python 3.13+, and FreeSASA wheels lag new interpreters too.

After pushing a new version, use **Reboot app** in the Cloud menu once:
Streamlit re-reads the script on every run but keeps already-imported
modules, and the editable-installed `fastpisa` package sits outside the
watched app folder, so a redeployed process can otherwise keep an old
`fastpisa.report` in memory (`ImportError: cannot import name ...`). The app
also reloads such modules itself when it detects a missing symbol.

All numbers come from `fastpisa.report.group_interface`, which is a plain
Python API you can also call from a notebook.

The app analyses the first coordinate model as supplied and does not generate
crystallographic symmetry mates or biological assemblies. Ordered water is
excluded. Contact maps and the implemented interaction-class subset use
COCOMAPS-compatible conventions; this is not a claim that every COCOMAPS 2.0
interaction class is implemented. Polymer-polymer interfaces have the
strongest PDBe PISA calibration, while ligand and ion estimates should be
treated as approximate.
