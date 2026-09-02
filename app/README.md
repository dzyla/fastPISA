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
fastPISA from the repository root.

All numbers come from `fastpisa.report.group_interface`, which is a plain
Python API you can also call from a notebook.
