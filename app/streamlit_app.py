"""fastPISA Interface Explorer -- manuscript-ready interface digests.

Run locally:   streamlit run app/streamlit_app.py
Streamlit Cloud: main file path ``app/streamlit_app.py`` (requirements in
``app/requirements.txt``).

Upload a structure (or type a PDB ID), assign chains to two groups (e.g.
antigen vs. antibody H+L) and get the buried surface per side, energies,
bond and contact counts, epitope / paratope residue lists, a contact map,
viewer commands and copy-pasteable Results / Methods text.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import os
import sys
import tempfile
import urllib.request

import pandas as pd
import streamlit as st

import fastpisa
from fastpisa.report import chain_inventory, group_interface, one_letter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app_helpers import (  # noqa: E402
    _excel_bytes, _fig_to_bytes, contact_map_figure, residue_barplot, viewer_html,
)

st.set_page_config(page_title="fastPISA Interface Explorer", page_icon="🧬", layout="wide")

CACHE_DIR = os.path.join(tempfile.gettempdir(), "fastpisa_app_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------
def _write_upload(uploaded) -> str:
    data = uploaded.getvalue()
    digest = hashlib.sha1(data).hexdigest()[:12]
    name = uploaded.name
    path = os.path.join(CACHE_DIR, f"{digest}_{name}")
    if not os.path.exists(path):
        with open(path, "wb") as fh:
            fh.write(data)
    return path


@st.cache_data(show_spinner=False)
def _fetch_pdb(pdb_id: str) -> str:
    pdb_id = pdb_id.strip().lower()
    path = os.path.join(CACHE_DIR, f"{pdb_id}.cif.gz")
    if not os.path.exists(path):
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif.gz"
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        with open(path, "wb") as fh:
            fh.write(data)
    return path


@st.cache_resource(show_spinner="Running fastPISA ...")
def _analyze(path: str, ligand_mode: str, exclude_water: bool, min_css: float):
    pdb_id = os.path.basename(path).split(".")[0].split("_", 1)[-1]
    return fastpisa.analyze(path, pdb_id=pdb_id, ligand_mode=ligand_mode,
                            exclude_water=exclude_water, min_css=min_css)


def _structure_text(path: str) -> str:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("fastPISA Interface Explorer")
st.caption("PISA-calibrated interface analysis, digested for manuscripts. "
           "Upload a structure, pick two groups of chains, copy the numbers.")

with st.sidebar:
    st.header("Structure")
    uploaded = st.file_uploader("PDB / mmCIF (optionally .gz)", type=["pdb", "cif", "ent", "gz", "mmcif"])
    pdb_id_in = st.text_input("... or fetch a PDB ID", placeholder="e.g. 1brs")
    st.header("Analysis options")
    ligand_mode = st.selectbox("Ligands / cofactors", ["separate", "merge"], index=0,
                               help="separate: every hetero group is its own molecule (classic PISA). "
                                    "merge: a chain's bound ligands belong to that chain.")
    exclude_water = st.checkbox("Exclude water", value=True)
    min_css = st.slider("Minimum CSS to keep an interface", 0.0, 1.0, 0.0, 0.05)

path = None
if uploaded is not None:
    path = _write_upload(uploaded)
elif pdb_id_in.strip():
    try:
        path = _fetch_pdb(pdb_id_in)
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Could not fetch {pdb_id_in}: {exc}")

if path is None:
    st.info("Upload a structure or enter a PDB ID in the sidebar to begin.")
    st.markdown(
        "**What you get:** buried surface per side and in total, solvation and "
        "stabilisation energies (with the hydrophobic / polar split), hydrogen bonds, "
        "salt bridges and disulfides, COCOMAPS contact classes, epitope / paratope "
        "residue lists, a contact-map figure, ChimeraX / PyMOL selections, and a "
        "Results paragraph plus Methods text you can paste into a manuscript.")
    st.stop()

res = _analyze(path, ligand_mode, exclude_water, float(min_css))
inventory = chain_inventory(res)

# ---- chain groups -------------------------------------------------------
with st.sidebar:
    st.header("Chain groups")
    label1 = st.text_input("Group 1 name", "antigen")
    label2 = st.text_input("Group 2 name", "antibody")
    inv_df = pd.DataFrame([{
        "molecule": r["label"], "type": r["class"], "residues": r["n_residues"],
        label1: False, label2: False} for r in inventory])
    edited = st.data_editor(
        inv_df, hide_index=True, width="stretch",
        column_config={"molecule": st.column_config.TextColumn(disabled=True),
                       "type": st.column_config.TextColumn(disabled=True),
                       "residues": st.column_config.NumberColumn(disabled=True),
                       label1: st.column_config.CheckboxColumn(),
                       label2: st.column_config.CheckboxColumn()},
        key="chain_groups")
    group1 = list(edited.loc[edited[label1], "molecule"])
    group2 = list(edited.loc[edited[label2], "molecule"])

st.subheader(f"{res.pdb_id}: {len(res)} interfaces found")
if not group1 or not group2:
    st.info("Tick the chains that belong to each group in the sidebar. The tables "
            "below list every interface in the structure meanwhile.")
    st.dataframe(res.to_dataframe(), width="stretch")
    st.stop()

try:
    gi = group_interface(res, group1, group2, label1, label2)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if gi.empty:
    st.warning(f"No interface between {label1} ({', '.join(group1)}) and {label2} ({', '.join(group2)}).")
    st.dataframe(res.to_dataframe(), width="stretch")
    st.stop()

tab_sum, tab_res, tab_bonds, tab_3d, tab_export, tab_all = st.tabs(
    ["Summary", "Residues", "Bonds & contacts", "3D view", "Export", "All interfaces"])

with tab_sum:
    c = st.columns(4)
    c[0].metric(f"Buried on {label1}", f"{gi.buried_side1:,.0f} A²")
    c[1].metric(f"Buried on {label2}", f"{gi.buried_side2:,.0f} A²")
    c[2].metric("Total buried surface", f"{gi.buried_total:,.0f} A²",
                help="Sum of both sides. PISA's 'interface area' is half of this.")
    c[3].metric("Interface area (PISA)", f"{gi.interface_area:,.0f} A²")
    c = st.columns(4)
    c[0].metric("ΔG solvation", f"{gi.dg_solv:+.1f} kcal/mol",
                help=f"apolar {gi.dg_apolar:+.1f}, polar {gi.dg_polar:+.1f}")
    c[1].metric("Stabilisation energy", f"{gi.stab_energy:+.1f} kcal/mol",
                help="ΔG solvation plus PISA's per-bond terms")
    c[2].metric("Hydrogen bonds", gi.n_hbonds)
    c[3].metric("Salt bridges", gi.n_salt_bridges,
                help=f"disulfides: {gi.n_disulfides}; residue pairs in contact: {gi.n_residue_pairs}")
    st.markdown("**Results text** (edit freely):")
    st.code(gi.results_paragraph(), language=None)
    st.markdown("**Contributing chain-pair interfaces**")
    st.dataframe(gi.pair_table(), width="stretch", hide_index=True)
    st.caption("P-value: probability that a random surface patch is as hydrophobic (low = specific). "
               "CSS: complexation significance score. Neither is additive over pairs.")

with tab_res:
    c1, c2 = st.columns(2)
    for col, side, label in ((c1, 1, label1), (c2, 2, label2)):
        with col:
            st.markdown(f"**{label} interface residues** ({len(gi.residues_side1 if side == 1 else gi.residues_side2)})")
            st.code(gi.residue_string(side), language=None)
            st.dataframe(gi.residue_table(side), width="stretch", hide_index=True, height=360)
    fig = residue_barplot(gi)
    st.pyplot(fig, width="stretch")
    c = st.columns(2)
    c[0].download_button("Download bar plot (PNG)", _fig_to_bytes(fig, "png"), "interface_residues.png", "image/png")
    c[1].download_button("Download bar plot (SVG)", _fig_to_bytes(fig, "svg"), "interface_residues.svg", "image/svg+xml")

with tab_bonds:
    bt = gi.bonds_table()
    kinds = st.multiselect("Bond types", ["hydrogen bond", "salt bridge", "disulfide"],
                           default=["hydrogen bond", "salt bridge", "disulfide"])
    view = st.radio("Summarise by", [f"per bond", f"per {label1} residue", f"per {label2} residue"], horizontal=True)
    sub = bt[bt["type"].isin(kinds)] if len(bt) else bt
    if view != "per bond" and len(sub):
        key = label1 if view.endswith(f"{label1} residue") else label2
        other = label2 if key == label1 else label1
        grouped = (sub.groupby([key, "type"])[other]
                   .agg(lambda s: ", ".join(dict.fromkeys(s)))
                   .reset_index()
                   .rename(columns={other: f"{other} partners"}))
        grouped["n"] = sub.groupby([key, "type"]).size().values
        st.dataframe(grouped, width="stretch", hide_index=True)
    else:
        st.dataframe(sub.drop(columns=["chain 1", "seq 1", "chain 2", "seq 2"]) if len(sub) else sub,
                     width="stretch", hide_index=True)
    st.markdown("**COCOMAPS interaction classes** (atom pairs within 5 A)")
    pop = gi.interaction_population
    if pop:
        st.dataframe(pd.DataFrame([{"class": k.replace("_", " "), "atom pairs": v}
                                   for k, v in sorted(pop.items(), key=lambda kv: -kv[1])]),
                     hide_index=True)
    fig = contact_map_figure(gi)
    if fig is not None:
        st.pyplot(fig, width="content")
        c = st.columns(2)
        c[0].download_button("Download contact map (PNG)", _fig_to_bytes(fig, "png"), "contact_map.png", "image/png")
        c[1].download_button("Download contact map (SVG)", _fig_to_bytes(fig, "svg"), "contact_map.svg", "image/svg+xml")

with tab_3d:
    fmt = "cif" if ".cif" in path or ".mmcif" in path else "pdb"
    st.components.v1.html(viewer_html(_structure_text(path), fmt, gi), height=540)
    st.caption(f"{label1} in orange, {label2} in blue; interface residues as sticks.")
    c1, c2 = st.columns(2)
    c1.markdown("**ChimeraX**")
    c1.code(gi.chimerax_command(), language="bash")
    c2.markdown("**PyMOL**")
    c2.code(gi.pymol_command(), language="bash")

with tab_export:
    import json
    sheets = {"summary": pd.DataFrame([gi.to_dict() | {"residues_side1": None, "residues_side2": None}]),
              "chain pairs": gi.pair_table(), "bonds": gi.bonds_table(),
              f"{label1} residues": gi.residue_table(1), f"{label2} residues": gi.residue_table(2),
              "contact map": gi.contact_map_table(), "all interfaces": res.to_dataframe()}
    c = st.columns(3)
    c[0].download_button("Excel workbook (all tables)", _excel_bytes(sheets), f"{res.pdb_id}_interface.xlsx",
                         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    c[1].download_button("Digest (JSON)", json.dumps(gi.to_dict(), indent=2), f"{res.pdb_id}_interface.json",
                         "application/json")
    c[2].download_button("Full fastPISA output (JSON)", json.dumps(res.interfaces_json, indent=1),
                         f"{res.pdb_id}_fastpisa_interfaces.json", "application/json")
    for name, df in sheets.items():
        if name in ("summary",):
            continue
        st.download_button(f"CSV: {name}", df.to_csv(index=False).encode(), f"{res.pdb_id}_{name.replace(' ', '_')}.csv",
                           "text/csv", key=f"csv_{name}")
    st.markdown("**Methods text**")
    st.code(gi.methods_paragraph(), language=None)
    st.caption("Cite: Krissinel & Henrick, J. Mol. Biol. 372:774 (2007) for PISA; "
               "Chawla et al., Bioinformatics (2025) for COCOMAPS 2.0; fastPISA (github.com/dzyla/fastPISA).")

with tab_all:
    st.dataframe(res.to_dataframe(), width="stretch", hide_index=True)
    st.code(repr(res), language=None)
