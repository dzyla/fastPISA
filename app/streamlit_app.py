"""fastPISA Interface Explorer -- manuscript-ready interface digests.

Run locally:   streamlit run app/streamlit_app.py
Streamlit Cloud: main file path ``app/streamlit_app.py`` (requirements in
``app/requirements.txt``).

Add one or more complexes (upload or PDB ID), assign chains to two named
groups per complex (e.g. antigen vs. antibody H+L) and get: buried surface
per side, energies with the hydrophobic/polar split, bonds and contact
classes, epitope / paratope residues, publication figures, a Mol* 3D view,
viewer commands, Results / Methods text -- and, with two or more
complexes, a comparison on the shared side (same antigen, different
binders), with the binders superposed in one scene.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import urllib.request

import pandas as pd
import streamlit as st

import fastpisa
from fastpisa.report import (
    GUIDE, ComplexEntry, chain_inventory, compare, group_interface, interpret,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figures as F  # noqa: E402
from alignment import detect_shared_chains, structure_text, superpose  # noqa: E402
from app_helpers import excel_bytes  # noqa: E402
from molstar_view import comparison_view_html, interface_view_html  # noqa: E402

st.set_page_config(page_title="fastPISA Interface Explorer", page_icon="🧬", layout="wide")

CACHE_DIR = os.path.join(tempfile.gettempdir(), "fastpisa_app_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]


# ---------------------------------------------------------------------------
# data access (cached)
# ---------------------------------------------------------------------------
def _write_upload(uploaded) -> str:
    data = uploaded.getvalue()
    digest = hashlib.sha1(data).hexdigest()[:12]
    path = os.path.join(CACHE_DIR, f"{digest}_{uploaded.name}")
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


@st.cache_data(show_spinner="Superposing on the shared chains ...")
def _superpose_cached(ref_path: str, mob_path: str, ref_chains: tuple, mob_chains: tuple):
    sp = superpose(ref_path, mob_path, list(ref_chains), list(mob_chains))
    return {"text": sp.aligned_text, "rmsd": sp.rmsd, "n": sp.n_aligned, "tm": sp.tm_score}


@st.cache_data(show_spinner="Matching chains by sequence ...")
def _detect_cached(ref_path: str, mob_path: str, ref_chains: tuple, candidates: tuple):
    return [(m.ref_chain, m.mob_chain, m.identity)
            for m in detect_shared_chains(ref_path, mob_path, list(ref_chains), list(candidates))]


def _fig_download(fig, name: str, key: str):
    c = st.columns([1, 1, 6])
    c[0].download_button("PNG", F.fig_bytes(fig, "png"), f"{name}.png", "image/png", key=f"{key}_png")
    c[1].download_button("SVG", F.fig_bytes(fig, "svg"), f"{name}.svg", "image/svg+xml", key=f"{key}_svg")


# ---------------------------------------------------------------------------
# session state: a list of complexes
# ---------------------------------------------------------------------------
if "complexes" not in st.session_state:
    st.session_state.complexes = []      # dicts: name, path, label1, label2, group1, group2

st.title("fastPISA Interface Explorer")
st.caption("PISA-calibrated interface analysis, digested for manuscripts. Add a complex, pick the two "
           "sides, copy the numbers. Add a second complex to compare binders on the same antigen.")

with st.sidebar:
    st.header("Add a complex")
    uploaded = st.file_uploader("PDB / mmCIF (optionally .gz)", type=["pdb", "cif", "ent", "gz", "mmcif"], key="up")
    pdb_id_in = st.text_input("... or a PDB ID", placeholder="e.g. 1brs", key="pdbid")
    name_in = st.text_input("Name (optional)", placeholder="e.g. Fab 2 complex", key="cname")
    if st.button("Add complex", type="primary"):
        path = None
        try:
            if uploaded is not None:
                path = _write_upload(uploaded)
            elif pdb_id_in.strip():
                path = _fetch_pdb(pdb_id_in)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load: {exc}")
        if path:
            name = name_in.strip() or os.path.basename(path).split(".")[0].split("_", 1)[-1]
            st.session_state.complexes.append({"name": name, "path": path, "label1": "antigen",
                                               "label2": "binder", "group1": [], "group2": []})
    st.header("Analysis options")
    ligand_mode = st.selectbox("Ligands / cofactors", ["separate", "merge"], index=0,
                               help="separate: every hetero group is its own molecule (classic PISA). "
                                    "merge: a chain's bound ligands belong to that chain.")
    exclude_water = st.checkbox("Exclude water", value=True)
    min_css = st.slider("Minimum CSS to keep an interface", 0.0, 1.0, 0.0, 0.05)

complexes = st.session_state.complexes
if not complexes:
    st.info("Add a complex in the sidebar (upload a file or enter a PDB ID) to begin.")
    with st.expander("What this app reports, and how to read it", expanded=True):
        st.markdown(GUIDE)
    st.stop()

# analyse every complex and let the user assign chains ------------------
with st.sidebar:
    st.header("Complexes")
    for i, cx in enumerate(complexes):
        with st.expander(f"{i + 1}. {cx['name']}", expanded=(i == len(complexes) - 1)):
            res = _analyze(cx["path"], ligand_mode, exclude_water, float(min_css))
            cx["res"] = res
            inv = chain_inventory(res)
            cx["inventory"] = inv
            cx["label1"] = st.text_input("Side 1 name", cx["label1"], key=f"l1_{i}")
            cx["label2"] = st.text_input("Side 2 name", cx["label2"], key=f"l2_{i}")
            df = pd.DataFrame([{"molecule": r["label"], "type": r["class"], "res": r["n_residues"],
                                cx["label1"]: r["label"] in cx["group1"],
                                cx["label2"]: r["label"] in cx["group2"]} for r in inv])
            edited = st.data_editor(
                df, hide_index=True, width="stretch", key=f"ed_{i}",
                column_config={"molecule": st.column_config.TextColumn(disabled=True),
                               "type": st.column_config.TextColumn(disabled=True),
                               "res": st.column_config.NumberColumn(disabled=True),
                               cx["label1"]: st.column_config.CheckboxColumn(),
                               cx["label2"]: st.column_config.CheckboxColumn()})
            cx["group1"] = list(edited.loc[edited[cx["label1"]], "molecule"])
            cx["group2"] = list(edited.loc[edited[cx["label2"]], "molecule"])
            if i > 0 and complexes[0]["group1"] and st.button("Auto-detect shared side from complex 1", key=f"auto_{i}"):
                ref = complexes[0]
                cands = tuple(r["label"] for r in inv if r["class"] != "Ligand")
                matches = _detect_cached(ref["path"], cx["path"],
                                         tuple(c for c in ref["group1"] if not c.startswith("[")), cands)
                cx["group1"] = [m for _, m, _ in matches]
                cx["group2"] = [r["label"] for r in inv if r["class"] != "Ligand" and r["label"] not in cx["group1"]]
                cx["label1"], cx["label2"] = ref["label1"], ref["label2"]
                st.rerun()
            if st.button("Remove", key=f"rm_{i}"):
                complexes.pop(i)
                st.rerun()
    active_idx = st.selectbox("Show details for", range(len(complexes)),
                              format_func=lambda i: f"{i + 1}. {complexes[i]['name']}") if len(complexes) > 1 else 0

cx = complexes[active_idx]
res = cx["res"]
label1, label2 = cx["label1"], cx["label2"]
st.subheader(f"{cx['name']} ({res.pdb_id}): {len(res)} interfaces in the structure")

if not cx["group1"] or not cx["group2"]:
    st.info(f"Tick the chains of **{label1}** and **{label2}** for this complex in the sidebar. "
            "Meanwhile, every interface fastPISA found:")
    st.dataframe(res.to_dataframe(), width="stretch", hide_index=True)
    st.stop()

try:
    gi = group_interface(res, cx["group1"], cx["group2"], label1, label2)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
cx["gi"] = gi
if gi.empty:
    st.warning(f"No interface between {label1} ({', '.join(cx['group1'])}) and {label2} ({', '.join(cx['group2'])}).")
    st.dataframe(res.to_dataframe(), width="stretch", hide_index=True)
    st.stop()

tabs = ["Summary", "Residues", "Bonds & contacts", "3D (Mol*)", "Export", "Guide"]
if len(complexes) > 1:
    tabs.insert(4, "Compare")
tab = dict(zip(tabs, st.tabs(tabs)))

# ---- Summary -----------------------------------------------------------------
with tab["Summary"]:
    c = st.columns(4)
    c[0].metric(f"Buried on {label1}", f"{gi.buried_side1:,.0f} Å²")
    c[1].metric(f"Buried on {label2}", f"{gi.buried_side2:,.0f} Å²")
    c[2].metric("Total buried surface", f"{gi.buried_total:,.0f} Å²",
                help="Sum of both sides. PISA's 'interface area' is half of this.")
    c[3].metric("Interface area (PISA)", f"{gi.interface_area:,.0f} Å²")
    c = st.columns(4)
    c[0].metric("ΔG solvation", f"{gi.dg_solv:+.1f} kcal/mol", help=f"apolar {gi.dg_apolar:+.1f}, polar {gi.dg_polar:+.1f}")
    c[1].metric("Stabilisation energy", f"{gi.stab_energy:+.1f} kcal/mol", help="ΔG solvation plus PISA's per-bond terms")
    c[2].metric("Hydrogen bonds", gi.n_hbonds)
    c[3].metric("Salt bridges", gi.n_salt_bridges,
                help=f"disulfides: {gi.n_disulfides}; residue pairs in contact: {gi.n_residue_pairs}")
    st.markdown("**What this interface looks like**")
    for f in interpret(gi):
        (st.warning if f["level"] == "warning" else st.info if f["level"] == "info" else st.write)(f["text"])
    st.markdown("**Results text** (edit freely)")
    st.code(gi.results_paragraph(), language=None)
    st.markdown("**Contributing chain-pair interfaces**")
    st.dataframe(gi.pair_table(), width="stretch", hide_index=True)
    st.caption("P-value: probability that a random surface patch is as hydrophobic (low = specific). "
               "CSS: complexation significance score. Neither is additive over pairs.")
    fig = F.composition(gi)
    st.pyplot(fig, width="stretch")
    _fig_download(fig, f"{res.pdb_id}_composition", "comp")

# ---- Residues ------------------------------------------------------------------
with tab["Residues"]:
    for side, label in ((1, label1), (2, label2)):
        fig = F.footprint(gi, side)
        st.pyplot(fig, width="stretch")
        _fig_download(fig, f"{res.pdb_id}_footprint_{label}", f"fp{side}")
    fig = F.residue_bars(gi)
    st.pyplot(fig, width="stretch")
    _fig_download(fig, f"{res.pdb_id}_residues", "rb")
    c1, c2 = st.columns(2)
    for col, side, label in ((c1, 1, label1), (c2, 2, label2)):
        with col:
            n = len(gi.residues_side1 if side == 1 else gi.residues_side2)
            st.markdown(f"**{label} interface residues** ({n})")
            st.code(gi.residue_string(side), language=None)
            st.dataframe(gi.residue_table(side), width="stretch", hide_index=True, height=340)

# ---- Bonds & contacts ------------------------------------------------------------
with tab["Bonds & contacts"]:
    fig = F.bond_network(gi)
    st.pyplot(fig, width="content")
    _fig_download(fig, f"{res.pdb_id}_bond_network", "bn")
    bt = gi.bonds_table()
    kinds = st.multiselect("Bond types", ["hydrogen bond", "salt bridge", "disulfide"],
                           default=["hydrogen bond", "salt bridge", "disulfide"])
    view = st.radio("Summarise by", ["per bond", f"per {label1} residue", f"per {label2} residue"], horizontal=True)
    sub = bt[bt["type"].isin(kinds)] if len(bt) else bt
    if view != "per bond" and len(sub):
        key = label1 if view.endswith(f"{label1} residue") else label2
        other = label2 if key == label1 else label1
        grouped = (sub.groupby([key, "type"])[other].agg(lambda s: ", ".join(dict.fromkeys(s)))
                   .reset_index().rename(columns={other: f"{other} partners"}))
        grouped["n"] = sub.groupby([key, "type"]).size().values
        st.dataframe(grouped, width="stretch", hide_index=True)
    else:
        st.dataframe(sub.drop(columns=["chain 1", "seq 1", "chain 2", "seq 2"]) if len(sub) else sub,
                     width="stretch", hide_index=True)
    st.markdown("**COCOMAPS interaction classes** (atom pairs within 5 Å)")
    pop = gi.interaction_population
    if pop:
        st.dataframe(pd.DataFrame([{"class": k.replace("_", " "), "atom pairs": v}
                                   for k, v in sorted(pop.items(), key=lambda kv: -kv[1])]), hide_index=True)
    fig = F.contact_map(gi)
    st.pyplot(fig, width="content")
    _fig_download(fig, f"{res.pdb_id}_contact_map", "cm")
    with st.expander("Residue-pair table"):
        st.dataframe(gi.contact_map_table(), width="stretch", hide_index=True)

# ---- 3D ---------------------------------------------------------------------------
with tab["3D (Mol*)"]:
    c = st.columns(4)
    show_surface = c[0].checkbox("Surface", value=False)
    show_bonds = c[1].checkbox("Bonds (dashed, with distance)", value=True)
    show_labels = c[2].checkbox("Residue labels", value=False)
    height = c[3].slider("Height", 400, 900, 620, 20)
    txt, fmt = structure_text(cx["path"])
    others = [r["chain"] for r in cx["inventory"]]
    st.components.v1.html(interface_view_html(txt, fmt, gi, height=height, show_surface=show_surface,
                                              show_bonds=show_bonds, show_labels=show_labels,
                                              other_chains=others), height=height + 40)
    st.caption("Mol* viewer. Interface residues as ball-and-stick; bonds as dashed lines labelled with the "
               "donor-acceptor distance. Use the expand icon for a full-screen view; the selection tool "
               "identifies residues.")
    c1, c2 = st.columns(2)
    c1.markdown("**ChimeraX**")
    c1.code(gi.chimerax_command(), language="bash")
    c2.markdown("**PyMOL**")
    c2.code(gi.pymol_command(), language="bash")

# ---- Compare -------------------------------------------------------------------
if "Compare" in tab:
    with tab["Compare"]:
        entries = []
        for c2x in complexes:
            if c2x.get("gi") is None and c2x["group1"] and c2x["group2"]:
                try:
                    c2x["gi"] = group_interface(c2x["res"], c2x["group1"], c2x["group2"], c2x["label1"], c2x["label2"])
                except ValueError:
                    pass
            if c2x.get("gi") is not None and not c2x["gi"].empty:
                entries.append(ComplexEntry(c2x["name"], c2x["gi"], c2x["res"]))
        if len(entries) < 2:
            st.info("Assign both sides for at least two complexes (use *Auto-detect shared side* for the "
                    "second complex) to compare them.")
        else:
            side = st.radio("Compare footprints on", [1, 2], horizontal=True,
                            format_func=lambda s: f"side {s} ({entries[0].gi.label1 if s == 1 else entries[0].gi.label2})")
            align = st.selectbox("Residue matching across complexes", ["auto", "number", "sequence"], index=0,
                                 help="auto: by residue number if the numbering agrees, else by sequence alignment")
            cmp = compare(entries, side=side, align=align)
            st.markdown("**Side by side**")
            st.dataframe(cmp.summary_table(), width="stretch", hide_index=True)
            st.code(cmp.prose(), language=None)
            fig = F.compare_bars(cmp)
            st.pyplot(fig, width="stretch")
            _fig_download(fig, "comparison_bars", "cb")
            st.markdown("**Footprint overlap on the shared side**")
            st.dataframe(cmp.overlap_table(), width="stretch", hide_index=True)
            fig = F.compare_footprints(cmp, side)
            st.pyplot(fig, width="stretch")
            _fig_download(fig, "comparison_footprints", "cf")
            fig = F.compare_heatmap(cmp, side)
            st.pyplot(fig, width="stretch")
            _fig_download(fig, "comparison_heatmap", "ch")
            with st.expander("Per-residue buried area matrix"):
                st.dataframe(cmp.residue_matrix(side), width="stretch")
            st.markdown("**All binders on one antigen (Mol\\*)**")
            if st.checkbox("Superpose and show", value=False,
                           help="Superposes every complex onto the first on the shared chains (pdb_align) "
                                "and draws all binders together."):
                ref = [c2 for c2 in complexes if c2["name"] == entries[0].name][0]
                ref_chains = tuple(sorted({r.chain for r in entries[0].gi.residues_side1}))
                txt0, fmt0 = structure_text(ref["path"])
                scene = [{"name": entries[0].name, "text": txt0, "fmt": fmt0, "gi": entries[0].gi, "color": PALETTE[0]}]
                rows = []
                for k, e in enumerate(entries[1:], 1):
                    c2 = [c3 for c3 in complexes if c3["name"] == e.name][0]
                    mob_chains = tuple(sorted({r.chain for r in e.gi.residues_side1}))
                    try:
                        sp = _superpose_cached(ref["path"], c2["path"], ref_chains, mob_chains)
                        scene.append({"name": e.name, "text": sp["text"], "fmt": "pdb", "gi": e.gi,
                                      "color": PALETTE[k % len(PALETTE)]})
                        rows.append({"complex": e.name, "shared chains": " ".join(mob_chains),
                                     "Cα RMSD to reference (Å)": round(sp["rmsd"], 2), "aligned residues": sp["n"],
                                     "TM-score": round(sp["tm"], 3) if sp["tm"] else None})
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"{e.name}: superposition failed ({exc})")
                if rows:
                    st.dataframe(pd.DataFrame(rows), hide_index=True)
                st.components.v1.html(comparison_view_html(scene, height=640), height=680)

# ---- Export ----------------------------------------------------------------------
with tab["Export"]:
    sheets = {"summary": pd.DataFrame([{k: v for k, v in gi.to_dict().items() if not k.startswith("residues_")}]),
              "chain pairs": gi.pair_table(), "bonds": gi.bonds_table(),
              f"{label1} residues": gi.residue_table(1), f"{label2} residues": gi.residue_table(2),
              "contact map": gi.contact_map_table(), "all interfaces": res.to_dataframe()}
    c = st.columns(3)
    c[0].download_button("Excel workbook (all tables)", excel_bytes(sheets), f"{res.pdb_id}_interface.xlsx",
                         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    c[1].download_button("Digest (JSON)", json.dumps(gi.to_dict(), indent=2), f"{res.pdb_id}_interface.json", "application/json")
    c[2].download_button("Full fastPISA output (JSON)", json.dumps(res.interfaces_json, indent=1),
                         f"{res.pdb_id}_fastpisa_interfaces.json", "application/json")
    for name, df in sheets.items():
        if name != "summary":
            st.download_button(f"CSV: {name}", df.to_csv(index=False).encode(),
                               f"{res.pdb_id}_{name.replace(' ', '_')}.csv", "text/csv", key=f"csv_{name}")
    st.markdown("**Methods text**")
    st.code(gi.methods_paragraph(), language=None)
    st.caption("Cite: Krissinel & Henrick, J. Mol. Biol. 372:774 (2007) for PISA; Chawla et al., "
               "Bioinformatics (2025) for COCOMAPS 2.0; fastPISA (github.com/dzyla/fastPISA); "
               "pdb_align (github.com/dzyla/pdb_align) for superposition in comparison mode.")

# ---- Guide ----------------------------------------------------------------------
with tab["Guide"]:
    st.markdown(GUIDE)
    st.markdown("### All interfaces in this structure")
    st.dataframe(res.to_dataframe(), width="stretch", hide_index=True)
