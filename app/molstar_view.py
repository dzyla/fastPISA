"""Mol* 3D view of a group interface, driven by MolViewSpec (MVS).

Mol* is loaded from the jsDelivr CDN (pinned version) inside a Streamlit
HTML component; the scene is described declaratively as an MVS tree
(components -> representations -> colours, focus, distance primitives for
bonds), which is the supported, stable way to script the viewer.

The structure itself is passed as a ``data:`` URL, so nothing has to be
served -- this works on Streamlit Cloud.
"""
from __future__ import annotations

import base64
import html
import json
from typing import Dict, List, Optional, Sequence

MOLSTAR_VERSION = "5.11.0"
MOLSTAR_JS = f"https://cdn.jsdelivr.net/npm/molstar@{MOLSTAR_VERSION}/build/viewer/molstar.js"
MOLSTAR_CSS = f"https://cdn.jsdelivr.net/npm/molstar@{MOLSTAR_VERSION}/build/viewer/molstar.css"

SIDE_COLORS = ("#E69F00", "#0072B2")          # Okabe-Ito orange / blue
SIDE_COLORS_LIGHT = ("#F5CB7A", "#8BB8E0")
BOND_COLORS = {"hbond": "#1B9E77", "salt_bridge": "#D62728", "disulfide": "#E6AB02"}


def _residue_selector(chain: str, seq: str, icode: str = "") -> dict:
    sel = {"auth_asym_id": chain, "auth_seq_id": int(seq)}
    if icode:
        sel["pdbx_PDB_ins_code"] = icode
    return sel


def _atom_selector(c, which: int) -> dict:
    chain = getattr(c, f"atom{which}_chain")
    seq = getattr(c, f"atom{which}_seq")
    name = getattr(c, f"atom{which}_name").strip()
    icode = getattr(c, f"atom{which}_icode", "") or ""
    sel = {"auth_asym_id": chain, "auth_seq_id": int(seq), "auth_atom_id": name}
    if icode:
        sel["pdbx_PDB_ins_code"] = icode
    return sel


def _component(selector, children: list, label: Optional[str] = None) -> dict:
    node = {"kind": "component", "params": {"selector": selector}, "children": children}
    return node


#: CPK colours for heteroatoms; carbons take the side colour so the chain
#: identity stays visible while chemistry (N/O/S) is readable.
ELEMENT_COLORS = {"N": "#3050F8", "O": "#FF0D0D", "S": "#FFFF30", "P": "#FF8000",
                  "F": "#90E050", "CL": "#1FF01F", "BR": "#A62929", "I": "#940094",
                  "ZN": "#7D80B0", "MG": "#8AFF00", "CA": "#3DFF00", "FE": "#E06633"}


def _repr(rtype: str, color: str, opacity: Optional[float] = None,
          children: Optional[list] = None, by_element: bool = False) -> dict:
    """Representation node coloured uniformly (carbons and everything else),
    optionally overriding heteroatoms with element colours."""
    kids = [{"kind": "color", "params": {"color": color}}]
    if by_element:
        for el, col in ELEMENT_COLORS.items():
            kids.append({"kind": "color", "params": {"color": col, "selector": {"type_symbol": el}}})
    if opacity is not None:
        kids.append({"kind": "opacity", "params": {"opacity": opacity}})
    return {"kind": "representation", "params": {"type": rtype},
            "children": kids + (children or [])}


def build_mvs(structure_text: str, fmt: str, gi, show_surface: bool = False,
              show_bonds: bool = True, show_labels: bool = False,
              bond_kinds: Sequence[str] = ("hbond", "salt_bridge", "disulfide"),
              side_colors: Sequence[str] = SIDE_COLORS,
              other_chains: Sequence[str] = ()) -> dict:
    """MolViewSpec document for one group interface.

    ``other_chains``: chains in neither group, drawn as grey cartoon (chains
    in a group are drawn only in the group colour -- drawing them twice makes
    the two cartoons z-fight and the grey one wins).
    """
    mime = "chemical/x-mmcif" if fmt == "mmcif" else "chemical/x-pdb"
    data_url = "data:" + mime + ";base64," + base64.b64encode(structure_text.encode()).decode()

    chains1 = sorted({r.chain for r in gi.residues_side1}) or sorted({_chain(c) for c in gi.group1})
    chains2 = sorted({r.chain for r in gi.residues_side2}) or sorted({_chain(c) for c in gi.group2})
    grp_sel1 = [{"auth_asym_id": c} for c in chains1]
    grp_sel2 = [{"auth_asym_id": c} for c in chains2]
    res_sel1 = [_residue_selector(r.chain, r.seq, r.icode) for r in gi.residues_side1]
    res_sel2 = [_residue_selector(r.chain, r.seq, r.icode) for r in gi.residues_side2]

    children: List[dict] = []
    # chains outside both groups faintly, then each group in its colour
    others = [c for c in other_chains if c not in chains1 and c not in chains2]
    if others:
        children.append(_component([{"auth_asym_id": c} for c in others], [_repr("cartoon", "#D9D9D9")]))
    if grp_sel1:
        children.append(_component(grp_sel1, [_repr("cartoon", SIDE_COLORS_LIGHT[0])]))
    if grp_sel2:
        children.append(_component(grp_sel2, [_repr("cartoon", SIDE_COLORS_LIGHT[1])]))
    # interface residues as ball-and-stick, strong colours
    focus_sel = res_sel1 + res_sel2
    if res_sel1:
        kids = [_repr("ball_and_stick", side_colors[0], by_element=True)]
        if show_labels:
            kids += [{"kind": "label", "params": {"text": f"{r.one}{r.seq}"}}
                     for r in gi.residues_side1[:0]]      # per-residue labels are added below
        children.append(_component(res_sel1, kids))
    if res_sel2:
        children.append(_component(res_sel2, [_repr("ball_and_stick", side_colors[1], by_element=True)]))
    if show_labels:
        for r in list(gi.residues_side1) + list(gi.residues_side2):
            children.append(_component([_residue_selector(r.chain, r.seq, r.icode)],
                                       [{"kind": "label", "params": {"text": f"{r.one}{r.seq}{r.icode}"}}]))
    if show_surface:
        if grp_sel1:
            children.append(_component(grp_sel1, [_repr("surface", side_colors[0], opacity=0.35)]))
        if grp_sel2:
            children.append(_component(grp_sel2, [_repr("surface", side_colors[1], opacity=0.35)]))
    # camera on the interface
    if focus_sel:
        children.append(_component(focus_sel, [{"kind": "focus", "params": {}}]))
    # bonds as distance primitives between the actual atoms
    prims = []
    if show_bonds:
        for c in gi.bonds(kinds=tuple(bond_kinds)):
            prims.append({"kind": "primitive", "params": {
                "kind": "distance_measurement",
                "start": _atom_selector(c, 1), "end": _atom_selector(c, 2),
                "color": BOND_COLORS.get(c.bond_type, "#444444"),
                "radius": 0.08, "dash_length": 0.25,
                "label_template": "{{distance}}", "label_size": 0.9,
                "label_color": BOND_COLORS.get(c.bond_type, "#444444"),
            }})
    if prims:
        children.append({"kind": "primitives", "params": {"opacity": 1.0}, "children": prims})

    structure = {"kind": "structure", "params": {"type": "model"}, "children": children}
    parse = {"kind": "parse", "params": {"format": fmt}, "children": [structure]}
    download = {"kind": "download", "params": {"url": data_url}, "children": [parse]}
    return {"root": {"kind": "root", "children": [download]},
            "metadata": {"version": "1", "title": f"{gi.label1} x {gi.label2} interface"}}


def molstar_html(mvs: dict, height: int = 600, legend: Optional[Dict[str, str]] = None) -> str:
    """Self-contained HTML that loads Mol* from the CDN and applies the MVS."""
    legend = legend or {}
    legend_html = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:14px">'
        f'<span style="width:12px;height:12px;background:{col};display:inline-block;'
        f'margin-right:5px;border-radius:2px"></span>{html.escape(str(name), quote=True)}</span>'
        for name, col in legend.items())
    # JSON is embedded in a script element. Escaping the HTML-significant
    # characters prevents a user label containing ``</script>`` from closing
    # that element while remaining valid JSON with identical decoded values.
    mvs_json = (json.dumps(mvs).replace("<", "\\u003c")
                .replace(">", "\\u003e").replace("&", "\\u0026"))
    return f"""<!DOCTYPE html>
<html><head>
<link rel="stylesheet" href="{MOLSTAR_CSS}">
<script src="{MOLSTAR_JS}"></script>
<style>
  body {{ margin:0; font-family: system-ui, sans-serif; }}
  #app {{ position:relative; width:100%; height:{height}px; }}
  #legend {{ font-size:12px; padding:6px 4px; color:#333; }}
</style></head>
<body>
<div id="legend">{legend_html}</div>
<div id="app"></div>
<script>
(async () => {{
  const viewer = await molstar.Viewer.create('app', {{
    layoutIsExpanded: false, layoutShowControls: false, layoutShowSequence: true,
    layoutShowLog: false, layoutShowLeftPanel: false, viewportShowExpand: true,
    viewportShowSelectionMode: true, viewportShowAnimation: false, pdbProvider: 'rcsb',
  }});
  const mvs = {mvs_json};
  await viewer.loadMvsData(JSON.stringify(mvs), 'mvsj', {{ replaceExisting: true }});
}})().catch(e => {{ document.getElementById('legend').innerText = 'Mol* failed: ' + e; console.error(e); }});
</script>
</body></html>"""


def _chain(label: str) -> str:
    if label.startswith("["):
        return label.split("]", 1)[1].split(":", 1)[0]
    return label


def interface_view_html(structure_text: str, fmt: str, gi, height: int = 600,
                        show_surface: bool = False, show_bonds: bool = True,
                        show_labels: bool = False, other_chains: Sequence[str] = ()) -> str:
    mvs = build_mvs(structure_text, fmt, gi, show_surface=show_surface,
                    show_bonds=show_bonds, show_labels=show_labels, other_chains=other_chains)
    legend = {f"{gi.label1} (interface residues)": SIDE_COLORS[0],
              f"{gi.label2} (interface residues)": SIDE_COLORS[1]}
    if show_bonds:
        legend.update({"hydrogen bond": BOND_COLORS["hbond"], "salt bridge": BOND_COLORS["salt_bridge"],
                       "disulfide": BOND_COLORS["disulfide"]})
    return molstar_html(mvs, height=height, legend=legend)


def comparison_view_html(entries: Sequence[dict], height: int = 640) -> str:
    """One Mol* scene with several complexes superposed on the shared side.

    ``entries``: list of dicts with ``name``, ``text`` (structure text, already
    in the reference frame), ``fmt``, ``gi`` (GroupInterface), ``color``. The
    first entry is the reference: its shared-side chains are drawn as grey
    cartoon + surface; every entry's binder chains are drawn as cartoon in
    the entry colour and its footprint on the shared side as ball-and-stick
    in the same colour.
    """
    downloads = []
    legend = {}
    for k, e in enumerate(entries):
        gi, color = e["gi"], e["color"]
        mime = "chemical/x-mmcif" if e["fmt"] == "mmcif" else "chemical/x-pdb"
        data_url = "data:" + mime + ";base64," + base64.b64encode(e["text"].encode()).decode()
        shared = sorted({r.chain for r in gi.residues_side1}) or sorted({_chain(c) for c in gi.group1})
        binder = sorted({r.chain for r in gi.residues_side2}) or sorted({_chain(c) for c in gi.group2})
        children: List[dict] = []
        if k == 0 and shared:
            sel = [{"auth_asym_id": c} for c in shared]
            children.append(_component(sel, [_repr("cartoon", "#BDBDBD"),
                                             _repr("surface", "#E0E0E0", opacity=0.25)]))
        if binder:
            children.append(_component([{"auth_asym_id": c} for c in binder],
                                       [_repr("cartoon", color, opacity=0.9)]))
        foot = [_residue_selector(r.chain, r.seq, r.icode) for r in gi.residues_side1]
        if foot and k == 0:
            children.append(_component(foot, [_repr("ball_and_stick", color, by_element=True)]))
        elif foot:
            # footprint of a superposed complex is drawn on ITS OWN (moved) antigen copy
            children.append(_component(foot, [_repr("ball_and_stick", color, by_element=True)]))
        if k == 0 and foot:
            children.append(_component(foot, [{"kind": "focus", "params": {}}]))
        structure = {"kind": "structure", "params": {"type": "model"}, "children": children}
        parse = {"kind": "parse", "params": {"format": e["fmt"]}, "children": [structure]}
        downloads.append({"kind": "download", "params": {"url": data_url}, "children": [parse]})
        legend[f"{e['name']}: binder + footprint"] = color
    legend = {f"{entries[0]['gi'].label1} (reference, shared side)": "#BDBDBD", **legend}
    mvs = {"root": {"kind": "root", "children": downloads},
           "metadata": {"version": "1", "title": "interface comparison"}}
    return molstar_html(mvs, height=height, legend=legend)
