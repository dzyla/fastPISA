"""Pure helpers for the Streamlit app (figures, 3D viewer HTML, Excel export).

Kept free of any Streamlit call so they can be unit-tested and reused.
"""
from __future__ import annotations

import io

import pandas as pd

# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _fig_to_bytes(fig, fmt: str) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=200, bbox_inches="tight")
    return buf.getvalue()


def residue_barplot(gi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, res, label, color in ((axes[0], gi.residues_side1, gi.label1, "#e69138"),
                                  (axes[1], gi.residues_side2, gi.label2, "#3d85c6")):
        if not res:
            ax.set_visible(False)
            continue
        xs = [f"{r.one}{r.seq}{r.icode}" + (f"\n{r.chain}" if len({q.chain for q in res}) > 1 else "")
              for r in res]
        ax.bar(range(len(res)), [r.bsa for r in res], color=color)
        ax.set_xticks(range(len(res)))
        ax.set_xticklabels(xs, rotation=90, fontsize=7)
        ax.set_ylabel("buried surface (A$^2$)")
        ax.set_title(f"{label}: {len(res)} interface residues, {sum(r.bsa for r in res):,.0f} A$^2$ buried",
                     fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def contact_map_figure(gi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cm = gi.contact_map_table()
    if cm.empty:
        return None
    r1 = list(dict.fromkeys(cm[gi.label1]))
    r2 = list(dict.fromkeys(cm[gi.label2]))
    key = lambda s: (s.split(":")[0], int("".join(ch for ch in s.split(":")[1][1:] if ch.isdigit()) or 0))  # noqa: E731
    r1.sort(key=key)
    r2.sort(key=key)
    i1 = {k: i for i, k in enumerate(r1)}
    i2 = {k: i for i, k in enumerate(r2)}
    colors = {"hydrogen_bond": "#1f77b4", "salt_bridge": "#d62728", "pi_pi": "#9467bd",
              "cation_pi": "#8c564b", "ch_pi": "#e377c2", "polar_vdw": "#2ca02c",
              "apolar_vdw": "#ff7f0e", "weak_hbond": "#17becf", "disulfide": "#bcbd22",
              "proximal": "#c7c7c7"}
    fig, ax = plt.subplots(figsize=(max(6, 0.32 * len(r2) + 2), max(5, 0.28 * len(r1) + 2)))
    for _, e in cm.iterrows():
        ax.scatter(i2[e[gi.label2]], i1[e[gi.label1]], s=25 + 10 * e["atom contacts"],
                   c=colors.get(e["dominant"], "#999"), edgecolors="k", linewidths=0.4)
    ax.set_xticks(range(len(r2)))
    ax.set_xticklabels(r2, rotation=90, fontsize=7)
    ax.set_yticks(range(len(r1)))
    ax.set_yticklabels(r1, fontsize=7)
    ax.set_xlabel(gi.label2)
    ax.set_ylabel(gi.label1)
    ax.invert_yaxis()
    ax.grid(alpha=0.2)
    present = [c for c in colors if c in set(cm["dominant"])]
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[c],
                              markeredgecolor="k", markersize=7, label=c.replace("_", " "))
                       for c in present],
              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    ax.set_title(f"{gi.label1} x {gi.label2}: {len(cm)} residue pairs (size = atom contacts)", fontsize=10)
    fig.tight_layout()
    return fig


def viewer_html(structure_text: str, fmt: str, gi, height: int = 520) -> str:
    """py3Dmol view: cartoon, interface residues as sticks coloured by side."""
    import json
    try:
        import py3Dmol  # noqa: F401
    except ImportError:
        return "<p>py3Dmol is not installed.</p>"
    sel = {1: [], 2: []}
    for side, res in ((1, gi.residues_side1), (2, gi.residues_side2)):
        for r in res:
            sel[side].append({"chain": r.chain, "resi": str(r.seq) + (r.icode or "")})
    chains1 = sorted({r.chain for r in gi.residues_side1}) or [c.split("]")[-1].split(":")[0] for c in gi.group1]
    chains2 = sorted({r.chain for r in gi.residues_side2}) or [c.split("]")[-1].split(":")[0] for c in gi.group2]
    js_model = json.dumps(structure_text)
    return f"""
<div id="viewer" style="width:100%;height:{height}px;position:relative"></div>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
  const v = $3Dmol.createViewer(document.getElementById('viewer'), {{backgroundColor: 'white'}});
  v.addModel({js_model}, "{fmt}");
  v.setStyle({{}}, {{cartoon: {{color: '#d9d9d9'}}}});
  v.setStyle({{chain: {json.dumps(chains1)}}}, {{cartoon: {{color: '#f6b26b'}}}});
  v.setStyle({{chain: {json.dumps(chains2)}}}, {{cartoon: {{color: '#9fc5e8'}}}});
  const s1 = {json.dumps(sel[1])}, s2 = {json.dumps(sel[2])};
  s1.forEach(r => v.addStyle({{chain: r.chain, resi: r.resi}}, {{stick: {{color: '#e69138', radius: 0.25}}}}));
  s2.forEach(r => v.addStyle({{chain: r.chain, resi: r.resi}}, {{stick: {{color: '#3d85c6', radius: 0.25}}}}));
  const all = s1.concat(s2);
  if (all.length) v.zoomTo({{or: all.map(r => ({{chain: r.chain, resi: r.resi}}))}}); else v.zoomTo();
  v.render();
</script>"""


def _excel_bytes(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()
