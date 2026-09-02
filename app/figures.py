"""Publication-style figures for a group interface (matplotlib, no Streamlit).

Every function returns a matplotlib Figure. Colours are Okabe-Ito
(colour-blind safe); styling is set locally with ``_style`` so the app's
global rcParams are untouched.
"""
from __future__ import annotations

import io
from typing import Dict, List, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SIDE = ("#E69F00", "#0072B2")
CLASS_COLORS = {
    "hydrophobic": "#7F7F7F", "aromatic": "#CC79A7", "polar": "#009E73",
    "positive": "#0072B2", "negative": "#D55E00", "glycine/proline": "#F0E442",
    "nucleotide": "#56B4E9", "other": "#BBBBBB",
}
RES_CLASS = {
    **{r: "hydrophobic" for r in ("ALA", "VAL", "LEU", "ILE", "MET", "MSE", "CYS")},
    **{r: "aromatic" for r in ("PHE", "TYR", "TRP", "HIS")},
    **{r: "polar" for r in ("SER", "THR", "ASN", "GLN")},
    **{r: "positive" for r in ("LYS", "ARG")},
    **{r: "negative" for r in ("ASP", "GLU")},
    **{r: "glycine/proline" for r in ("GLY", "PRO")},
    **{r: "nucleotide" for r in ("A", "G", "C", "U", "DA", "DG", "DC", "DT", "DU")},
}
BOND_COLORS = {"hydrogen bond": "#1B9E77", "salt bridge": "#D62728", "disulfide": "#E6AB02"}
CONTACT_COLORS = {"hydrogen_bond": "#1B9E77", "salt_bridge": "#D62728", "pi_pi": "#7570B3",
                  "cation_pi": "#E7298A", "ch_pi": "#A6761D", "polar_vdw": "#66A61E",
                  "apolar_vdw": "#E6AB02", "weak_hbond": "#1F78B4", "disulfide": "#B15928",
                  "halogen_bond": "#6A3D9A", "metal_mediated": "#FF7F00", "proximal": "#CFCFCF"}


def res_class(name: str) -> str:
    return RES_CLASS.get(name.strip().upper(), "other")


def _style(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=11, loc="left", fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=8)


def fig_bytes(fig, fmt: str = "png", dpi: int = 300) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    return buf.getvalue()


# ---------------------------------------------------------------------------
def footprint(gi, side: int, chain_lengths: Optional[Dict[str, tuple]] = None):
    """Interface footprint along the sequence: BSA per residue vs residue number.

    ``chain_lengths``: {chain: (first_seq, last_seq)} to draw the full chain
    extent; otherwise the axis spans the interface residues +- 5.
    """
    res = gi.residues_side1 if side == 1 else gi.residues_side2
    label = gi.label1 if side == 1 else gi.label2
    color = SIDE[side - 1]
    chains = sorted({r.chain for r in res})
    n = max(1, len(chains))
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.2 * n + 0.6), squeeze=False)
    for ax, ch in zip(axes[:, 0], chains):
        rr = [r for r in res if r.chain == ch]
        xs = np.array([_num(r.seq) for r in rr])
        ys = np.array([r.bsa for r in rr])
        lo, hi = (chain_lengths or {}).get(ch, (xs.min() - 5, xs.max() + 5))
        ax.bar(xs, ys, width=1.0, color=[CLASS_COLORS[res_class(r.name)] for r in rr], edgecolor="none")
        ax.set_xlim(lo - 0.5, hi + 0.5)
        top = sorted(rr, key=lambda r: -r.bsa)[:6]
        for r in top:
            ax.annotate(f"{r.one}{r.seq}", (_num(r.seq), r.bsa), fontsize=7, ha="center",
                        va="bottom", xytext=(0, 2), textcoords="offset points")
        _style(ax, f"{label}, chain {ch}: {len(rr)} interface residues, {ys.sum():,.0f} A$^2$ buried",
               "residue number", "BSA (A$^2$)")
        ax.axhline(0, color="k", lw=0.5)
    handles = [Patch(color=c, label=k) for k, c in CLASS_COLORS.items() if any(res_class(r.name) == k for r in res)]
    axes[0, 0].legend(handles=handles, fontsize=7, ncol=len(handles), loc="upper right", frameon=False)
    fig.tight_layout()
    return fig


def residue_bars(gi):
    """Per-residue buried area for both sides, hot spots annotated; the bar
    fill shows the fraction of the residue's surface that is buried."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, res, label, color in ((axes[0], gi.residues_side1, gi.label1, SIDE[0]),
                                  (axes[1], gi.residues_side2, gi.label2, SIDE[1])):
        if not res:
            ax.set_visible(False)
            continue
        rr = sorted(res, key=lambda r: -r.bsa)
        x = np.arange(len(rr))
        bsa = np.array([r.bsa for r in rr])
        asa = np.array([r.asa for r in rr])
        ax.bar(x, asa, color="#EEEEEE", edgecolor="none", label="isolated ASA")
        ax.bar(x, bsa, color=color, edgecolor="none", label="buried (BSA)")
        for i, r in enumerate(rr):
            if r.n_bonds:
                ax.text(i, asa[i] + 2, "•" * min(r.n_bonds, 4), ha="center", va="bottom", fontsize=7, color="#333")
        multi = len({r.chain for r in rr}) > 1
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r.one}{r.seq}{r.icode}" + (f"\n{r.chain}" if multi else "") for r in rr],
                           rotation=90, fontsize=7)
        _style(ax, f"{label}: {len(rr)} residues, {bsa.sum():,.0f} A$^2$ buried", "", "surface area (A$^2$)")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.text(0.5, -0.02, "dots: number of H-bonds / salt bridges / disulfides the residue makes (max 4 shown)",
             ha="center", fontsize=7, color="#555")
    fig.tight_layout()
    return fig


def composition(gi, surface_classes: Optional[Dict[int, Dict[str, float]]] = None):
    """Interface composition by residue class (fraction of buried area), one
    bar per side, plus the apolar / polar split of the solvation energy."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    classes = list(CLASS_COLORS)
    ys = [gi.label1, gi.label2]
    for i, res in enumerate((gi.residues_side1, gi.residues_side2)):
        tot = sum(r.bsa for r in res) or 1.0
        left = 0.0
        for cl in classes:
            v = sum(r.bsa for r in res if res_class(r.name) == cl) / tot
            if v > 0:
                ax.barh(i, v, left=left, color=CLASS_COLORS[cl], edgecolor="white", lw=0.5)
                if v > 0.06:
                    ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white")
                left += v
    ax.set_yticks([0, 1])
    ax.set_yticklabels(ys)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _style(ax, "Buried area by residue class", "fraction of buried surface", "")
    ax.grid(False)
    ax.legend(handles=[Patch(color=c, label=k) for k, c in CLASS_COLORS.items()],
              fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False)
    ax = axes[1]
    vals = [gi.dg_apolar, gi.dg_polar, gi.stab_energy - gi.dg_solv, gi.stab_energy]
    names = ["apolar\nburial", "polar\nburial", "H-bonds /\nsalt bridges", "stabilisation\nenergy"]
    cols = ["#7F7F7F", "#009E73", "#1B9E77", "#000000"]
    ax.bar(range(4), vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.3 if v >= 0 else -0.3), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, fontsize=7)
    ax.axhline(0, color="k", lw=0.6)
    _style(ax, "Energy decomposition", "", "kcal/mol")
    fig.tight_layout()
    return fig


def bond_network(gi):
    """Bipartite residue graph of H-bonds, salt bridges and disulfides."""
    bt = gi.bonds_table(one_letter_codes=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * max(len(set(bt[gi.label1])) if len(bt) else 1,
                                                          len(set(bt[gi.label2])) if len(bt) else 1) + 1)))
    if bt.empty:
        ax.text(0.5, 0.5, "no hydrogen bonds, salt bridges or disulfides", ha="center", va="center")
        ax.axis("off")
        return fig
    left = list(dict.fromkeys(sorted(bt[gi.label1], key=_reskey)))
    right = list(dict.fromkeys(sorted(bt[gi.label2], key=_reskey)))
    yl = {r: i for i, r in enumerate(left)}
    yr = {r: i for i, r in enumerate(right)}
    sl = max(len(left), len(right)) / max(len(left), 1)
    sr = max(len(left), len(right)) / max(len(right), 1)
    for _, row in bt.iterrows():
        y1, y2 = yl[row[gi.label1]] * sl, yr[row[gi.label2]] * sr
        ax.plot([0, 1], [y1, y2], color=BOND_COLORS.get(row["type"], "#444"),
                lw=2.2 if row["type"] == "salt bridge" else 1.4,
                ls="-" if row["moiety"] == "side chain-side chain" else "--", alpha=0.85)
    for r, i in yl.items():
        ax.text(-0.03, i * sl, r, ha="right", va="center", fontsize=8, color=SIDE[0])
        ax.plot(0, i * sl, "o", color=SIDE[0], ms=6)
    for r, i in yr.items():
        ax.text(1.03, i * sr, r, ha="left", va="center", fontsize=8, color=SIDE[1])
        ax.plot(1, i * sr, "o", color=SIDE[1], ms=6)
    ax.set_xlim(-0.35, 1.35)
    ax.axis("off")
    fig.suptitle(f"Bond network: {gi.n_hbonds} H-bonds, {gi.n_salt_bridges} salt bridges, "
                 f"{gi.n_disulfides} disulfides", fontsize=11, x=0.01, ha="left", fontweight="bold")
    ax.text(0, -1.0 * max(sl, sr), gi.label1, ha="center", va="bottom", fontsize=9, color=SIDE[0], fontweight="bold")
    ax.text(1, -1.0 * max(sl, sr), gi.label2, ha="center", va="bottom", fontsize=9, color=SIDE[1], fontweight="bold")
    ax.set_ylim(max(len(left) * sl, len(right) * sr), -1.6 * max(sl, sr))
    handles = [Line2D([0], [0], color=c, lw=2, label=k) for k, c in BOND_COLORS.items()]
    handles += [Line2D([0], [0], color="#444", lw=1.4, ls="--", label="involves backbone")]
    ax.legend(handles=handles, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)
    fig.tight_layout()
    return fig


def contact_map(gi):
    cm = gi.contact_map_table()
    fig, ax = plt.subplots(figsize=(7, 6))
    if cm.empty:
        ax.text(0.5, 0.5, "no residue contacts", ha="center")
        ax.axis("off")
        return fig
    r1 = sorted(set(cm[gi.label1]), key=_reskey)
    r2 = sorted(set(cm[gi.label2]), key=_reskey)
    fig.set_size_inches(max(6, 0.3 * len(r2) + 2.5), max(5, 0.26 * len(r1) + 1.8))
    i1 = {k: i for i, k in enumerate(r1)}
    i2 = {k: i for i, k in enumerate(r2)}
    for _, e in cm.iterrows():
        ax.scatter(i2[e[gi.label2]], i1[e[gi.label1]], s=18 + 9 * e["atom contacts"],
                   c=CONTACT_COLORS.get(e["dominant"], "#999"), edgecolors="k", linewidths=0.35, zorder=3)
    ax.set_xticks(range(len(r2)))
    ax.set_xticklabels(r2, rotation=90, fontsize=7)
    ax.set_yticks(range(len(r1)))
    ax.set_yticklabels(r1, fontsize=7)
    ax.set_xlim(-0.7, len(r2) - 0.3)
    ax.set_ylim(len(r1) - 0.3, -0.7)
    ax.set_xlabel(gi.label2, color=SIDE[1], fontweight="bold")
    ax.set_ylabel(gi.label1, color=SIDE[0], fontweight="bold")
    ax.grid(alpha=0.2, lw=0.5)
    present = [c for c in CONTACT_COLORS if c in set(cm["dominant"])]
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=CONTACT_COLORS[c],
                              markeredgecolor="k", markersize=7, label=c.replace("_", " ")) for c in present],
              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7, frameon=False)
    ax.set_title(f"Residue contact map: {len(cm)} pairs (marker size = atom contacts)",
                 fontsize=11, loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# comparison figures
# ---------------------------------------------------------------------------
def compare_bars(cmp):
    """Grouped bars of the headline numbers for every complex."""
    df = cmp.summary_table()
    names = list(df["complex"])
    metrics = [("buried side 1 (A^2)", "buried on side 1 (A$^2$)"), ("buried side 2 (A^2)", "buried on side 2 (A$^2$)"),
               ("dG solv (kcal/mol)", "$\\Delta G_{solv}$ (kcal/mol)"), ("stab energy (kcal/mol)", "stabilisation (kcal/mol)"),
               ("H-bonds", "H-bonds"), ("salt bridges", "salt bridges")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.3 * len(metrics), 3.6))
    palette = plt.cm.tab10.colors
    for ax, (col, title) in zip(axes, metrics):
        ax.bar(range(len(names)), df[col], color=[palette[i % 10] for i in range(len(names))])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        _style(ax, title)
        ax.axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    return fig


def compare_footprints(cmp, side: int = 1):
    """Aligned per-residue BSA tracks for the shared side, one row per complex."""
    mat = cmp.residue_matrix(side)
    if mat.empty:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "no shared residues", ha="center")
        ax.axis("off")
        return fig
    n = mat.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.6 * n + 0.8), sharex=True, squeeze=False)
    xs = np.arange(len(mat))
    palette = plt.cm.tab10.colors
    for i, (ax, col) in enumerate(zip(axes[:, 0], mat.columns)):
        ax.bar(xs, mat[col].values, width=1.0, color=palette[i % 10], edgecolor="none")
        _style(ax, col, "", "BSA (A$^2$)")
    axes[-1, 0].set_xticks(xs[:: max(1, len(xs) // 40)])
    axes[-1, 0].set_xticklabels([mat.index[i] for i in xs[:: max(1, len(xs) // 40)]], rotation=90, fontsize=7)
    label = cmp.gis[0].label1 if side == 1 else cmp.gis[0].label2
    fig.suptitle(f"Interface footprint on {label} (residues aligned across complexes)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def compare_heatmap(cmp, side: int = 1):
    mat = cmp.residue_matrix(side)
    fig, ax = plt.subplots(figsize=(max(6, 0.22 * len(mat) + 2), 0.5 * mat.shape[1] + 1.5))
    if mat.empty:
        ax.text(0.5, 0.5, "no shared residues", ha="center")
        ax.axis("off")
        return fig
    im = ax.imshow(mat.values.T, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(mat.shape[1]))
    ax.set_yticklabels(mat.columns, fontsize=8)
    ax.set_xticks(range(len(mat)))
    ax.set_xticklabels(mat.index, rotation=90, fontsize=6)
    fig.colorbar(im, ax=ax, label="BSA (A$^2$)", fraction=0.03, pad=0.02)
    label = cmp.gis[0].label1 if side == 1 else cmp.gis[0].label2
    ax.set_title(f"Per-residue buried area on {label}, all complexes", fontsize=11, loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def _num(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        return int("".join(ch for ch in s if ch.isdigit()) or 0)


def _reskey(s: str):
    ch, rest = (s.split(":", 1) + [""])[:2] if ":" in s else ("", s)
    return (ch, _num(rest[1:] if rest and rest[0].isalpha() else rest))
