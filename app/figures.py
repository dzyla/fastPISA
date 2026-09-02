"""Publication-style figures for a group interface (matplotlib, no Streamlit).

Every function returns a matplotlib Figure whose SIZE is computed from its
content (number of residues, chains, pairs), so that text never overlaps:
the app renders them as PNG at a fixed DPI and shows them at native size
(scrolling horizontally when wide) instead of squeezing them into a column.
Colours are Okabe-Ito (colour-blind safe).
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
DPI = 150


def res_class(name: str) -> str:
    return RES_CLASS.get(name.strip().upper(), "other")


def _style(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)


def fig_bytes(fig, fmt: str = "png", dpi: int = 300) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    return buf.getvalue()


def fig_png(fig, dpi: int = DPI) -> bytes:
    """PNG for on-screen display at native size (see module docstring)."""
    return fig_bytes(fig, "png", dpi=dpi)


# ---------------------------------------------------------------------------
def footprint(gi, side: int, chain_axis: Optional[Dict[str, Sequence[tuple]]] = None):
    """Interface footprint along the FULL sequence of each chain of one side.

    ``chain_axis``: {chain: [(seq, icode, one_letter), ...]} for the whole
    chain (from ``report.chain_residue_axis``); without it the axis spans the
    interface residues +- 10. Bars are coloured by residue class; the six
    largest contributions are labelled, staggered so labels never overlap.
    """
    res = gi.residues_side1 if side == 1 else gi.residues_side2
    label = gi.label1 if side == 1 else gi.label2
    chains = sorted({r.chain for r in res})
    n = max(1, len(chains))
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.6 * n + 0.5), squeeze=False, constrained_layout=True)
    for ax, ch in zip(axes[:, 0], chains):
        rr = [r for r in res if r.chain == ch]
        xs = np.array([_num(r.seq) for r in rr])
        ys = np.array([r.bsa for r in rr])
        if chain_axis and ch in chain_axis and chain_axis[ch]:
            nums = [_num(s) for _, s, _, _ in chain_axis[ch]]
            lo, hi = min(nums), max(nums)
        else:
            lo, hi = xs.min() - 10, xs.max() + 10
        ax.bar(xs, ys, width=max(1.0, (hi - lo) / 400), color=[CLASS_COLORS[res_class(r.name)] for r in rr],
               edgecolor="none")
        ax.set_xlim(lo - 1, hi + 1)
        ymax = ys.max() if len(ys) else 1
        ax.set_ylim(0, ymax * 1.35)
        top = sorted(rr, key=lambda r: -r.bsa)[:6]
        top.sort(key=lambda r: _num(r.seq))
        last_x, level = -1e9, 0
        for r in top:
            x = _num(r.seq)
            level = (level + 1) % 3 if (x - last_x) < (hi - lo) * 0.04 else 0
            last_x = x
            ax.annotate(f"{r.one}{r.seq}", (x, r.bsa), xytext=(0, 4 + 10 * level), textcoords="offset points",
                        fontsize=7, ha="center", va="bottom",
                        arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#888"} if level else None)
        _style(ax, f"{label}, chain {ch}: {len(rr)} interface residues, {ys.sum():,.0f} A$^2$ buried, "
                   f"residues {lo}-{hi}", "residue number", "BSA (A$^2$)")
    present = [k for k in CLASS_COLORS if any(res_class(r.name) == k for r in res)]
    fig.legend(handles=[Patch(color=CLASS_COLORS[k], label=k) for k in present], fontsize=7,
               ncol=len(present), loc="outside upper right", frameon=False)
    return fig


def residue_bars(gi, top_n: int = 30):
    """Horizontal bars: the ``top_n`` residues per side by buried area, grey
    = isolated ASA behind the coloured BSA, bond count printed at the bar end."""
    sides = ((gi.residues_side1, gi.label1, SIDE[0]), (gi.residues_side2, gi.label2, SIDE[1]))
    nmax = max(1, min(top_n, max(len(sides[0][0]), len(sides[1][0]))))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.26 * nmax + 1.4), constrained_layout=True)
    for ax, (res, label, color) in zip(axes, sides):
        if not res:
            ax.set_visible(False)
            continue
        rr = sorted(res, key=lambda r: -r.bsa)[:top_n]
        y = np.arange(len(rr))
        asa = np.array([r.asa for r in rr])
        bsa = np.array([r.bsa for r in rr])
        ax.barh(y, asa, color="#EBEBEB", edgecolor="none", label="isolated ASA")
        ax.barh(y, bsa, color=color, edgecolor="none", label="buried (BSA)")
        multi = len({r.chain for r in rr}) > 1
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r.name.title()}{r.seq}{r.icode}" + (f" ({r.chain})" if multi else "") for r in rr],
                           fontsize=8)
        for i, r in enumerate(rr):
            if r.n_bonds:
                ax.text(max(asa[i], bsa[i]) + 2, i, f"{r.n_bonds} bond{'s' if r.n_bonds > 1 else ''}",
                        va="center", fontsize=7, color="#444")
        ax.invert_yaxis()
        ax.set_xlim(0, max(asa.max(), bsa.max()) * 1.25)
        ax.grid(axis="x", alpha=0.25, lw=0.6)
        ax.grid(axis="y", alpha=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(f"{label}: top {len(rr)} of {len(res)} residues, {sum(r.bsa for r in res):,.0f} A$^2$ buried",
                     fontsize=10, loc="left", fontweight="bold")
        ax.set_xlabel("surface area (A$^2$)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    return fig


def composition(gi):
    """Residue-class composition of the buried area per side, and the
    apolar / polar / bond decomposition of the energy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [2.2, 1]},
                             constrained_layout=True)
    ax = axes[0]
    classes = list(CLASS_COLORS)
    for i, res in enumerate((gi.residues_side1, gi.residues_side2)):
        tot = sum(r.bsa for r in res) or 1.0
        left = 0.0
        for cl in classes:
            v = sum(r.bsa for r in res if res_class(r.name) == cl) / tot
            if v > 0:
                ax.barh(i, v, left=left, color=CLASS_COLORS[cl], edgecolor="white", lw=0.5, height=0.6)
                if v > 0.07:
                    ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white")
                left += v
    ax.set_yticks([0, 1])
    ax.set_yticklabels([gi.label1, gi.label2], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 1.6)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _style(ax, "Buried area by residue class", "fraction of buried surface", "")
    ax.grid(False)
    present = [k for k in classes if any(res_class(r.name) == k for r in gi.residues_side1 + gi.residues_side2)]
    ax.legend(handles=[Patch(color=CLASS_COLORS[k], label=k) for k in present],
              fontsize=7, ncol=min(4, len(present)), loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    ax = axes[1]
    vals = [gi.dg_apolar, gi.dg_polar, gi.stab_energy - gi.dg_solv, gi.stab_energy]
    names = ["apolar\nburial", "polar\nburial", "bonds", "stabilisation\nenergy"]
    cols = ["#7F7F7F", "#009E73", "#1B9E77", "#000000"]
    ax.bar(range(4), vals, color=cols, width=0.65)
    span = max(abs(v) for v in vals) or 1
    for i, v in enumerate(vals):
        ax.text(i, v + (0.03 if v >= 0 else -0.03) * span, f"{v:+.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(min(0, min(vals)) - 0.2 * span, max(0, max(vals)) + 0.2 * span)
    ax.axhline(0, color="k", lw=0.6)
    _style(ax, "Energy decomposition", "", "kcal/mol")
    return fig


def bond_network(gi):
    """Bipartite residue graph of H-bonds, salt bridges and disulfides."""
    bt = gi.bonds_table(one_letter_codes=True)
    if bt.empty:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "no hydrogen bonds, salt bridges or disulfides", ha="center", va="center")
        ax.axis("off")
        return fig
    left = list(dict.fromkeys(sorted(bt[gi.label1], key=_reskey)))
    right = list(dict.fromkeys(sorted(bt[gi.label2], key=_reskey)))
    n = max(len(left), len(right))
    fig, ax = plt.subplots(figsize=(7, 0.32 * n + 1.6), constrained_layout=True)
    yl = {r: i for i, r in enumerate(left)}
    yr = {r: i for i, r in enumerate(right)}
    sl = n / max(len(left), 1)
    sr = n / max(len(right), 1)
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
    ax.set_ylim(n + 0.5, -1.8)
    ax.text(0, -1.1, gi.label1, ha="center", va="center", fontsize=9, color=SIDE[0], fontweight="bold")
    ax.text(1, -1.1, gi.label2, ha="center", va="center", fontsize=9, color=SIDE[1], fontweight="bold")
    ax.axis("off")
    fig.suptitle(f"Bond network: {gi.n_hbonds} H-bonds, {gi.n_salt_bridges} salt bridges, "
                 f"{gi.n_disulfides} disulfides", fontsize=10, x=0.01, ha="left", fontweight="bold")
    handles = [Line2D([0], [0], color=c, lw=2, label=k) for k, c in BOND_COLORS.items()]
    handles += [Line2D([0], [0], color="#444", lw=1.4, ls="--", label="involves backbone")]
    fig.legend(handles=handles, fontsize=7, loc="outside lower center", ncol=4, frameon=False)
    return fig


# ---------------------------------------------------------------------------
# contact maps
# ---------------------------------------------------------------------------
def contact_maps_per_pair(gi) -> List[tuple]:
    """One interface-residue contact map per chain pair: [(title, Figure)].

    A single map over every chain of both groups is unreadable (a dimeric
    antigen with two Fabs gives a mostly-empty 100 x 100 grid); per pair each
    panel is compact and every label is legible.
    """
    out = []
    for p in gi.pairs:
        cm = p.contact_map
        if not cm:
            continue
        g1 = {_chain(c) for c in gi.group1}
        rows: Dict[str, int] = {}
        cols: Dict[str, int] = {}
        pts = []
        for e in cm:
            flip = e["residue_1_chain"] not in g1
            a, b = ("2", "1") if flip else ("1", "2")
            k1 = f"{e[f'residue_{a}_chain']}:{_one(e[f'residue_{a}_type'])}{e[f'residue_{a}_seq']}"
            k2 = f"{e[f'residue_{b}_chain']}:{_one(e[f'residue_{b}_type'])}{e[f'residue_{b}_seq']}"
            pts.append((k1, k2, e["num_contacts"], e.get("dominant_interaction")))
        r1 = sorted({k for k, _, _, _ in pts}, key=_reskey)
        r2 = sorted({k for _, k, _, _ in pts}, key=_reskey)
        rows = {k: i for i, k in enumerate(r1)}
        cols = {k: i for i, k in enumerate(r2)}
        fig, ax = plt.subplots(figsize=(0.3 * len(r2) + 2.8, 0.3 * len(r1) + 1.8), constrained_layout=True)
        for k1, k2, nc, dom in pts:
            ax.scatter(cols[k2], rows[k1], s=22 + 10 * nc, c=CONTACT_COLORS.get(dom, "#999"),
                       edgecolors="k", linewidths=0.35, zorder=3)
        ax.set_xticks(range(len(r2)))
        ax.set_xticklabels(r2, rotation=90, fontsize=8)
        ax.set_yticks(range(len(r1)))
        ax.set_yticklabels(r1, fontsize=8)
        ax.set_xlim(-0.7, len(r2) - 0.3)
        ax.set_ylim(len(r1) - 0.3, -0.7)
        ax.set_xlabel(f"{gi.label2} (chain {r2[0].split(':')[0]})", color=SIDE[1], fontweight="bold", fontsize=9)
        ax.set_ylabel(f"{gi.label1} (chain {r1[0].split(':')[0]})", color=SIDE[0], fontweight="bold", fontsize=9)
        ax.grid(alpha=0.2, lw=0.5)
        present = [c for c in CONTACT_COLORS if c in {d for _, _, _, d in pts}]
        ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=CONTACT_COLORS[c],
                                  markeredgecolor="k", markersize=7, label=c.replace("_", " ")) for c in present],
                  loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7, frameon=False)
        title = f"{' + '.join(p.chains)}: {len(pts)} residue pairs, {p.interface_area:,.0f} A$^2$"
        ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
        out.append((" + ".join(p.chains), fig))
    return out


def full_contact_map(gi, axis1: Sequence[tuple], axis2: Sequence[tuple], by: str = "class"):
    """COCOMAPS-1-style map over the FULL sequences of both sides.

    ``axis1`` / ``axis2``: every residue of the group-1 / group-2 chains as
    (chain, seq, icode, one_letter) (``report.chain_residue_axis``). Each
    contacting residue pair is a cell coloured by its dominant interaction
    class (``by="class"``) or by minimum distance (``by="distance"``); chain
    boundaries are drawn, ticks every 10 residues, so the reader sees WHERE
    along each sequence the interface sits (CDR loops, epitope segments).
    """
    i1 = {(c, s, ic): i for i, (c, s, ic, _) in enumerate(axis1)}
    i2 = {(c, s, ic): i for i, (c, s, ic, _) in enumerate(axis2)}
    n1, n2 = len(axis1), len(axis2)
    g1 = {_chain(c) for c in gi.group1}
    cells = []
    for p in gi.pairs:
        for e in p.contact_map:
            flip = e["residue_1_chain"] not in g1
            a, b = ("2", "1") if flip else ("1", "2")
            k1 = (e[f"residue_{a}_chain"], str(e[f"residue_{a}_seq"]), e.get(f"residue_{a}_icode") or "")
            k2 = (e[f"residue_{b}_chain"], str(e[f"residue_{b}_seq"]), e.get(f"residue_{b}_icode") or "")
            if k1 in i1 and k2 in i2:
                cells.append((i1[k1], i2[k2], e.get("dominant_interaction"), e["min_distance"], e["num_contacts"]))
    w = min(16, max(7, n2 / 40))
    h = min(16, max(5, n1 / 40))
    fig, ax = plt.subplots(figsize=(w + 1.5, h + 1.0), constrained_layout=True)
    ax.set_xlim(-0.5, n2 - 0.5)
    ax.set_ylim(n1 - 0.5, -0.5)
    ax.set_facecolor("#FAFAFA")
    if by == "distance":
        d = np.array([c[3] for c in cells]) if cells else np.array([])
        sc = ax.scatter([c[1] for c in cells], [c[0] for c in cells], c=d, cmap="viridis_r", s=14,
                        marker="s", vmin=2.5, vmax=5.0, linewidths=0)
        fig.colorbar(sc, ax=ax, label="minimum distance (A)", fraction=0.03, pad=0.01)
    else:
        for c in cells:
            ax.scatter(c[1], c[0], s=14, marker="s", c=CONTACT_COLORS.get(c[2], "#999"), linewidths=0)
        present = [c for c in CONTACT_COLORS if c in {x[2] for x in cells}]
        ax.legend(handles=[Line2D([0], [0], marker="s", color="w", markerfacecolor=CONTACT_COLORS[c],
                                  markersize=7, label=c.replace("_", " ")) for c in present],
                  loc="upper left", bbox_to_anchor=(1.005, 1), fontsize=7, frameon=False)
    # chain boundaries + ticks every 10 residues labelled with the author number
    for axis, idx_of, vertical in ((axis1, None, False), (axis2, None, True)):
        pass
    _sequence_ticks(ax, axis1, vertical=False)
    _sequence_ticks(ax, axis2, vertical=True)
    ax.set_xlabel(gi.label2, color=SIDE[1], fontweight="bold", fontsize=10)
    ax.set_ylabel(gi.label1, color=SIDE[0], fontweight="bold", fontsize=10)
    ax.set_title(f"Full-sequence contact map: {gi.label1} ({n1} residues) x {gi.label2} ({n2} residues), "
                 f"{len(cells)} contacting pairs", fontsize=10, loc="left", fontweight="bold")
    ax.tick_params(labelsize=7)
    return fig


def _sequence_ticks(ax, axis: Sequence[tuple], vertical: bool):
    """Ticks every 10 residues (author numbering) and lines at chain breaks."""
    ticks, labels, bounds = [], [], []
    prev_chain = None
    for i, (ch, seq, ic, _) in enumerate(axis):
        if ch != prev_chain:
            if prev_chain is not None:
                bounds.append(i - 0.5)
            prev_chain = ch
        n = _num(seq)
        if n % 10 == 0 and not ic:
            ticks.append(i)
            labels.append(f"{ch}:{n}")
    if vertical:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=90)
        for b in bounds:
            ax.axvline(b, color="#444", lw=0.8)
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        for b in bounds:
            ax.axhline(b, color="#444", lw=0.8)


# ---------------------------------------------------------------------------
# comparison figures
# ---------------------------------------------------------------------------
def compare_bars(cmp):
    df = cmp.summary_table()
    names = list(df["complex"])
    metrics = [("buried side 1 (A^2)", "buried on side 1 (A$^2$)"), ("buried side 2 (A^2)", "buried on side 2 (A$^2$)"),
               ("dG solv (kcal/mol)", "$\\Delta G_{solv}$ (kcal/mol)"), ("stab energy (kcal/mol)", "stabilisation (kcal/mol)"),
               ("H-bonds", "H-bonds"), ("salt bridges", "salt bridges")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.4 * len(metrics), 3.4), constrained_layout=True)
    palette = plt.cm.tab10.colors
    for ax, (col, title) in zip(axes, metrics):
        ax.bar(range(len(names)), df[col], color=[palette[i % 10] for i in range(len(names))])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        _style(ax, title)
        ax.axhline(0, color="k", lw=0.5)
    return fig


def compare_footprints(cmp, side: int = 1):
    mat = cmp.residue_matrix(side)
    if mat.empty:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "no shared residues", ha="center")
        ax.axis("off")
        return fig
    n = mat.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(max(10, 0.18 * len(mat) + 2), 1.7 * n + 0.8), sharex=True,
                             squeeze=False, constrained_layout=True)
    xs = np.arange(len(mat))
    palette = plt.cm.tab10.colors
    for i, (ax, col) in enumerate(zip(axes[:, 0], mat.columns)):
        ax.bar(xs, mat[col].values, width=0.85, color=palette[i % 10], edgecolor="none")
        _style(ax, col, "", "BSA (A$^2$)")
    axes[-1, 0].set_xticks(xs)
    axes[-1, 0].set_xticklabels(list(mat.index), rotation=90, fontsize=7)
    label = cmp.gis[0].label1 if side == 1 else cmp.gis[0].label2
    fig.suptitle(f"Interface footprint on {label}, residues aligned across complexes", fontsize=10, x=0.01,
                 ha="left", fontweight="bold")
    return fig


def compare_heatmap(cmp, side: int = 1):
    mat = cmp.residue_matrix(side)
    if mat.empty:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "no shared residues", ha="center")
        ax.axis("off")
        return fig
    fig, ax = plt.subplots(figsize=(max(8, 0.2 * len(mat) + 2), 0.45 * mat.shape[1] + 1.6), constrained_layout=True)
    im = ax.imshow(mat.values.T, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(mat.shape[1]))
    ax.set_yticklabels(mat.columns, fontsize=8)
    ax.set_xticks(range(len(mat)))
    ax.set_xticklabels(mat.index, rotation=90, fontsize=7)
    fig.colorbar(im, ax=ax, label="BSA (A$^2$)", fraction=0.03, pad=0.02)
    label = cmp.gis[0].label1 if side == 1 else cmp.gis[0].label2
    ax.set_title(f"Per-residue buried area on {label}, all complexes", fontsize=10, loc="left", fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
def _num(s: str) -> int:
    try:
        return int(s)
    except (ValueError, TypeError):
        return int("".join(ch for ch in str(s) if ch.isdigit()) or 0)


def _reskey(s: str):
    ch, rest = (s.split(":", 1) + [""])[:2] if ":" in s else ("", s)
    return (ch, _num(rest[1:] if rest and rest[0].isalpha() else rest))


def _chain(label: str) -> str:
    if label.startswith("["):
        return label.split("]", 1)[1].split(":", 1)[0]
    return label


def _one(name: str) -> str:
    from fastpisa.report import one_letter
    return one_letter(name)
