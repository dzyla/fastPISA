#!/usr/bin/env python3
"""Render the README comparison figures from the committed calibration tables.

    python examples/make_figures.py            # -> docs/figures/*.png

Everything here is offline: the scatter plots come from
``tests/data/calibration/features.json.gz`` (6.9k interfaces, PISA's values
vs ours) and ``residue_fit.json.gz`` (119k residues); the contact-map and
H-bond panels run fastPISA on the committed 1brs reference structure.
Numbers printed alongside are the ones quoted in the README.
"""
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fastpisa.energy.asp_table import SIGMA
from fastpisa.reference.calibrate import (
    dg_metrics, load_feature_table, load_residue_fit_rows, predict_dg, predict_p,
)
from fastpisa.scoring.scoring import P_VALUE_Z_SCALE

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "figures")
os.makedirs(OUT, exist_ok=True)

POLY_C, LIG_C = "#1f77b4", "#d62728"


def _style(ax, xl, yl, title):
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25)


def _identity(ax, lo, hi):
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def fig_energetics(F):
    poly = np.array([r["is_polymer_pair"] for r in F])
    dg_ref = np.array([r["dg_ref"] for r in F], float)
    dg = predict_dg(F, SIGMA)
    area_ref = np.array([r["area_ref"] for r in F], float)
    area = np.array([r["area_fp"] for r in F], float)
    pv_ref = np.array([r["pv_ref"] if r["pv_ref"] is not None else np.nan for r in F], float)
    pv = predict_p(F, SIGMA, P_VALUE_Z_SCALE, dg=dg)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for mask, c, lab, z in ((~poly, LIG_C, "ligand-involving", 1), (poly, POLY_C, "polymer-polymer", 2)):
        axes[0].scatter(area_ref[mask], area[mask], s=5, c=c, alpha=0.35, label=lab, zorder=z)
        axes[1].scatter(dg_ref[mask], dg[mask], s=5, c=c, alpha=0.35, label=lab, zorder=z)
        ok = mask & np.isfinite(pv_ref)
        axes[2].scatter(pv_ref[ok], pv[ok], s=5, c=c, alpha=0.35, label=lab, zorder=z)
    _identity(axes[0], 0, 3000)
    _identity(axes[1], -80, 20)
    _identity(axes[2], 0, 1)
    mp = dg_metrics(dg, dg_ref, poly)
    rel = np.abs(area - area_ref) / np.maximum(area_ref, 1)
    ok = poly & np.isfinite(pv_ref)
    _style(axes[0], "PISA interface area (A$^2$)", "fastPISA",
           f"Interface area\nmedian |rel. err| polymer {np.median(rel[poly]):.1%}, ligand {np.median(rel[~poly]):.1%}")
    _style(axes[1], "PISA $\\Delta G_{solv}$ (kcal/mol)", "fastPISA",
           f"Solvation energy (polymer pairs)\nr = {mp['pearson']:.3f}, R$^2_{{1:1}}$ = {mp['r2_identity']:.3f}, median |err| = {mp['median_abs_err']:.2f}")
    _style(axes[2], "PISA P-value", "fastPISA",
           f"Hydrophobicity P-value (polymer pairs)\nmedian |err| = {np.median(np.abs(pv[ok]-pv_ref[ok])):.3f}")
    axes[0].legend(loc="upper left", fontsize=9, markerscale=3)
    fig.suptitle("fastPISA vs original PISA -- 674 PDB entries, 6.9k interfaces (in-sample view; CV numbers in the text)",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT, "energetics_vs_pisa.png")
    fig.savefig(path, dpi=130)
    print("wrote", path)
    print(f"  polymer dG: {mp}")
    return path


def fig_residues(rows):
    bsa_fp, bsa_ref, solv_fp, solv_ref = [], [], [], []
    from fastpisa.energy.asp_table import sigma_of_fine
    for _, r in rows:
        if not r["is_polymer"] or not r["bsa_ref"] or r["bsa_ref"] < 5:
            continue
        b = sum(r["bsa_by_type"].values())
        bsa_fp.append(b)
        bsa_ref.append(r["bsa_ref"])
        solv_fp.append(sum(sigma_of_fine(t) * a for t, a in r["bsa_by_type"].items()))
        solv_ref.append(r["solv_ref"])
    bsa_fp, bsa_ref = np.array(bsa_fp), np.array(bsa_ref)
    solv_fp, solv_ref = np.array(solv_fp), np.array(solv_ref)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].hexbin(bsa_ref, bsa_fp, gridsize=70, bins="log", cmap="Blues", extent=(0, 200, 0, 200))
    _identity(axes[0], 0, 200)
    rel = np.abs(bsa_fp - bsa_ref) / bsa_ref
    _style(axes[0], "PISA residue BSA (A$^2$)", "fastPISA",
           f"Per-residue buried area, {len(rel):,} residues\nmedian |rel. err| {np.median(rel):.2%}")
    axes[1].hexbin(solv_ref, solv_fp, gridsize=70, bins="log", cmap="Blues", extent=(-4, 3, -4, 3))
    _identity(axes[1], -4, 3)
    r = np.corrcoef(solv_fp, solv_ref)[0, 1]
    _style(axes[1], "PISA residue $\\Delta G_{solv}$ (kcal/mol)", "fastPISA",
           f"Per-residue solvation energy\nr = {r:.3f}, median |err| = {np.median(np.abs(solv_fp-solv_ref)):.3f}")
    fig.tight_layout()
    path = os.path.join(OUT, "residues_vs_pisa.png")
    fig.savefig(path, dpi=130)
    print("wrote", path)
    return path


def fig_ligands(F):
    import collections
    import re
    by = collections.defaultdict(list)
    for r in F:
        if r["is_polymer_pair"]:
            continue
        rel = abs(r["area_fp"] - r["area_ref"]) / max(r["area_ref"], 1)
        for l in set(re.findall(r"\[([A-Z0-9]+)\]", r["pair"])):
            by[l].append(rel)
    top = sorted(((l, v) for l, v in by.items() if len(v) >= 40), key=lambda kv: np.median(kv[1]))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar([l for l, _ in top], [np.median(v) * 100 for _, v in top],
           color=[LIG_C if np.median(v) > 0.1 else POLY_C for _, v in top])
    ax.set_ylabel("median |rel. error| in interface area (%)")
    ax.set_title("Ligand / ion interface area vs PISA, by ligand type (n >= 40 interfaces)", fontsize=11)
    ax.tick_params(axis="x", rotation=60)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(OUT, "ligand_area_by_type.png")
    fig.savefig(path, dpi=130)
    print("wrote", path)
    for l, v in top:
        print(f"  {l:5} n={len(v):4d} median {np.median(v):.1%}")
    return path


def fig_contact_map():
    """1brs barnase-barstar: COCOMAPS contact map + H-bond table figure."""
    import fastpisa
    from fastpisa.reference.ebi_pisa import cached_pdb_path

    pdb = cached_pdb_path("1brs")
    if pdb is None:
        print("1brs not cached; skipping contact-map figure")
        return None
    res = fastpisa.analyze(pdb, pdb_id="1brs")
    iface = res.interface_between("A", "D")
    cm = iface.contact_map
    r1 = sorted({(e["residue_1_chain"], e["residue_1_seq"], e["residue_1_type"]) for e in cm}, key=lambda t: t[1])
    r2 = sorted({(e["residue_2_chain"], e["residue_2_seq"], e["residue_2_type"]) for e in cm}, key=lambda t: t[1])
    i1 = {k: i for i, k in enumerate(r1)}
    i2 = {k: i for i, k in enumerate(r2)}
    classes = ["hydrogen_bond", "salt_bridge", "pi_pi", "cation_pi", "ch_pi",
               "polar_vdw", "apolar_vdw", "weak_hbond", "proximal"]
    colors = {"hydrogen_bond": "#1f77b4", "salt_bridge": "#d62728", "pi_pi": "#9467bd",
              "cation_pi": "#8c564b", "ch_pi": "#e377c2", "polar_vdw": "#2ca02c",
              "apolar_vdw": "#ff7f0e", "weak_hbond": "#17becf", "proximal": "#c7c7c7"}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.15, 1]})
    for e in cm:
        x = i2[(e["residue_2_chain"], e["residue_2_seq"], e["residue_2_type"])]
        y = i1[(e["residue_1_chain"], e["residue_1_seq"], e["residue_1_type"])]
        ax.scatter(x, y, s=30 + 12 * e["num_contacts"], c=colors.get(e["dominant_interaction"], "#999"),
                   edgecolors="k", linewidths=0.4)
    ax.set_xticks(range(len(r2)))
    ax.set_xticklabels([f"{t}{s}" for _, s, t in r2], rotation=90, fontsize=8)
    ax.set_yticks(range(len(r1)))
    ax.set_yticklabels([f"{t}{s}" for _, s, t in r1], fontsize=8)
    ax.set_xlabel(f"chain {r2[0][0]} (barstar)")
    ax.set_ylabel(f"chain {r1[0][0]} (barnase)")
    ax.set_title("1brs A+D residue contact map (COCOMAPS classes; size = atom contacts)", fontsize=10)
    ax.invert_yaxis()
    ax.grid(alpha=0.2)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[c], markeredgecolor="k",
                              markersize=8, label=c) for c in classes],
              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    # H-bond / salt-bridge table
    ax2.axis("off")
    rows = [(c.bond_type.replace("_", " "), f"{c.atom1_chain}:{c.atom1_residue}{c.atom1_seq}.{c.atom1_name.strip()}",
             f"{c.atom2_chain}:{c.atom2_residue}{c.atom2_seq}.{c.atom2_name.strip()}", f"{c.distance:.2f}")
            for c in sorted(iface.contacts, key=lambda c: (c.bond_type, c.distance))
            if c.bond_type in ("hbond", "salt_bridge")]
    tbl = ax2.table(cellText=rows[:26], colLabels=["type", "atom 1", "atom 2", "d (A)"],
                    loc="upper center", cellLoc="left", colWidths=[0.22, 0.3, 0.3, 0.13])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.15)
    # cross-check the very same pairs against PISA's own bond list
    from fastpisa.reference.bonds_audit import audit_entry
    aud = [a for a in audit_entry("1brs") if a["pair"] == "A+D"][0]
    hb, sb = aud["hb"], aud["sb"]
    ax2.set_title(f"{iface.number_hydrogen_bonds} H-bonds, {iface.number_salt_bridges} salt bridges  |  "
                  f"vs PISA's list: H-bonds {hb['n_matched']}/{hb['n_ref']} matched, "
                  f"salt bridges {sb['n_matched']}/{sb['n_ref']}", fontsize=10)
    fig.suptitle(f"{iface!r}", fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT, "contact_map_1brs.png")
    fig.savefig(path, dpi=130)
    print("wrote", path)
    return path


def main():
    F = load_feature_table()
    rows = load_residue_fit_rows()
    if not F or not rows:
        sys.exit("calibration tables missing")
    fig_energetics(F)
    fig_residues(rows)
    fig_ligands(F)
    fig_contact_map()


if __name__ == "__main__":
    main()
