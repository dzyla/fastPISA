"""Visualization helpers for fastPISA (item 4.3 of fastpisa_improvements.md).

.. note::
    ``plot_contact_heatmap`` requires ``matplotlib`` (install with
    ``pip install fastpisa[viz]``); both ``write_pymol_script`` and
    ``write_molstar_html`` are pure text generation and have no extra deps.
"""
from __future__ import annotations

import json
import os
from typing import Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# colour helpers (no matplotlib dependency)
# ---------------------------------------------------------------------------
def _value_to_hex(value: float, vmin: float, vmax: float) -> str:
    """Map a value in [vmin, vmax] to a blue->white->red hex colour string."""
    if vmax <= vmin:
        t = 0.0
    else:
        t = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    r, g, b = 255, 255, 255
    if t < 0.5:
        tt = 2.0 * t
        r, g, b = int(255 * tt), int(255 * tt), 255
    else:
        tt = 2.0 * (t - 0.5)
        r, g, b = 255, int(255 * (1 - tt)), int(255 * (1 - tt))
    return f"0x{r:02x}{g:02x}{b:02x}"


def _mol_interface_residues(interface) -> List[Tuple[str, str, float, str]]:
    """Return (chain, resi, value, resname) for each interface residue.

    ``value`` is per-residue buried surface area (for PyMOL colouring).
    """
    out = []
    for mol in interface.molecules:
        comp = mol.get("residue_label_comp_ids") or []
        seq = mol.get("residue_seq_ids") or []
        ba = mol.get("buried_surface_areas") or []
        chain = mol.get("auth_asym_id", "")
        for k in range(len(comp)):
            resi = str(seq[k]) if k < len(seq) else None
            if not resi or resi == "?":
                continue
            val = ba[k] if k < len(ba) and ba[k] is not None else 0.0
            out.append((chain, resi, val, comp[k]))
    return out


def write_pymol_script(structure_path: str, interface, out_path: str,
                       by: str = "bsa") -> str:
    """Write a PyMOL ``.pml`` colouring the interface residues by BSA.

    Colors interface residues blue (small buried area) -> red (large buried
    area). Run with ``pymol <out>.pml`` (requires the model file too).
    """
    if by not in ("bsa",):
        raise ValueError(f"by must be 'bsa', got {by!r}")
    residues = _mol_interface_residues(interface)
    if not residues:
        raise ValueError("interface has no per-residue data to colour")

    vals = [r[2] for r in residues]
    vmin, vmax = min(vals), max(vals)

    lines = [
        "# fastPISA PyMOL script",
        f"load {os.path.abspath(structure_path)}",
        "hide everything",
        "show cartoon",
        "show sticks, (elem S)",
        "# interface residue colouring by buried surface area (blue=low, red=high)",
    ]
    for chain, resi, val, *_ in residues:
        col = _value_to_hex(val, vmin, vmax)
        lines.append(f"color {col}, (chain {chain} and resi {resi})")

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return os.path.abspath(out_path)


# ---------------------------------------------------------------------------
# matplotlib contact-map heatmap (optional dependency)
# ---------------------------------------------------------------------------
def plot_contact_heatmap(interface, atoms, out_path: Optional[str] = None,
                         show: bool = True, cmap: str = "viridis",
                         figsize: Tuple[float, float] = (8, 6)):
    """Plot a residue-residue contact-count heatmap for an interface.

    Rows/columns are the interface residues of molecules 1 and 2; cell values
    are the number of atom-atom contacts between the pair. When ``out_path``
    is given the figure is saved there, otherwise it is shown (``show``).
    """
    try:
        import matplotlib
        if out_path:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dep
        raise ImportError(
            "plot_contact_heatmap needs matplotlib. Install it with "
            "`pip install fastpisa[viz]` or `pip install matplotlib`."
        ) from None

    from fastpisa.cocomaps.contact_map import build_contact_matrix
    mat, rows, cols = build_contact_matrix(interface.contacts, atoms)
    if mat.size == 0:
        raise ValueError("interface has no contact matrix to plot")

    def label(r):
        return f"{r[0]}{r[1]}"

    row_labels = [label(r) for r in rows]
    col_labels = [label(r) for r in cols]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat.T, cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(row_labels)))
    ax.set_yticks(range(len(col_labels)))
    ax.set_xticklabels(row_labels, rotation=90, fontsize=6)
    ax.set_yticklabels(col_labels, fontsize=6)
    ax.set_xlabel("Molecule 1 residues")
    ax.set_ylabel("Molecule 2 residues")
    ax.set_title(f"Interface {interface.interface_id}: residue-residue contacts")
    fig.colorbar(im, ax=ax, label="atom-atom contacts")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return os.path.abspath(out_path)
    if show:
        plt.show()
    return None


# ---------------------------------------------------------------------------
# standalone Mol* HTML viewer (optional, self-contained)
# ---------------------------------------------------------------------------
# Minimal self-contained HTML that loads the structure string and embeds the
# MolStar viewer from CDN, then selects + colours the interface residues.
_MOLSTAR_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>fastPISA interface {iid}</title>
<style>
  body {{ margin: 0; font-family: sans-serif; }}
  #app {{ position: absolute; top: 0; left: 0; right: 0; bottom: 0; }}
</style>
</head>
<body>
<div id="app"></div>
<script src="https://cdn.jsdelivr.net/npm/molstar@3/build/molstar.js"></script>
<script>
  const structData = {struct_json};
  const ifaceSelections = {selections_json};
  async function init() {{
    const plugin = await molstar.PluginContext.create({{
      target: document.getElementById('app'),
      layout: {{ isExpanded: true, showControls: true }},
    }});
    await plugin.init();

    const data = await plugin.builders.data.rawData({{ data: structData }});
    const trajectory = await plugin.builders.structure.parseTrajectory(data, 'pdb');
    const model = await plugin.builders.structure.createModel(trajectory);
    const structure = await plugin.builders.structure.createStructure(model);
    await plugin.builders.structure.representation.addRepresentation(structure, {{ type: 'cartoon' }});

    for (const sel of ifaceSelections) {{
      const selExpr = 'chain ' + sel.chain + ' and resi ' + sel.resi;
      const component = await plugin.builders.structure.tryCreateComponentFromExpression(
        structure, selExpr
      );
      if (component) {{
        await plugin.builders.structure.representation.addRepresentation(component,
          {{ type: 'ball-and-stick', color: {{ color: sel.color }} }});
      }}
    }}
  }}
  init().catch(e => {{ document.body.textContent = 'Mol* failed: ' + e; }});
</script>
</body>
</html>
"""


def write_molstar_html(structure_path: str, interface, out_path: str) -> str:
    """Write a self-contained HTML file opening the structure in MolStar.

    The model is inlined as a PDB string and the interface residues are drawn
    as ball-and-stick coloured blue->red by buried surface area. Requires an
    internet connection the first time (MolStar is loaded from a CDN).
    """
    from fastpisa.parser.pdb_parser import parse_pdb, parse_mmcif
    if str(structure_path).endswith((".cif", ".cif.gz")):
        st = parse_mmcif(structure_path)
        pdb_text = _structure_to_pdb(st)
    else:
        pdb_text = open(structure_path).read()

    residues = _mol_interface_residues(interface)
    vals = [r[2] for r in residues] or [0.0]
    vmin, vmax = min(vals), max(vals)
    selections = [
        {"chain": chain, "resi": int(resi), "color": "#" + _value_to_hex(v, vmin, vmax)[2:]}
        for (chain, resi, v, _resn) in residues if resi.isdigit()
    ]

    html = _MOLSTAR_HTML_TEMPLATE.format(
        iid=interface.interface_id,
        struct_json=json.dumps(pdb_text),
        selections_json=json.dumps(selections),
    )
    with open(out_path, "w") as fh:
        fh.write(html)
    return os.path.abspath(out_path)


def _structure_to_pdb(structure) -> str:
    """Best-effort PDB serialisation of a parsed structure (for Mol* inline)."""
    import tempfile
    import gemmi
    st = gemmi.Structure()
    model = gemmi.Model(1)
    st.add_model(model)
    for chain in structure.chains:
        gchain = gemmi.Chain(chain.auth_asym_id)
        # reconstruct residues from atoms
        by_res = {}
        for atom in chain.atoms:
            by_res.setdefault((atom.res_seq, atom.icode or ""), []).append(atom)
        for (seq, icode), atoms in sorted(by_res.items()):
            cres = gemmi.Residue()
            cres.seqid = gemmi.SeqId(seq, icode)
            cres.name = atoms[0].res_name
            for a in atoms:
                gat = gemmi.Atom()
                gat.name = a.atom_name
                gat.element = gemmi.Element(a.element)
                gat.pos = gemmi.Position(a.x, a.y, a.z)
                cres.add_atom(gat)
            gchain.add_residue(cres)
        model.add_chain(gchain)
    fd, tmp = tempfile.mkstemp(suffix=".pdb")
    try:
        os.close(fd)
        st.write_pdb(tmp)
        with open(tmp) as fh:
            return fh.read()
    finally:
        os.remove(tmp)
