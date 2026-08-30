#!/usr/bin/env python3
"""
fastPISA CLI: Local reproduction of PISA with COCOMAPS mode.

Three modes analyse a PDB/mmCIF structure (all find identical interfaces —
they share one analysis core):

  --mode combined  (default) One unified report: PISA thermodynamic/surface
                   analysis AND the COCOMAPS contact map on every interface.

  --mode pisa      PISA thermodynamic/surface analysis only, output in the
                   PDBe PISA JSON schema ('assembly' + 'interfaces').

  --mode cocomaps  COCOMAPS 2.0 contact-map analysis. Reports each interface
                   as a residue-residue contact map with atomic
                   interaction-type classification (H-bond, salt bridge,
                   pi-pi, ...). The output is JSON-compatible with the PISA
                   schema and additionally carries the contact-map fields.

Usage:
    python -m fastpisa.cli /path/to/structure.pdb --pdb_id 6nxr --output_dir ./out
    python -m fastpisa.cli /path/to/structure.pdb --mode cocomaps --pdb_id 6nxr -o ./out
    python -m fastpisa.cli /path/to/file.cif --mode pisa --no-water -o ./out
"""

import argparse
import json
import os
import sys
import time


def _version() -> str:
    """Return the installed fastPISA version via importlib.metadata."""
    try:
        from importlib.metadata import version
        return version("fastpisa")
    except Exception:  # pragma: no cover - not installed / editable edge cases
        return "0.0.0+unknown"


def main():
    parser = argparse.ArgumentParser(
        description="fastPISA: PISA + COCOMAPS interface analysis (local)",
        prog="fastpisa",
    )
    parser.add_argument("input", help="Path to PDB or mmCIF file")

    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {_version()}",
        help="Show version and exit",
    )

    # Analysis mode & core parameters
    parser.add_argument(
        "--mode", choices=["combined", "pisa", "cocomaps"], default="combined",
        help="Analysis mode: 'combined' (default; PISA energetics + COCOMAPS "
             "contact maps in one report), 'pisa', or 'cocomaps'",
    )
    parser.add_argument(
        "--pdb_id", default="unknown",
        help="PDB identifier (default: 'unknown')",
    )
    parser.add_argument(
        "--assembly_id", default="1", help="Assembly ID (default: '1')",
    )
    parser.add_argument(
        "--probe_radius", type=float, default=1.4,
        help="Probe radius for ASA calculation (default: 1.4 A)",
    )
    parser.add_argument(
        "--point_density", type=int, default=480,
        help="Number of points on probe sphere (default: 480)",
    )
    parser.add_argument(
        "--interface_cutoff", type=float, default=5.0,
        help="Interface atom cutoff distance (default: 5.0 A)",
    )
    parser.add_argument(
        "--no-water", dest="exclude_water", action="store_true", default=True,
        help="Exclude ordered water from interface search (default: True)",
    )
    parser.add_argument(
        "--with-water", dest="exclude_water", action="store_false",
        help="Include ordered water in interface search",
    )
    parser.add_argument(
        "--min_css", type=float, default=0.0,
        help="Only keep interfaces with CSS >= this significance score "
             "(default 0.0 = keep all). Use e.g. 0.5 to drop weak/"
             "crystal-packing artifacts.",
    )

    # AlphaFold confidence filtering (item 4.4)
    parser.add_argument(
        "--pae", metavar="JSON", default=None,
        help="Path to AlphaFold '*_predicted_aligned_error.json'; enables "
             "confidence filtering by PAE / ipTM.",
    )
    parser.add_argument(
        "--min-pae", type=float, default=5.0,
        help="With --pae: keep only interfaces whose mean inter-residue PAE "
             "is <= this (A). Lower = more confident (default 5.0).",
    )
    parser.add_argument(
        "--min-iptm", type=float, default=None,
        help="With --pae: drop all interfaces when the model ipTM is below "
             "this (e.g. 0.8). Default: no ipTM cut.",
    )
    parser.add_argument(
        "--min-plddt", type=float, default=None,
        help="Keep only interfaces whose mean per-residue pLDDT (read from the "
             "B-factor column) is >= this. Portable across predictors -- no "
             "JSON needed. E.g. 70.0.",
    )

    # Visualisation (item 4.3)
    parser.add_argument(
        "--pymol-script", metavar="PATH", default=None,
        help="Write a PyMOL .pml colouring the top interface's residues by "
             "buried surface area.",
    )
    parser.add_argument(
        "--heatmap", metavar="PATH", default=None,
        help="Save a matplotlib residue-residue contact heatmap (needs "
             "matplotlib; fastpisa[viz]).",
    )
    parser.add_argument(
        "--molstar", metavar="PATH", default=None,
        help="Write a self-contained Mol* HTML viewer for the top interface.",
    )
    parser.add_argument(
        "--hotspots", type=int, metavar="N", default=0,
        help="Print the top N hotspot interface residues (by buried area).",
    )

    # Output controls
    parser.add_argument(
        "-o", "--output_dir", "--output-dir", default=".",
        help="Output directory for JSON files",
    )
    parser.add_argument(
        "--json-summary", action="store_true",
        help="Print a compact JSON summary instead of the text summary",
    )
    parser.add_argument(
        "--time", dest="show_time", action="store_true",
        help="Report wall-clock analysis time",
    )

    # Compatibility flags (accepted, no-op or minimal)
    parser.add_argument(
        "--asis", action="store_true",
        help="Calculate interfaces only (accepted for compatibility)",
    )
    parser.add_argument(
        "--extended", action="store_true",
        help="Include extended -list data (accepted for compatibility)",
    )
    parser.add_argument(
        "--json_only", action="store_true",
        help="Only output JSON files (default; accepted for compatibility)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed progress information",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Use the class API for consistency between CLI and Python usage.
    from fastpisa.api import PISAInterfaceAnalyzer

    analyzer = PISAInterfaceAnalyzer(
        path=args.input,
        pdb_id=args.pdb_id,
        assembly_id=args.assembly_id,
        probe_radius=args.probe_radius,
        point_density=args.point_density,
        interface_cutoff=args.interface_cutoff,
        mode=args.mode,
        exclude_water=args.exclude_water,
        min_css=args.min_css,
    )

    t0 = time.monotonic()
    analyzer.analyze()
    t_wall = time.monotonic() - t0

    # Optional AlphaFold confidence filtering (mutates analyzer.interfaces).
    if args.pae:
        analyzer.load_pae(args.pae)
        kept = analyzer.filter_by_pae(max_pae=args.min_pae)
        if args.min_iptm is not None:
            analyzer.filter_by_iptm(min_iptm=args.min_iptm)
        if analyzer.interfaces:
            print(f"PAE filter: kept {len(analyzer.interfaces)}/{len(kept) or 1} interface(s) "
                  f"with mean PAE <= {args.min_pae:.1f} A")

    # Portable B-factor / pLDDT confidence filter (no JSON required).
    if args.min_plddt is not None:
        analyzer.load_plddt()
        analyzer.filter_by_plddt(min_plddt=args.min_plddt)
        print(f"pLDDT filter: kept {analyzer.n_interfaces()} interface(s) with "
              f"mean pLDDT >= {args.min_plddt:.1f}")

    os.makedirs(args.output_dir, exist_ok=True)
    written = analyzer.write_json(args.output_dir)

    for label, path in written.items():
        print(f"Written: {path}")

    # Visualisation outputs (item 4.3)
    if analyzer.interfaces:
        if args.pymol_script:
            print(f"Written: {analyzer.write_pymol_script(args.pymol_script)}")
        if args.molstar:
            print(f"Written: {analyzer.write_molstar_html(args.molstar)}")
        if args.heatmap:
            print(f"Written: {analyzer.plot_contact_heatmap(1, out_path=args.heatmap)}")

    n_iface = analyzer.n_interfaces()
    asm = analyzer.assembly_json["assembly"]

    if args.json_summary:
        out = {
            "mode": args.mode,
            "pdb_id": args.pdb_id,
            "interfaces": n_iface,
            "accessible_surface_area": asm["accessible_surface_area"],
            "buried_surface_area": asm["buried_surface_area"],
            "dissociation_energy": asm["dissociation_energy"],
            "interface_count": n_iface,
        }
        if args.pae and analyzer.pae_data is not None:
            out["pae_filtered_interfaces"] = n_iface
        if args.show_time:
            out["wall_seconds"] = round(t_wall, 3)
        print(json.dumps(out, indent=2))
        return

    print("\n=== Summary ===")
    print(f"Mode: {args.mode}")
    print(f"Interfaces found: {n_iface}")
    print(f"Assembly dissociation energy: {asm['dissociation_energy']}")
    print(f"Total ASA: {asm['accessible_surface_area']}")
    print(f"Total BSA: {asm['buried_surface_area']}")

    if args.hotspots and analyzer.interfaces:
        print(f"\nTop {args.hotspots} hotspot residues (by buried area):")
        for hs in analyzer.hot_spot_residues(top_n=args.hotspots):
            print(f"  {hs['chain']}{hs['seq']} "
                  f"({hs['residue']}) BSA={hs['bsa']:.1f} A^2  interfaces={hs['interfaces']}")

    if args.mode in ("cocomaps", "combined"):
        for iface in analyzer.interfaces:
            iid = iface.interface_id
            cm = iface.cocomaps or {}
            pop = cm.get("interaction_population", {})
            print(f"\n  Interface {iid}: {cm.get('num_residue_pairs', 0)} residue pairs")
            if pop:
                print(f"    Interaction population: {pop}")

    if args.show_time:
        print(f"\nWall-clock analysis time: {t_wall:.3f} s")


if __name__ == "__main__":
    main()