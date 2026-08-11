#!/usr/bin/env python3
"""
fastPISA CLI: Local reproduction of PISA with COCOMAPS mode.

Two complementary modes analyse a PDB/mmCIF structure:

  --mode pisa      (default) PISA thermodynamic/surface analysis, output in the
                   PDBe PISA JSON schema ('assembly' + 'interfaces').

  --mode cocomaps  COCOMAPS 2.0 contact-map analysis. Identifies the same
                   interfaces as PISA (same 5 A atom cutoff and surface
                   machinery) but reports each interface as a residue-residue
                   contact map with atomic interaction-type classification
                   (H-bond, salt bridge, pi-pi, ...). The output is
                   JSON-compatible with the PISA schema and additionally
                   carries the contact-map fields.

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


def main():
    parser = argparse.ArgumentParser(
        description="fastPISA: PISA + COCOMAPS interface analysis (local)",
        prog="fastpisa",
    )
    parser.add_argument("input", help="Path to PDB or mmCIF file")

    # Analysis mode & core parameters
    parser.add_argument(
        "--mode", choices=["pisa", "cocomaps"], default="pisa",
        help="Analysis mode: 'pisa' (default) or 'cocomaps'",
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

    os.makedirs(args.output_dir, exist_ok=True)
    written = analyzer.write_json(args.output_dir)

    for label, path in written.items():
        print(f"Written: {path}")

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

    if args.mode == "cocomaps":
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