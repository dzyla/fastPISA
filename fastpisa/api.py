"""
High-level Python API for fastPISA.

Provides a class-based interface so fastPISA can be used directly from Python
(introspection, interactive use, programmatic pipelines) without the CLI.

The central class is :class:`PISAInterfaceAnalyzer`. It parses a structure
once, runs the selected analysis mode, and returns structured results.

Examples
--------
>>> from fastpisa.api import PISAInterfaceAnalyzer
>>> ana = PISAInterfaceAnalyzer("/path/to/6nxr.pdb", pdb_id="6nxr")
>>> result = ana.analyze()                    # PISA mode (default)
>>> interfaces = ana.interfaces               # list of Interface objects
>>> for iface in interfaces:
...     print(iface.interface_id, iface.interface_area, iface.number_interface_residues)

>>> ana.mode = "cocomaps"
>>> result = ana.analyze()
>>> for iface in ana.interfaces:
...     cm = iface.cocomaps["contact_map"]    # residue-residue contact map
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastpisa.interface.contacts import Interface
from fastpisa.pipeline import analyze_structure
from fastpisa.cocomaps.pipeline import analyze_structure_cocomaps


class PISAInterfaceAnalyzer:
    """Analyze a biomolecular structure for interfaces (PISA or COCOMAPS mode).

    Parameters
    ----------
    path : str | os.PathLike
        Path to a PDB or mmCIF structure file.
    pdb_id : str
        PDB identifier used in output filenames (default "unknown").
    assembly_id : str
        Assembly identifier (default "1").
    probe_radius : float
        Probe sphere radius for the ASA calculation (default 1.4 A).
    point_density : int
        Number of points on the probe sphere (default 480).
    interface_cutoff : float
        Distance cutoff for interface-atom detection (default 5.0 A).
    mode : str
        Analysis mode: ``"pisa"`` (default) or ``"cocomaps"``.
    exclude_water : bool
        Exclude ordered water (HOH etc.) from the interface search
        (default True).
    min_css : float
        Minimum CSS score for an interface to be kept (significance filter).
        ``0.0`` (default) keeps every detected interface.

    Attributes
    ----------
    structure : PDBStructure
        The parsed structure.
    interfaces : list[Interface]
        Populated after :meth:`analyze` — a list of :class:`Interface` objects.
    result : dict
        Populated after :meth:`analyze` — the raw ``interfaces`` + ``assembly``
        JSON documents.
    """

    def __init__(
        self,
        path: str,
        pdb_id: str = "unknown",
        assembly_id: str = "1",
        probe_radius: float = 1.4,
        point_density: int = 480,
        interface_cutoff: float = 5.0,
        mode: str = "pisa",
        exclude_water: bool = True,
        min_css: float = 0.0,
    ):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Structure file not found: {self.path}")
        self.pdb_id = pdb_id
        self.assembly_id = assembly_id
        self.probe_radius = probe_radius
        self.point_density = point_density
        self.interface_cutoff = interface_cutoff
        self.mode = mode
        self.exclude_water = exclude_water
        self.min_css = min_css

        # Populated by analyze()
        self.interfaces: List[Interface] = []
        self.result: Dict[str, Any] = {}
        self._interfaces_json: dict = {}
        self._assembly_json: dict = {}

    # -- public API --------------------------------------------------------
    def analyze(self, recompute: bool = True) -> dict:
        """Run the analysis and populate :attr:`interfaces` and :attr:`result`.

        Returns the raw result dict ``{"interfaces": ..., "assembly": ...}``.

        Parameters
        ----------
        recompute : bool
            If True (default), re-run analysis each call. If False, return the
            cached result if present.
        """
        if not recompute and self.result:
            return self.result

        kwargs = dict(
            input_file=str(self.path),
            pdb_id=self.pdb_id,
            assembly_id=self.assembly_id,
            probe_radius=self.probe_radius,
            point_density=self.point_density,
            interface_cutoff=self.interface_cutoff,
            exclude_water=self.exclude_water,
            min_css=self.min_css,
        )
        if self.mode == "cocomaps":
            result = analyze_structure_cocomaps(**kwargs)
        elif self.mode == "pisa":
            result = analyze_structure(**kwargs)
        else:
            raise ValueError(f"Unknown mode: {self.mode!r} (expected 'pisa' or 'cocomaps')")

        self.result = result
        self.interfaces = result.get("interfaces_obj", [])
        self._interfaces_json = result["interfaces"]
        self._assembly_json = result["assembly"]
        return result

    # -- accessors ---------------------------------------------------------
    @property
    def interfaces_json(self) -> dict:
        """The full interfaces.json document (as a dict)."""
        if not self._interfaces_json:
            self.analyze()
        return self._interfaces_json

    @property
    def assembly_json(self) -> dict:
        """The full assembly.json document (as a dict)."""
        if not self._assembly_json:
            self.analyze()
        return self._assembly_json

    def get_interface(self, interface_id: int) -> Interface:
        """Return the :class:`Interface` with the given ID."""
        for iface in self.interfaces:
            if iface.interface_id == interface_id:
                return iface
        raise KeyError(f"No interface with id {interface_id}")

    def n_interfaces(self) -> int:
        """Number of detected interfaces (must call :meth:`analyze` first)."""
        return len(self.interfaces)

    # -- output ------------------------------------------------------------
    def write_json(self, output_dir: str = ".") -> Dict[str, str]:
        """Write the two JSON documents to ``output_dir``.

        Returns a dict mapping ``"interfaces"`` / ``"assembly"`` to the file
        paths written.
        """
        self.analyze()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = f"{self.pdb_id}-assembly{self.assembly_id}"

        import json

        interfaces_path = out_dir / f"{base}-interfaces.json"
        assembly_path = out_dir / f"{base}.json"
        with open(interfaces_path, "w") as f:
            json.dump(self._interfaces_json, f, indent=2)
        with open(assembly_path, "w") as f:
            json.dump(self._assembly_json, f, indent=2)
        return {"interfaces": str(interfaces_path), "assembly": str(assembly_path)}

    def summary(self) -> str:
        """Return a human-readable summary string of the analysis."""
        self.analyze()
        lines = [
            f"fastPISA ({self.mode} mode)",
            f"  pdb_id            : {self.pdb_id}",
            f"  interfaces        : {len(self.interfaces)}",
            f"  assembly ASA      : {self.assembly_json['assembly']['accessible_surface_area']:.1f} A^2",
            f"  assembly BSA      : {self.assembly_json['assembly']['buried_surface_area']:.1f} A^2",
            f"  dissociation E    : {self.assembly_json['assembly']['dissociation_energy']} kcal/mol",
        ]
        n_atoms = len(self._parsed_atoms())
        lines.insert(3, f"  atoms             : {n_atoms}")
        for iface in self.interfaces:
            lines.append(
                f"    if{iface.interface_id}: area={iface.interface_area:.1f} "
                f"residues={iface.number_interface_residues} "
                f"atom-pairs={len(iface.contacts)}"
            )
        return "\n".join(lines)

    def _parsed_atoms(self) -> list:
        """Expose the parsed atoms via a lightweight re-parse (cached)."""
        if not hasattr(self, "_structure_cache"):
            from fastpisa.parser.pdb_parser import parse_pdb, parse_mmcif
            if str(self.path).endswith((".cif", ".cif.gz")):
                self._structure_cache = parse_mmcif(self.path)
            else:
                self._structure_cache = parse_pdb(self.path)
        return self._structure_cache.atoms

    def __repr__(self) -> str:
        state = "unanalyzed"
        if self.interfaces:
            state = f"{len(self.interfaces)} interfaces"
        return f"<PISAInterfaceAnalyzer {self.path.name} [{self.mode}] {state}>"


def analyze_interface(
    path: str,
    pdb_id: str = "unknown",
    mode: str = "pisa",
    **kwargs,
) -> Dict[str, Any]:
    """One-shot function: analyze a structure and return the raw result.

    Convenience wrapper around :class:`PISAInterfaceAnalyzer` for scripts that
    only need the JSON documents (no object handling).

    Parameters
    ----------
    path : str
        Path to PDB/mmCIF file.
    pdb_id : str
        PDB identifier.
    mode : str
        ``"pisa"`` or ``"cocomaps"``.
    **kwargs
        Passed to :class:`PISAInterfaceAnalyzer` (probe_radius, interface_cutoff,
        exclude_water, ...).

    Returns
    -------
    dict
        ``{"interfaces": <interfaces.json>, "assembly": <assembly.json>,
        "interfaces_obj": [<Interface>, ...]}``
    """
    ana = PISAInterfaceAnalyzer(path, pdb_id=pdb_id, mode=mode, **kwargs)
    return ana.analyze()