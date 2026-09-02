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
from fastpisa.core import analyze as _core_analyze, MODES


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
        Analysis mode: ``"combined"`` (default; PISA energetics AND the
        COCOMAPS contact map on every interface), ``"pisa"``, or
        ``"cocomaps"``. All modes find identical interfaces (single shared
        core).
    exclude_water : bool
        Exclude ordered water (HOH etc.) from the interface search
        (default True).
    min_css : float
        Minimum CSS score for an interface to be kept (significance filter).
        ``0.0`` (default) keeps every detected interface.
    ligand_mode : str
        ``"separate"`` (default; classic PISA -- each bound hetero group is
        its own monomer) or ``"merge"`` (a chain's bound ligands/cofactors
        belong to that chain's molecule, the jsPISA-on-assembly convention).

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
        mode: str = "combined",
        exclude_water: bool = True,
        min_css: float = 0.0,
        ligand_mode: str = "separate",
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
        self.ligand_mode = ligand_mode

        # Populated by analyze()
        self.interfaces: List[Interface] = []
        self.result: Dict[str, Any] = {}
        self._interfaces_json: dict = {}
        self._assembly_json: dict = {}

        # AlphaFold confidence (set via load_pae)
        self.pae_data = None          # fastpisa.pae.PAEData
        self._pae_map: dict = {}
        # Portable per-residue confidence (set via load_plddt; from B-factor).
        self._plddt_map: dict = {}

    # -- public API --------------------------------------------------------
    # -- pythonic conveniences --------------------------------------------
    def __iter__(self):
        return iter(self.interfaces)

    def __len__(self) -> int:
        return len(self.interfaces)

    def __getitem__(self, i):
        return self.interfaces[i]

    def __repr__(self) -> str:
        n = len(self.interfaces)
        head = (f"<fastPISA {self.pdb_id}: {n} interface{'s' if n != 1 else ''}"
                f" ({self.mode} mode)>")
        if not n:
            return head
        body = "\n".join("  " + repr(i) for i in self.interfaces[:12])
        more = f"\n  ... {n - 12} more" if n > 12 else ""
        return head + "\n" + body + more

    def interface_between(self, chain_a: str, chain_b: str):
        """The interface between two chains (PISA labels), or None."""
        want = {chain_a, chain_b}
        for i in self.interfaces:
            if set(i.chains) == want:
                return i
        return None

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
            ligand_mode=self.ligand_mode,
        )
        if self.mode not in MODES:
            raise ValueError(
                f"Unknown mode: {self.mode!r} (expected one of {MODES})")
        result = _core_analyze(mode=self.mode, **kwargs)

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

        ``recompute=False`` is used internally so this never re-runs the
        analysis (which would clobber any PAE/pLDDT/CSS filtering applied to
        :attr:`interfaces`) and does not do duplicate work.
        """
        self.analyze(recompute=False)
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

    # -- tabular / hot-spot access -----------------------------------------
    def to_dataframe(self):
        """Return a pandas DataFrame, one row per interface.

        Columns: interface_id, molecule ids, area, energies, p-value, CSS,
        residue/contact counts. Requires ``pandas`` (``pip install pandas`` or
        the ``fastpisa[dataframe]`` extra). If it is unavailable a clear
        :class:`ImportError` is raised.
        """
        import pandas as pd
        self.analyze()
        rows = []
        for i in self.interfaces:
            rows.append({
                "interface_id": i.interface_id,
                "molecule1_id": i.molecule1_id,
                "molecule2_id": i.molecule2_id,
                "interface_area": i.interface_area,
                "solvation_energy": i.solvation_energy,
                "solvation_energy_apolar": i.solvation_energy_apolar,
                "solvation_energy_polar": i.solvation_energy_polar,
                "stabilization_energy": i.stabilization_energy,
                "p_value": i.p_value,
                "css": i.css,
                "n_interface_residues": i.number_interface_residues,
                "n_hydrogen_bonds": i.number_hydrogen_bonds,
                "n_salt_bridges": i.number_salt_bridges,
                "n_disulfide_bonds": i.number_disulfide_bonds,
                "n_other_bonds": i.number_other_bonds,
                "n_atom_contacts": len(i.contacts),
            })
        return pd.DataFrame(rows)

    def to_residue_dataframe(self):
        """Return a pandas DataFrame, one row per interface residue.

        Columns: interface_id, molecule, chain, residue, seq, asa, bsa,
        solvation_energy. Requires ``pandas`` (see :meth:`to_dataframe`).
        """
        import pandas as pd
        self.analyze()
        rows = []
        for i in self.interfaces:
            for imol, mol in enumerate(i.molecules, start=1):
                comp = mol.get("residue_label_comp_ids") or []
                seq = mol.get("residue_seq_ids") or []
                sa = mol.get("accessible_surface_areas") or []
                ba = mol.get("buried_surface_areas") or []
                se = mol.get("solvation_energies") or []
                chain = mol.get("auth_asym_id", "")
                for k in range(len(comp)):
                    rows.append({
                        "interface_id": i.interface_id,
                        "molecule": imol,
                        "chain": chain,
                        "residue": comp[k],
                        "seq": seq[k] if k < len(seq) else None,
                        "asa": sa[k] if k < len(sa) else None,
                        "bsa": ba[k] if k < len(ba) else None,
                        "solvation_energy": se[k] if k < len(se) else None,
                    })
        return pd.DataFrame(rows)

    def hot_spot_residues(self, top_n: int = 10, by: str = "bsa") -> List[dict]:
        """Return the top-N interface residues ranked by buried surface area.

        ``bsa`` residues buried most across all interfaces; ``solv`` ranks the
        most negative (most stabilising) per-residue solvation energy. Each
        entry is ``{chain, seq, residue, bsa, solvation_energy, interfaces}``.
        Values are summed over every interface the residue participates in.
        """
        self.analyze()
        by = by.lower()
        if by not in ("bsa", "solv"):
            raise ValueError(f"by must be 'bsa' or 'solv', got {by!r}")
        res: Dict[tuple, dict] = {}
        for iface in self.interfaces:
            for _, mol in enumerate(iface.molecules, start=1):
                comp = mol.get("residue_label_comp_ids") or []
                seq = mol.get("residue_seq_ids") or []
                ba = mol.get("buried_surface_areas") or []
                se = mol.get("solvation_energies") or []
                chain = mol.get("auth_asym_id", "")
                for k in range(len(comp)):
                    key = (chain, str(seq[k]) if k < len(seq) else "?", comp[k])
                    e = res.setdefault(key, {"bsa": 0.0, "solv": 0.0, "faces": set()})
                    if k < len(ba) and ba[k] is not None:
                        e["bsa"] += ba[k]
                    if k < len(se) and se[k] is not None:
                        e["solv"] += se[k]
                    e["faces"].add(iface.interface_id)
        if by == "solv":
            ranked = sorted(res.items(), key=lambda kv: kv[1]["solv"])
        else:
            ranked = sorted(res.items(), key=lambda kv: kv[1]["bsa"], reverse=True)
        return [
            {
                "chain": key[0], "seq": key[1], "residue": key[2],
                "bsa": round(v["bsa"], 2),
                "solvation_energy": round(v["solv"], 4),
                "interfaces": sorted(v["faces"]),
            }
            for key, v in ranked[:top_n]
        ]

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

    def _parsed_structure(self):
        """Return the parsed structure, parsed and cached once."""
        if not hasattr(self, "_structure_cache"):
            from fastpisa.parser.pdb_parser import parse_pdb, parse_mmcif
            if str(self.path).endswith((".cif", ".cif.gz")):
                self._structure_cache = parse_mmcif(self.path)
            else:
                self._structure_cache = parse_pdb(self.path)
        return self._structure_cache

    def _parsed_atoms(self) -> list:
        """Expose the parsed atoms (single shared parse, cached)."""
        return self._parsed_structure().atoms

    # -- AlphaFold confidence (PAE / ipTM) --------------------------------
    def load_pae(self, json_path) -> "PISAInterfaceAnalyzer":
        """Load an AlphaFold ``*_predicted_aligned_error.json`` for confidence
        filtering. Returns ``self`` for chaining."""
        from fastpisa.pae import load_pae, build_pae_index_map
        self.pae_data = load_pae(json_path)
        self._pae_map = build_pae_index_map(self._parsed_structure())
        return self

    def pae_scores(self) -> Dict[int, Optional[float]]:
        """Mean PAE (A) per interface over its contacting residue pairs.

        Lower = more confidently predicted. Requires :meth:`load_pae` first.
        """
        from fastpisa.pae import interface_pae_score
        if self.pae_data is None or not self.pae_data.has_pae:
            raise ValueError("load_pae(...) must be called before pae_scores")
        self.analyze()
        atoms = self._parsed_atoms()
        return {
            i.interface_id: interface_pae_score(i, atoms, self._pae_map, self.pae_data)
            for i in self.interfaces
        }

    def filter_by_pae(self, max_pae: float = 5.0) -> List[Interface]:
        """Keep only interfaces whose mean PAE is <= ``max_pae``.

        Requires :meth:`load_pae`. Mutates and returns :attr:`interfaces`.
        """
        from fastpisa.pae import interface_pae_score
        if self.pae_data is None or not self.pae_data.has_pae:
            raise ValueError("load_pae(...) must be called before filter_by_pae")
        self.analyze()
        atoms = self._parsed_atoms()
        kept = []
        for i in self.interfaces:
            score = interface_pae_score(i, atoms, self._pae_map, self.pae_data)
            if score is None or score <= max_pae:
                kept.append(i)
        self.interfaces = kept
        return kept

    def filter_by_iptm(self, min_iptm: float = 0.8) -> List[Interface]:
        """Drop all interfaces if the model's ipTM is below ``min_iptm``.

        Per DeepMind convention, ipTM < 0.8 indicates an unreliable model.
        Requires :meth:`load_pae`. Mutates and returns :attr:`interfaces`.
        """
        from fastpisa.pae import interface_pae_score  # noqa: F401 (import consistency)
        if self.pae_data is None:
            raise ValueError("load_pae(...) must be called before filter_by_iptm")
        self.analyze()
        if self.pae_data.iptm is not None and self.pae_data.iptm < min_iptm:
            self.interfaces = []
        return self.interfaces

    # -- portable confidence from the B-factor / pLDDT column -------------
    def load_plddt(self) -> "PISAInterfaceAnalyzer":
        """Read per-residue confidence from the B-factor column (pLDDT).

        AlphaFold / ColabFold / Protenix store pLDDT (0-100) in B-factors, so
        this works for any predictor with no extra JSON -- unlike ``load_pae``,
        whose ``*_predicted_aligned_error.json`` only some pipelines emit.
        Raises ValueError if the model carries no meaningful B-factors.
        """
        from fastpisa.pae import build_plddt_map
        self._plddt_map = build_plddt_map(self._parsed_structure())
        if not self._plddt_map:
            raise ValueError("no residues with a B-factor to read")
        vals = list(self._plddt_map.values())
        if max(vals) == min(vals):
            raise ValueError(
                "B-factors are constant (e.g. all zero) -- this model carries no "
                "pLDDT-like confidence to filter on"
            )
        return self

    @property
    def has_plddt(self) -> bool:
        return bool(self._plddt_map)

    def model_plddt(self) -> Optional[float]:
        """Overall mean pLDDT (a global confidence proxy), or None."""
        from fastpisa.pae import model_plddt
        return model_plddt(self._plddt_map)

    def plddt_scores(self) -> Dict[int, Optional[float]]:
        """Mean interface pLDDT per interface. Requires :meth:`load_plddt`."""
        from fastpisa.pae import interface_plddt
        if not self._plddt_map:
            raise ValueError("load_plddt(...) must be called before plddt_scores")
        self.analyze()
        atoms = self._parsed_atoms()
        return {
            i.interface_id: interface_plddt(i, atoms, self._plddt_map)
            for i in self.interfaces
        }

    def filter_by_plddt(self, min_plddt: float = 70.0) -> List[Interface]:
        """Keep only interfaces whose mean per-residue pLDDT >= ``min_plddt``.

        Uses the B-factor column (portable across predictors). Requires
        :meth:`load_plddt`. Mutates and returns :attr:`interfaces`.
        """
        from fastpisa.pae import interface_plddt
        if not self._plddt_map:
            raise ValueError("load_plddt(...) must be called before filter_by_plddt")
        self.analyze()
        atoms = self._parsed_atoms()
        kept = []
        for i in self.interfaces:
            score = interface_plddt(i, atoms, self._plddt_map)
            if score is None or score >= min_plddt:
                kept.append(i)
        self.interfaces = kept
        return kept

    def weight_energies_by_confidence(self, pae_threshold: float = 10.0, plddt_threshold: float = 50.0) -> None:
        """Weight physical interaction energies and CSS based on prediction confidence.

        For low-confidence interfaces (high PAE or low pLDDT), stabilization and
        solvation energies, as well as the CSS score, are scaled down. This avoids
        overestimating the stability of domains that are not confidently predicted.
        """
        from fastpisa.pae import interface_pae_score, interface_plddt
        self.analyze()
        atoms = self._parsed_atoms()

        for i in self.interfaces:
            weight = 1.0

            # Prioritize PAE if available (lower is better, typically 0-30 A)
            if self.pae_data is not None and self.pae_data.has_pae:
                pae = interface_pae_score(i, atoms, self._pae_map, self.pae_data)
                if pae is not None:
                    # e.g., if pae is > threshold, weight drops linearly to 0.1 at threshold*2
                    if pae > pae_threshold:
                        weight = max(0.1, 1.0 - ((pae - pae_threshold) / pae_threshold))
            
            # Fallback to pLDDT if available (higher is better, typically 0-100)
            elif self._plddt_map:
                plddt = interface_plddt(i, atoms, self._plddt_map)
                if plddt is not None:
                    # e.g., if plddt < threshold, weight drops linearly
                    if plddt < plddt_threshold:
                        weight = max(0.1, plddt / plddt_threshold)

            if weight < 1.0:
                i.stabilization_energy = round(i.stabilization_energy * weight, 2)
                i.solvation_energy = round(i.solvation_energy * weight, 2)
                i.solvation_energy_apolar = round(i.solvation_energy_apolar * weight, 2)
                i.solvation_energy_polar = round(i.solvation_energy_polar * weight, 2)
                i.css = round(i.css * weight, 3)

    # -- visualisation helpers --------------------------------------------
    def write_pymol_script(self, out_path: str, by: str = "bsa") -> str:
        """Write a PyMOL ``.pml`` colouring the first interface's residues by
        BSA. See :func:`fastpisa.viz.write_pymol_script`."""
        from fastpisa.viz import write_pymol_script
        self.analyze()
        return write_pymol_script(str(self.path), self.interfaces[0], out_path, by=by)

    def write_molstar_html(self, out_path: str) -> str:
        """Write a self-contained Mol* HTML viewer for the first interface."""
        from fastpisa.viz import write_molstar_html
        self.analyze()
        return write_molstar_html(str(self.path), self.interfaces[0], out_path)

    def plot_contact_heatmap(self, interface_id: int = 1, out_path: Optional[str] = None,
                             **kwargs):
        """Plot a residue-residue contact heatmap for an interface (needs
        matplotlib; ``fastpisa[viz]``). See
        :func:`fastpisa.viz.plot_contact_heatmap`."""
        from fastpisa.viz import plot_contact_heatmap
        self.analyze()
        iface = self.get_interface(interface_id)
        return plot_contact_heatmap(iface, self._parsed_atoms(), out_path=out_path, **kwargs)


def analyze_interface(
    path: str,
    pdb_id: str = "unknown",
    mode: str = "combined",
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

def analyze(path, pdb_id: str = None, **kwargs) -> "PISAInterfaceAnalyzer":
    """One-call analysis: ``fastpisa.analyze("x.pdb")`` -> analyzer with
    ``.interfaces`` populated (iterate it, index it, ``.to_dataframe()``,
    ``.write_json()``). ``kwargs`` are :class:`PISAInterfaceAnalyzer` options
    (``mode``, ``ligand_mode``, ``min_css``, ...)."""
    import os as _os
    if pdb_id is None:
        pdb_id = _os.path.basename(str(path)).split(".")[0]
    ana = PISAInterfaceAnalyzer(path, pdb_id=pdb_id, **kwargs)
    ana.analyze()
    return ana
