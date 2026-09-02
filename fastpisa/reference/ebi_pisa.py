"""Fetch and parse original-PISA reference data from the EBI PISA service.

The EBI PDBe PISA CGI (``https://www.ebi.ac.uk/pdbe/pisa/cgi-bin/``) exposes
the original PISA engine's results for every deposited PDB entry as XML:
per-interface ``int_area``, ``int_solv_en``, ``pvalue``, ``stab_en``, ``css``
and the full hydrogen-bond / salt-bridge / disulfide atom lists, plus
per-molecule and per-residue ASA/BSA/solvation values.

This module fetches that XML (cached under ``tests/data/reference/``) and
parses it into plain dicts used by the calibration and validation code.

PISA analyses the whole crystal, so most interfaces involve symmetry mates.
fastPISA (by scope decision) analyses only the deposited coordinates, so
comparisons use :func:`identity_interfaces`: interfaces whose two molecules
both carry the identity operation (``x,y,z`` / ``X,Y,Z``).
"""

from __future__ import annotations

import gzip
import os
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

EBI_PISA_URL = "https://www.ebi.ac.uk/pdbe/pisa/cgi-bin/interfaces.pisa?{pdbid}"
RCSB_PDB_URL = "https://files.rcsb.org/download/{pdbid}.pdb"

# Modern PDBe PISA JSON API (covers recent entries; the classic CGI above has
# a frozen database). It reports the analysis of a BIOLOGICAL ASSEMBLY, so
# fastPISA must be run on the matching assembly coordinate file.
PDBE_PISA_JSON_URL = "https://www.ebi.ac.uk/pdbe/api/pisa/interfaces/{pdbid}/{assembly}"
RCSB_ASSEMBLY_URL = "https://files.rcsb.org/download/{pdbid}-assembly{assembly}.cif.gz"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REFERENCE_DIR = os.path.join(_REPO_ROOT, "tests", "data", "reference")


def _f(el, tag, default=None):
    txt = el.findtext(tag)
    if txt is None or txt.strip() == "":
        return default
    try:
        return float(txt)
    except ValueError:
        return default


def _s(el, tag):
    txt = el.findtext(tag)
    return txt.strip() if txt else ""


def fetch_pisa_xml(pdb_id: str, cache_dir: str = REFERENCE_DIR,
                   timeout: int = 120) -> str:
    """Return the EBI PISA interfaces XML for ``pdb_id`` (cached, gzipped)."""
    pdb_id = pdb_id.lower()
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{pdb_id}.pisa.xml.gz")
    if os.path.exists(path):
        with gzip.open(path, "rt") as fh:
            return fh.read()
    url = EBI_PISA_URL.format(pdbid=pdb_id)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    if "<pisa_interfaces>" not in text:
        raise RuntimeError(f"EBI PISA returned no interface XML for {pdb_id}")
    root = ET.fromstring(text)
    status = root.findtext("status")
    if status and status.strip().lower() != "ok":
        raise RuntimeError(f"EBI PISA status {status!r} for {pdb_id}")
    entry_status = root.findtext("pdb_entry/status")
    if entry_status and entry_status.strip().lower() != "ok":
        # The classic CGI's database is frozen; recent entries come back as
        # "Entry not found". Do NOT cache that -- use the PDBe JSON API
        # (fetch_pisa_assembly_json) for such entries instead.
        raise RuntimeError(
            f"EBI PISA (classic CGI) has no data for {pdb_id}: {entry_status}")
    with gzip.open(path, "wt") as fh:
        fh.write(text)
    return text


def fetch_pdb_file(pdb_id: str, cache_dir: str = REFERENCE_DIR,
                   timeout: int = 120) -> str:
    """Download the PDB file for ``pdb_id`` from RCSB (cached, gzipped).

    Returns the path to the cached ``.pdb.gz``.
    """
    pdb_id = pdb_id.lower()
    pdb_dir = os.path.join(cache_dir, "pdb")
    os.makedirs(pdb_dir, exist_ok=True)
    path = os.path.join(pdb_dir, f"{pdb_id}.pdb.gz")
    if os.path.exists(path):
        return path
    url = RCSB_PDB_URL.format(pdbid=pdb_id.upper())
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = resp.read()
    with gzip.open(path, "wb") as fh:
        fh.write(data)
    return path


def _parse_bonds(iface_el, tag: str) -> List[dict]:
    """Parse one bond list (``h-bonds``, ``salt-bridges``, ``ss-bonds``...)."""
    parent = iface_el.find(tag)
    bonds = []
    if parent is None:
        return bonds
    for b in parent.findall("bond"):
        bonds.append({
            "chain1": _s(b, "chain-1"), "res1": _s(b, "res-1"),
            "seqnum1": _s(b, "seqnum-1"), "inscode1": _s(b, "inscode-1"),
            "atname1": _s(b, "atname-1"),
            "chain2": _s(b, "chain-2"), "res2": _s(b, "res-2"),
            "seqnum2": _s(b, "seqnum-2"), "inscode2": _s(b, "inscode-2"),
            "atname2": _s(b, "atname-2"),
            "dist": _f(b, "dist"),
        })
    return bonds


def _parse_molecule(mol_el, include_residues: bool) -> dict:
    mol = {
        "id": _s(mol_el, "id"),
        "chain_id": _s(mol_el, "chain_id"),
        "class": _s(mol_el, "class"),
        "symop": _s(mol_el, "symop"),
        "symop_no": _s(mol_el, "symop_no"),
        "int_natoms": int(_f(mol_el, "int_natoms", 0) or 0),
        "int_nres": int(_f(mol_el, "int_nres", 0) or 0),
        "int_area": _f(mol_el, "int_area"),
        "int_solv_en": _f(mol_el, "int_solv_en"),
        "pvalue": _f(mol_el, "pvalue"),
    }
    if include_residues:
        residues = []
        res_parent = mol_el.find("residues")
        if res_parent is not None:
            for r in res_parent.findall("residue"):
                bsa = _f(r, "bsa", 0.0)
                if not bsa:
                    continue  # keep only residues actually buried
                residues.append({
                    "name": _s(r, "name"),
                    "seq_num": _s(r, "seq_num"),
                    "ins_code": _s(r, "ins_code"),
                    "asa": _f(r, "asa"),
                    "bsa": bsa,
                    "solv_en": _f(r, "solv_en"),
                })
        mol["residues"] = residues
    return mol


def parse_pisa_xml(text: str, include_residues: bool = False) -> List[dict]:
    """Parse EBI PISA interfaces XML into a list of interface dicts."""
    root = ET.fromstring(text)
    out = []
    for iface in root.iter("interface"):
        out.append({
            "id": int(_f(iface, "id", 0) or 0),
            "type": _s(iface, "type"),
            "n_occ": int(_f(iface, "n_occ", 1) or 1),
            "int_area": _f(iface, "int_area"),
            "int_solv_en": _f(iface, "int_solv_en"),
            "pvalue": _f(iface, "pvalue"),
            "stab_en": _f(iface, "stab_en"),
            "css": _f(iface, "css"),
            "h_bonds": _parse_bonds(iface, "h-bonds"),
            "salt_bridges": _parse_bonds(iface, "salt-bridges"),
            "ss_bonds": _parse_bonds(iface, "ss-bonds"),
            "cov_bonds": _parse_bonds(iface, "cov-bonds"),
            "molecules": [
                _parse_molecule(m, include_residues)
                for m in iface.findall("molecule")
            ],
        })
    return out


def _is_identity(symop: str) -> bool:
    return symop.replace(" ", "").lower() == "x,y,z"


def identity_interfaces(interfaces: List[dict]) -> List[dict]:
    """Interfaces where both molecules carry the identity symmetry operation.

    These are the interfaces present in the deposited coordinates -- the ones
    fastPISA computes (crystal-symmetry mates are out of fastPISA's scope).
    """
    return [
        i for i in interfaces
        if len(i["molecules"]) == 2
        and all(_is_identity(m["symop"]) for m in i["molecules"])
    ]


def load_cached_reference(pdb_id: str, cache_dir: str = REFERENCE_DIR,
                          include_residues: bool = False) -> Optional[List[dict]]:
    """Load a cached reference (no network). Returns None if not cached."""
    path = os.path.join(cache_dir, f"{pdb_id.lower()}.pisa.xml.gz")
    if not os.path.exists(path):
        return None
    with gzip.open(path, "rt") as fh:
        return parse_pisa_xml(fh.read(), include_residues=include_residues)


def cached_pdb_path(pdb_id: str, cache_dir: str = REFERENCE_DIR) -> Optional[str]:
    """Path to the cached PDB file (``.pdb.gz``), or None if not cached."""
    path = os.path.join(cache_dir, "pdb", f"{pdb_id.lower()}.pdb.gz")
    return path if os.path.exists(path) else None


# ---------------------------------------------------------------------------
# Modern PDBe PISA JSON API (assembly-based; works for recent entries)
# ---------------------------------------------------------------------------
def fetch_pisa_assembly_json(pdb_id: str, assembly: str = "1",
                             cache_dir: str = REFERENCE_DIR,
                             timeout: int = 120) -> dict:
    """PDBe PISA JSON for one biological assembly (cached, gzipped).

    Returns the inner document: ``{"assembly_id", "pisa_version",
    "assembly": {..., "interfaces": [...]}}``.
    """
    import json

    pdb_id = pdb_id.lower()
    jdir = os.path.join(cache_dir, "json")
    os.makedirs(jdir, exist_ok=True)
    path = os.path.join(jdir, f"{pdb_id}-a{assembly}.pisa.json.gz")
    if os.path.exists(path):
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    url = PDBE_PISA_JSON_URL.format(pdbid=pdb_id, assembly=assembly)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    doc = data.get(pdb_id)
    if not doc or "assembly" not in doc:
        raise RuntimeError(f"PDBe PISA JSON has no assembly doc for {pdb_id}")
    with gzip.open(path, "wt") as fh:
        json.dump(doc, fh)
    return doc


def fetch_assembly_cif(pdb_id: str, assembly: str = "1",
                       cache_dir: str = REFERENCE_DIR,
                       timeout: int = 300) -> str:
    """Download the biological-assembly mmCIF from RCSB (cached).

    These files can be large, so the cache directory
    (``tests/data/reference/assemblies``) is gitignored -- refetch on demand.
    """
    pdb_id = pdb_id.lower()
    adir = os.path.join(cache_dir, "assemblies")
    os.makedirs(adir, exist_ok=True)
    path = os.path.join(adir, f"{pdb_id}-assembly{assembly}.cif.gz")
    if os.path.exists(path):
        return path
    url = RCSB_ASSEMBLY_URL.format(pdbid=pdb_id.upper(), assembly=assembly)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = resp.read()
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def normalize_json_interfaces(doc: dict) -> List[dict]:
    """PDBe PISA JSON interfaces -> the dict shape parse_pisa_xml produces."""
    out = []
    for i in doc["assembly"]["interfaces"]:
        out.append({
            "id": int(i["interface_id"]),
            "int_area": i.get("interface_area"),
            "int_solv_en": i.get("solvation_energy"),
            "stab_en": i.get("stabilization_energy"),
            "pvalue": i.get("p_value"),
            "css": i.get("css"),  # absent in the JSON API -> None
            "n_h_bonds": i.get("number_hydrogen_bonds", 0),
            "n_salt_bridges": i.get("number_salt_bridges", 0),
            "n_ss_bonds": i.get("number_disulfide_bonds", 0),
            "molecules": [
                {"chain_id": m.get("chain_id", ""), "class": m.get("molecule_class", ""),
                 "symop": "x,y,z"}
                for m in i.get("molecules", [])
            ],
        })
    return out
