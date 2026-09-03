"""
PDB file parser for fastPISA.

Reads ATOM and HETATM records from a PDB file and constructs
a list of atoms with chain, residue, and element information.
Supports standard PDB columns (columns are 1-indexed in PDB format).
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Atom:
    """A single atom in a PDB structure."""
    atom_name: str
    altloc: str
    res_name: str
    chain_id: str
    res_seq: int
    icode: str
    x: float
    y: float
    z: float
    occupancy: float
    bfactor: float
    element: str
    # Derived
    residue_label_asym_id: str = ""
    label_asym_id: str = ""
    label_seq_id: int = 0
    label_comp_id: str = ""
    auth_asym_id: str = ""
    auth_seq_id: int = 0
    group: str = "ATOM"  # "ATOM" or "HETATM"

    @property
    def name_clean(self) -> str:
        """Atom name with trailing spaces removed and digits stripped if needed."""
        return self.atom_name.strip()

    def distance_sq(self, other: "Atom") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return dx * dx + dy * dy + dz * dz

    def distance(self, other: "Atom") -> float:
        return self.distance_sq(other) ** 0.5


@dataclass
class Chain:
    """A polypeptide/polynucleotide chain."""
    auth_asym_id: str
    label_asym_id: str
    label_comp_id: str = ""
    chain_id: str = ""
    group: str = "polymer"  # "polymer" or "ligand"
    atoms: List[Atom] = field(default_factory=list)

    @property
    def residues(self) -> List:
        """Return unique residues (by seq id + icode)."""
        seen = {}
        for atom in self.atoms:
            key = (atom.res_seq, atom.icode)
            if key not in seen:
                seen[key] = {
                    "auth_seq_id": atom.auth_seq_id,
                    "label_seq_id": atom.label_seq_id,
                    "res_name": atom.res_name,
                    "label_comp_id": atom.label_comp_id,
                    "group": atom.group,
                    "seq_num": atom.res_seq,
                    "ins_code": atom.icode,
                }
        return list(seen.values())

    def get_ligands(self):
        """Return ligand residue info for this chain."""
        return [r for r in self.residues if r["group"] == "HETATM"]


@dataclass
class PDBStructure:
    """A parsed PDB structure."""
    chains: List[Chain] = field(default_factory=list)
    header: dict = field(default_factory=dict)
    space_group: str = ""
    crystal_info: dict = field(default_factory=dict)
    source: str = ""

    @property
    def atoms(self) -> List[Atom]:
        result = []
        for chain in self.chains:
            result.extend(chain.atoms)
        return result

    def get_chain(self, auth_asym_id: str) -> Optional[Chain]:
        for chain in self.chains:
            if chain.auth_asym_id == auth_asym_id:
                return chain
        return None


def _parse_cryst1_record(line: str) -> dict:
    """Parse a CRYST1 record."""
    info = {}
    try:
        info["a"] = float(line[6:11])
        info["b"] = float(line[11:16])
        info["c"] = float(line[16:21])
        info["alpha"] = float(line[21:26])
        info["beta"] = float(line[26:31])
        info["gamma"] = float(line[31:36])
        info["space_group"] = line[36:66].strip()
        info["z"] = int(line[66:70])
    except (ValueError, IndexError):
        pass
    return info


_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}
_NUCLEIC_ACIDS = {
    "A", "G", "C", "T", "U", "RA", "RG", "RC", "RT", "RU",
    "DA", "DG", "DC", "DT", "DU",
}
_POLYMER_RESIDUES = _AMINO_ACIDS | _NUCLEIC_ACIDS


def _is_polymer_residue(res_name: str) -> bool:
    """Check if a residue is a standard amino acid or nucleic acid."""
    return res_name.upper() in _POLYMER_RESIDUES


def is_mmcif_path(path) -> bool:
    """Return whether *path* has a supported mmCIF filename suffix."""
    return str(path).lower().endswith((
        ".cif", ".cif.gz", ".mmcif", ".mmcif.gz",
    ))


def _site_key(atom: Atom) -> tuple:
    """Identity of one atom site before alternate-conformer selection."""
    return (
        atom.auth_asym_id,
        atom.label_asym_id,
        atom.auth_seq_id,
        atom.label_seq_id,
        atom.icode,
        atom.res_name,
        atom.atom_name.strip(),
    )


def _prefer_altloc(candidate: Atom, current: Atom) -> bool:
    """Whether *candidate* wins the deterministic alternate-location rule."""
    candidate_blank = candidate.altloc == " "
    current_blank = current.altloc == " "
    if candidate_blank != current_blank:
        return candidate_blank
    if candidate.occupancy != current.occupancy:
        return candidate.occupancy > current.occupancy
    return candidate.altloc < current.altloc


def _chains_from_atoms(atoms: List[Atom]) -> List[Chain]:
    """Build parser chains after alternate conformers have been resolved."""
    chains_by_id = {}
    for atom in atoms:
        chain_type = "polymer" if _is_polymer_residue(atom.res_name) else "ligand"
        if atom.chain_id not in chains_by_id:
            chains_by_id[atom.chain_id] = Chain(
                auth_asym_id=atom.auth_asym_id,
                label_asym_id=atom.label_asym_id,
                chain_id=atom.chain_id,
                group=chain_type,
            )
        elif chain_type == "ligand":
            chains_by_id[atom.chain_id].group = "ligand"
        chains_by_id[atom.chain_id].atoms.append(atom)
    for chain in chains_by_id.values():
        chain.atoms.sort(key=lambda a: (a.res_seq, a.icode, a.atom_name))
    return list(chains_by_id.values())


def parse_pdb(path: str) -> PDBStructure:
    """Parse a PDB file and return a PDBStructure.

    Parameters
    ----------
    path : str
        Path to the PDB file.

    Returns
    -------
    PDBStructure
    """
    structure = PDBStructure()
    selected_atoms = {}
    seen_model = False

    if str(path).endswith(".gz"):
        import gzip
        _open = lambda p: gzip.open(p, "rt")  # noqa: E731
    else:
        _open = lambda p: open(p, "r")  # noqa: E731

    with _open(path) as f:
        for line_number, line in enumerate(f, start=1):
            rec = line[:6].strip()
            if rec == "MODEL":
                if seen_model:
                    break
                seen_model = True
                continue
            if rec == "ENDMDL" and seen_model:
                break
            if rec not in ("ATOM", "HETATM"):
                if rec == "HEADER":
                    structure.source = line[10:70].strip()
                elif rec == "CRYST1":
                    structure.crystal_info = _parse_cryst1_record(line)
                    structure.space_group = structure.crystal_info.get("space_group", "")
                elif rec == "COMPND":
                    pass
                elif rec == "SOURCE":
                    pass
                continue

            # Parse ATOM/HETATM record (PDB v3.3 format)
            atom_name = line[12:16].strip()
            altloc = line[16].strip() if line[16].strip() else " "
            res_name = line[17:20].strip()
            chain_id = line[21].strip()
            # int() rather than isdigit(): negative residue numbers ("  -4",
            # common for expression tags and DNA numbered about a centre)
            # are valid and must not collapse onto residue 0.
            try:
                res_seq = int(line[22:26])
            except ValueError:
                res_seq = 0
            icode = line[26].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            occupancy = float(line[54:60]) if line[54:60].strip() else 1.0
            bfactor = float(line[60:66]) if line[60:66].strip() else 0.0

            # Element from PDB columns 77-78 (authoritative in PDB format).
            # Guessing from the atom name confuses alpha carbon (" CA ") with
            # calcium and silently changes radii and solvation parameters.
            el = line[76:78].strip().upper()
            if not el:
                raise ValueError(
                    f"Invalid PDB atom record at line {line_number}: element "
                    "columns 77-78 are blank"
                )

            group = rec

            atom = Atom(
                atom_name=atom_name,
                altloc=altloc,
                res_name=res_name,
                chain_id=chain_id,
                res_seq=res_seq,
                icode=icode,
                x=x,
                y=y,
                z=z,
                occupancy=occupancy,
                bfactor=bfactor,
                element=el,
                label_asym_id=chain_id,
                label_seq_id=res_seq,
                label_comp_id=res_name,
                auth_asym_id=chain_id,
                auth_seq_id=res_seq,
                group=group,
            )
            key = _site_key(atom)
            current = selected_atoms.get(key)
            if current is None or _prefer_altloc(atom, current):
                selected_atoms[key] = atom

    structure.chains = _chains_from_atoms(list(selected_atoms.values()))
    return structure


def parse_mmcif(path: str) -> PDBStructure:
    """Parse an mmCIF file using gemmi.

    Supports both experimental and predicted (AlphaFold) mmCIF files.
    """
    try:
        import gemmi
    except ImportError:
        raise ImportError(
            "gemmi is required for mmCIF parsing. Install with: pip install gemmi"
        )

    structure = PDBStructure()
    doc = gemmi.cif.read(str(path))   # gemmi rejects pathlib.Path
    block = doc.sole_block()

    # Header
    try:
        _id = block.find_values("_entry.id")
        if _id:
            structure.source = str(_id[0])
    except Exception:
        pass

    # Crystal info
    try:
        sg = block.find_values("_exptl.space_group_type")
        structure.space_group = " ".join(str(x) for x in sg) if sg else ""
    except Exception:
        structure.space_group = ""
    try:
        if block.find_values("_cell.length_a"):
            structure.crystal_info = {
                "a": float(block.find_values("_cell.length_a")[0]),
                "b": float(block.find_values("_cell.length_b")[0]),
                "c": float(block.find_values("_cell.length_c")[0]),
                "alpha": float(block.find_values("_cell.angle_alpha")[0]),
                "beta": float(block.find_values("_cell.angle_beta")[0]),
                "gamma": float(block.find_values("_cell.angle_gamma")[0]),
                "space_group": structure.space_group,
                "z": int(block.find_values("_cell.Z_pdbx")[0]) if block.find_values("_cell.Z_pdbx") else 1,
            }
    except Exception:
        pass

    # The atom_site table retains label/auth identifiers and model/altloc
    # fields that Gemmi's high-level Structure view may normalize away.
    structure.chains = _parse_mmcif_atom_site_manual(block)
    return structure


def _parse_mmcif_atom_site_manual(block) -> list:
    """Manually parse the _atom_site loop from an mmCIF block (no gemmi read_structure).

    Returns a list of Chain objects.
    """
    tags = [
        "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id",
        "label_comp_id", "label_asym_id", "label_seq_id", "Cartn_x",
        "Cartn_y", "Cartn_z", "occupancy", "B_iso_or_equiv", "auth_seq_id",
        "auth_comp_id", "auth_asym_id", "auth_atom_id", "pdbx_PDB_ins_code",
        "pdbx_PDB_model_num",
    ]
    columns = {
        tag: [str(value) for value in block.find_values(f"_atom_site.{tag}")]
        for tag in tags
    }
    n_rows = len(columns["Cartn_x"])
    if not n_rows:
        return []

    def value(tag, row, default=""):
        column = columns[tag]
        raw = column[row] if row < len(column) else default
        return default if raw in ("", ".", "?") else raw

    def integer(tag, row, default=0):
        try:
            return int(value(tag, row, str(default)))
        except ValueError:
            return default

    def number(tag, row, default=0.0):
        try:
            return float(value(tag, row, str(default)))
        except ValueError:
            return default

    first_model = value("pdbx_PDB_model_num", 0, "1")
    selected_atoms = {}
    for row in range(n_rows):
        if value("pdbx_PDB_model_num", row, first_model) != first_model:
            continue
        group = value("group_PDB", row)
        if group not in ("ATOM", "HETATM"):
            continue
        label_chain = value("label_asym_id", row)
        auth_chain = value("auth_asym_id", row, label_chain)
        label_res = value("label_comp_id", row).upper()
        auth_res = value("auth_comp_id", row, label_res).upper()
        auth_seq = integer("auth_seq_id", row)
        label_seq = integer("label_seq_id", row, auth_seq)
        atom_name = value("auth_atom_id", row, value("label_atom_id", row))
        altloc = value("label_alt_id", row, " ")
        element = value("type_symbol", row).upper()
        if not element:
            raise ValueError(f"Invalid mmCIF atom_site row {row + 1}: type_symbol is blank")
        atom = Atom(
            atom_name=atom_name,
            altloc=altloc,
            res_name=auth_res,
            chain_id=auth_chain,
            res_seq=auth_seq,
            icode=value("pdbx_PDB_ins_code", row),
            x=number("Cartn_x", row),
            y=number("Cartn_y", row),
            z=number("Cartn_z", row),
            occupancy=number("occupancy", row, 1.0),
            bfactor=number("B_iso_or_equiv", row),
            element=element,
            residue_label_asym_id=label_chain,
            label_asym_id=label_chain,
            label_seq_id=label_seq,
            label_comp_id=label_res,
            auth_asym_id=auth_chain,
            auth_seq_id=auth_seq,
            group=group,
        )
        key = _site_key(atom)
        current = selected_atoms.get(key)
        if current is None or _prefer_altloc(atom, current):
            selected_atoms[key] = atom
    return _chains_from_atoms(list(selected_atoms.values()))
