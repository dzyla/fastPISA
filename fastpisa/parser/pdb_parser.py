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
    chains_by_id: dict = {}

    if str(path).endswith(".gz"):
        import gzip
        _open = lambda p: gzip.open(p, "rt")  # noqa: E731
    else:
        _open = lambda p: open(p, "r")  # noqa: E731

    with _open(path) as f:
        for line in f:
            rec = line[:6].strip()
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

            # Element from PDB columns 77-78 (authoritative in PDB format)
            # Fallback to deriving from atom name if not present
            el = line[76:78].strip().upper()
            if not el:
                el = atom_name[0:2].strip()
                if len(el) == 2 and not el[1].isalpha():
                    el = el[0]
                el = el.upper()
                if not el or not el[0].isalpha():
                    el = atom_name[0].upper() if atom_name else "C"

            group = rec

            # Determine chain type
            is_poly = _is_polymer_residue(res_name)
            chain_type = "polymer" if is_poly else "ligand"

            if chain_id not in chains_by_id:
                chains_by_id[chain_id] = Chain(
                    auth_asym_id=chain_id,
                    label_asym_id=chain_id,
                    chain_id=chain_id,
                    group=chain_type,
                )

            chain = chains_by_id[chain_id]

            # For ligands, set the chain group to ligand if any residue is a ligand
            if chain_type == "ligand":
                chain.group = "ligand"

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
            chain.atoms.append(atom)

    # Sort atoms by residue
    for chain in chains_by_id.values():
        chain.atoms.sort(key=lambda a: (a.res_seq, a.icode, a.atom_name))

    structure.chains = list(chains_by_id.values())
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

    # Atom site table via gemmi structure parse (robust for predicted CIFs)
    try:
        st = gemmi.read_structure(path)
    except Exception:
        st = None

    chains_by_id: dict = {}
    if st is not None:
        for model in st:
            for chain in model:
                cid = chain.name
                if cid not in chains_by_id:
                    chains_by_id[cid] = Chain(
                        auth_asym_id=cid,
                        label_asym_id=cid,
                        chain_id=cid,
                        group="polymer" if chain.get_polymer() is not None else "ligand",
                    )
                chain_obj = chains_by_id[cid]
                for res in chain:
                    res_name = res.name.upper()
                    auth_seq = int(res.seqid.num) if res.seqid else 0
                    label_seq = int(res.seqid.num) if res.seqid else 0
                    for atom in res:
                        x, y, z = atom.pos[0], atom.pos[1], atom.pos[2]
                        element = atom.element.name.upper() if atom.element.name else "C"
                        atom_name = atom.name
                        chain_obj.atoms.append(Atom(
                            atom_name=atom_name,
                            altloc=" ",
                            res_name=res_name,
                            chain_id=cid,
                            res_seq=auth_seq,
                            icode="",
                            x=x, y=y, z=z,
                            occupancy=atom.occ if atom.occ else 1.0,
                            bfactor=atom.b_iso if atom.b_iso else 0.0,
                            element=element,
                            label_asym_id=cid,
                            label_seq_id=label_seq,
                            label_comp_id=res_name,
                            auth_asym_id=cid,
                            auth_seq_id=auth_seq,
                            group=atom.element.name,
                        ))
        if chains_by_id:
            structure.chains = list(chains_by_id.values())
            return structure

    # Fallback: manual mmCIF atom_site table parsing
    structure.chains = _parse_mmcif_atom_site_manual(block)
    return structure


def _parse_mmcif_atom_site_manual(block) -> list:
    """Manually parse the _atom_site loop from an mmCIF block (no gemmi read_structure).

    Returns a list of Chain objects.
    """
    tags = ["group_PDB", "auth_asym_id", "label_comp_id", "auth_seq_id",
            "Cartn_x", "Cartn_y", "Cartn_z", "type_symbol", "label_atom_id"]
    rows = block.find("_atom_site.", tags)  # gemmi.cif.Table
    idx = {t: i for i, t in enumerate(tags)}

    chains_by_id = {}
    for row in rows:
        group = str(row[idx["group_PDB"]])
        if group not in ("ATOM", "HETATM"):
            continue
        cid = str(row[idx["auth_asym_id"]])
        res_name = str(row[idx["label_comp_id"]]).upper()
        try:
            auth_seq = int(str(row[idx["auth_seq_id"]]))
        except ValueError:
            auth_seq = 0
        x = float(str(row[idx["Cartn_x"]]))
        y = float(str(row[idx["Cartn_y"]]))
        z = float(str(row[idx["Cartn_z"]]))
        el = str(row[idx["type_symbol"]]).upper() or "C"
        atom_name = str(row[idx["label_atom_id"]])

        if cid not in chains_by_id:
            chains_by_id[cid] = Chain(
                auth_asym_id=cid, label_asym_id=cid, chain_id=cid,
                group="polymer" if res_name in _POLYMER_RESIDUES else "ligand",
            )
        chain = chains_by_id[cid]
        chain.atoms.append(Atom(
            atom_name=atom_name, altloc=" ", res_name=res_name, chain_id=cid,
            res_seq=auth_seq, icode="", x=x, y=y, z=z, occupancy=1.0, bfactor=0.0,
            element=el, label_asym_id=cid, label_seq_id=auth_seq,
            label_comp_id=res_name, auth_asym_id=cid, auth_seq_id=auth_seq,
            group=group,
        ))
    return list(chains_by_id.values())