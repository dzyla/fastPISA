"""Manuscript-ready digests of the interface between two groups of chains.

The question a paper asks is rarely "what is interface 7"; it is "how much
surface does the antibody bury on the antigen, with how many hydrogen bonds,
and which residues". That is a *group* question: the Fab's heavy and light
chains both touch the antigen, so the answer is the union of several
chain-pair interfaces. :func:`group_interface` builds that union and states
every convention it uses.

    import fastpisa
    from fastpisa.report import group_interface

    res = fastpisa.analyze("complex.pdb")
    gi = group_interface(res, ["A"], ["H", "L"], "antigen", "Fab")
    gi.buried_side1, gi.n_hbonds, gi.results_paragraph()

Conventions (PISA's, so the numbers are comparable with the literature):

* **interface area** = half the total surface buried on both sides,
  summed over the contributing chain pairs (what PISA prints);
* **buried surface, per side** = the sum of each side's buried area -- the
  "buries N A^2 on the antigen" number; the two sides add up to
  2 x interface area;
* energies are sums over pairs (they are additive by construction);
  P-values and CSS are not, and are reported per pair only;
* bond counts follow PISA's rules (independent predicates: a charged pair
  that is also an H-bond counts in both tables); COCOMAPS contact classes
  are counted over atom pairs within 5 A.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E",
    "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
    "DA": "A", "DG": "G", "DC": "C", "DT": "T", "DU": "U",
    "A": "A", "G": "G", "C": "C", "U": "U",
}

_BACKBONE = {"N", "CA", "C", "O", "OXT"}


def one_letter(res_name: str) -> str:
    return THREE_TO_ONE.get(res_name.strip().upper(), "X")


def _chain_of(chain_id: str) -> str:
    """Author chain of a PISA molecule label (``"[ZN]A:301"`` -> ``"A"``)."""
    if chain_id.startswith("["):
        return chain_id.split("]", 1)[1].split(":", 1)[0]
    return chain_id


# ---------------------------------------------------------------------------
# Chain inventory (for the UI and for sanity)
# ---------------------------------------------------------------------------
def chain_inventory(result) -> List[dict]:
    """One row per molecule fastPISA analysed: label, chain, class, size.

    ``label`` is the PISA-style molecule label (``A``, ``[HEM]A:601``) used
    everywhere else in this module for group membership.
    """
    import numpy as np
    from fastpisa.interface.contacts import (
        get_molecules, get_molecule_masks, filter_water_molecules)

    structure = result._parsed_structure()
    atoms = structure.atoms
    mols = filter_water_molecules(
        get_molecules(structure, merge_ligands=(getattr(result, "ligand_mode", "separate") == "merge")),
        exclude_water=True)
    masks = get_molecule_masks(atoms, mols)
    rows = []
    for m, mask in zip(mols, masks):
        idx = np.flatnonzero(mask)
        heavy = [i for i in idx if atoms[i].element.strip().upper() not in ("H", "D")]
        nres = len({(atoms[i].auth_asym_id, atoms[i].res_seq, atoms[i].icode) for i in heavy})
        rows.append({
            "label": m["chain_id"],
            "chain": _chain_of(m["chain_id"]),
            "class": m.get("molecule_class", ""),
            "n_residues": nres,
            "n_atoms": len(heavy),
        })
    return rows


# ---------------------------------------------------------------------------
# The digest
# ---------------------------------------------------------------------------
@dataclass
class ResidueEntry:
    side: int
    chain: str
    name: str
    seq: str
    icode: str
    asa: float
    bsa: float
    dg: float
    n_bonds: int = 0

    @property
    def one(self) -> str:
        return one_letter(self.name)

    @property
    def label(self) -> str:
        return f"{self.chain}:{self.one}{self.seq}{self.icode}"

    @property
    def fraction_buried(self) -> float:
        return self.bsa / self.asa if self.asa else 0.0


@dataclass
class GroupInterface:
    label1: str
    label2: str
    group1: List[str]
    group2: List[str]
    pairs: list = field(default_factory=list)          # Interface objects
    interface_area: float = 0.0
    buried_side1: float = 0.0
    buried_side2: float = 0.0
    dg_solv: float = 0.0
    dg_apolar: float = 0.0
    dg_polar: float = 0.0
    stab_energy: float = 0.0
    n_hbonds: int = 0
    n_salt_bridges: int = 0
    n_disulfides: int = 0
    n_residue_pairs: int = 0
    interaction_population: Dict[str, int] = field(default_factory=dict)
    residues_side1: List[ResidueEntry] = field(default_factory=list)
    residues_side2: List[ResidueEntry] = field(default_factory=list)

    # -- derived -----------------------------------------------------------
    @property
    def buried_total(self) -> float:
        """Total surface buried on both sides (= 2 x interface area)."""
        return self.buried_side1 + self.buried_side2

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def empty(self) -> bool:
        return not self.pairs

    def bonds(self, kinds: Sequence[str] = ("hbond", "salt_bridge", "disulfide")) -> list:
        """AtomContact objects of the given kinds, oriented side1 -> side2."""
        g1 = {_chain_of(c) for c in self.group1}
        out = []
        for p in self.pairs:
            for c in p.contacts:
                if c.bond_type not in kinds:
                    continue
                out.append(_orient(c, g1))
        return out

    # -- tables --------------------------------------------------------------
    def pair_table(self):
        import pandas as pd
        rows = []
        for p in self.pairs:
            rows.append({
                "chain pair": " + ".join(p.chains),
                "interface area (A^2)": p.interface_area,
                "buried side 1 (A^2)": _side_bsa(p, self.group1),
                "buried side 2 (A^2)": _side_bsa(p, self.group2),
                "dG solv (kcal/mol)": p.solvation_energy,
                "dG apolar": p.solvation_energy_apolar,
                "dG polar": p.solvation_energy_polar,
                "stab energy (kcal/mol)": p.stabilization_energy,
                "P-value": p.p_value,
                "CSS": p.css,
                "H-bonds": p.number_hydrogen_bonds,
                "salt bridges": p.number_salt_bridges,
                "disulfides": p.number_disulfide_bonds,
                "residue pairs": len(p.contact_map),
            })
        return pd.DataFrame(rows)

    def bonds_table(self, one_letter_codes: bool = True):
        import pandas as pd
        rows = []
        for c in self.bonds():
            r1 = one_letter(c.atom1_residue) if one_letter_codes else c.atom1_residue
            r2 = one_letter(c.atom2_residue) if one_letter_codes else c.atom2_residue
            rows.append({
                self.label1: f"{c.atom1_chain}:{r1}{c.atom1_seq}{c.atom1_icode}",
                f"{self.label1} atom": c.atom1_name.strip(),
                self.label2: f"{c.atom2_chain}:{r2}{c.atom2_seq}{c.atom2_icode}",
                f"{self.label2} atom": c.atom2_name.strip(),
                "type": {"hbond": "hydrogen bond", "salt_bridge": "salt bridge",
                         "disulfide": "disulfide"}.get(c.bond_type, c.bond_type),
                "moiety": f"{_moiety(c.atom1_name)}-{_moiety(c.atom2_name)}",
                "distance (A)": round(c.distance, 2),
                "chain 1": c.atom1_chain, "seq 1": c.atom1_seq,
                "chain 2": c.atom2_chain, "seq 2": c.atom2_seq,
            })
        df = pd.DataFrame(rows)
        if len(df):
            df = df.sort_values(["type", "chain 1", "seq 1", "chain 2", "seq 2"]).reset_index(drop=True)
        return df

    def residue_table(self, side: int):
        import pandas as pd
        res = self.residues_side1 if side == 1 else self.residues_side2
        return pd.DataFrame([{
            "chain": r.chain, "residue": r.name, "code": r.one, "seq": r.seq,
            "icode": r.icode, "ASA isolated (A^2)": r.asa, "BSA (A^2)": r.bsa,
            "fraction buried": round(r.fraction_buried, 2),
            "dG solv (kcal/mol)": r.dg, "bonds": r.n_bonds,
        } for r in res])

    def contact_map_table(self):
        import pandas as pd
        rows = []
        g1 = {_chain_of(c) for c in self.group1}
        for p in self.pairs:
            for e in p.contact_map:
                flip = e["residue_1_chain"] not in g1
                a = ("2" if flip else "1"), ("1" if flip else "2")
                rows.append({
                    self.label1: f"{e[f'residue_{a[0]}_chain']}:{one_letter(e[f'residue_{a[0]}_type'])}{e[f'residue_{a[0]}_seq']}",
                    self.label2: f"{e[f'residue_{a[1]}_chain']}:{one_letter(e[f'residue_{a[1]}_type'])}{e[f'residue_{a[1]}_seq']}",
                    "min distance (A)": round(e["min_distance"], 2),
                    "atom contacts": e["num_contacts"],
                    "dominant": e.get("dominant_interaction"),
                    **{k: v for k, v in e.get("interaction_counts", {}).items()},
                })
        return pd.DataFrame(rows)

    def residue_string(self, side: int, sep: str = ", ") -> str:
        res = self.residues_side1 if side == 1 else self.residues_side2
        return sep.join(f"{r.one}{r.seq}{r.icode}" + (f"({r.chain})" if _multi_chain(res) else "")
                        for r in res)

    # -- prose ---------------------------------------------------------------
    def results_sentence(self) -> str:
        if self.empty:
            return f"No interface was detected between {self.label1} and {self.label2}."
        n1, n2 = len(self.residues_side1), len(self.residues_side2)
        bonds = []
        if self.n_hbonds:
            bonds.append(f"{self.n_hbonds} hydrogen bond{'s' if self.n_hbonds != 1 else ''}")
        if self.n_salt_bridges:
            bonds.append(f"{self.n_salt_bridges} salt bridge{'s' if self.n_salt_bridges != 1 else ''}")
        if self.n_disulfides:
            bonds.append(f"{self.n_disulfides} disulfide{'s' if self.n_disulfides != 1 else ''}")
        bond_txt = (" and is stabilised by " + _join(bonds)) if bonds else ""
        return (f"The {self.label1}-{self.label2} interface buries a total of "
                f"{self.buried_total:,.0f} A^2 of solvent-accessible surface "
                f"({self.buried_side1:,.0f} A^2 on {self.label1} and {self.buried_side2:,.0f} A^2 "
                f"on {self.label2}; interface area {self.interface_area:,.0f} A^2), "
                f"involves {n1} {self.label1} and {n2} {self.label2} residues, "
                f"has a solvation free-energy gain of {self.dg_solv:.1f} kcal/mol "
                f"({self.dg_apolar:.1f} kcal/mol from apolar burial){bond_txt}.")

    def results_paragraph(self) -> str:
        s = self.results_sentence()
        if self.empty:
            return s
        hot1 = _hot(self.residues_side1)
        hot2 = _hot(self.residues_side2)
        extra = (f" The largest contributions to the buried surface come from "
                 f"{hot1} on {self.label1} and {hot2} on {self.label2}.")
        pop = self.interaction_population
        apolar = pop.get("apolar_vdw", 0) + pop.get("ch_pi", 0) + pop.get("pi_pi", 0)
        if apolar:
            extra += (f" The contact map contains {self.n_residue_pairs} residue pairs, "
                      f"including {pop.get('pi_pi', 0)} pi-pi, {pop.get('cation_pi', 0)} cation-pi "
                      f"and {pop.get('apolar_vdw', 0)} apolar van der Waals atom contacts.")
        if self.n_pairs > 1:
            parts = [f"{' + '.join(p.chains)} ({p.interface_area:,.0f} A^2)" for p in self.pairs]
            extra += f" It comprises {self.n_pairs} chain-pair interfaces: {_join(parts)}."
        return s + extra

    @staticmethod
    def methods_paragraph() -> str:
        from fastpisa import __version__
        return (
            f"Interfaces were analysed with fastPISA {__version__}, a Python "
            "re-implementation of the PISA algorithm (Krissinel & Henrick, J. Mol. "
            "Biol. 2007) calibrated against the PDBe PISA service. Solvent-accessible "
            "surface areas were computed with a 1.4 A probe and NACCESS/Chothia atomic "
            "radii on heavy atoms; an interface is defined by the surface buried on "
            "association, its area being half the total buried on both molecules. "
            "The solvation free-energy gain is the sum of per-atom solvation parameters "
            "times buried area; the stabilisation energy adds PISA's per-bond terms for "
            "hydrogen bonds, salt bridges and disulfides. Hydrogen bonds were assigned "
            "on donor-acceptor distance (<= 3.89 A) and antecedent geometry, salt "
            "bridges between oppositely charged side-chain atoms within 4.0 A, following "
            "PISA. Residue-residue contact maps and interaction classes (pi-pi, "
            "cation-pi, CH-pi, van der Waals) follow COCOMAPS 2.0 (Chawla et al., "
            "Bioinformatics 2025) within a 5 A cutoff.")

    # -- viewer commands -------------------------------------------------------
    def _side_residues(self):
        by = {1: {}, 2: {}}
        for r in self.residues_side1:
            by[1].setdefault(r.chain, []).append(f"{r.seq}{r.icode}")
        for r in self.residues_side2:
            by[2].setdefault(r.chain, []).append(f"{r.seq}{r.icode}")
        return by

    def chimerax_command(self) -> str:
        by = self._side_residues()
        lines = []
        for side, color, name in ((1, "orange", "side1"), (2, "cornflowerblue", "side2")):
            sel = "".join(f"/{ch}:{','.join(seqs)}" for ch, seqs in sorted(by[side].items()))
            if sel:
                lines.append(f"name {name} {sel}")
                lines.append(f"color {name} {color}")
        if lines:
            lines.append("show side1 | side2 atoms")
            lines.append("style side1 | side2 stick")
        return "\n".join(lines) if lines else "# no interface residues"

    def pymol_command(self) -> str:
        by = self._side_residues()
        lines = []
        for side, color, name in ((1, "orange", "side1"), (2, "marine", "side2")):
            parts = [f"(chain {ch} and resi {'+'.join(seqs)})" for ch, seqs in sorted(by[side].items())]
            if parts:
                lines.append(f"select {name}, {' or '.join(parts)}")
                lines.append(f"color {color}, {name}")
        if lines:
            lines.append("show sticks, side1 or side2")
        return "\n".join(lines) if lines else "# no interface residues"

    def to_dict(self) -> dict:
        return {
            "label1": self.label1, "label2": self.label2,
            "group1": list(self.group1), "group2": list(self.group2),
            "pairs": [" + ".join(p.chains) for p in self.pairs],
            "interface_area": round(self.interface_area, 2),
            "buried_side1": round(self.buried_side1, 2),
            "buried_side2": round(self.buried_side2, 2),
            "buried_total": round(self.buried_total, 2),
            "dg_solv": round(self.dg_solv, 2), "dg_apolar": round(self.dg_apolar, 2),
            "dg_polar": round(self.dg_polar, 2), "stab_energy": round(self.stab_energy, 2),
            "n_hbonds": self.n_hbonds, "n_salt_bridges": self.n_salt_bridges,
            "n_disulfides": self.n_disulfides, "n_residue_pairs": self.n_residue_pairs,
            "interaction_population": dict(self.interaction_population),
            "residues_side1": [r.__dict__ for r in self.residues_side1],
            "residues_side2": [r.__dict__ for r in self.residues_side2],
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _join(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _multi_chain(res: Sequence[ResidueEntry]) -> bool:
    return len({r.chain for r in res}) > 1


def _hot(res: Sequence[ResidueEntry], n: int = 4) -> str:
    top = sorted(res, key=lambda r: -r.bsa)[:n]
    return _join([f"{r.name.title()}{r.seq}{r.icode}" + (f" ({r.chain})" if _multi_chain(res) else "")
                  for r in top]) if top else "no residues"


def _moiety(atom_name: str) -> str:
    return "backbone" if atom_name.strip().upper() in _BACKBONE else "side chain"


def _orient(c, group1_chains):
    """Copy of an AtomContact with atom 1 on side 1."""
    if c.atom1_chain in group1_chains:
        return c
    import copy
    d = copy.copy(c)
    for a, b in (("atom1_idx", "atom2_idx"), ("atom1_name", "atom2_name"),
                 ("atom1_residue", "atom2_residue"), ("atom1_chain", "atom2_chain"),
                 ("atom1_seq", "atom2_seq"), ("atom1_icode", "atom2_icode")):
        setattr(d, a, getattr(c, b))
        setattr(d, b, getattr(c, a))
    return d


def _side_bsa(iface, group) -> float:
    total = 0.0
    for m in iface.molecules:
        if m.get("chain_id") in group:
            total += float(sum(m.get("buried_surface_areas", [])))
    return total


def _molecule_residues(iface, side: int, label: str, bond_counts: Dict[tuple, int]) -> List[ResidueEntry]:
    out = []
    for m in iface.molecules:
        if m.get("chain_id") != label:
            continue
        chain = _chain_of(label)
        names = m.get("residue_label_comp_ids", [])
        seqs = m.get("residue_seq_ids", [])
        ics = m.get("residue_ins_codes", [])
        asa = m.get("accessible_surface_areas", [])
        bsa = m.get("buried_surface_areas", [])
        dg = m.get("solvation_energies", [])
        for j in range(len(names)):
            if not bsa[j]:
                continue
            ic = (ics[j] or "") if j < len(ics) else ""
            out.append(ResidueEntry(
                side=side, chain=chain, name=names[j], seq=str(seqs[j]), icode=ic,
                asa=float(asa[j]), bsa=float(bsa[j]), dg=float(dg[j]),
                n_bonds=bond_counts.get((chain, str(seqs[j]), ic), 0)))
    return out


def group_interface(result, group1: Iterable[str], group2: Iterable[str],
                    label1: str = "group 1", label2: str = "group 2") -> GroupInterface:
    """Digest of the interface between two groups of molecules.

    ``group1`` / ``group2`` hold fastPISA molecule labels as shown by
    :func:`chain_inventory` (``"A"``, ``"[HEM]A:601"``). Bare chain IDs match
    both the polymer chain and, in ``ligand_mode="merge"``, its cofactors.
    """
    g1, g2 = [str(c) for c in group1], [str(c) for c in group2]
    if set(g1) & set(g2):
        raise ValueError(f"chains in both groups: {sorted(set(g1) & set(g2))}")
    gi = GroupInterface(label1, label2, g1, g2)
    pop: Dict[str, int] = {}
    merged1: Dict[tuple, ResidueEntry] = {}
    merged2: Dict[tuple, ResidueEntry] = {}
    for iface in result.interfaces:
        c1, c2 = iface.chains
        if (c1 in g1 and c2 in g2) or (c1 in g2 and c2 in g1):
            gi.pairs.append(iface)
    for iface in gi.pairs:
        gi.interface_area += iface.interface_area
        gi.buried_side1 += _side_bsa(iface, g1)
        gi.buried_side2 += _side_bsa(iface, g2)
        gi.dg_solv += iface.solvation_energy
        gi.dg_apolar += iface.solvation_energy_apolar
        gi.dg_polar += iface.solvation_energy_polar
        gi.stab_energy += iface.stabilization_energy
        gi.n_hbonds += iface.number_hydrogen_bonds
        gi.n_salt_bridges += iface.number_salt_bridges
        gi.n_disulfides += iface.number_disulfide_bonds
        gi.n_residue_pairs += len(iface.contact_map)
        for k, v in iface.interaction_population.items():
            pop[k] = pop.get(k, 0) + int(v)
        bond_counts: Dict[tuple, int] = {}
        for c in iface.contacts:
            if c.bond_type in ("hbond", "salt_bridge", "disulfide"):
                for ch, seq, ic in ((c.atom1_chain, str(c.atom1_seq), c.atom1_icode or ""),
                                    (c.atom2_chain, str(c.atom2_seq), c.atom2_icode or "")):
                    bond_counts[(ch, seq, ic)] = bond_counts.get((ch, seq, ic), 0) + 1
        for label in iface.chains:
            side = 1 if label in g1 else 2
            store = merged1 if side == 1 else merged2
            for r in _molecule_residues(iface, side, label, bond_counts):
                key = (r.chain, r.seq, r.icode)
                if key in store:      # same residue buried against two partners
                    store[key].bsa += r.bsa
                    store[key].dg += r.dg
                    store[key].n_bonds += r.n_bonds
                else:
                    store[key] = r
    gi.interaction_population = pop

    def _sorted(d):
        return sorted(d.values(), key=lambda r: (r.chain, _int(r.seq), r.icode))
    gi.residues_side1 = _sorted(merged1)
    gi.residues_side2 = _sorted(merged2)
    for r in gi.residues_side1 + gi.residues_side2:
        r.bsa = round(r.bsa, 2)
        r.dg = round(r.dg, 3)
    for k in ("interface_area", "buried_side1", "buried_side2", "dg_solv",
              "dg_apolar", "dg_polar", "stab_energy"):
        setattr(gi, k, round(getattr(gi, k), 2))
    return gi


def _int(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Interpretation: what the numbers mean for THIS interface
# ---------------------------------------------------------------------------
def interpret(gi: "GroupInterface") -> List[dict]:
    """Automatic observations about a group interface.

    Each item is ``{"level": "info"|"note"|"warning", "text": ...}``. The
    thresholds are the usual rules of thumb from the PISA literature and
    the antibody-antigen structural literature, and are stated in the text
    so a reader can disagree with them.
    """
    out: List[dict] = []
    if gi.empty:
        return [{"level": "warning", "text": "No interface between the two groups: no surface is buried on association."}]
    area = gi.interface_area
    if area < 400:
        out.append({"level": "warning", "text":
                    f"Interface area {area:,.0f} A^2 is small. Crystal-packing contacts are typically "
                    "< 400-500 A^2; biologically relevant interfaces are usually > 600 A^2 (Krissinel & Henrick 2007)."})
    elif area < 700:
        out.append({"level": "note", "text":
                    f"Interface area {area:,.0f} A^2 is in the range where packing contacts and weak "
                    "biological interfaces overlap; use the P-value, conservation and biochemistry to decide."})
    else:
        out.append({"level": "info", "text":
                    f"Interface area {area:,.0f} A^2 ({gi.buried_total:,.0f} A^2 total buried) is typical of a "
                    "stable biological interface (antibody-antigen and most obligate dimers bury 1,400-2,000 A^2 in total)."})
    asym = abs(gi.buried_side1 - gi.buried_side2) / max(gi.buried_total, 1)
    if asym > 0.15:
        big = gi.label1 if gi.buried_side1 > gi.buried_side2 else gi.label2
        out.append({"level": "note", "text":
                    f"Burial is asymmetric: {big} buries {max(gi.buried_side1, gi.buried_side2):,.0f} A^2 vs "
                    f"{min(gi.buried_side1, gi.buried_side2):,.0f} A^2 -- one side wraps around or inserts into the other."})
    pvals = [p.p_value for p in gi.pairs if p.interface_area > 200]
    if pvals:
        pmin = min(pvals)
        if pmin < 0.3:
            out.append({"level": "info", "text":
                        f"Hydrophobicity P-value {pmin:.2f} (lowest over the pairs): the interface is markedly more "
                        "hydrophobic than a random surface patch, as expected for a specific interaction."})
        elif pmin > 0.6:
            out.append({"level": "note", "text":
                        f"Hydrophobicity P-value {pmin:.2f}: the buried surface is no more hydrophobic than random "
                        "surface. Polar / charged interfaces (many antibody epitopes, protein-DNA) score this way; "
                        "it does not by itself mean the interface is not biological."})
    if gi.dg_solv:
        frac = gi.dg_apolar / gi.dg_solv if gi.dg_solv < 0 else float("nan")
        if gi.dg_solv > 0:
            out.append({"level": "note", "text":
                        f"Solvation free-energy gain is positive ({gi.dg_solv:+.1f} kcal/mol): desolvating the polar / "
                        "charged atoms costs more than burying the apolar ones gains. The interface is then held by "
                        "hydrogen bonds, salt bridges and shape complementarity rather than the hydrophobic effect."})
        elif frac > 2:
            out.append({"level": "info", "text":
                        f"The hydrophobic effect dominates: apolar burial contributes {gi.dg_apolar:+.1f} kcal/mol "
                        f"against a polar desolvation cost of {gi.dg_polar:+.1f} kcal/mol."})
    n_res = len(gi.residues_side1) + len(gi.residues_side2)
    dens = (gi.n_hbonds + gi.n_salt_bridges) / max(area / 100, 1e-9)
    if gi.n_hbonds + gi.n_salt_bridges == 0:
        out.append({"level": "note", "text": "No hydrogen bonds or salt bridges: the interface is purely apolar / "
                                              "van der Waals, or polar groups are bridged by water not present in the model."})
    elif dens > 1.5:
        out.append({"level": "info", "text":
                    f"Polar-rich interface: {gi.n_hbonds} H-bonds and {gi.n_salt_bridges} salt bridges over "
                    f"{area:,.0f} A^2 ({dens:.1f} per 100 A^2; ~1 per 100 A^2 is typical)."})
    hot1 = [r for r in gi.residues_side1 if r.bsa >= 60]
    hot2 = [r for r in gi.residues_side2 if r.bsa >= 60]
    if hot1 or hot2:
        out.append({"level": "info", "text":
                    "Residues burying >= 60 A^2 (hot-spot candidates): "
                    + (", ".join(f"{r.name.title()}{r.seq}{r.icode} ({gi.label1}, {r.bsa:.0f})" for r in hot1) or "none on " + gi.label1)
                    + "; " + (", ".join(f"{r.name.title()}{r.seq}{r.icode} ({gi.label2}, {r.bsa:.0f})" for r in hot2) or "none on " + gi.label2)
                    + ". Buried area is a proxy; alanine scanning or conservation is needed to confirm."})
    arom = sum(r.bsa for r in gi.residues_side1 + gi.residues_side2 if r.name.upper() in ("TYR", "TRP", "PHE", "HIS"))
    if gi.buried_total and arom / gi.buried_total > 0.3:
        out.append({"level": "info", "text":
                    f"Aromatic residues account for {arom / gi.buried_total:.0%} of the buried surface -- common in "
                    "antibody paratopes (Tyr/Trp-rich CDRs) and in hot spots."})
    if n_res and gi.n_pairs > 1:
        out.append({"level": "info", "text":
                    f"The group interface spans {gi.n_pairs} chain pairs; the per-pair table shows how the burial "
                    "is distributed between them (e.g. heavy vs. light chain)."})
    return out


GUIDE = """
### What the numbers mean

**Interface area** (PISA convention) is *half* the total solvent-accessible
surface buried when the two sides associate: `(BSA_side1 + BSA_side2) / 2`.
Papers quote either this or the total; both are given here, say which one
you use. Buried area is computed on heavy atoms with a 1.4 A probe and the
NACCESS/Chothia radii PISA uses.

**Buried surface per side** is the area each partner loses. For an antibody,
"buries 850 A^2 on the antigen" is the epitope size; the paratope side is
usually similar but not identical.

**Solvation free-energy gain (dG_solv)** is the hydrophobic-effect estimate:
sum over buried atoms of an atomic solvation parameter times buried area.
Negative = favourable. It is **not a binding free energy** -- it omits
electrostatics, entropy and conformational change -- and should be used to
compare interfaces, not to predict K_d. The **apolar / polar split** shows
how much comes from burying carbon/sulfur (favourable) versus desolvating
polar and charged atoms (a cost).

**Stabilisation energy** = dG_solv + PISA's per-bond terms (-0.44 kcal/mol
per hydrogen bond, -0.15 per salt bridge, -4.0 per disulfide). Same caveat.

**P-value** (hydrophobicity): probability that a random patch of the same
size on the protein surface would be at least as hydrophobic. Low (< 0.3)
means the interface is unusually hydrophobic, i.e. interaction-specific;
high values are common for polar interfaces and are not a verdict on
biological relevance.

**CSS** (complexation significance score, 0-1) is PISA's heuristic for
"is this interface part of the stable assembly"; fastPISA's value is a
calibrated surrogate for ranking, not an exact reproduction.

**Hydrogen bonds** follow PISA: donor-acceptor distance <= 3.89 A on heavy
atoms with antecedent-angle checks and per-atom capacities. **Salt bridges**:
Lys/Arg/His N to Asp/Glu O within 4.0 A. A charged pair can count in both
tables, exactly as in PISA. **Contact classes** (pi-pi, cation-pi, CH-pi,
van der Waals) follow COCOMAPS 2.0 within 5 A.

### Typical values (for orientation, not thresholds)

| interface | total buried | residues / side | H-bonds |
|---|---|---|---|
| crystal-packing contact | < 800 A^2 | < 10 | 0-3 |
| antibody - protein antigen | 1,400 - 2,000 A^2 | 15 - 25 | 5 - 15 |
| enzyme - inhibitor (e.g. barnase-barstar) | ~1,600 A^2 | ~20 | 10 - 15 |
| obligate homodimer | 2,000 - 5,000 A^2 | 25 - 60 | 10 - 30 |

### Things to check before quoting a number

* **Missing residues / loops** in the model shrink the interface silently.
* **Alternate conformations**: the first altloc is used.
* **Hydrogens** are ignored for surfaces; explicit H improve H-bond geometry.
* **Ligands / cofactors**: in `separate` mode (default, classic PISA) a heme
  or glycan is its own molecule and is *not* part of "chain A"; use `merge`
  to count a chain's bound groups with it.
* **Waters** are excluded; water-mediated H-bonds are not counted.
* **Symmetry mates** are not generated: only contacts present in the file.
* **Numbering**: residue numbers are the author numbering in the file.

### Suggested wording

> The X-Y interface buries N A^2 of total solvent-accessible surface
> (a A^2 on X, b A^2 on Y; PISA interface area N/2 A^2) and is stabilised by
> h hydrogen bonds and s salt bridges (PISA criteria, computed with
> fastPISA). The epitope comprises residues ... of X; the paratope ...
"""


# ---------------------------------------------------------------------------
# Comparison of several complexes
# ---------------------------------------------------------------------------
def _nw_align(a: str, b: str, match: int = 2, mismatch: int = -1, gap: int = -2):
    """Needleman-Wunsch global alignment; returns list of (i, j) index pairs."""
    n, m = len(a), len(b)
    S = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        S[i][0] = i * gap
    for j in range(1, m + 1):
        S[0][j] = j * gap
    for i in range(1, n + 1):
        ai = a[i - 1]
        row, prev = S[i], S[i - 1]
        for j in range(1, m + 1):
            row[j] = max(prev[j - 1] + (match if ai == b[j - 1] else mismatch),
                         prev[j] + gap, row[j - 1] + gap)
    i, j, pairs = n, m, []
    while i > 0 and j > 0:
        if S[i][j] == S[i - 1][j - 1] + (match if a[i - 1] == b[j - 1] else mismatch):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif S[i][j] == S[i - 1][j] + gap:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def _chain_sequence(result, chain: str):
    """[(seq, icode, one_letter)] of a chain from the parsed structure."""
    st = result._parsed_structure()
    seen = {}
    for a in st.atoms:
        if a.auth_asym_id.strip() != chain or a.element.strip().upper() in ("H", "D"):
            continue
        key = (a.res_seq, a.icode or "")
        if key not in seen:
            seen[key] = one_letter(a.res_name)
    return [(str(k[0]), k[1], v) for k, v in sorted(seen.items(), key=lambda kv: (kv[0][0], kv[0][1]))]


@dataclass
class ComplexEntry:
    name: str
    gi: GroupInterface
    result: object = None       # the analyzer (for sequences); optional


@dataclass
class Comparison:
    entries: List[ComplexEntry]
    side: int = 1                       # the SHARED side residues are aligned on
    align: str = "auto"                 # "number" | "sequence" | "auto"
    _maps: Dict[str, Dict[tuple, str]] = field(default_factory=dict)

    @property
    def gis(self) -> List[GroupInterface]:
        return [e.gi for e in self.entries]

    @property
    def names(self) -> List[str]:
        return [e.name for e in self.entries]

    # -- residue keys shared across complexes ----------------------------------
    def _residue_key_map(self, e: ComplexEntry, side: int) -> Dict[tuple, str]:
        """Map (chain, seq, icode) of complex ``e`` -> common key string."""
        cache_key = f"{e.name}:{side}"
        if cache_key in self._maps:
            return self._maps[cache_key]
        ref = self.entries[0]
        res_e = e.gi.residues_side1 if side == 1 else e.gi.residues_side2
        mapping: Dict[tuple, str] = {}
        use_seq = self.align == "sequence"
        if self.align == "auto" and e is not ref and e.result is not None and ref.result is not None:
            # numbering agrees if the residue names match at the same numbers
            ref_res = ref.gi.residues_side1 if side == 1 else ref.gi.residues_side2
            ref_names = {(r.chain, r.seq, r.icode): r.one for r in ref_res}
            hits = [(r.chain, r.seq, r.icode) in ref_names and ref_names[(r.chain, r.seq, r.icode)] == r.one for r in res_e]
            if hits and sum(hits) / len(hits) < 0.6:
                use_seq = True
        if use_seq and e is not ref and e.result is not None and ref.result is not None:
            ref_chains = sorted({r.chain for r in (ref.gi.residues_side1 if side == 1 else ref.gi.residues_side2)})
            for ch_e in sorted({r.chain for r in res_e}):
                seq_e = _chain_sequence(e.result, ch_e)
                best = None
                for ch_r in ref_chains:
                    seq_r = _chain_sequence(ref.result, ch_r)
                    pairs = _nw_align("".join(x[2] for x in seq_e), "".join(x[2] for x in seq_r))
                    ident = sum(seq_e[i][2] == seq_r[j][2] for i, j in pairs) / max(len(pairs), 1)
                    if best is None or ident > best[0]:
                        best = (ident, ch_r, seq_r, pairs)
                if best is None:
                    continue
                _, ch_r, seq_r, pairs = best
                for i, j in pairs:
                    mapping[(ch_e, seq_e[i][0], seq_e[i][1])] = f"{ch_r}:{seq_r[j][2]}{seq_r[j][0]}{seq_r[j][1]}"
        for r in res_e:
            mapping.setdefault((r.chain, r.seq, r.icode), f"{r.chain}:{r.one}{r.seq}{r.icode}")
        self._maps[cache_key] = mapping
        return mapping

    def residue_matrix(self, side: Optional[int] = None):
        """DataFrame: rows = shared-side residues (common key), columns = complexes, values = BSA."""
        import pandas as pd
        side = side or self.side
        data: Dict[str, Dict[str, float]] = {}
        for e in self.entries:
            m = self._residue_key_map(e, side)
            col: Dict[str, float] = {}
            for r in (e.gi.residues_side1 if side == 1 else e.gi.residues_side2):
                k = m[(r.chain, r.seq, r.icode)]
                col[k] = col.get(k, 0.0) + r.bsa
            data[e.name] = col
        df = pd.DataFrame(data).fillna(0.0)
        if df.empty:
            return df
        df = df.loc[sorted(df.index, key=lambda s: (s.split(":")[0], _int("".join(ch for ch in s.split(":")[1][1:] if ch.isdigit()) or "0")))]
        return df

    def overlap_table(self, side: Optional[int] = None):
        """Pairwise shared / unique residue counts and Jaccard on the shared side."""
        import pandas as pd
        side = side or self.side
        sets = {}
        for e in self.entries:
            m = self._residue_key_map(e, side)
            sets[e.name] = {m[(r.chain, r.seq, r.icode)] for r in (e.gi.residues_side1 if side == 1 else e.gi.residues_side2)}
        rows = []
        names = self.names
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = sets[names[i]], sets[names[j]]
                inter, union = a & b, a | b
                rows.append({"complex A": names[i], "complex B": names[j],
                             "residues A": len(a), "residues B": len(b), "shared": len(inter),
                             "only A": len(a - b), "only B": len(b - a),
                             "Jaccard": round(len(inter) / len(union), 2) if union else 0.0,
                             "shared residues": ", ".join(sorted(inter, key=lambda s: _int("".join(c for c in s if c.isdigit()) or "0")))})
        return pd.DataFrame(rows)

    def summary_table(self):
        import pandas as pd
        rows = []
        for e in self.entries:
            g = e.gi
            rows.append({
                "complex": e.name, "side 1": g.label1, "side 2": g.label2,
                "chain pairs": g.n_pairs, "interface area (A^2)": g.interface_area,
                "buried side 1 (A^2)": g.buried_side1, "buried side 2 (A^2)": g.buried_side2,
                "total buried (A^2)": g.buried_total,
                "dG solv (kcal/mol)": g.dg_solv, "dG apolar": g.dg_apolar, "dG polar": g.dg_polar,
                "stab energy (kcal/mol)": g.stab_energy,
                "H-bonds": g.n_hbonds, "salt bridges": g.n_salt_bridges, "disulfides": g.n_disulfides,
                "residues side 1": len(g.residues_side1), "residues side 2": len(g.residues_side2),
                "residue pairs": g.n_residue_pairs,
            })
        return pd.DataFrame(rows)

    def prose(self) -> str:
        if len(self.entries) < 2:
            return ""
        ref = self.entries[0]
        out = []
        for e in self.entries[1:]:
            a, b = ref.gi, e.gi
            d_tot = b.buried_total - a.buried_total
            d1 = b.buried_side1 - a.buried_side1
            ov = self.overlap_table(self.side)
            row = ov[(ov["complex A"] == ref.name) & (ov["complex B"] == e.name)]
            jac = float(row["Jaccard"].iloc[0]) if len(row) else 0.0
            shared = int(row["shared"].iloc[0]) if len(row) else 0
            side_label = a.label1 if self.side == 1 else a.label2
            out.append(
                f"Relative to {ref.name}, {e.name} buries {abs(d_tot):,.0f} A^2 {'more' if d_tot >= 0 else 'less'} "
                f"surface in total ({abs(d1):,.0f} A^2 {'more' if d1 >= 0 else 'less'} on {b.label1}), "
                f"with {b.n_hbonds} vs {a.n_hbonds} hydrogen bonds and {b.n_salt_bridges} vs {a.n_salt_bridges} "
                f"salt bridges; its footprint on {side_label} shares {shared} residues with that of {ref.name} "
                f"(Jaccard {jac:.2f}).")
        return " ".join(out)


def compare(entries: Sequence[ComplexEntry], side: int = 1, align: str = "auto") -> Comparison:
    """Compare the group interfaces of several complexes on their shared side."""
    if len(entries) < 2:
        raise ValueError("need at least two complexes to compare")
    return Comparison(list(entries), side=side, align=align)
