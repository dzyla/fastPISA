"""Polymer membership comes from the FILE, not only from a hardcoded list.

``interface/contacts.py`` carries sets of standard amino-acid and nucleotide
names plus a hand-maintained list of common modified residues. Anything not
on those lists is treated as a bound hetero group and becomes its own
molecule -- which fabricates interfaces, splits a chain's buried area across
fragments, and is exactly the trap CLAUDE.md warns about.

A list can never be complete. Measured on a blind draw of 60 random PDB
entries, **7 (12%)** declare at least one SEQRES residue absent from those
sets: ACE, AGM, DAH, DHA, DLE, DVA, ETA, FVA, GL3, MGN, MHS, TRX, TYS, 6OG.
2izq is a D-peptide whose chains are built from DLE and DVA: splitting them
off turned its 201 PISA interfaces into 715 and put its main interface 6%
off (724.7 vs 773.6 A^2).

Both formats declare their polymers explicitly -- ``SEQRES`` in PDB,
``_entity_poly_seq`` / ``_pdbx_poly_seq_scheme`` in mmCIF -- and PISA treats
those residues as part of the chain. So does fastPISA now, with the
hardcoded sets as the fallback for files that declare nothing.
"""

from __future__ import annotations

import pytest

from fastpisa.interface.contacts import get_molecule_masks, get_molecules
from fastpisa.parser.pdb_parser import parse_pdb


def _atom(serial, name, res, chain, seq, x, element="C", record="ATOM"):
    return (
        f"{record:<6}{serial:>5} {name:<4} {res:>3} "
        f"{chain:1}{seq:>4}{' ':1}   {x:>8.3f}{0.0:>8.3f}{0.0:>8.3f}"
        f"{1.0:>6.2f}{20.0:>6.2f}          {element:>2}\n"
    )


def _d_peptide(tmp_path, with_seqres=True):
    """Two chains of D-amino acids, declared in SEQRES or not."""
    lines = []
    if with_seqres:
        lines.append("SEQRES   1 A    3  DVA DLE ALA\n")
        lines.append("SEQRES   1 B    3  DVA DLE ALA\n")
    serial = 1
    for chain, offset in (("A", 0.0), ("B", 6.0)):
        for seq, res in enumerate(("DVA", "DLE", "ALA"), start=1):
            record = "ATOM" if res == "ALA" else "HETATM"
            for name, dx in (("N", 0.0), ("CA", 1.5), ("C", 2.6), ("CB", 1.9)):
                element = "N" if name == "N" else "C"
                lines.append(_atom(serial, name, res, chain, seq,
                                   offset + 4.0 * seq + dx, element, record))
                serial += 1
    path = tmp_path / ("seqres.pdb" if with_seqres else "no_seqres.pdb")
    path.write_text("".join(lines) + "END\n")
    return path


def test_seqres_residues_are_parsed_per_chain(tmp_path):
    structure = parse_pdb(str(_d_peptide(tmp_path)))
    assert structure.polymer_residues == {"DVA", "DLE", "ALA"}


def test_a_d_peptide_chain_is_one_polymer_molecule(tmp_path):
    """D-amino acids are the chain, not ligands bound to it."""
    structure = parse_pdb(str(_d_peptide(tmp_path)))
    molecules = get_molecules(structure)

    assert len(molecules) == 2, [m["chain_id"] for m in molecules]
    assert {m["chain_id"] for m in molecules} == {"A", "B"}
    for mol in molecules:
        assert mol["molecule_class"] == "Protein"
        assert mol["chain_type"] == "polymer"


def test_masks_cover_every_declared_polymer_atom(tmp_path):
    """Buried area must be attributed to the chain, not lost with a fragment."""
    structure = parse_pdb(str(_d_peptide(tmp_path)))
    atoms = structure.atoms
    molecules = get_molecules(structure)
    masks = get_molecule_masks(atoms, molecules,
                              polymer_residues=structure.polymer_residues)

    assert sum(int(m.sum()) for m in masks) == len(atoms)
    for mask in masks:
        assert int(mask.sum()) == 12          # 3 residues x 4 atoms


def test_without_seqres_the_standard_sets_still_apply(tmp_path):
    """No declaration: fall back to the hardcoded sets, as before.

    The ALA residues remain a polymer; the D-amino acids become their own
    molecules. This is the old behaviour, kept for files that declare
    nothing (and for coordinate fragments).
    """
    structure = parse_pdb(str(_d_peptide(tmp_path, with_seqres=False)))
    assert structure.polymer_residues == set()
    molecules = get_molecules(structure)
    assert len(molecules) > 2
    assert any(m["chain_type"] == "ligand" for m in molecules)


def test_seqres_does_not_promote_a_genuine_ligand(tmp_path):
    """A bound hetero group is absent from SEQRES and stays its own molecule."""
    lines = ["SEQRES   1 A    2  ALA ALA\n"]
    serial = 1
    for seq in (1, 2):
        for name, dx in (("N", 0.0), ("CA", 1.5)):
            lines.append(_atom(serial, name, "ALA", "A", seq,
                               4.0 * seq + dx, "N" if name == "N" else "C"))
            serial += 1
    lines.append(_atom(99, "ZN", "ZN", "A", 301, 10.0, "ZN", "HETATM"))
    path = tmp_path / "ligand.pdb"
    path.write_text("".join(lines) + "END\n")

    molecules = get_molecules(parse_pdb(str(path)))
    classes = {m["chain_id"]: m for m in molecules}
    assert "A" in classes and classes["A"]["chain_type"] == "polymer"
    assert any(m["chain_type"] == "ligand" and m.get("ccd_id") == "ZN"
               for m in molecules)


def test_water_is_never_promoted_by_seqres(tmp_path):
    """Defensive: a malformed SEQRES naming HOH must not make water polymer."""
    lines = ["SEQRES   1 A    2  ALA HOH\n"]
    lines.append(_atom(1, "N", "ALA", "A", 1, 0.0, "N"))
    lines.append(_atom(2, "CA", "ALA", "A", 1, 1.5, "C"))
    lines.append(_atom(3, "O", "HOH", "A", 500, 8.0, "O", "HETATM"))
    path = tmp_path / "water.pdb"
    path.write_text("".join(lines) + "END\n")

    molecules = get_molecules(parse_pdb(str(path)))
    assert not any(m.get("ccd_id") == "HOH" and m["chain_type"] == "polymer"
                   for m in molecules)
    assert any(m["chain_id"] == "A" and m["chain_type"] == "polymer"
               for m in molecules)


def test_mmcif_declares_its_polymer_residues():
    """mmCIF carries the same information in ``_pdbx_poly_seq_scheme``."""
    pytest.importorskip("gemmi")
    import os

    from fastpisa.parser.pdb_parser import parse_mmcif

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                        "reference", "assemblies", "8jap-assembly1.cif.gz")
    if not os.path.exists(path):
        pytest.skip("cached assembly mmCIF not available")
    structure = parse_mmcif(path)
    assert structure.polymer_residues
    # Standard residues must be in there; a bound sugar must not.
    assert "ALA" in structure.polymer_residues or "GLY" in structure.polymer_residues
    assert "NAG" not in structure.polymer_residues
