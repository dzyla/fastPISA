"""ligand_mode: classic separate-monomer vs merged-chain molecule definitions."""
import os

import pytest

from fastpisa.parser.pdb_parser import parse_pdb
from fastpisa.interface.contacts import get_molecules, get_molecule_masks
from fastpisa.api import analyze_interface
from fastpisa.reference.ebi_pisa import cached_pdb_path

from conftest import KTZ

HB = cached_pdb_path("1a3n")  # hemoglobin: 4 protein chains + 4 HEM ligands
needs_hb = pytest.mark.skipif(HB is None, reason="1a3n reference PDB not cached")


def test_no_ligands_modes_agree():
    """Without hetero groups the two conventions are identical."""
    sep = analyze_interface(KTZ, pdb_id="1ktz", mode="pisa")
    mer = analyze_interface(KTZ, pdb_id="1ktz", mode="pisa", ligand_mode="merge")
    assert [i.interface_area for i in sep["interfaces_obj"]] == \
           [i.interface_area for i in mer["interfaces_obj"]]


@needs_hb
def test_merge_folds_ligands_into_chains():
    st = parse_pdb(HB)
    sep = get_molecules(st)
    mer = get_molecules(st, merge_ligands=True)
    assert sum(1 for m in sep if m["chain_type"] == "ligand") >= 4  # HEMs
    assert all(m["chain_type"] == "chain" for m in mer)
    assert len(mer) == 4  # one molecule per chain, hetero groups included

    masks = get_molecule_masks(st.atoms, mer)
    # every non-water atom belongs to exactly one merged molecule
    import numpy as np
    total = np.zeros(len(st.atoms), dtype=int)
    for m in masks:
        total += m.astype(int)
    n_water = sum(1 for a in st.atoms if a.res_name.upper() in ("HOH", "WAT"))
    assert (total <= 1).all()
    assert total.sum() == len(st.atoms) - n_water


@needs_hb
def test_merge_reduces_interface_count_but_keeps_chain_pairs():
    sep = analyze_interface(HB, pdb_id="1a3n", mode="pisa")
    mer = analyze_interface(HB, pdb_id="1a3n", mode="pisa", ligand_mode="merge")
    pairs_sep = {tuple(sorted([i.molecules[0]["chain_id"], i.molecules[1]["chain_id"]]))
                 for i in sep["interfaces_obj"]}
    pairs_mer = {tuple(sorted([i.molecules[0]["chain_id"], i.molecules[1]["chain_id"]]))
                 for i in mer["interfaces_obj"]}
    # merged mode reports only chain-chain interfaces
    assert all("[" not in a and "[" not in b for a, b in pairs_mer)
    # the protein-protein pairs survive
    assert {("A", "B"), ("C", "D")} <= pairs_mer
    assert len(mer["interfaces_obj"]) < len(sep["interfaces_obj"])
