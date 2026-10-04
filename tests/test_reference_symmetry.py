"""The reference parser must expose PISA's per-molecule crystal transform.

Each ``<molecule>`` in PISA's interfaces XML carries the symmetry operation
that places it: the triplet (``symop``), its index (``symop_no``), the cell
translation (``cell_i/j/k``) and the full 3x3 rotation plus translation in
ORTHOGONAL coordinates (``rxx``...``rzz``, ``tx/ty/tz``).

Only ``symop``/``symop_no`` were parsed, which is why the benchmark could
only filter PISA's list down to identity-identity interfaces and throw the
other 60% away. The matrix is what lets a generated symmetry mate be matched
to the one PISA actually used, so it has to come through.
"""

from __future__ import annotations

import gzip
import os

import numpy as np
import pytest

from fastpisa.reference.ebi_pisa import parse_pisa_xml

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "reference")


def _interfaces(pdb_id):
    with gzip.open(os.path.join(REF, f"{pdb_id}.pisa.xml.gz"), "rt") as fh:
        return parse_pisa_xml(fh.read())


def test_identity_molecule_carries_the_identity_transform():
    mol = _interfaces("1acb")[0]["molecules"][0]
    assert mol["symop_no"] == "1"
    assert mol["cell"] == (0, 0, 0)
    assert np.allclose(mol["rotation"], np.eye(3))
    assert np.allclose(mol["translation"], np.zeros(3))


def test_symmetry_mate_carries_a_non_identity_transform():
    """1acb is P 1 21 1: its packing interfaces use the 2-fold screw axis."""
    mates = [
        m for i in _interfaces("1acb") for m in i["molecules"]
        if m["symop_no"] != "1"
    ]
    assert mates, "1acb must have symmetry-mate molecules"
    for m in mates:
        rot = np.asarray(m["rotation"], dtype=float)
        assert rot.shape == (3, 3)
        # A crystallographic rotation is orthogonal with |det| = 1.
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-6)
        assert abs(abs(np.linalg.det(rot)) - 1.0) < 1e-6
        assert not np.allclose(rot, np.eye(3)) or any(m["cell"])


def test_cell_translations_are_integers():
    for iface in _interfaces("1aay"):
        for m in iface["molecules"]:
            assert all(isinstance(v, int) for v in m["cell"])


def test_transform_actually_places_the_mate_in_contact():
    """Applying PISA's matrix must reproduce the contact PISA reported.

    This is the property the matrix is used for, and it catches a
    transposed rotation or a translation read in the wrong space -- both of
    which leave an orthogonal matrix but move the molecule elsewhere.

    (Comparing instead against ``gemmi.Op(symop)`` would tangle two
    conventions: PISA's triplet already carries the cell translation, e.g.
    ``-X,Y-1/2,-Z+1``, while gemmi normalises it into [0,1) and PISA also
    reports the integer part separately in ``cell_i/j/k``.)
    """
    import numpy as np
    from scipy.spatial import cKDTree

    from fastpisa.parser.pdb_parser import parse_pdb

    structure = parse_pdb(os.path.join(REF, "pdb", "1acb.pdb.gz"))
    by_chain = {}
    for atom in structure.atoms:
        if atom.element.strip().upper() in ("H", "D"):
            continue
        if atom.res_name.strip().upper() in ("HOH", "WAT"):
            continue
        by_chain.setdefault(atom.auth_asym_id, []).append(
            (atom.x, atom.y, atom.z))
    by_chain = {k: np.array(v, dtype=float) for k, v in by_chain.items()}

    checked = 0
    for iface in _interfaces("1acb"):
        mols = iface["molecules"]
        if len(mols) != 2:
            continue
        placed = []
        for m in mols:
            xyz = by_chain.get(m["chain_id"])
            if xyz is None:
                break
            rot = np.asarray(m["rotation"], dtype=float)
            tr = np.asarray(m["translation"], dtype=float)
            placed.append(xyz @ rot.T + tr)
        if len(placed) != 2:
            continue
        a, b = placed
        # The threshold is the SHADOW cutoff, not 5 A. PISA defines an
        # interface by buried area, and two atoms shadow each other's
        # accessible surface out to r1 + r2 + 2*probe ~ 6.6 A. 1acb's
        # interface 9 is real at 14.2 A^2 with a minimum atom distance of
        # 5.02 A -- a 5 A contact screen would miss it outright.
        shadow = 2 * 1.87 + 2 * 1.4 + 0.1
        min_dist = cKDTree(b).query(a)[0].min()
        assert min_dist < shadow, (
            f"interface {iface['id']} ({mols[0]['chain_id']}/"
            f"{mols[0]['symop']} + {mols[1]['chain_id']}/{mols[1]['symop']}) "
            f"min atom distance {min_dist:.2f} A once PISA's own transforms "
            "are applied")
        checked += 1
    assert checked == 9, f"1acb has 9 interfaces, checked {checked}"
