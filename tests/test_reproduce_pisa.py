"""Reproducibility tests: fastPISA vs the original CCP4 PISA binary.

These are the publishability checks. For each structure we run the ORIGINAL
PISA v2.2.0 binary (`/programs/xtal/ccp4-9/bin/pisa`) and parse its
`-list interfaces` output (interface count + area per chain pair), then compare
against fastPISA on the same input. The hard acceptance criteria:

  * interface COUNT must match the original binary, and
  * per-pair interface AREA must agree within ~10% (elliptic-surface-area
    conventions differ slightly between engines).

These tests are skipped when the original binary or PDB conversion is
unavailable so the suite still runs standalone on other machines.
"""
import glob
import os
import re
import subprocess
import sys

import pytest

import gemmi

from fastpisa.api import PISAInterfaceAnalyzer

from conftest import (
    PISA_BIN, CASP17_DIR, needs_pisa_bin, needs_casp17, REPO_ROOT,
    run_pisa_binary_analyse, pisa_list_interfaces, KTZ,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cif_to_pdb(cif_path, out_dir):
    """Convert an mmCIF to PDB (original PISA needs PDB for these models)."""
    st = gemmi.read_structure(cif_path)
    out = os.path.join(out_dir, os.path.basename(cif_path).replace(".cif", ".pdb"))
    st.write_pdb(out)
    return out


def parse_pisa_list(text):
    """Parse `pisa <session> -list interfaces` output.

    Rows are ``|``-delimited, e.g.::

        2  2 |      B      |      A   X,Y,Z  1_555 |   493.4  -4.3  5  8  0

    Returns a list of dicts:
      {chain1, chain2, symop, area, deltag, nhb, nsb, nds}.
    For structures with crystal data the symmetry op is non-identity and
    aggregated (multi-interface) entries appear; for no-crystal models every
    row is the identity ``X,Y,Z`` operation.
    """
    interfaces = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue
        # parts[0] = "##  Id" ; parts[1] = monomer1 ; parts[2] = monomer2+symop ;
        # parts[3] = "area deltag nhb nsb nds"
        chain1 = parts[1].strip()
        tail = parts[2].split()
        chain2 = tail[0]
        symop = " ".join(tail[1:]) if len(tail) > 1 else "X,Y,Z"
        num = parts[3].split()
        if len(num) < 5:
            continue
        try:
            interfaces.append({
                "chain1": chain1,
                "chain2": chain2,
                "symop": symop,
                "area": float(num[0]),
                "deltag": float(num[1]),
                "nhb": int(num[2]),
                "nsb": int(num[3]),
                "nds": int(num[4]),
            })
        except ValueError:
            continue
    return interfaces


def fastpisa_interfaces(path, mode="pisa"):
    """Return list of (sorted chain pair, area) from fastPISA."""
    a = PISAInterfaceAnalyzer(path, pdb_id="x", mode=mode).analyze()
    out = []
    for i in a["interfaces_obj"]:
        pair = tuple(sorted([i.molecules[0]["chain_id"], i.molecules[1]["chain_id"]]))
        out.append({"pair": pair, "area": i.interface_area,
                    "nhb": i.number_hydrogen_bonds,
                    "nsb": i.number_salt_bridges,
                    "nds": i.number_disulfide_bonds})
    return out


def collect_one(cif_path, tmp_path, session):
    """Run original PISA on a CIF and return its parsed interface list."""
    pdb = cif_to_pdb(cif_path, str(tmp_path))
    r = run_pisa_binary_analyse(pdb, session, str(tmp_path))
    if r.returncode != 0:
        pytest.skip(f"original PISA -analyse failed for {os.path.basename(cif_path)}")
    text = pisa_list_interfaces(session, str(tmp_path))
    return parse_pisa_list(text)


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------
@needs_pisa_bin
def test_1ktz_matches_original_pisa(tmp_path):
    """1ktz dimer: asymmetric-unit (X,Y,Z) interface area vs original PISA.

    1ktz has crystal data, so the original binary also reports many
    symmetry-copy interfaces that fastPISA does not generate yet. We therefore
    compare only the identity (X,Y,Z) asymmetric-unit interface -- this is the
    one both tools compute for the input as given. (Full symmetry-copy
    assembly prediction is a separate, documented scope gap.)
    """
    session = "__r_1ktz"
    ref = collect_one(KTZ, tmp_path, session)

    # FastPISA finds the single A-B X,Y,Z interface.
    ours = fastpisa_interfaces(KTZ)
    assert len(ours) == 1, f"expected 1 interface from fastPISA, got {ours}"

    # Find original-PISA rows that are the asymmetric-unit (X,Y,Z) A-B contact.
    identity = [i for i in ref if i["symop"].startswith("X,Y,Z")
                and {i["chain1"], i["chain2"]} == {"A", "B"}]
    assert identity, f"original PISA found no X,Y,Z A-B interface in {ref}"
    ref_area = identity[0]["area"]

    rel = abs(ours[0]["area"] - ref_area) / ref_area
    assert rel < 0.10, (
        f"1ktz X,Y,Z area mismatch: fastPISA={ours[0]['area']:.1f} "
        f"PISA={ref_area:.1f} ({rel*100:.1f}%)"
    )


@needs_pisa_bin
@needs_casp17
@pytest.mark.parametrize("case", ["H1443", "H1400", "H2343", "H1346"])
def test_casp17_matches_original_pisa(case, tmp_path):
    """CASP17 antibody AlphaFold models: interface set + areas vs PISA.

    These models have NO crystal data, so the original binary reports exactly
    the asymmetric-unit interfaces -- a direct apples-to-apples comparison
    (counts AND per-pair areas) with fastPISA.
    """
    cif = sorted(glob.glob(os.path.join(
        CASP17_DIR, case, case, "seed_101", "predictions", "*.cif")))[0]
    ref = collect_one(cif, tmp_path, f"__r_{case}")
    assert len(ref) >= 1, f"original PISA found no interfaces for {case}"
    ref_by_pair = {tuple(sorted([i["chain1"], i["chain2"]])): i for i in ref}

    ours = fastpisa_interfaces(cif)
    ours_by_pair = {i["pair"]: i for i in ours}

    # Interface COUNT must match.
    assert len(ours) == len(ref), (
        f"{case}: interface count fastPISA={len(ours)} vs PISA={len(ref)}; "
        f"ours={sorted(ours_by_pair)} ref={sorted(ref_by_pair)}"
    )

    # Per-pair AREA must agree within 10% (elliptic-surface convention differs
    # slightly between engines).
    for pair, r in ref_by_pair.items():
        assert pair in ours_by_pair, (
            f"{case}: PISA found interface {pair}, fastPISA did not "
            f"(ours={sorted(ours_by_pair)})"
        )
        o = ours_by_pair[pair]
        rel = abs(o["area"] - r["area"]) / r["area"]
        assert rel < 0.10, (
            f"{case}: interface {pair} area mismatch: "
            f"fastPISA={o['area']:.1f} PISA={r['area']:.1f} ({rel*100:.1f}%)"
        )
