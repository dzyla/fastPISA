"""Real-world PDB files that must parse, or fail with a usable message.

The failure mode this guards against is silent: a residue number the parser
cannot read became 0, and because alternate-conformer selection keys on
(chain, sequence number, insertion code), several residues then looked like
ONE residue and all but one of their atoms were discarded. Three atoms in,
two out, no warning -- in a structure large enough to need hybrid-36 that is
thousands of atoms.
"""

from __future__ import annotations

import pytest

from fastpisa.parser.pdb_parser import decode_sequence_number, parse_pdb


def _atom_line(serial, name, res, chain, seq_field, x, element=" C",
               icode=" "):
    """One ATOM record built at true PDB v3.3 columns."""
    line = [" "] * 80

    def put(text, start):        # start is 1-based, as the spec numbers them
        for offset, char in enumerate(text):
            line[start - 1 + offset] = char

    put("ATOM  ", 1)
    put(f"{serial:5d}", 7)
    put(f"{name:<4s}", 13)
    put(f"{res:>3s}", 18)
    put(chain, 22)
    put(f"{seq_field:>4s}", 23)
    put(icode, 27)
    put(f"{x:8.3f}", 31)
    put(f"{0.0:8.3f}", 39)
    put(f"{0.0:8.3f}", 47)
    put(f"{1.0:6.2f}", 55)
    put(f"{20.0:6.2f}", 61)
    put(f"{element:>2s}", 77)
    return "".join(line).rstrip() + "\n"


# ---------------------------------------------------------------------------
# hybrid-36 decoding
# ---------------------------------------------------------------------------
def test_plain_decimal_sequence_numbers():
    assert decode_sequence_number("   1") == 1
    assert decode_sequence_number("9999") == 9999
    assert decode_sequence_number(" 100") == 100


def test_negative_sequence_numbers_still_work():
    """Expression tags and DNA numbered about a centre use these."""
    assert decode_sequence_number("  -4") == -4
    assert decode_sequence_number("-999") == -999


def test_hybrid36_upper_case_range():
    """10000 onwards: 'A000'...'ZZZZ' (PDB's documented overflow encoding)."""
    assert decode_sequence_number("A000") == 10000
    assert decode_sequence_number("A001") == 10001
    assert decode_sequence_number("A00A") == 10010
    assert decode_sequence_number("ZZZZ") == 1223055


def test_hybrid36_lower_case_range():
    """After 'ZZZZ' the encoding continues in lower case."""
    assert decode_sequence_number("a000") == 1223056
    assert decode_sequence_number("a001") == 1223057


def test_hybrid36_round_trips_against_a_reference_implementation():
    """Decode is the inverse of the documented encoding, across the range."""
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def encode(value):
        if value < 10000:
            return f"{value:4d}"
        value -= 10000
        if value < 26 * 36 ** 3:
            value += 10 * 36 ** 3
            out = ""
            for _ in range(4):
                value, rem = divmod(value, 36)
                out = digits[rem] + out
            return out
        raise ValueError("beyond the upper-case range")

    for value in (10000, 10001, 12345, 99999, 466559, 1223055):
        assert decode_sequence_number(encode(value)) == value, value


def test_an_unreadable_sequence_number_is_an_error_not_a_zero():
    """Silently becoming 0 merges unrelated residues; say so instead."""
    with pytest.raises(ValueError, match="sequence number"):
        decode_sequence_number("??")


# ---------------------------------------------------------------------------
# ...and the same through the parser
# ---------------------------------------------------------------------------
def test_residues_past_9999_stay_distinct(tmp_path):
    """The bug: 'A000' and 'A001' both became 0, so one CA was deleted."""
    path = tmp_path / "big.pdb"
    path.write_text(
        _atom_line(1, "CA", "ALA", "A", "9999", 0.0)
        + _atom_line(2, "CA", "ALA", "A", "A000", 5.0)
        + _atom_line(3, "CA", "ALA", "A", "A001", 10.0)
        + "END\n"
    )

    atoms = parse_pdb(str(path)).atoms

    assert len(atoms) == 3, "an atom was discarded as a duplicate residue"
    assert sorted(a.res_seq for a in atoms) == [9999, 10000, 10001]
    residues = {(a.auth_asym_id, a.res_seq, a.icode) for a in atoms}
    assert len(residues) == 3


def test_a_large_chain_keeps_every_residue(tmp_path):
    """A ribosomal chain crosses the 9999 boundary; none may be lost."""
    path = tmp_path / "chain.pdb"
    lines = []
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def encode(value):
        if value < 10000:
            return f"{value:4d}"
        value = value - 10000 + 10 * 36 ** 3
        out = ""
        for _ in range(4):
            value, rem = divmod(value, 36)
            out = digits[rem] + out
        return out

    expected = list(range(9995, 10015))
    for index, seq in enumerate(expected, start=1):
        lines.append(_atom_line(index, "CA", "GLY", "A", encode(seq),
                                float(index) * 4.0))
    path.write_text("".join(lines) + "END\n")

    atoms = parse_pdb(str(path)).atoms

    assert len(atoms) == len(expected)
    assert sorted(a.res_seq for a in atoms) == expected


# ---------------------------------------------------------------------------
# Truncated records
# ---------------------------------------------------------------------------
def test_a_truncated_atom_record_names_the_line(tmp_path):
    """A short line must not surface as a confusing element complaint."""
    path = tmp_path / "short.pdb"
    path.write_text("ATOM      1  CA  ALA A   1      10.000\nEND\n")

    with pytest.raises(ValueError, match="line 1"):
        parse_pdb(str(path))


def test_a_record_with_unreadable_coordinates_names_the_line(tmp_path):
    path = tmp_path / "bad_xyz.pdb"
    line = _atom_line(1, "CA", "ALA", "A", "   1", 0.0)
    broken = line[:30] + "   abcde" + line[38:]
    path.write_text(broken + "END\n")

    with pytest.raises(ValueError, match="line 1"):
        parse_pdb(str(path))
