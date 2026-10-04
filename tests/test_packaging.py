"""Packaging metadata a scientific user depends on.

Three of these are adoption blockers rather than code defects:

* **No LICENSE file.** ``pyproject.toml`` and ``CITATION.cff`` both declared
  MIT while the repository carried no licence text, so by default the code
  was all-rights-reserved: an institutional legal review stops there and
  nobody can legally reuse it.
* **Version drift.** ``CITATION.cff`` said 0.2.0 while the package said
  0.4.0, so a citation pointed at a version that never produced the numbers
  being cited.
* **An unusable author record.** ``family-names: "zyla"`` with empty
  ``given-names`` renders as a broken name in every reference manager that
  reads CFF.
"""

from __future__ import annotations

import os
import re

import pytest

import fastpisa

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name):
    with open(os.path.join(REPO, name), encoding="utf-8") as fh:
        return fh.read()


def _pyproject_version():
    for line in _read("pyproject.toml").splitlines():
        if line.startswith("version"):
            return line.split("=", 1)[1].strip().strip('"')
    raise AssertionError("pyproject.toml has no version")


def _citation_field(field):
    for line in _read("CITATION.cff").splitlines():
        if line.startswith(f"{field}:"):
            return line.split(":", 1)[1].strip().strip('"')
    return None


def test_a_licence_file_exists_and_matches_the_declared_licence():
    """Without this file the declared MIT licence is not actually granted."""
    assert os.path.exists(os.path.join(REPO, "LICENSE")), (
        "no LICENSE file: pyproject.toml declares MIT but the repository "
        "grants nothing, so the code is all-rights-reserved by default")
    text = _read("LICENSE")
    assert "MIT License" in text
    assert "Permission is hereby granted, free of charge" in text
    assert re.search(r"Copyright \(c\) \d{4}", text), "no copyright line"
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in text


def test_the_version_is_the_same_everywhere():
    """A citation must point at the version that produced the numbers."""
    package = fastpisa.__version__
    assert _pyproject_version() == package, (
        f"pyproject {_pyproject_version()} != fastpisa {package}")
    assert _citation_field("version") == package, (
        f"CITATION.cff {_citation_field('version')} != fastpisa {package}")


def test_the_citation_names_a_person_a_reference_manager_can_render():
    author_block = _read("CITATION.cff")
    assert "family-names:" in author_block
    assert "given-names:" in author_block
    family = re.search(r'family-names:\s*"?([^"\n]+)"?', author_block)
    given = re.search(r'given-names:\s*"?([^"\n]+)"?', author_block)
    assert family and family.group(1).strip(), "empty family-names"
    assert given and given.group(1).strip(), (
        "empty given-names: renders as a broken name in every CFF consumer")
    assert family.group(1).strip()[0].isupper(), "family name not capitalised"


def test_the_citation_declares_the_same_licence_as_the_package():
    assert _citation_field("license") == "MIT"


def test_a_changelog_exists_and_covers_the_current_version():
    """Results changed between versions; a user has to be able to see that."""
    assert os.path.exists(os.path.join(REPO, "CHANGELOG.md")), (
        "no CHANGELOG: an earlier release computed ASA ~6x too large without "
        "freesasa, and a user has no way to learn that their numbers moved")
    text = _read("CHANGELOG.md")
    assert fastpisa.__version__ in text
    # The ASA correction changed published numbers; it must be called out.
    assert "freesasa" in text.lower() or "shrake" in text.lower()


def test_the_package_exposes_its_version():
    assert re.fullmatch(r"\d+\.\d+\.\d+", fastpisa.__version__)
