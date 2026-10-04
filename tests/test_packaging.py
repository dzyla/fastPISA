"""Packaging metadata a scientific user depends on.

Three of these are adoption blockers rather than code defects:

* **No LICENSE file.** ``pyproject.toml`` and ``CITATION.cff`` both declared
  a licence while the repository carried no licence text, so by default the
  code was all-rights-reserved: an institutional legal review stops there
  and nobody can legally reuse it.

The licence is **AGPL-3.0-or-later** with an additional attribution term
under its section 7(b). The choice is deliberate and these tests pin the
parts that carry the intent: strong copyleft so a modified version stays
open, the section 13 network clause so hosting it as a service also
triggers disclosure, and a preserved citation notice.
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
    """Without this file the declared licence is not actually granted."""
    assert os.path.exists(os.path.join(REPO, "LICENSE")), (
        "no LICENSE file: pyproject.toml declares a licence but the "
        "repository grants nothing, so the code is all-rights-reserved")
    text = _read("LICENSE")
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in text
    assert "Version 3, 19 November 2007" in text
    assert re.search(r"Copyright \(c\) \d{4}", text), "no copyright line"


def test_the_licence_text_is_the_unmodified_agpl():
    """A paraphrased licence is not a licence. Pin the operative clauses.

    These three are why AGPL was chosen over MIT or plain GPL: section 13
    closes the hosting loophole, section 5(c) is the copyleft that keeps a
    modified version open, and section 7(b) is what makes the attribution
    requirement binding rather than a request.
    """
    text = _read("LICENSE")
    # The licence wraps its clauses, so compare on collapsed whitespace.
    flat = " ".join(text.split())
    assert "13. Remote Network Interaction" in text
    assert "modified version must prominently offer all users" in flat
    assert "7. Additional Terms" in text
    assert ("Requiring preservation of specified reasonable legal notices "
            "or author attributions") in flat
    # The full text, not an excerpt.
    assert len(text) > 34000, f"LICENSE is only {len(text)} bytes"


def test_the_licence_carries_the_citation_term():
    """The 7(b) term and how to cite must be in the LICENSE itself.

    A citation request living only in the README is not a licence term, and
    a downstream user reads the LICENSE.
    """
    text = _read("LICENSE")
    head = text[:text.index("GNU AFFERO GENERAL PUBLIC LICENSE")]
    assert "7(b)" in head or "Section 7(b)" in head
    assert "CITATION.cff" in head
    assert "AGPL-3.0-or-later" in head


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
    assert _citation_field("license") == "AGPL-3.0-or-later"


def test_the_packaging_metadata_declares_the_licence_by_spdx_id():
    """A classifier or SPDX id is what dependency scanners read."""
    text = _read("pyproject.toml")
    assert "AGPL-3.0-or-later" in text
    assert "MIT" not in text, "stale MIT declaration in pyproject.toml"


def test_the_author_is_named_for_citation():
    """``name = "dzyla"`` is a username, not an author of a cited work."""
    text = _read("pyproject.toml")
    assert "Dawid Zyla" in text


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


def test_every_shipped_module_carries_the_spdx_tag():
    """A copied file must carry its licence.

    The root LICENSE covers the project as distributed, but a copyleft
    licence's weak point is a single module lifted into another codebase: a
    file with no notice looks unlicensed to whoever finds it, and to every
    automated licence scanner. One SPDX line per file closes that.
    """
    missing = []
    for root, dirs, files in os.walk(os.path.join(REPO, "fastpisa")):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as fh:
                head = fh.read(400)
            if "SPDX-License-Identifier: AGPL-3.0-or-later" not in head:
                missing.append(os.path.relpath(path, REPO))
    assert not missing, f"{len(missing)} modules without an SPDX tag: {missing[:5]}"
