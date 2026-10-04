"""A large benchmark must cache outside the repository's test data.

The 37-entry reference set is committed; a 2000-entry run is 1-2 GB of
fetched PISA XML and PDB files that must not land in ``tests/data/reference``
and must not be hardcoded to one machine's paths. Both the environment
variable and the explicit argument are checked, because the harness sets the
variable and library callers should not have to.
"""

from __future__ import annotations

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_reference_dir_honours_the_environment_variable(tmp_path):
    """Resolved in a fresh process: the default is an import-time constant."""
    env = dict(os.environ, FASTPISA_REFERENCE_DIR=str(tmp_path))
    out = subprocess.run(
        [sys.executable, "-c",
         "from fastpisa.reference.ebi_pisa import reference_dir; "
         "print(reference_dir())"],
        capture_output=True, text=True, cwd=REPO, env=env, timeout=120)
    assert out.returncode == 0, out.stderr[-1500:]
    assert out.stdout.strip() == str(tmp_path)


def test_the_default_is_the_committed_reference_set():
    env = {k: v for k, v in os.environ.items()
           if k != "FASTPISA_REFERENCE_DIR"}
    out = subprocess.run(
        [sys.executable, "-c",
         "from fastpisa.reference.ebi_pisa import reference_dir; "
         "print(reference_dir())"],
        capture_output=True, text=True, cwd=REPO, env=env, timeout=120)
    assert out.returncode == 0, out.stderr[-1500:]
    assert out.stdout.strip() == os.path.join(REPO, "tests", "data",
                                              "reference")


def test_compare_crystal_entry_takes_an_explicit_cache_dir(tmp_path):
    """An empty cache dir must yield None, not silently read the committed set."""
    import pytest
    pytest.importorskip("gemmi")
    from fastpisa.reference.compare import compare_crystal_entry

    assert compare_crystal_entry("1acb", allow_fetch=False,
                                 cache_dir=str(tmp_path)) is None
    # ...while the committed set still resolves.
    assert compare_crystal_entry("1acb", allow_fetch=False) is not None
