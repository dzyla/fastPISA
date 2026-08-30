"""Shared pytest fixtures for fastPISA tests."""
import os
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "tests", "data")
KTZ = os.path.join(DATA_DIR, "1ktz.pdb")

# Optional external resources for integration / reproducibility tests,
# supplied via environment variables (unset = the tests skip):
#   FASTPISA_EXTERNAL_CIF        an AlphaFold/cryo-EM multi-chain complex CIF
#                                (H-free model, used by H-free-chemistry tests)
#   FASTPISA_EXTERNAL_MODELS_GLOB glob of predicted-model CIFs without crystal
#                                data (used by the binary reproducibility test)
#   FASTPISA_PISA_BIN            path to an original CCP4 `pisa` binary
EXTERNAL_CIF = os.environ.get("FASTPISA_EXTERNAL_CIF", "")
EXTERNAL_MODELS_GLOB = os.environ.get("FASTPISA_EXTERNAL_MODELS_GLOB", "")
PISA_BIN = os.environ.get("FASTPISA_PISA_BIN", "")


def _have(path):
    return bool(path) and os.path.exists(path)


needs_external_cif = pytest.mark.skipif(
    not _have(EXTERNAL_CIF),
    reason="no external complex CIF (set FASTPISA_EXTERNAL_CIF)"
)
needs_external_models = pytest.mark.skipif(
    not EXTERNAL_MODELS_GLOB,
    reason="no external model set (set FASTPISA_EXTERNAL_MODELS_GLOB)"
)
needs_pisa_bin = pytest.mark.skipif(
    not _have(PISA_BIN),
    reason="original CCP4 PISA binary not available (set FASTPISA_PISA_BIN)"
)


# Minimal CCP4 PISA config template. PISA (v2.2.0) hard-errors with
# "No configuration file" (exit 3) if none is provided, so every invocation
# MUST set PISA_CONF_FILE. We point DATA_ROOT at the test's own tmp dir with a
# unique SESSION_PREFIX so each test's sessions are isolated and `-list`
# resolves them from the same working dir. The SRS/MOLREF/PISTORE dirs must
# point at the CCP4 share tree (they hold the chemical / surface data).
# The CCP4 root is derived from PISA_BIN (<ccp4>/bin/pisa -> <ccp4>).
CCP4_ROOT = os.path.dirname(os.path.dirname(PISA_BIN))

_PISA_CFG_BODY = """\
DATA_ROOT
{cwd}
SRS_DIR
{ccp4}/share/ccp4srs/
MOLREF_DIR
{ccp4}/share/pisa/
PISTORE_DIR
{ccp4}/share/pisa/
RASMOL_COM
/dummy/rasmol
JMOL_COM
/dummy/jmol
CCP4MG_COM
/dummy/ccp4mg
SESSION_PREFIX
__fp_
"""


def _write_pisa_config(cwd):
    """Write a self-contained PISA config into ``cwd`` and return its path."""
    cfg_path = os.path.join(cwd, "pisa_fp_test.cfg")
    with open(cfg_path, "w") as fh:
        fh.write(_PISA_CFG_BODY.format(cwd=cwd, ccp4=CCP4_ROOT))
    return cfg_path


def _pisa_env(cwd):
    """Return an env with PISA_CONF_FILE pointing at a fresh config in ``cwd``.

    The session prefix ``__fp_`` keeps sessions from colliding with anything
    else in the same DATA_ROOT and guarantees ``pisa <sess> -list`` resolves
    the session written by a prior ``-analyse`` from the same (tmp) dir.
    """
    env = dict(os.environ)
    env.pop("PISA_CONFIG", None)
    env["PISA_CONF_FILE"] = _write_pisa_config(cwd)
    env["PATH"] = os.path.dirname(PISA_BIN) + ":" + env.get("PATH", "")
    return env


def run_pisa_binary_analyse(cif_pdb, session_name, cwd):
    """Run the original CCP4 PISA binary -analyse.

    A config file is required (the binary otherwise exits 3 with "No
    configuration file"). We write a per-run config whose DATA_ROOT is ``cwd``
    with a unique SESSION_PREFIX, so the session lands in ``cwd`` and the
    subsequent ``-list`` resolves it from the same place.

    For structures without crystal data (predicted models) this yields
    exactly the ASU interfaces -- the cleanest apples-to-apples comparison
    with fastPISA.
    """
    env = _pisa_env(cwd)
    cmd = [PISA_BIN, session_name, "-analyse", cif_pdb]
    r = subprocess.run(
        cmd, capture_output=True, text=True, cwd=cwd, env=env, timeout=300
    )
    return r


def pisa_list_interfaces(session_name, cwd):
    """Run `pisa <session> -list interfaces` and return its text."""
    env = _pisa_env(cwd)
    cmd = [PISA_BIN, session_name, "-list", "interfaces"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env, timeout=120)
    return r.stdout
