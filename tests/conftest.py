"""Shared pytest fixtures for fastPISA tests."""
import os
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "tests", "data")
KTZ = os.path.join(DATA_DIR, "1ktz.pdb")

# Optional external structure used for integration / reproducibility tests.
OPENDDE_AB = "/home/dzyla/dzyla-lab_home/Code/OpenDDE/outputs/antibody_complexes/results/MeV3920_F4-B05/MeV3920_F4-B05/seed_101/predictions/MeV3920_F4-B05_sample_0.cif"
CASP17_DIR = "/home/dzyla/dzyla-lab_home/Code/OpenDDE/outputs/casp17_antibodies/results"

# Optional original CCP4 PISA binary + config (for reproducibility tests).
PISA_BIN = "/programs/xtal/ccp4-9/bin/pisa"
PISA_CFG = "/tmp/pisa_ref/mypisa.cfg"


def _have(path):
    return path is not None and os.path.exists(path)


needs_opendde = pytest.mark.skipif(
    not _have(OPENDDE_AB), reason="OpenDDE antibody-complex CIF not available"
)
needs_casp17 = pytest.mark.skipif(
    not _have(CASP17_DIR), reason="CASP17 antibodies results dir not available"
)
needs_pisa_bin = pytest.mark.skipif(
    not _have(PISA_BIN), reason="original CCP4 PISA binary not available"
)


# Minimal CCP4 PISA config template. PISA (v2.2.0) hard-errors with
# "No configuration file" (exit 3) if none is provided, so every invocation
# MUST set PISA_CONF_FILE. We point DATA_ROOT at the test's own tmp dir with a
# unique SESSION_PREFIX so each test's sessions are isolated and `-list`
# resolves them from the same working dir. The SRS/MOLREF/PISTORE dirs must
# point at the CCP4 share tree (they hold the chemical / surface data).
_PISA_CFG_BODY = """\
DATA_ROOT
{cwd}
SRS_DIR
/programs/xtal/ccp4-9/share/ccp4srs/
MOLREF_DIR
/programs/xtal/ccp4-9/share/pisa/
PISTORE_DIR
/programs/xtal/ccp4-9/share/pisa/
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
        fh.write(_PISA_CFG_BODY.format(cwd=cwd))
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
    env["PATH"] = "/programs/xtal/ccp4-9/bin:" + env.get("PATH", "")
    return env


def run_pisa_binary_analyse(cif_pdb, session_name, cwd):
    """Run the original CCP4 PISA binary -analyse.

    A config file is required (the binary otherwise exits 3 with "No
    configuration file"). We write a per-run config whose DATA_ROOT is ``cwd``
    with a unique SESSION_PREFIX, so the session lands in ``cwd`` and the
    subsequent ``-list`` resolves it from the same place.

    For structures without crystal data (CASP17 AlphaFold models) this yields
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
