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


def run_pisa_binary_analyse(cif_pdb, session_name, cwd):
    """Run the original CCP4 PISA binary -analyse.

    IMPORTANT: we do NOT set PISA_CONF_FILE / PISA_CONFIG here, because the
    reference config's DATA_ROOT redirects sessions away from the working dir.
    Without a config env the binary stores the session under ``cwd`` (verified
    experimentally), which is what lets the subsequent ``-list`` resolve it.

    For structures without crystal data (CASP17 AlphaFold models) this yields
    exactly the ASU interfaces -- the cleanest apples-to-apples comparison
    with fastPISA.
    """
    env = dict(os.environ)
    env.pop("PISA_CONFIG", None)
    env.pop("PISA_CONF_FILE", None)
    env["PATH"] = "/programs/xtal/ccp4-9/bin:" + env.get("PATH", "")
    cmd = [PISA_BIN, session_name, "-analyse", cif_pdb]
    r = subprocess.run(
        cmd, capture_output=True, text=True, cwd=cwd, env=env, timeout=300
    )
    return r


def pisa_list_interfaces(session_name, cwd):
    """Run `pisa <session> -list interfaces` and return its text."""
    env = dict(os.environ)
    env.pop("PISA_CONFIG", None)
    env.pop("PISA_CONF_FILE", None)
    env["PATH"] = "/programs/xtal/ccp4-9/bin:" + env.get("PATH", "")
    cmd = [PISA_BIN, session_name, "-list", "interfaces"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env, timeout=120)
    return r.stdout
