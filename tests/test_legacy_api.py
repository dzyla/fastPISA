"""The uncalibrated legacy helpers must warn, and the live ones must not.

``fastpisa.scoring.scoring`` and ``fastpisa.energy.energy`` still export
pre-calibration models that no pipeline uses -- an ad-hoc P-value, a
hand-weighted CSS composite, a bipartition-free dissociation sum. Their own
docstrings call them UNCALIBRATED, but they are importable from a package
whose entire claim is calibration against PISA, so a user can reach for
``calculate_p_value`` and get numbers that do not match any PISA output.

Deprecation warnings, as ``calculate_bsa`` already does.
"""

from __future__ import annotations

import warnings

import pytest


LEGACY = [
    ("fastpisa.scoring.scoring", "calculate_p_value", (-20.0, 800.0, 20000.0)),
    ("fastpisa.scoring.scoring", "calculate_css", (800.0, -20.0, 0.3, 50, 40, 20000.0)),
    ("fastpisa.scoring.scoring", "classify_interface", (0.3, 0.7, 900.0)),
    ("fastpisa.energy.energy", "calculate_dissociation_energy",
     ([800.0], [-20.0], [12.0])),
    ("fastpisa.energy.energy", "calculate_stabilization_energy", (-20.0, [])),
    ("fastpisa.energy.energy", "calculate_entropy", (800.0, 20, 20)),
]


@pytest.mark.parametrize("module_name,func_name,args", LEGACY,
                         ids=[f"{m.rsplit('.', 1)[-1]}.{n}" for m, n, _ in LEGACY])
def test_legacy_helper_warns(module_name, func_name, args):
    import importlib

    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    with pytest.warns(DeprecationWarning, match=func_name):
        func(*args)


LIVE = [
    ("fastpisa.scoring.scoring", "calculate_css_pisa", (-20.0, 800.0)),
    ("fastpisa.energy.energy", "bond_energy", (5, 2, 0)),
    ("fastpisa.energy.entropy", "dissociation_entropy", ([20000.0, 20000.0],)),
]


@pytest.mark.parametrize("module_name,func_name,args", LIVE,
                         ids=[n for _, n, _ in LIVE])
def test_calibrated_helper_does_not_warn(module_name, func_name, args):
    import importlib

    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        func(*args)


def test_a_full_analysis_emits_no_deprecation_warnings():
    """The pipeline must not reach its own deprecated code."""
    import fastpisa

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        fastpisa.analyze("tests/data/1ktz.pdb", pdb_id="1ktz", mode="combined")


def test_dead_private_helpers_are_gone():
    """Removed rather than left to rot (both were unreachable)."""
    from fastpisa.interface import contacts
    from fastpisa.surface import shrake_rupley

    assert not hasattr(contacts, "_element_from_name")
    assert not hasattr(shrake_rupley, "calculate_asa_batched")


def test_symmetry_module_is_not_shipped():
    """It advertised crystal-symmetry assembly prediction that does not exist.

    fastPISA analyses the coordinates it is given; symmetry-mate enumeration
    is a documented non-feature. A dead module promising it in its docstring
    invites a user to assume packing interfaces are covered.
    """
    import importlib

    with pytest.raises(ImportError):
        importlib.import_module("fastpisa.assembly.symmetry")
