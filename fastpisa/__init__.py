# Copyright (c) 2026 Dawid Zyla. Part of fastPISA.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""fastPISA: Local reproduction of PISA with COCOMAPS mode.

Reads a PDB/mmCIF file and identifies biomolecular interfaces. All modes run
one shared analysis core (:mod:`fastpisa.core`) so they find identical
interfaces by construction:

  - combined mode:  (default) one unified report per interface — PISA
                    thermodynamics AND the COCOMAPS contact map.
  - PISA mode:      thermodynamic/surface analysis, output in the PDBe PISA
                    JSON schema ('assembly' + 'interfaces' documents).
  - COCOMAPS mode:  COCOMAPS 2.0 residue-residue contact-map analysis with
                    atomic interaction-type classification (H-bond, salt
                    bridge, pi-pi, cation-pi, ch-pi, ...). Output is a
                    superset of the PISA JSON schema plus an
                    'interface_contact_map' field per interface.

Getting interfaces, the short way::

    import fastpisa

    for interface in fastpisa.interfaces("complex.pdb"):
        print(interface.label, interface.interface_area,
              interface.solvation_energy)
        for bond in interface.hydrogen_bonds:
            print("   ", bond.label)

The full report, when you want the PDBe-shaped JSON documents, the assembly
totals or the per-residue tables::

    result = fastpisa.analyze("complex.pdb", pdb_id="X")
    result.interfaces            # the same Interface objects
    result.write_json("out/")

Or hold the analyzer when you want to re-run with different options, load
AlphaFold confidence, or filter::

    from fastpisa import PISAInterfaceAnalyzer
    analyzer = PISAInterfaceAnalyzer("model.cif", pdb_id="X")
    analyzer.analyze()
    analyzer.load_pae("pae.json").filter_by_pae(max_pae=5.0)

CLI: python -m fastpisa.cli <pdb_file> --mode {combined,pisa,cocomaps}
"""

__version__ = "0.5.0"

__all__ = [
    "__version__",
    "analyze",
    "interfaces",
    "Interface",
    "AtomContact",
    "PISAInterfaceAnalyzer",
]


def __getattr__(name):
    """Expose the classes lazily, so importing fastpisa stays cheap.

    ``Interface`` and ``AtomContact`` pull in numpy and scipy through the
    interface module; a user who only wants ``__version__`` should not pay
    for that.
    """
    if name == "Interface":
        from fastpisa.interface.contacts import Interface

        return Interface
    if name == "AtomContact":
        from fastpisa.interface.contacts import AtomContact

        return AtomContact
    if name == "PISAInterfaceAnalyzer":
        from fastpisa.api import PISAInterfaceAnalyzer

        return PISAInterfaceAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


def analyze(path, pdb_id=None, **kwargs):
    """``fastpisa.analyze("complex.pdb")`` -- see :func:`fastpisa.api.analyze`."""
    from fastpisa.api import analyze as _analyze
    return _analyze(path, pdb_id=pdb_id, **kwargs)


def interfaces(path, **kwargs):
    """Every interface in a structure, **largest first**.

    The one-call entry point: returns a plain ``list`` of
    :class:`fastpisa.interface.contacts.Interface` objects, each carrying its
    area, solvation and stabilisation energies, P-value, CSS, bond counts and
    the bonds themselves.

    Sorted by buried area descending, because the interface a reader cares
    about is almost always the biggest one -- unlike the detection order the
    JSON documents use, which is an implementation detail.

    ``kwargs`` are those of :class:`fastpisa.api.PISAInterfaceAnalyzer`:
    ``mode``, ``ligand_mode``, ``symmetry``, ``predict_assemblies``,
    ``interface_cutoff``, ``probe_radius``, ``point_density``,
    ``exclude_water``, ``min_css``.

    >>> found = interfaces("complex.pdb")           # doctest: +SKIP
    >>> found[0].label, round(found[0].interface_area)
    ('A + B', 781)
    """
    from fastpisa.api import PISAInterfaceAnalyzer

    analyzer = PISAInterfaceAnalyzer(path, **kwargs)
    analyzer.analyze()
    return sorted(analyzer.interfaces,
                  key=lambda i: (-i.interface_area, i.interface_id))

