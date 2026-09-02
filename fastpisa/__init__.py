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

Use:
  from fastpisa.api import PISAInterfaceAnalyzer
  ana = PISAInterfaceAnalyzer("in.pdb", pdb_id="X")   # combined mode
  ana.analyze(); ana.interfaces; ana.write_json("out/")

CLI: python -m fastpisa.cli <pdb_file> --mode {combined,pisa,cocomaps}
"""

__version__ = "0.4.0"


def analyze(path, pdb_id=None, **kwargs):
    """``fastpisa.analyze("complex.pdb")`` -- see :func:`fastpisa.api.analyze`."""
    from fastpisa.api import analyze as _analyze
    return _analyze(path, pdb_id=pdb_id, **kwargs)

