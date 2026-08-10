"""fastPISA: Local reproduction of PISA with COCOMAPS mode.

Reads a PDB/mmCIF file and identifies biomolecular interfaces using two
complementary approaches that share the same interface-detection and
surface machinery (so they find identical interfaces):

  - PISA mode:      thermodynamic/surface analysis, output in the PDBe PISA
                    JSON schema ('assembly' + 'interfaces' documents).
  - COCOMAPS mode:  COCOMAPS 2.0 residue-residue contact-map analysis with
                    atomic interaction-type classification (H-bond, salt
                    bridge, pi-pi, cation-pi, ch-pi, ...). Output is a
                    superset of the PISA JSON schema plus an
                    'interface_contact_map' field per interface.

Use:
  from fastpisa import pipeline, cocomaps
  results = pipeline.analyze_structure(...)
  results = cocomaps.analyze_structure_cocomaps(...)

CLI: python -m fastpisa.cli <pdb_file> --mode {pisa,cocomaps}
"""

__version__ = "1.0.0"