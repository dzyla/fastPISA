"""Assembly-level analysis.

fastPISA analyses the coordinates it is GIVEN -- an asymmetric unit, a
biological-assembly file, a predicted model. It does not generate
crystallographic symmetry mates and does not search the crystal for
assemblies, so the packing interfaces original PISA reports for an
experimental entry are outside its scope (see the README's benchmark note:
the comparison is restricted to PISA's identity-symmetry interfaces).

What IS here, in :mod:`fastpisa.energy.dissociation`, is the assembly
dissociation pathway for the given coordinates: the cheapest set of
interfaces to break, and the resulting dG_diss and T*dS under PISA's own
relation.

A ``symmetry`` module used to live here. It was never imported, never
tested, and its docstring advertised space-group operator generation and
assembly prediction -- which invited the assumption that packing interfaces
were covered. It was removed rather than left as an advertisement for
absent functionality; ``git log`` has it if it is ever revived.
"""
