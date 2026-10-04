"""Assembly-level analysis.

fastPISA analyses the coordinates it is GIVEN by default -- an asymmetric
unit, a biological-assembly file, a predicted model. With
``symmetry="crystal"`` it additionally generates crystallographic symmetry
mates and reports the crystal's packing interfaces, and with
``predict_assemblies=True`` it searches the crystal for the assemblies those
interfaces admit. Both are opt-in; without a usable cell and space group
they are no-ops, which is the right answer for a predicted model.

What IS here: :mod:`fastpisa.assembly.crystal` generates the symmetry mates,
:mod:`fastpisa.assembly.graph` turns the crystal's interfaces into a contact
graph of placed molecules, and :mod:`fastpisa.assembly.predict` enumerates the
finite assemblies that graph admits and ranks them by the dissociation energy
of :mod:`fastpisa.energy.dissociation`. What is still absent is a biological-
versus-crystal verdict calibrated against an independent dataset; the
predicted assemblies are its input, not a substitute for it.

A ``symmetry`` module used to live here. It was never imported, never
tested, and its docstring advertised space-group operator generation and
assembly prediction -- which invited the assumption that packing interfaces
were covered. It was removed rather than left as an advertisement for
absent functionality; ``git log`` has it if it is ever revived.
"""
