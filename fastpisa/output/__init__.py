"""JSON serialisation of fastPISA results in the PDBe PISA schema.

This package was previously absent from the repository: .gitignore's `output/` rule,
intended for runtime output directories, also matched this SOURCE directory, so it was
never committed and `from fastpisa.api import PISAInterfaceAnalyzer` raised
ModuleNotFoundError from any clean clone. The rule now carries an explicit negation.
"""

from fastpisa.output.json_output import build_interfaces_json, build_assembly_json

__all__ = ["build_interfaces_json", "build_assembly_json"]
