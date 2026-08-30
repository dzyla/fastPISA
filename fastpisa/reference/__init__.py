"""Reference data access for validating fastPISA against original PISA."""

from fastpisa.reference.ebi_pisa import (  # noqa: F401
    fetch_pisa_xml, parse_pisa_xml, identity_interfaces,
    load_cached_reference, REFERENCE_DIR,
)
