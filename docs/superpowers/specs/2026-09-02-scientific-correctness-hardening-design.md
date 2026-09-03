# Scientific Correctness Hardening Design

Date: 2026-09-02. Status: approved for implementation.

## Goal

Correct the scientifically material, local defects found in the fastPISA
Streamlit audit without adding biological-assembly generation, prediction
confidence workflows, or other new subsystems.

## Scope boundary

This pass preserves the existing single-core architecture and calibrated
energy/scoring constants. It does not add symmetry expansion, biological
assembly selection, PAE upload, or new calibration. Existing JSON keys and
the dominant `AtomContact.bond_type` remain compatible.

## 1. Coordinate parsing

- PDB and mmCIF inputs use only the first coordinate model.
- Alternate conformers are resolved once per atom site. A blank conformer is
  preferred; otherwise the conformer with greatest occupancy is selected,
  with the lexicographically first altloc as a deterministic tie-break.
- mmCIF parsing preserves author chain, author sequence, insertion code,
  label chain, label sequence, altloc, occupancy, B-factor, and ATOM/HETATM
  status. The `_atom_site` table is the source of truth where Gemmi's
  high-level structure view loses label/auth distinctions.
- `.cif`, `.cif.gz`, `.mmcif`, and `.mmcif.gz` use the mmCIF parser in both
  the core and the high-level API.
- A PDB ATOM/HETATM record without columns 77-78 populated is rejected with a
  line-specific error. The element is never inferred from the atom name.

## 2. Surface and residue accounting

- Hydrogens and deuteriums remain available for hydrogen-bond geometry but
  receive no isolated, paired, or combined surface area.
- Combined-structure ASA is calculated on the heavy-atom subset with global
  atom indices, matching the already-heavy molecule masks.
- Per-residue `asa` is the isolated ASA of every heavy atom in that residue,
  not only atoms whose surface changes at a particular interface.
- When one residue participates in more than one chain-pair interface, its
  isolated ASA is retained once while BSA, solvation contribution, and bond
  participation are accumulated. `fraction_buried` is therefore bounded by
  physically meaningful whole-residue accounting except where pairwise group
  aggregation itself overlaps the same patch; such overlap is not silently
  described as a unique union surface.

## 3. Independent bond classifications

PISA hydrogen-bond, salt-bridge, and disulfide predicates remain independent.
The existing dominant `bond_type` field stays unchanged for compatibility and
visual styling. Reporting and exports obtain bond rows from the independent
bond assignments, so a charged hydrogen-bonded atom pair appears in both the
hydrogen-bond and salt-bridge tables and residue bond counts reflect both.

## 4. App robustness and repeated work

- Excel worksheet names are stripped of forbidden characters, limited to 31
  characters, made non-empty, and deduplicated case-insensitively.
- User labels inserted into Mol* HTML are escaped; serialized MVS data remains
  JSON encoded.
- Auto-detecting a shared side clears any previously computed group digest.
- App exports use stable worksheet identifiers rather than user labels.
- Read-only analyzer helpers reuse an existing analysis result rather than
  recomputing it. Explicit `analyze()` retains its current recompute default.
- Water remains excluded in the Streamlit workflow and is described as such;
  the misleading toggle is removed rather than partially supporting water as
  an interface partner.
- The deprecated Streamlit HTML component call is replaced with the supported
  local-HTML embedding API available in the pinned Streamlit range.

## 5. Scientific communication and provenance

- Area, hydrophobicity P-value, aromatic enrichment, and high-burial residue
  observations are described as structural evidence, not proof of biological
  relevance or energetic hot spots.
- Interfaces involving ligands or ions receive an explicit warning that the
  strongest parity is for polymer-polymer interfaces and that ligand/ion
  estimates have larger observed errors.
- COCOMAPS wording says the contact-map conventions and implemented subset of
  interaction classes are COCOMAPS-compatible; it does not claim complete
  implementation of every COCOMAPS 2.0 class.
- Methods/export provenance records the fastPISA version, input coordinate
  scope (first model; no symmetry generation), probe radius, point density,
  contact cutoff, ligand mode, water policy, and the actual surface backend
  and algorithm detectable at runtime. It must not change the calibrated ASA
  algorithm by calling FreeSASA `Parameters.setAlgorithm`.

## 6. Verification

Each behavior is introduced through a regression test that fails for the
observed defect before production code is changed. Verification includes the
focused parser, surface, report, API, and app tests; the complete offline test
suite; `tests/test_vs_pdbe_pisa.py`; `tests/test_vs_cocomaps2.py`; and
`examples/compare_vs_pisa.py`. Optional external/CCP4 tests may remain skipped
when their documented environment variables are unset.
