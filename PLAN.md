# fastPISA: Local PISA Reproduction Plan

## Overview

PISA (Protein Interfaces, Surfaces and Assemblies) by Krissinel & Henrick (2007)
analyzes macromolecular crystal structures to identify interfaces, calculate their
properties, and predict biological assemblies. This plan describes reproducing PISA's
core output as a standalone Python tool that reads a PDB file and emits the same JSON
schema as the PDBe PISA API (`{pdb_id}-assembly{assembly_id}.json` and
`{pdb_id}-assembly{assembly_id}-interfaces.json`).

The tool also provides a **COCOMAPS mode** (COCOMAPS 2.0, Chawla et al. 2025,
`https://pmc.ncbi.nlm.nih.gov/articles/PMC12684709/`), a complementary residue-residue
contact-map analysis that identifies the same interfaces as PISA (sharing the same
interface-detection and surface machinery) and reports each interface as a residue
contact map with atomic interaction-type classification (H-bond, salt bridge, pi-pi,
cation-pi, ch-pi, ...).

The original PISA is a large C++ codebase (~100+ files) using crystallographic symmetry,
sphere-rolling surface area, atomic solvation parameters (ASPs), and statistical scoring.
Full bit-exact reproduction is infeasible without the original binary and its ASP data files,
but the **core algorithm** (ASA/BSA, interface area, solvation/binding energy, entropy,
P-value, CSS, assembly prediction) can be faithfully re-implemented in Python with numpy.

---

## Target Output Format

Match the PDBe PISA JSON schema exactly:

### `assembly.json`
```json
{
  "PISA": {
    "pdb_id": "6nxr",
    "assembly_id": "1",
    "pisa_version": "2.0",
    "assembly": {
      "id": "1",
      "size": "2",
      "score": "",
      "macromolecular_size": "2",
      "dissociation_energy": 15.61,
      "accessible_surface_area": 19395.3,
      "buried_surface_area": 3514.17,
      "entropy": 12.98,
      "dissociation_area": 1427.5,
      "solvation_energy_gain": -35.28,
      "number_of_uc": "0",
      "number_of_dissociated_elements": "2",
      "symmetry_number": "1",
      "formula": "A(2)a(2)b(2)",
      "composition": "A-2A[NA](2)[GOL](2)",
      "R350": ""
    }
  }
}
```

### `interfaces.json`
```json
{
  "PISA": {
    "pdb_id": "6nxr",
    "assembly_id": "1",
    "pisa_version": "2.0",
    "assembly": {
      "mmsize": "2",
      "dissociation_energy": 15.61,
      "accessible_surface_area": 19395.3,
      "buried_surface_area": 3514.17,
      "entropy": 12.98,
      "dissociation_area": 1427.5,
      "solvation_energy_gain": -35.28,
      "formula": "A(2)a(2)b(2)",
      "composition": "A-2A[NA](2)[GOL](2)",
      "interface_count": 1,
      "interfaces": [{
        "interface_id": "1",
        "interface_area": 1427.5,
        "solvation_energy": -18.22,
        "stabilization_energy": -28.59,
        "p_value": 0.095,
        "number_interface_residues": 248,
        "number_hydrogen_bonds": 20,
        "number_covalent_bonds": 0,
        "number_disulfide_bonds": 0,
        "number_salt_bridges": 4,
        "number_other_bonds": 261,
        "hydrogen_bonds": { "bond_distances": [...], "atom_site_1_*": [...], "atom_site_2_*": [...] },
        "salt_bridges": { ... },
        "disulfide_bonds": { ... },
        "covalent_bonds": { ... },
        "other_bonds": { ... },
        "molecules": [{
          "molecule_id": 1,
          "molecule_class": "Protein",
          "chain_id": "A-2",
          "residue_label_comp_ids": [...],
          "residue_seq_ids": [...],
          "residue_label_seq_ids": [...],
          "residue_ins_codes": [...],
          "residue_bonds": [...],
          "solvation_energies": [...],
          "accessible_surface_areas": [...],
          "buried_surface_areas": [...],
          "auth_asym_id": "A",
          "int_natoms": 476,
          "int_nres": 248
        }]
      }]
    }
  }
}
```

---

## Architecture

```
fastpisa/
├── README.md
├── pyproject.toml
├── requirements.txt
├── fastpisa/
│   ├── __init__.py
│   ├── parser/
│   │   ├── __init__.py
│   │   ├── pdb_parser.py        # PDB/mmCIF parsing → atom/chains/ligands
│   │   └── atom.py              # Atom dataclass
│   ├── surface/
│   │   ├── __init__.py
│   │   ├── shrake_rupley.py     # ASA calculation (sphere rolling)
│   │   └── per_residue.py       # Per-residue ASA/BSA aggregation
│   ├── interface/
│   │   ├── __init__.py
│   │   ├── contacts.py          # Molecule detection, masks, H-bonds/salt bridges/disulfides
│   │   └── interface.py         # Interface area
│   ├── codomaps/                # <---- COCOMAPS 2.0 mode
│   │   ├── __init__.py
│   │   ├── interactions.py      # Atomic interaction-type classifier (14 types)
│   │   ├── contact_map.py       # Residue-residue contact map + contact matrix
│   │   └── pipeline.py          # COCOMAPS analysis pipeline (reuses PISA shared modules)
│   ├── energy/
│   │   ├── __init__.py
│   │   ├── asp_table.py         # Atomic solvation parameters
│   │   └── energy.py            # ΔGsolv, ΔGint, TΔS, ΔGdiss
│   ├── scoring/
│   │   ├── __init__.py
│   │   └── scoring.py           # P-value, CSS
│   ├── assembly/
│   │   ├── __init__.py
│   │   └── symmetry.py          # Crystallographic symmetry operators
│   ├── output/
│   │   ├── __init__.py
│   │   ├── json_output.py       # Build assembly.json + interfaces.json
│   │   └── schema.py            # Pydantic models matching PDBe schema
│   └── cli.py                   # CLI entry point (--mode {pisa,cocomaps})
├── tests/
└── data/
    └── asp_values.py            # Precomputed ASP table
```

## COCOMAPS mode

The COCOMAPS mode (`fastpisa/cocomaps/`) implements the analysis approach of COCOMAPS 2.0
(Chawla et al., Bioinformatics 2025), a complementary view of the same interfaces:

- **Same interfaces**: reuses the identical interface-atom detection (5 A cutoff), molecule
  detection, and Shrake-Rupley ASA/BSA machinery as the PISA pipeline, so both modes always
  find the same interfaces for a structure. The COCOMAPS output JSON is a superset of the
  PISA JSON schema (all `assembly` / `interfaces` fields present) plus an
  `interface_contact_map` field on each interface.
- **Residue-residue contact map**: for each interface, a contact map of which residue of
  molecule 1 contacts which residue of molecule 2, with per-pair min distance, contact count,
  and dominant interaction type.
- **Interaction-type classification** (`interactions.py`): each atom-atom contact is typed
  (hydrogen_bond, weak_hbond, salt_bridge, disulfide, halogen_bond, pi_pi, cation_pi, ch_pi,
  polar_vdw, apolar_vdw, water_mediated, metal_mediated, clash, distal). Classification rules
  derive from the COCOMAPS 2.0 interaction criteria; without an external H-add / HBPLUS
  backend this is a rule-based subset of COCOMAPS 2.0's 16 classes.
- **Contact matrix** (`build_contact_matrix`): a dense 2D residue-residue contact matrix
  (contact map) for inter-chain residue pairs.

CLI: `python -m fastpisa.cli <pdb> --mode cocomaps`.

### Key interaction classifier notes
- Specific interactions (disulfide, salt_bridge, halogen_bond, metal_mediated, hydrogen_bond)
  are evaluated before the generic `clash` fallback so that genuine short-range interactions
  are not miscalled as steric clashes.
- H-bond detection requires a donor on at least one side; two acceptor-only oxygens are not an
  H-bond.

### Known surface-area convention issue
The combined-structure BSA (total vdW area - ASA) currently reports 0.0 for many structures
because the probe-sphere-based ASA includes the full 4π(r+probe)² surface; this is a
pre-existing PISA-mode convention mismatch with PDBe (ASA ~3x PDBe). It affects the numerical
value of `interface_area`/energies identically in both modes and does not change which
interfaces are detected. Reproducing the exact PISA surface convention is tracked as a
separate task.

---

## Core Algorithm Modules

### 1. PDB/mmCIF Parser (`fastpisa/parser/`)
- Parse ATOM/HETATM records
- Identify protein chains, RNA/DNA chains, ligands
- Assign `auth_asym_id`, `auth_seq_id`, `label_asym_id`, `label_seq_id`
- Extract residue info: name, sequence number, insertion code
- Support standard amino acids, nucleic acids, and common ligands

### 2. Surface Area Calculation (`fastpisa/surface/`)
- Implement **Shrake-Rupley** algorithm: place a probe sphere (r=1.4 Å) on each atom,
  roll over the van der Waals surface, integrate accessible area
- Use numpy for vectorized sphere intersection calculations
- Compute per-atom ASA → per-residue ASA/BSA
- ASP-based solvation energy: ΔGsolv = Σ(ASP_k × σ_k) where σ_k is buried area of atom type k

### 3. Interface Detection (`fastpisa/interface/`)
- Interface atoms: atoms within 5 Å (default) of an atom in another chain
- Interface area = (ASA_A + ASA_B - ASA_AB) / 2  (per-interface buried area)
- Per-residue BSA and ASA for interface residues
- Atom-atom contacts:
  - Hydrogen bonds: distance < 3.5 Å between donor-H and acceptor
  - Salt bridges: distance < 4.0 Å between opposite charges
  - Disulfide bonds: Cys S-S distance < 3.0 Å
  - Other bonds: all other contacts within cutoff (3.5–5.0 Å)

### 4. Energy Calculation (`fastpisa/energy/`)
- **Solvation energy (ΔGsolv)**: Σ(ASP_k × buried_area_k) — negative = favorable
- **Contact energy (ΔGcont)**: proportional to interface area, approximated from atom-atom contacts
- **Electrostatic energy (ΔGes)**: simplified Coulombic term for charged atoms
- **Binding energy (ΔGint)**: ΔGsolv + ΔGcont + ΔGes
- **Entropy (TΔS)**: related to buried surface area and complex size

### 5. Scoring (`fastpisa/scoring/`)
- **P-value**: probability that the interface's solvation energy is due to chance.
  Uses the statistical distribution of interface energies from the Krissinel 2007 paper.
  P ≈ 0.5 means typical; P < 0.5 means more hydrophobic than expected; P > 0.5 less.
- **CSS (complexation significance score)**: combines interface area, ΔGint, P-value,
  and number of contacts to score biological significance.

### 6. Assembly Prediction (`fastpisa/assembly/`)
- Read crystallographic space group from PDB HEADER/CRYST1 records
- Apply symmetry operators to generate neighboring ASU copies
- Build contact graph between symmetry copies
- Predict assemblies by finding clusters of stable interfaces
- Calculate dissociation energy for each assembly
- Use Stock-based method: concentration × symmetry × stability

### 7. JSON Output (`fastpisa/output/`)
- Build `assembly.json` with all assembly-level metrics
- Build `interfaces.json` with per-interface details including:
  - Per-residue ASA, BSA, solvation energy
  - Per-bond atom-level details (H-bonds, salt bridges, disulfides, other bonds)
  - Molecule-level interface atom/residue counts

---

## Implementation Priorities

### Phase 1 (core, MVP):
- PDB parser
- ASA/BSA calculation (Shrake-Rupley with numpy)
- ASP table
- Interface detection
- Solvation energy, binding energy, entropy
- P-value, CSS
- JSON output (assembly.json + interfaces.json)
- Per-residue data in interfaces

### Phase 2 (enhanced):
- Crystallographic symmetry & assembly prediction
- H-bond / salt bridge / disulfide / other bond detection
- Per-bond atom-level detail in JSON
- mmCIF parser
- Extended data (-list) parsing

### Phase 3 (refinement):
- Validation against example PDB structures
- Performance optimization
- CLI interface with options (--asis, ligand handling)

---

## Key Design Decisions

1. **Python + numpy**: Fast enough for typical PDB structures (~10K atoms); vectorized
   surface area calculation.
2. **Pure PDB parser**: No external dependencies for basic parsing; optional `gemmi` for mmCIF.
3. **ASP table**: Use literature values from the Krissinel & Henrick 2007 paper and
   supplementary tables.
4. **Pydantic models**: Enforce the exact PDBe PISA JSON schema for output validation.
5. **Modular**: Each algorithm component is independently testable.

---

## Dependencies
- numpy ≥ 1.21
- scipy ≥ 1.7
- pydantic ≥ 2.0 (for schema validation)
- Optional: gemmi (for mmCIF parsing)
- Optional: networkx (for assembly graph)

---

## Testing Strategy
- Compare ASA/BSA values against known structures
- Compare interface counts, H-bond counts against PDBe PISA API outputs
- Validate JSON schema with pydantic
- Unit tests for each module