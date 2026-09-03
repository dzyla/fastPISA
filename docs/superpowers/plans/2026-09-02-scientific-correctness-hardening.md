# Scientific Correctness Hardening Implementation Plan

Execution status: complete on 2026-09-02. All six tasks were implemented;
the full suite passed with 122 tests and 12 documented optional skips.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct scientifically material parsing, surface-accounting, bond-reporting, and Streamlit robustness defects without changing calibrated fastPISA physics or adding new analysis subsystems.

**Architecture:** Keep the existing single analysis core and public output schema. Normalize coordinate records at the parser boundary, correct heavy-atom and whole-residue accounting in existing surface functions, derive report rows from independent bond predicates while retaining the dominant contact label, and place presentation-only validation in pure app helpers.

**Tech Stack:** Python 3.9+, NumPy, SciPy, Gemmi, FreeSASA/pure-Python ASA fallback, pandas, openpyxl, Streamlit, pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-scientific-correctness-hardening-design.md`

## Global Constraints

- All modes use the same `fastpisa/core.py` interface detection and energy calculation.
- Interfaces retain PISA pair-dASA semantics.
- Do not modify calibrated ASP sigmas, per-bond constants, P-value scaling, or CSS logistic coefficients.
- Hydrogen and deuterium atoms stay available for bond geometry but carry no surface.
- PDB elements come only from columns 77-78.
- Do not call FreeSASA `Parameters.setAlgorithm`.
- Biological assemblies, symmetry expansion, and PAE upload are out of scope.
- Preserve existing JSON keys and `AtomContact.bond_type` compatibility.

---

### Task 1: Normalize PDB and mmCIF coordinate records

**Files:**
- Modify: `fastpisa/parser/pdb_parser.py`
- Modify: `fastpisa/core.py`
- Modify: `fastpisa/api.py`
- Create: `tests/test_parser_models_altlocs.py`

**Interfaces:**
- Produces: `_is_mmcif_path(path) -> bool` and parser output containing one selected atom per `(chain, residue, atom name)` site from the first model.
- Preserves: `parse_pdb(path) -> PDBStructure` and `parse_mmcif(path) -> PDBStructure`.

- [ ] **Step 1: Write failing PDB parser tests**

Create synthetic fixed-column PDB records and assert that parsing stops after
the first `ENDMDL`, chooses blank altloc over named conformers, otherwise
chooses greatest occupancy with an alphabetical tie-break, and raises
`ValueError` containing the source line number when columns 77-78 are blank.

```python
def test_pdb_uses_first_model_and_resolves_altlocs(tmp_path):
    path = write_pdb(tmp_path, [
        atom_line(1, "CA", "A", 0.40, 1.0),
        atom_line(2, "CA", "B", 0.60, 2.0),
        atom_line(3, "N", " ", 0.20, 3.0),
        atom_line(4, "N", "A", 0.80, 4.0),
        "ENDMDL\n",
        atom_line(5, "C", " ", 1.00, 99.0),
    ], model=True)
    atoms = parse_pdb(path).atoms
    assert [(a.atom_name, a.altloc, a.x) for a in atoms] == [
        ("CA", "B", 2.0), ("N", " ", 3.0)]

def test_pdb_rejects_missing_element_column(tmp_path):
    path = write_pdb(tmp_path, [atom_line(1, "CA", " ", 1.0, 1.0, element="")])
    with pytest.raises(ValueError, match=r"line 1.*columns 77-78"):
        parse_pdb(path)
```

- [ ] **Step 2: Run the PDB tests and verify the observed failures**

Run: `pytest tests/test_parser_models_altlocs.py -k pdb -q`

Expected: first-model/altloc assertions fail because every record is retained;
missing-element assertion fails because the atom-name fallback is used.

- [ ] **Step 3: Implement PDB first-model, element validation, and altloc selection**

Track model state while reading, collect atoms keyed by
`(chain_id, res_seq, icode, res_name, atom_name)`, and replace a selected
record only according to this literal ranking:

```python
def _altloc_rank(atom: Atom) -> tuple:
    return (atom.altloc == " ", atom.occupancy, _reverse_alpha(atom.altloc))
```

Implement the tie-break without changing atom coordinates or inventing an
element. Raise a clear line-specific `ValueError` for a blank element field.

- [ ] **Step 4: Run the PDB parser tests**

Run: `pytest tests/test_parser_models_altlocs.py -k pdb -q`

Expected: PASS.

- [ ] **Step 5: Write failing mmCIF parser and suffix-dispatch tests**

Use a minimal `_atom_site` loop with two model numbers, distinct auth/label
chain and sequence identifiers, insertion code, altloc, occupancy, and
ATOM/HETATM records. Assert the first model and altloc rule, all identifiers,
and successful analysis dispatch for `.mmcif` and `.mmcif.gz` paths.

```python
def test_mmcif_preserves_atom_site_identifiers_and_first_model(tmp_path):
    atoms = parse_mmcif(write_cif(tmp_path)).atoms
    assert len(atoms) == 2
    ca = next(a for a in atoms if a.atom_name == "CA")
    assert (ca.auth_asym_id, ca.label_asym_id) == ("A", "AA")
    assert (ca.auth_seq_id, ca.label_seq_id, ca.icode, ca.altloc) == (7, 3, "B", "A")
    assert ca.group == "ATOM"

@pytest.mark.parametrize("suffix", [".mmcif", ".mmcif.gz"])
def test_analyzer_dispatches_mmcif_suffix(monkeypatch, tmp_path, suffix):
    # Patch only the core analysis boundary and assert the selected parser via
    # a valid minimal file; do not mock parser output.
```

- [ ] **Step 6: Run the mmCIF tests and verify the observed failures**

Run: `pytest tests/test_parser_models_altlocs.py -k 'mmcif or suffix' -q`

Expected: identifier/model assertions fail and `.mmcif` is dispatched as PDB.

- [ ] **Step 7: Make `_atom_site` the mmCIF coordinate source and centralize suffix dispatch**

Read these columns with `block.find("_atom_site.", tags)`:

```python
tags = [
    "group_PDB", "id", "type_symbol", "label_atom_id",
    "label_alt_id", "label_comp_id", "label_asym_id", "label_seq_id",
    "Cartn_x", "Cartn_y", "Cartn_z", "occupancy", "B_iso_or_equiv",
    "auth_seq_id", "auth_comp_id", "auth_asym_id", "auth_atom_id",
    "pdbx_PDB_ins_code", "pdbx_PDB_model_num",
]
```

Normalize `.` and `?` to missing values, select the lowest encountered model,
apply the same altloc ranking as PDB, and add one `_is_mmcif_path` suffix
predicate used by core and API parsing.

- [ ] **Step 8: Run parser and existing parsing tests**

Run: `pytest tests/test_parser_models_altlocs.py tests/test_ligand_mode.py tests/test_pisa_fidelity.py -q`

Expected: PASS.

### Task 2: Enforce heavy-atom and whole-residue ASA accounting

**Files:**
- Modify: `fastpisa/core.py`
- Modify: `fastpisa/surface/per_residue.py`
- Modify: `fastpisa/report.py`
- Create: `tests/test_surface_accounting.py`
- Modify: `tests/test_report.py`

**Interfaces:**
- Changes: `compute_per_residue_surface(..., mol_atoms)` now uses `mol_atoms`
  to sum isolated ASA for every heavy atom of interface residues.
- Preserves: all public interface-area and JSON field names.

- [ ] **Step 1: Write a failing explicit-hydrogen assembly ASA test**

Analyze a tiny two-chain PDB twice, once with explicit hydrogens and once with
those hydrogen records removed. Use a geometry in which hydrogen coordinates
do not alter heavy-atom positions and assert identical assembly ASA/BSA and
interface area within the ASA backend tolerance.

```python
assert with_h["assembly"]["assembly"]["accessible_surface_area"] == pytest.approx(
    without_h["assembly"]["assembly"]["accessible_surface_area"], abs=0.05)
assert with_h["interfaces_obj"][0].interface_area == pytest.approx(
    without_h["interfaces_obj"][0].interface_area, abs=0.05)
```

- [ ] **Step 2: Run the hydrogen test and verify it fails on assembly ASA**

Run: `pytest tests/test_surface_accounting.py -k hydrogen -q`

Expected: interface area agrees but assembly ASA/BSA differs.

- [ ] **Step 3: Calculate combined ASA over globally indexed heavy atoms only**

Replace the combined ASA call with:

```python
heavy_ids = np.flatnonzero(heavy).tolist()
asa_combined = calculate_asa(
    atoms=[atoms[i] for i in heavy_ids], atom_indices=heavy_ids, **asa_kwargs)
```

Keep the full atom list and KD tree for explicit-H bond geometry.

- [ ] **Step 4: Run the hydrogen test**

Run: `pytest tests/test_surface_accounting.py -k hydrogen -q`

Expected: PASS.

- [ ] **Step 5: Write a failing whole-residue isolated-ASA test**

Create a unit fixture where one residue has one buried/interface atom and one
non-interface atom with known isolated ASA. Assert residue ASA includes both
atoms while BSA and solvation include only the interface burial.

```python
result = compute_per_residue_surface(atoms, {0: 10.0, 1: 20.0}, {0: 4.0}, {0}, [0, 1])
assert result["accessible_surface_areas"] == [30.0]
assert result["buried_surface_areas"] == [4.0]
```

- [ ] **Step 6: Run the residue test and verify it fails with ASA 10.0**

Run: `pytest tests/test_surface_accounting.py -k whole_residue -q`

Expected: FAIL with accessible ASA 10.0 instead of 30.0.

- [ ] **Step 7: Group all molecule atoms for selected interface residues**

Build the selected residue-key set from `interface_atom_indices`, then group
indices from `mol_atoms` whose residue key is selected. Continue computing BSA
from `atom_bsa_buried`, so non-interface atoms contribute zero burial.

- [ ] **Step 8: Write and fix a multi-pair residue merge regression**

Use cached `1vfb` with group A against B+C. Assert A:TYR50 keeps the maximum
identical isolated ASA across contributing pair records rather than the first
pair's partial value, while BSA remains the sum of pair contributions.

```python
tyr50 = next(r for r in gi.residues_side1 if (r.chain, r.seq) == ("A", "50"))
pair_asa = [residue_asa(p, "A", "50") for p in gi.pairs]
assert tyr50.asa == pytest.approx(max(pair_asa), abs=0.01)
assert tyr50.bsa <= sum(pair_asa) + 0.1
```

When merging duplicate `ResidueEntry` values, set `asa = max(old.asa,
new.asa)` and sum BSA/dG/bonds.

- [ ] **Step 9: Run surface and report tests**

Run: `pytest tests/test_surface_accounting.py tests/test_report.py -q`

Expected: PASS.

### Task 3: Export independent PISA bond predicates

**Files:**
- Modify: `fastpisa/interface/contacts.py`
- Modify: `fastpisa/core.py`
- Modify: `fastpisa/report.py`
- Modify: `app/molstar_view.py`
- Modify: `tests/test_report.py`
- Modify: `tests/test_app_components.py`

**Interfaces:**
- Adds: `AtomContact.bond_types: tuple[str, ...]` with a compatibility default.
- Preserves: dominant `AtomContact.bond_type` and existing JSON `bond_type`.
- Changes: report bond iteration yields one view row per independent class.

- [ ] **Step 1: Strengthen the existing 1brs report regression**

Change the bounded bond-row assertion to exact independent totals and require
at least one atom pair to appear twice with different `type` values.

```python
assert len(bt) == gi.n_hbonds + gi.n_salt_bridges + gi.n_disulfides
dual = bt.groupby(["chain 1", "residue 1", "atom 1", "chain 2", "residue 2", "atom 2"])["type"].nunique()
assert dual.max() >= 2
```

- [ ] **Step 2: Run the bond table test and verify it fails**

Run: `pytest tests/test_report.py::test_tables_and_prose -q`

Expected: FAIL because dominant labels hide dual H-bond/salt-bridge rows.

- [ ] **Step 3: Store independent flags without changing dominant labels**

Populate `bond_types` from the existing `bond_flags` result in priority-free
order `("hbond", "salt_bridge", "disulfide")`; retain current priority for
`bond_type`. Include `bond_types` only in internal/dataclass serialization
where it cannot alter the pinned PISA JSON schema.

- [ ] **Step 4: Expand report bond rows by independent type**

Add a lightweight copied contact or `(contact, kind)` iterator used by
`GroupInterface.bonds`, `bonds_table`, residue bond counts, and Mol* bond
styling. Each requested independent class gets exactly one row/primitive.

- [ ] **Step 5: Run bond, energy, and viewer regressions**

Run: `pytest tests/test_report.py tests/test_app_components.py tests/test_pipeline.py tests/test_improvements.py -q`

Expected: PASS with existing PISA summary counts unchanged.

### Task 4: Harden app exports, state, HTML, and analyzer accessors

**Files:**
- Modify: `app/app_helpers.py`
- Modify: `app/molstar_view.py`
- Modify: `app/streamlit_app.py`
- Modify: `fastpisa/api.py`
- Modify: `tests/test_app_components.py`
- Create: `tests/test_api_cached_accessors.py`

**Interfaces:**
- Adds: `safe_sheet_names(names: Iterable[str]) -> list[str]`.
- Adds: `apply_shared_side(cx: dict, ref: dict, inventory: list[dict], matches: list[tuple]) -> None` as a pure state helper.
- Preserves: `excel_bytes(sheets) -> bytes`.

- [ ] **Step 1: Write failing worksheet-name tests**

Assert forbidden-character removal, 31-character truncation, non-empty
fallback, and case-insensitive deduplication. Open the returned bytes with
openpyxl to assert exact names and successful workbook creation.

```python
data = excel_bytes({"bad/name": df, "BAD\\NAME": df, "": df})
book = openpyxl.load_workbook(io.BytesIO(data), read_only=True)
assert book.sheetnames == ["bad name", "BAD NAME (2)", "Sheet"]
```

- [ ] **Step 2: Run the worksheet test and verify the openpyxl failure**

Run: `pytest tests/test_app_components.py -k excel -q`

Expected: FAIL with an invalid-character worksheet-title error.

- [ ] **Step 3: Implement deterministic worksheet normalization**

Replace `[]:*?/\\` with spaces, collapse whitespace, truncate after adding a
` (N)` suffix, and deduplicate with `casefold()`. Use normalized names only at
the workbook boundary.

- [ ] **Step 4: Write and fix Mol* HTML escaping regression**

Pass a label/name containing `<script>&"` through `interface_view_html` and
`comparison_view_html`; assert no raw script tag occurs in legend HTML while
the MVS JSON remains parseable. Use `html.escape(str(name), quote=True)` for
legend text and never hand-escape the JSON payload.

- [ ] **Step 5: Write and fix shared-side stale-state regression**

Move the state mutation into `apply_shared_side`; assert it sets both groups
and labels and removes an existing `gi` key. Call it from the Streamlit button
handler before `st.rerun()`.

- [ ] **Step 6: Write failing no-recompute accessor tests**

Analyze a real small structure once, replace the analyzer instance's analysis
boundary with a function that raises, then call `to_dataframe`,
`to_residue_dataframe`, `hot_spot_residues`, `summary`, confidence score
readers, and JSON properties as applicable. The already-populated result must
be reused.

- [ ] **Step 7: Run the accessor test and verify repeated analysis fails**

Run: `pytest tests/test_api_cached_accessors.py -q`

Expected: FAIL at the first helper that calls `analyze()` with its default.

- [ ] **Step 8: Use `analyze(recompute=False)` in read-only helpers**

Change only convenience/accessor calls. Keep explicit user calls to
`analyze()` recomputing by default for backwards compatibility.

- [ ] **Step 9: Make Streamlit presentation changes**

Use fixed export sheet keys (`side 1 residues`, `side 2 residues`), describe
water as always excluded in the app and pass `exclude_water=True`, and replace
deprecated local component embedding with the supported Streamlit API after
checking the installed signature. Do not change calculation defaults outside
the app.

- [ ] **Step 10: Run app and API tests**

Run: `pytest tests/test_app_components.py tests/test_api_cached_accessors.py tests/test_pae_viz_batch.py -q`

Expected: PASS.

### Task 5: Qualify scientific claims and record provenance

**Files:**
- Modify: `fastpisa/report.py`
- Modify: `fastpisa/surface/freesasa_backend.py`
- Modify: `app/streamlit_app.py`
- Modify: `app/README.md`
- Modify: `tests/test_report.py`
- Modify: `tests/test_app_components.py`

**Interfaces:**
- Adds: `surface_backend_info() -> dict[str, str]` returning backend,
  algorithm, and optional version without mutating FreeSASA parameters.
- Adds: analysis options/provenance to `GroupInterface.to_dict()` using
  analyzer attributes already available through the result object.

- [ ] **Step 1: Write failing backend-provenance test**

Assert the runtime helper says `python`/`Shrake-Rupley` when FreeSASA is
unavailable and reports the actual `freesasa.Parameters().algorithm()` value
when available. Monkeypatch availability only at the dispatch boundary; do
not call `setAlgorithm`.

- [ ] **Step 2: Run the backend test and verify the helper is absent**

Run: `pytest tests/test_surface_accounting.py -k backend_info -q`

Expected: FAIL on import.

- [ ] **Step 3: Implement read-only surface backend introspection**

Return strings suitable for JSON and prose. Guard version/algorithm lookup so
the pure-Python fallback never requires FreeSASA.

- [ ] **Step 4: Write report behavior assertions**

Assert methods text contains first-model/no-symmetry scope and actual backend;
interpretation avoids `stable biological interface` and `hot-spot candidates`;
COCOMAPS text says `implemented subset`; and a ligand-involving digest emits a
calibration-limit warning.

- [ ] **Step 5: Run report assertions and verify the old claims fail**

Run: `pytest tests/test_report.py tests/test_app_components.py -k 'methods or interpretation or ligand' -q`

Expected: FAIL on the old unqualified wording and missing provenance/warning.

- [ ] **Step 6: Pass analyzer provenance into the group digest**

Attach a plain options dictionary to the analyzer result after analysis and
copy it into `GroupInterface`. Include at least input filename/suffix scope,
probe radius, point density, contact cutoff, ligand mode, water policy,
surface backend, surface algorithm, and fastPISA version. Do not add these
keys to pinned PDBe-compatible interface JSON.

- [ ] **Step 7: Rewrite interpretation and Methods copy**

Use `compatible with`/`supports` language for area and P-value observations,
rename high-BSA rows to `high-burial residues`, state that mutational evidence
is required for energetic hot spots, qualify ligand/ion calibration, and call
the contact classes the implemented COCOMAPS-compatible subset.

- [ ] **Step 8: Update app guide/README and run report/app tests**

Run: `pytest tests/test_report.py tests/test_app_components.py -q`

Expected: PASS.

### Task 6: Full regression and parity verification

**Files:**
- Modify only files required to repair failures caused by Tasks 1-5.

**Interfaces:**
- Consumes all prior tasks; produces fresh verification evidence.

- [ ] **Step 1: Run focused scientific parity tests**

Run: `pytest tests/test_vs_pdbe_pisa.py tests/test_vs_cocomaps2.py -q`

Expected: all cached offline parity tests pass; documented optional fixtures
may skip.

- [ ] **Step 2: Run the complete suite**

Run: `pytest tests/ -q`

Expected: zero failures; only documented optional integration skips.

- [ ] **Step 3: Run the human-readable PISA comparison**

Run: `python examples/compare_vs_pisa.py`

Expected: completes and preserves the established 265 matched-interface
benchmark and polymer-polymer accuracy regime.

- [ ] **Step 4: Smoke-test the Streamlit app**

Run: `streamlit run app/streamlit_app.py --server.headless true --server.port 8765`

Verify the health endpoint returns HTTP 200 and run the existing Streamlit
AppTest flow with `tests/data/1ktz.pdb`, including an invalid side label export.

- [ ] **Step 5: Inspect the final diff and requirement coverage**

Run: `git diff --check` and `git status --short`.

Confirm every design requirement has an implementation/test or is explicitly
listed as deferred by the approved scope. Do not stage or modify the unrelated
untracked `fastpisa/output/schema.py`.
