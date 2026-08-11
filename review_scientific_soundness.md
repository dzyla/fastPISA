# fastPISA — Scientific-Soundness Review
Review date: 2026-08-10

> **STATUS (2026-08-10): all 7 recommended changes below are implemented and
> verified.** Fixes 1–5 land in commit `3eda7a0` (delegate PISA-mode disulfide/H-bond/
> salt-bridge chemistry to the COCOMAPS classifier, remove the double-counted
> electrostatic term, and replace the `4·π·r²−ASA` BSA convention with
> `isolated−combined` ASA). Change 6 (`min_css` significance filter) lands in both
> pipelines + CLI/API. Change 7 (count-invariant CI) lands as `tests/test_pipeline.py`.
> Reproducing the original binary is also fixed: `tests/conftest.py` now writes a
> self-contained PISA config (PISA exits 3 without one), so `test_reproduce_pisa.py`
> runs and passes (was silently SKIPPED — 2026-08-10). Verified: 1ktz interface area
> 483.5 Å² vs original PISA 493.4 (~2%), zero bogus disulfides, BSA now sane
> (541/18057 Å² vs the old 345638 garbage).
Test structure: MeV3920_F4-B05_sample_0.cif (OpenDDE AlphaFold antibody complex, 5 chains A/B/C/D/E, 12384 atoms)
Head-to-head: original CCP4 PISA v2.2.0 binary (/programs/xtal/ccp4-9/bin/pisa)

## Headline result
On the random test CIF, fastPISA runs correctly (both modes parse, detect interfaces, write
JSON), but the PISA-mode **atom-chemistry classifier has serious scientific bugs** that produce
physically impossible results. COCOMAPS mode is much closer to being chemically sound. The
shared interface-*detection* machinery is sound (both modes agree on interface IDs), but
PISA-mode *contact classification* and the *energy/scoring* numbers built on top of it are not
reliable.

Original PISA binary: 10 potential contacts -> 10 interfaces.
fastPISA:           9 interfaces (PISA and COCOMAPS modes, identical IDs).

## CONFIRMED BUGS (verified by direct inspection, not speculation)

### 1. CRITICAL — `is_disulfide` classifies ANY atom pair < 3.0 A as a disulfide  (contacts.py:171-177)
```python
def is_disulfide(atom1_name, atom2_name, distance):
    """Check if a contact is a disulfide bond (S-S)."""
    return distance < DISULFIDE_DISTANCE
```
The function IGNORES atom names and elements. The correct rule (used in the COCOMAPS
classifier, interactions.py:157) is: both atoms are S and both residues are CYS.

Evidence on the test structure:
  - Interface 5 reported **40 "disulfide bonds"**; interface 1 = 39; interface 9 = 10.
  - Directed verification: **ZERO of them are real Cys-Sg <-> Cys-Sg contacts.** The
    "disulfides" are Leu CD2-C, Lys C-N, Arg N-Glu C, Ser O-Arg C, Tyr C-C, Asn-Phe, etc.
  - Only interface 6 even touches a Cys, and it's Cys-Tyr / Ser-Tyr, not Cys-Cys.

Impact: `number_disulfide_bonds` is pure garbage (39-40 impossible S-S bonds on a single
interface). This does NOT feed the energy terms (calculate_contact_energy only sums
hbond+salt_bridge), but it pollutes the JSON output and any downstream consumer that reads
`number_disulfide_bonds`. It also means the codebase's own invariant (COCOMAPS bond classes
are chemically sensible) is violated in PISA-mode output.

### 2. HIGH — H-bond detection requires an explicit H atom (contacts.py:129-143)
PISA-mode `is_hydrogen_bond` only counts a contact as an H-bond if an atom NAME ends in "H"
(i.e. an explicit hydrogen atom exists) AND the partner is N/O. Root cause:
```python
h1 = atom1_name.strip().endswith("H")
if not (h1 or h2): return False
```
OpenDDE / AlphaFold / most cryo-EM and many X-ray models contain **NO H atoms** (verified:
0 of 12384 atoms). So PISA mode finds almost no H-bonds.

Evidence: Interface 1  PISA mode reports **2 hbonds**; COCOMAPS mode (same structure, same
contacts) reports **22 hbonds**. Only 2 found because the single-letter "OH" name
coincidentally ends in "H" and the code mistakes the hydroxyl O for a hydrogen.

Impact: PISA-mode H-bond counts (and the H-bond energy/contact terms that depend on them) are
badly undercounted on H-free structures — the vast majority of modern models.

### 3. HIGH — Salt-bridge detection is far too loose (contacts.py:163-168)
`is_salt_bridge` treats **ANY N-O atom pair < 4.0 A** as a salt bridge:
```python
charged_pairs = {("N", "O"), ("O", "N")}
if (el1, el2) not in charged_pairs: return False
```
This does not check the atoms are charged side-chain atoms (Lys NZ / Arg NH1,NH2 / Asp/Glu
carboxylates). It therefore flags backbone N...O and generic polar N-O contacts as salt
bridges.

Evidence on interface 5: salt bridges include LYSB-N <-> GLNC-OE1, PHEB-O <-> ALAC-N, SERB-N <-
GLYC-O, TYRB-OH <-> PHEC-N — these are backbone/polar contacts, NOT salt bridges. Backbone
amide-carbonyl pairs are H-bond partners double-counted as salt bridges.

Impact: inflates `number_salt_bridges` AND the energy terms — `calculate_contact_energy`
assigns -0.5 to -2.0 kcal/mol per salt bridge (energy.py:85-92), so binding/stabilization
energies are systematically too negative.

### 4. MEDIUM — Binding/stabilization energies double-count & omit terms
- `calculate_binding_energy` (energy.py:98) adds `calculate_contact_energy` (hbond+salt) AND
  then `electrostatic_energy = -0.5 * n_salt` again (salt bridges counted twice).
- `calculate_stabilization_energy` (energy.py:186) = solv + contact, using the same loose salt
  metric, so stabilization is over-stabilized by the salt-bridge bug.
- The ASP table and magnitude choices are calibrated ad-hocs; the skill itself flags these as
  approximations. On the test CIF, solvation energies are implausibly favourable for
  "interfaces" that are really just the 3 antigen copies A/B/C stacking against each other
  (area ~2600 A^2 between copies is a model artifact, not a real interaction).

### 5. MEDIUM — Whole-structure "buried surface area" exceeds accessible area (pipeline.py:128-130)
```
Combined ASA: 73895 A^2,  BSA: 345638 A^2   (BSA is 4.7x ASA!)
```
`assembly_bsa = total_vdw_sphere_area - ASA` where total_vdw = sum(4*pi*r^2) over ALL atoms.
That treats every atom's full sphere as "surface" and calls the difference "buried area", so
the number is meaningless and huge. It feeds `assembly_bsa`, the total-ASA normalisation in
p-Value/CSS, and (per-residue) interface quantities. Same for `interface.py:285` where
per-atom "BSA" = 4*pi*r^2 - ASA.

## WHAT IS SOUND (confirmed)
- Interface *detection* (KD-tree, 5 A cutoff, molecule/mask logic) works; interface IDs are
  identical between modes — the codebase invariant holds.
- Molecule classification by residue composition (not sticky chain.group), water exclusion,
  and element-from-columns-77-78 are all correct per the skill / AGENTS.md.
- The COCOMAPS interaction classifier (interactions.py) uses the chemically correct rules:
  disulfides require Cys-Cys S-S, salt bridges use a charged-atom table, H-bonds use a
  donor/acceptor table. Its populations are sensible (interface 1: 507 polar_vdw, 471 apolar,
  108 ch_pi, 27 clash, 22 hbond, 11 salt).
- Runtime is fine (~2 s including both modes), FreeSASA backend works.

## RECOMMENDED CHANGES (priority order)
1. **Fix `is_disulfide`** to require `element1 == element2 == "S"` and both residues CYS,
   matching the COCOMAPS rule in interactions.py:157. (contacts.py:171)
2. **Fix H-bond detection** to use the same donor/acceptor rule-based approach as
   `_hbond()`/`HBOND_ATOMS_AA` in interactions.py — it does NOT require explicit H atoms.
   De-duplicate: have the PISA-mode classifier delegate to the COCOMAPS classifier so the two
   modes always agree (single source of truth for atom-chemistry).
3. **Fix `is_salt_bridge`** to check the atom is a genuinely charged side-chain atom using the
   `CHARGED_ATOMS` table (ARG NH1/NH2, LYS NZ, HIS ND1/NE2, ASP OD1/OD2, GLU OE1/OE2), not
   "any N-O".
4. **Remove the double-counted electrostatic term** in `calculate_binding_energy`
   (salt bridges currently counted in both contact_energy and electrostatic_energy).
5. **Recompute the per-atom "BSA"/buried-area convention** or stop labeling `4*pi*r^2 - ASA`
   as BSA; either implement a proper probe-expanded surface or explicitly document that the
   assembly-level BSA is a convention-specific number and exclude it from p-value/CSS
   normalisation. Verify ASA/BSA absolute values against the PDBe reference (the skill notes a
   ~3x mismatch).
6. **Add a significance / biological filter** (p-value + CSS threshold, as original PISA does)
   or at least an assembly-symmetry step, now that 9-10 "interfaces" on this test are mostly
   weak/artifact contacts between antigen copies. This is the difference between fastPISA
   listing 9 interfaces and PISA reporting meaningful ones.
7. **Reconcile PISA vs COCOMAPS H-bond/salt/disulfide counts** in CI — add the invariant check
   from AGENTS.md as an actual test (counts, not just IDs) so these regressions are caught.

## Verification guidance
After fixing, re-run on 1ktz.pdb and the F4-B05 CIF and check:
- `number_disulfide_bonds` drops to a small number and matches number of Cys-Sg-Cys-Sg atom
  pairs (should be 0-3 per interface on these structures).
- PISA-mode hbond count ≈ COCOMAPS-mode hbond count on H-free structures.
- Salt bridges restricted to charged side-chain pairs.