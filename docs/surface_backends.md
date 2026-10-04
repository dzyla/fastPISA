# Surface backends: which one runs, and why it does not matter

fastPISA computes solvent-accessible surface area with one of two engines.
Every fitted solvation parameter is applied by whichever engine happens to be
installed, so "they agree" is a claim that has to be measured rather than
assumed. This page records the measurement.

## The two engines

| | reference engine | accelerator |
|---|---|---|
| module | `fastpisa/surface/shrake_rupley.py` | `fastpisa/surface/freesasa_backend.py` |
| algorithm | Shrake–Rupley | Lee–Richards |
| quadrature | `point_density` sphere points (default 480) | 20 slices (pinned) |
| radii | NACCESS/Chothia set, `surface_radius()` | the same, passed in |
| dependency | numpy + scipy | `freesasa` (C library) |

The **Python implementation is the reference**: it defines the
`--point_density` quadrature, and the ASP sigmas are validated against it (see
below). FreeSASA is an accelerator used by default when importable.

Pin one explicitly for reproducibility:

```bash
export FASTPISA_SASA_BACKEND=python     # or freesasa, or auto (default)
```

`PISAInterfaceAnalyzer.analysis_provenance()` and `report.methods_paragraph()`
report the engine, the algorithm and the quadrature that actually ran, so a
methods section quotes the truth rather than an intention.

## Why FreeSASA is pinned to Lee–Richards

Two traps, both of which bit this project:

1. **FreeSASA's library default is Lee–Richards (20 slices), not
   Shrake–Rupley.** Code that only called `setNPoints()` was configuring a
   parameter the active algorithm ignores — so `--point_density` was a silent
   no-op, while the docs claimed Shrake–Rupley. Verified: ASA sums are
   bit-identical for `point_density` 20, 480 and 5000. The algorithm is now set
   explicitly on every call; never inherit a library default you depend on.

2. **FreeSASA 2.2.1's Shrake–Rupley kernel segfaults on spatially sparse
   inputs.** Two atoms 20 Å apart is enough, at every point count, with both
   `list` and `ndarray` input. fastPISA's per-molecule and per-pair ASA calls
   produce exactly such subsets routinely, and a segfault cannot be caught — it
   would abort a user's analysis mid-run.

`tests/test_sasa_backends.py::test_freesasa_survives_spatially_sparse_subsets`
crashes the whole suite if anyone switches the backend to Shrake–Rupley, which
is the alarm we want rather than a silent regression.

A single atom is short-circuited to the closed form `4π(r + probe)²` in both
engines: it is exact (there is nothing to occlude it), faster, and a lone metal
ion (`[ZN]A:301`) is a routine molecule here.

## Correctness of the Python engine

Pinned to closed-form geometry, not to the other backend — so a bug cannot
hide behind an agreeing pair:

* a lone atom returns `4π(r + probe)²` exactly;
* two atoms at separation `d` hide exactly the analytic spherical cap
  `2πR₁h`, `h = R₁ − (d² + R₁² − R₂²)/2d` with `Rᵢ = rᵢ + probe`;
* an atom enclosed by a shell of neighbours returns zero;
* two K⁺ ions (r = 2.75 Å) still shadow each other at 7.95 Å, inside
  `r₁ + r₂ + 2·probe = 8.3 Å`.

Two bugs this pins down, both fixed:

* the burial test compared a probe-centre point against the **bare** van der
  Waals sphere `r_j` instead of `r_j + probe`. A test point is the *centre* of
  a probe sphere, so it is inaccessible within `r_j + probe`. Measured on 400
  atoms of 1ktz: 27037 Å² instead of 4508 Å² — **ASA ~6× too large**, and with
  it every ΔG, P-value, CSS and BSA.
* the neighbour cutoff was `2·r_max + probe + 1.0`, short of the geometric
  requirement `2·r_max + 2·probe`, dropping real occluders. The cutoff is now
  clamped up to the requirement: it is a performance hint, and a short one is a
  physics error rather than a speed/accuracy trade.

## Agreement, measured

Per-atom ASA over all heavy atoms of 1ktz (1493 atoms), Python Shrake–Rupley
480 points vs FreeSASA Lee–Richards 20 slices:

| quantity | value |
|---|---|
| total ASA | 11079.7 vs 11079.2 Å² (**0.00%**) |
| median per-atom &#124;Δ&#124; | 0.09 Å² |
| 95th percentile | 0.79 Å² |
| max | 1.58 Å² |

For reference, FreeSASA's own Shrake–Rupley at 5000 points gives 11490.6 Å² on
the same atoms where Lee–Richards/20 gives 11490.9 Å² (0.002%), so neither
engine's quadrature is the limiting error.

## The fitted sigmas hold under either engine

The sigmas were originally fitted with FreeSASA. The full offline accuracy
benchmark against original PISA (`tests/test_vs_pdbe_pisa.py`,
`fastpisa.reference.compare`) was re-run under each engine on identical inputs
— same 265 matched identity interfaces:

| metric vs original PISA | FreeSASA | Python | Δ |
|---|---|---|---|
| interface area, median rel. err | 2.11% | 2.10% | −0.02% |
| polymer–polymer area, median rel. err | 1.36% | 1.53% | +0.17% |
| ΔG Pearson r | 0.9684 | 0.9681 | −0.0002 |
| ΔG median &#124;err&#124; (kcal/mol) | 0.472 | 0.475 | +0.003 |
| polymer–polymer ΔG Pearson r | 0.9971 | 0.9972 | +0.0001 |
| stabilization Pearson r | 0.9814 | 0.9813 | −0.0001 |
| P-value median &#124;err&#124; | 0.079 | 0.082 | +0.003 |
| CSS Spearman | 0.728 | 0.729 | +0.001 |
| H-bonds, mean &#124;diff&#124; | 0.593 | 0.593 | 0.000 |
| H-bonds within ±1 | 91.7% | 91.7% | 0.000 |
| salt bridges, mean &#124;diff&#124; | 0.045 | 0.045 | 0.000 |
| disulfides exact | 100% | 100% | 0.000 |
| wall clock | 13.4 s | 175.7 s | ×13 |

**No separate calibration is needed for the Python-only path.** The two engines
are statistically indistinguishable against PISA — the differences are far
smaller than fastPISA's own error against PISA — and the bond counts are
identical, because bonds are geometric and do not depend on the surface engine
at all. Reproduce with:

```bash
FASTPISA_SASA_BACKEND=python  pytest tests/test_vs_pdbe_pisa.py -q
FASTPISA_SASA_BACKEND=freesasa pytest tests/test_vs_pdbe_pisa.py -q
```

Both legs run in CI on every supported Python version.

## The cost of going without FreeSASA

~13× slower on the benchmark above (175.7 s vs 13.4 s for 37 entries). The
difference is the ASA kernel itself, not Python overhead around it: the
subset coordinate packing was measured at ~2% of a call, and reusing the
caller's precomputed coordinate and radius arrays (which the Python engine now
does) made no measurable difference to the FreeSASA path. Install `freesasa`
for throughput; use the Python engine when a wheel is unavailable, when a
dependency-free install matters, or to check a surprising number against a
second implementation.
