"""Reproducible refit and honest evaluation of fastPISA's fitted constants.

Three constants sets in fastPISA are *fitted*, not derived:

* ``fastpisa.energy.asp_table.SIGMA`` -- the atomic solvation parameters;
* ``fastpisa.scoring.scoring.P_VALUE_Z_SCALE`` -- the effective-z deflation
  in the P-value model;
* the logistic coefficients in ``fastpisa.scoring.scoring.calculate_css_pisa``.

Until this module existed the values were hardcoded with a prose description
of how they had been obtained; nothing in the repository could reproduce or
audit them. Everything here is driven off a compact **feature table** whose
records are sufficient statistics, so a fit can be re-run offline without
re-deriving surfaces.

Sufficient statistics
---------------------
The solvation gain is linear in the per-class sigmas::

    dG_solv = sum_c  sigma_c * A_c

where ``A_c`` is the interface's buried area in solvation class ``c``. So the
per-class buried areas (``bsa_by_class``) are all a sigma fit needs. The
P-value model needs only the buried-patch moments (``b_sum``, ``b_sq_sum``)
and the class-resolved *surface* area of the two parent molecules
(``surf_asa_by_class``), because the surface sigma mean/variance are
class-weighted averages. :func:`predict_dg` and :func:`predict_p` reproduce
the pipeline's own numbers exactly from these fields.

Evaluation discipline
---------------------
Interfaces inside one PDB entry are not independent observations (shared
chains, shared chemistry, crystallographic copies of the same contact), so
every cross-validation here **groups by entry**: a fold never contains
interfaces from an entry used to fit it. Reporting in-sample fit quality for
a model with 11 free sigmas would materially overstate accuracy.
"""

from __future__ import annotations

import gzip
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from fastpisa.energy.asp_table import SIGMA, DELTA, class_of_fine, sigma_of_fine
from fastpisa.reference.ebi_pisa import (
    REFERENCE_DIR, cached_pdb_path, identity_interfaces, load_cached_reference,
)

#: Solvation classes in a fixed order (the design-matrix column order).
CLASSES: Tuple[str, ...] = tuple(k for k in SIGMA if k != "X")

CALIBRATION_DIR = os.path.join(os.path.dirname(REFERENCE_DIR), "calibration")
FEATURE_TABLE = os.path.join(CALIBRATION_DIR, "features.json.gz")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _key(chain_ids) -> frozenset:
    return frozenset(c.replace(" ", "") for c in chain_ids)


def extract_entry(pdb_id: str) -> List[dict]:
    """Feature + target records for every matched identity interface.

    Returns [] when the entry is not cached. Requires the deposited PDB file
    and the cached EBI PISA XML; runs the shared core once.
    """
    from fastpisa.core import run_core

    ref_all = load_cached_reference(pdb_id)
    pdb = cached_pdb_path(pdb_id)
    if ref_all is None or pdb is None:
        return []
    refk = {_key(m["chain_id"] for m in i["molecules"]): i
            for i in identity_interfaces(ref_all)}
    if not refk:
        return []

    state = run_core(pdb, mode="pisa", collect_calibration=True)
    out = []
    for iface in state.interfaces:
        k = _key(m["chain_id"] for m in iface.molecules)
        ri = refk.get(k)
        if ri is None:
            continue
        cal = iface.calibration
        out.append({
            "pdb_id": pdb_id,
            "pair": "+".join(sorted(k)),
            "is_polymer_pair": "[" not in "+".join(sorted(k)),
            # features (sufficient statistics)
            "bsa_by_class": {c: round(v, 4)
                             for c, v in cal["bsa_by_class"].items()},
            "surf_asa_by_class": {c: round(v, 3)
                                  for c, v in cal["surf_asa_by_class"].items()},
            # fine-typed forms: exact under the hierarchical (class + delta)
            # model and sufficient for refitting either level
            "bsa_by_type": {t: round(v, 4)
                            for t, v in _bsa_by_type(cal["residues"]).items()},
            "surf_asa_by_type": {t: round(v, 3)
                                 for t, v in cal["surf_asa_by_type"].items()},
            "b_sum": round(cal["b_sum"], 4),
            "b_sq_sum": round(cal["b_sq_sum"], 4),
            "area_fp": iface.interface_area,
            # fastPISA bond counts (stab_en = dG + bond energy)
            "nhb_fp": iface.number_hydrogen_bonds,
            "nsb_fp": iface.number_salt_bridges,
            "nss_fp": iface.number_disulfide_bonds,
            # targets from the original PISA engine
            "area_ref": ri["int_area"],
            "dg_ref": ri["int_solv_en"],
            "stab_ref": ri["stab_en"],
            "pv_ref": ri["pvalue"],
            "css_ref": ri["css"],
            "nhb_ref": len(ri["h_bonds"]),
            "nsb_ref": len(ri["salt_bridges"]),
            "nss_ref": len(ri["ss_bonds"]),
        })
    return out


def _bsa_by_type(residues: Sequence[dict]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for r in residues:
        for t, v in r["bsa_by_type"].items():
            out[t] = out.get(t, 0.0) + v
    return out


def save_feature_table(records: List[dict], path: str = FEATURE_TABLE) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as fh:
        json.dump(records, fh)
    return path


def load_feature_table(path: str = FEATURE_TABLE) -> Optional[List[dict]]:
    if not os.path.exists(path):
        return None
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Design matrices and predictions
# ---------------------------------------------------------------------------
def design_matrix(records: Sequence[dict],
                  classes: Sequence[str] = CLASSES) -> np.ndarray:
    """Rows = interfaces, columns = buried area per solvation class (A^2)."""
    X = np.zeros((len(records), len(classes)))
    idx = {c: j for j, c in enumerate(classes)}
    for i, r in enumerate(records):
        bt = r.get("bsa_by_type")
        if bt is not None:
            # derive the class areas under the CURRENT scheme, so a scheme
            # change does not require re-extracting the table
            for t, a in bt.items():
                j = idx.get(class_of_fine(t))
                if j is not None:
                    X[i, j] += a
            continue
        for c, a in r["bsa_by_class"].items():
            j = idx.get(c)
            if j is not None:
                X[i, j] = a
    return X


def predict_dg(records: Sequence[dict], sigma: Dict[str, float],
               delta: Optional[Dict[str, float]] = None) -> np.ndarray:
    """dG_solv under a class sigma table (+ optional fine deviations).

    With ``delta`` (default: the shipped DELTA) and typed records this is
    exactly the pipeline's value; class-only records give the class-level
    model.
    """
    if delta is None:
        delta = DELTA
    out = np.empty(len(records))
    for i, r in enumerate(records):
        bt = r.get("bsa_by_type")
        if bt is not None and delta:
            out[i] = sum(sigma_of_fine(t, sigma, delta) * a for t, a in bt.items())
        else:
            out[i] = sum(sigma.get(c, 0.0) * a for c, a in r["bsa_by_class"].items())
    return out


def _surface_sigma_moments(record: dict, sigma: Dict[str, float],
                           delta: Optional[Dict[str, float]] = None,
                           ) -> Tuple[float, float]:
    if delta is None:
        delta = DELTA
    w, s = [], []
    st = record.get("surf_asa_by_type")
    if st is not None and delta:
        for t, a in st.items():
            if t == "H" or a <= 0:
                continue
            w.append(a)
            s.append(sigma_of_fine(t, sigma, delta))
    else:
        for c, a in record["surf_asa_by_class"].items():
            if c == "H" or a <= 0:
                continue
            w.append(a)
            s.append(sigma.get(c, 0.0))
    if not w:
        return 0.0, 0.0
    w = np.asarray(w, float)
    s = np.asarray(s, float)
    m = float(np.average(s, weights=w))
    v = float(np.average((s - m) ** 2, weights=w))
    return m, v


def undeflated_z(records: Sequence[dict], sigma: Dict[str, float],
                 dg: Optional[np.ndarray] = None) -> np.ndarray:
    """The P-value z-score BEFORE the deflation constant is applied.

    Split out because it does not depend on ``P_VALUE_Z_SCALE``: fitting that
    one constant is then a scan over ``Phi(z0 * scale)`` with no per-record
    work, instead of recomputing the surface moments at every grid point.
    Records whose surface variance vanishes get z0 = 0, i.e. P = 0.5.
    """
    if dg is None:
        dg = predict_dg(records, sigma)
    z0 = np.zeros(len(records))
    for i, r in enumerate(records):
        m, v = _surface_sigma_moments(r, sigma)
        var = r["b_sq_sum"] * v
        if var > 1e-12:
            z0[i] = (dg[i] - r["b_sum"] * m) / np.sqrt(var)
    return z0


def predict_p(records: Sequence[dict], sigma: Dict[str, float],
              z_scale: float, dg: Optional[np.ndarray] = None,
              z0: Optional[np.ndarray] = None) -> np.ndarray:
    """P-value under a given sigma table and z deflation."""
    from scipy.special import erf

    if z0 is None:
        z0 = undeflated_z(records, sigma, dg=dg)
    z = z0 * z_scale
    return np.clip(0.5 * (1.0 + erf(z / np.sqrt(2))), 0.0, 1.0)


def predict_css(records: Sequence[dict], coef: Sequence[float],
                dg: Optional[np.ndarray] = None) -> np.ndarray:
    """CSS surrogate: sigmoid(b0 + b1*dG + b2*log1p(area))."""
    if dg is None:
        dg = predict_dg(records, SIGMA)
    area = np.array([r["area_fp"] for r in records], float)
    x = coef[0] + coef[1] * dg + coef[2] * np.log1p(np.maximum(area, 0.0))
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------
def fit_sigma(records: Sequence[dict],
              classes: Sequence[str] = CLASSES,
              ridge: float = 0.0) -> Tuple[Dict[str, float], dict]:
    """Least-squares fit of the per-class sigmas to PISA's ``int_solv_en``.

    No intercept: PISA's solvation gain is a pure sum of per-atom terms, so an
    intercept would be a model error, not a parameter. ``ridge`` adds an L2
    penalty (in units of area^2) which only matters for classes with little
    support -- report :func:`class_support` alongside any fit.

    Returns ``(sigma_dict, diagnostics)``. Classes absent from the data keep
    their incumbent value, and that is recorded in ``diagnostics["held"]``.
    """
    X = design_matrix(records, classes)
    y = np.array([r["dg_ref"] for r in records], float)
    present = X.any(axis=0)
    Xp = X[:, present]
    cols = [c for c, p in zip(classes, present) if p]

    if ridge > 0:
        A = Xp.T @ Xp + ridge * np.eye(Xp.shape[1])
        beta = np.linalg.solve(A, Xp.T @ y)
        rank = Xp.shape[1]
    else:
        beta, _, rank, _ = np.linalg.lstsq(Xp, y, rcond=None)

    resid = y - Xp @ beta
    dof = max(len(y) - Xp.shape[1], 1)
    s2 = float(resid @ resid) / dof
    try:
        cov = s2 * np.linalg.inv(Xp.T @ Xp)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(len(cols), np.nan)

    sigma = dict(SIGMA)
    for c, b in zip(cols, beta):
        sigma[c] = float(b)
    held = [c for c, p in zip(classes, present) if not p]

    diag = {
        "n_interfaces": len(records),
        "n_entries": len({r["pdb_id"] for r in records}),
        "columns": cols,
        "std_err": {c: float(e) for c, e in zip(cols, se)},
        "rank": int(rank),
        "condition_number": float(np.linalg.cond(Xp)),
        "residual_rms": float(np.sqrt(np.mean(resid ** 2))),
        "held": held,
    }
    return sigma, diag


def class_support(records: Sequence[dict],
                  classes: Sequence[str] = CLASSES) -> Dict[str, dict]:
    """Per-class evidence: how many interfaces and how much area inform it.

    A sigma fitted from a handful of interfaces is a fitted constant in name
    only; this is what separates ``C`` (every interface) from ``ZN``.
    """
    X = design_matrix(records, classes)
    out = {}
    for j, c in enumerate(classes):
        col = X[:, j]
        nz = col > 0
        out[c] = {
            "n_interfaces": int(nz.sum()),
            "n_entries": len({r["pdb_id"] for r, m in zip(records, nz) if m}),
            "total_area": float(col.sum()),
            "median_area_when_present": float(np.median(col[nz])) if nz.any() else 0.0,
        }
    return out


def fit_p_scale(records: Sequence[dict], sigma: Dict[str, float],
                grid: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Fit the P-value z deflation by minimising median |P - P_ref|.

    A median (not mean-square) objective: PISA's ligand/ion P-values follow
    different statistics and behave as outliers here, and a squared loss lets
    them steer a one-parameter fit.
    """
    rec = [r for r in records if r.get("pv_ref") is not None]
    if not rec:
        return float("nan"), float("nan")
    from scipy.special import erf

    ref = np.array([r["pv_ref"] for r in rec], float)
    z0 = undeflated_z(rec, sigma)
    if grid is None:
        grid = np.linspace(0.02, 2.0, 199)
    # Phi(z0 * scale) for every grid point at once.
    z = np.outer(np.asarray(grid, float), z0)
    losses = np.median(np.abs(0.5 * (1.0 + erf(z / np.sqrt(2))) - ref), axis=1)
    j = int(np.argmin(losses))
    return float(grid[j]), float(losses[j])


def fit_css(records: Sequence[dict], sigma: Dict[str, float],
            l2: float = 1e-3) -> Tuple[List[float], dict]:
    """Fit the CSS logistic surrogate by gradient descent on log-loss.

    PISA's CSS is a *fraction* in [0, 1] derived from its crystal-wide
    assembly analysis, so this is a fractional-response (quasi-binomial)
    logistic regression, fitted with a small L2 penalty for numerical
    stability. It is a surrogate, not a reproduction -- fastPISA does not
    enumerate symmetry mates.
    """
    from scipy.optimize import minimize

    rec = [r for r in records if r.get("css_ref") is not None]
    if not rec:
        return [float("nan")] * 3, {}
    y = np.clip(np.array([r["css_ref"] for r in rec], float), 1e-6, 1 - 1e-6)
    dg = predict_dg(rec, sigma)
    area = np.array([r["area_fp"] for r in rec], float)
    Z = np.column_stack([np.ones(len(rec)), dg, np.log1p(np.maximum(area, 0))])

    def nll(b):
        t = np.clip(Z @ b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-t))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) \
            + l2 * float(b[1:] @ b[1:])

    res = minimize(nll, np.array([-6.9, -0.17, 0.85]), method="Nelder-Mead",
                   options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-10})
    return [float(v) for v in res.x], {"log_loss": float(res.fun),
                                       "n": len(rec), "success": bool(res.success)}


# ---------------------------------------------------------------------------
# Grouped cross-validation
# ---------------------------------------------------------------------------
def entry_folds(records: Sequence[dict], k: int = 10,
                seed: int = 0) -> List[List[int]]:
    """Split record indices into ``k`` folds that never split a PDB entry."""
    entries = sorted({r["pdb_id"] for r in records})
    rng = np.random.default_rng(seed)
    rng.shuffle(entries)
    assign = {e: i % k for i, e in enumerate(entries)}
    folds: List[List[int]] = [[] for _ in range(k)]
    for i, r in enumerate(records):
        folds[assign[r["pdb_id"]]].append(i)
    return [f for f in folds if f]


def cross_validate_sigma(records: Sequence[dict], k: int = 10, seed: int = 0,
                         ridge: float = 0.0) -> dict:
    """Grouped K-fold CV of the sigma fit. Returns out-of-fold predictions."""
    folds = entry_folds(records, k=k, seed=seed)
    pred = np.full(len(records), np.nan)
    for fold in folds:
        test = set(fold)
        train = [r for i, r in enumerate(records) if i not in test]
        if not train:
            continue
        sig, _ = fit_sigma(train, ridge=ridge)
        held = [records[i] for i in fold]
        pred[fold] = predict_dg(held, sig)
    y = np.array([r["dg_ref"] for r in records], float)
    return {"pred": pred, "target": y, "k": len(folds)}


def dg_metrics(pred: np.ndarray, target: np.ndarray,
               mask: Optional[np.ndarray] = None) -> dict:
    """Agreement statistics for a dG prediction.

    Reports the regression *slope* and *intercept* of prediction on target as
    well as correlation: a Pearson r near 1 is easy to reach when interface
    sizes span two orders of magnitude, and hides a systematic scale error.
    """
    p, t = np.asarray(pred, float), np.asarray(target, float)
    if mask is not None:
        p, t = p[mask], t[mask]
    ok = np.isfinite(p) & np.isfinite(t)
    p, t = p[ok], t[ok]
    if p.size < 3:
        return {"n": int(p.size)}
    err = p - t
    A = np.column_stack([t, np.ones_like(t)])
    slope, intercept = np.linalg.lstsq(A, p, rcond=None)[0]
    ss_res = float(((t - p) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    return {
        "n": int(p.size),
        "pearson": float(np.corrcoef(p, t)[0, 1]),
        "median_abs_err": float(np.median(np.abs(err))),
        "mean_err": float(err.mean()),          # bias
        "rmse": float(np.sqrt((err ** 2).mean())),
        "slope": float(slope),
        "intercept": float(intercept),
        # R^2 about the 1:1 line, NOT about a refitted line: this is the
        # fraction of PISA's variance actually reproduced.
        "r2_identity": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Residue-level reference (PISA's per-residue ASA / BSA / solvation energy)
# ---------------------------------------------------------------------------
#: Residue-level table (LOCAL artifact, gitignored: ~10-20 MB). Interfaces
#: with their residues matched to PISA's per-residue records, buried area
#: per FINE atom type. Any class scheme is a groupby over it.
RESIDUE_TABLE = os.path.join(CALIBRATION_DIR, "residues.json.gz")


def _chain_of_molecule(chain_id: str) -> str:
    """PISA molecule chain_id -> author chain: ``"A"`` or ``"[SO4]A:301"``."""
    if chain_id.startswith("["):
        return chain_id.split("]", 1)[1].split(":", 1)[0].strip()
    return chain_id.strip()


def extract_entry_residues(pdb_id: str) -> List[dict]:
    """Interface records with PISA-matched residue-level features.

    One record per matched identity interface. Its ``residues`` list holds
    every residue that buries area in the pair AND is present in PISA's
    per-residue table, with our buried area per fine atom type and PISA's
    isolated ASA, BSA and solvation energy. PISA prints the residue
    solvation term with the opposite sign to the interface total (a
    hydrophobic residue shows a positive ``solv_en`` while a hydrophobic
    interface shows a negative ``int_solv_en``); ``solv_ref`` here is
    flipped into the interface convention so sigma * BSA fits it directly.
    """
    from fastpisa.core import run_core

    ref_all = load_cached_reference(pdb_id, include_residues=True)
    pdb = cached_pdb_path(pdb_id)
    if ref_all is None or pdb is None:
        return []
    refk = {_key(m["chain_id"] for m in i["molecules"]): i
            for i in identity_interfaces(ref_all)}
    if not refk:
        return []

    state = run_core(pdb, mode="pisa", collect_calibration=True)
    out = []
    for iface in state.interfaces:
        k = _key(m["chain_id"] for m in iface.molecules)
        ri = refk.get(k)
        if ri is None:
            continue
        ref_res = {}
        for m in ri["molecules"]:
            ch = _chain_of_molecule(m["chain_id"])
            for r in m.get("residues", []):
                try:
                    seq = int(r["seq_num"])
                except (TypeError, ValueError):
                    continue
                ref_res[(ch, seq, (r["ins_code"] or "").strip())] = r
        cal = iface.calibration
        matched, n_fp = [], 0
        for r in cal["residues"]:
            n_fp += 1
            rr = ref_res.get((r["chain"], r["seqnum"], (r["icode"] or "").strip()))
            if rr is None:
                continue
            matched.append({
                "name": r["name"], "chain": r["chain"], "seqnum": r["seqnum"],
                "asa_fp": round(r["asa_iso"], 3), "bsa_fp": round(r["bsa"], 4),
                "bsa_by_type": {t: round(v, 4) for t, v in r["bsa_by_type"].items()},
                "asa_ref": rr["asa"], "bsa_ref": rr["bsa"],
                "solv_ref": -(rr["solv_en"] or 0.0),
            })
        out.append({
            "pdb_id": pdb_id,
            "pair": "+".join(sorted(k)),
            "is_polymer_pair": "[" not in "+".join(sorted(k)),
            "area_fp": iface.interface_area, "area_ref": ri["int_area"],
            "dg_ref": ri["int_solv_en"], "pv_ref": ri["pvalue"],
            "css_ref": ri["css"], "stab_ref": ri["stab_en"],
            "nhb_fp": iface.number_hydrogen_bonds,
            "nsb_fp": iface.number_salt_bridges,
            "nss_fp": iface.number_disulfide_bonds,
            "surf_asa_by_type": {t: round(v, 3)
                                 for t, v in cal["surf_asa_by_type"].items()},
            "b_sum": round(cal["b_sum"], 4),
            "b_sq_sum": round(cal["b_sq_sum"], 4),
            "n_res_fp": n_fp, "n_res_ref": len(ref_res),
            "residues": matched,
        })
    return out


def save_residue_table(records: List[dict], path: str = RESIDUE_TABLE) -> str:
    return save_feature_table(records, path)


def load_residue_table(path: str = RESIDUE_TABLE) -> Optional[List[dict]]:
    return load_feature_table(path)


# ---------------------------------------------------------------------------
# Class schemes: a scheme is a function fine_type -> class
# ---------------------------------------------------------------------------
def group_by_scheme(by_type: Dict[str, float], class_of) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for t, v in by_type.items():
        c = class_of(t)
        out[c] = out.get(c, 0.0) + v
    return out


def interface_records_from_residues(rtab: Sequence[dict], class_of,
                                    ) -> List[dict]:
    """Interface-level records (the ``features.json.gz`` shape) under a scheme.

    Rebuilt from the residue table so alternative schemes can be evaluated
    with :func:`fit_sigma` / :func:`cross_validate_sigma` unchanged. NOTE:
    ``bsa_by_class`` here sums over PISA-MATCHED residues only, so an
    interface whose residues did not all match carries slightly less area
    than the pipeline's own figure.
    """
    out = []
    for rec in rtab:
        by_type: Dict[str, float] = {}
        for r in rec["residues"]:
            for t, v in r["bsa_by_type"].items():
                by_type[t] = by_type.get(t, 0.0) + v
        d = {k: v for k, v in rec.items() if k not in ("residues", "surf_asa_by_type")}
        d["bsa_by_class"] = group_by_scheme(by_type, class_of)
        d["surf_asa_by_class"] = group_by_scheme(rec["surf_asa_by_type"], class_of)
        out.append(d)
    return out


#: Compact, COMMITTED residue-level fit table: one row per PISA-matched
#: interface residue, ``[pdb_id, is_polymer_pair, solv_ref, bsa_ref, bsa_by_type]``
#: (``bsa_ref`` is PISA's buried area for the residue, so the geometry can be
#: regression-tested offline too).
#: Everything the residue-level sigma fit needs, ~2 MB; the full residue
#: table (with names, ASA, PISA BSA -- for audits) stays local.
RESIDUE_FIT_TABLE = os.path.join(CALIBRATION_DIR, "residue_fit.json.gz")


def compact_residue_table(rtab: Sequence[dict]) -> List[list]:
    rows = []
    for rec in rtab:
        for r in rec["residues"]:
            rows.append([rec["pdb_id"], int(rec["is_polymer_pair"]),
                         round(r["solv_ref"], 4), round(r["bsa_ref"] or 0.0, 2),
                         {t: round(v, 2) for t, v in r["bsa_by_type"].items()
                          if v >= 0.01}])
    return rows


def save_residue_fit_table(rtab: Sequence[dict],
                           path: str = RESIDUE_FIT_TABLE) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as fh:
        json.dump(compact_residue_table(rtab), fh, separators=(",", ":"))
    return path


def load_residue_fit_rows(path: str = RESIDUE_FIT_TABLE) -> Optional[list]:
    """Flat residue rows ``(pdb_id, {solv_ref, bsa_by_type, is_polymer})``."""
    if not os.path.exists(path):
        return None
    with gzip.open(path, "rt") as fh:
        raw = json.load(fh)
    return [(pid, {"solv_ref": sv, "bsa_ref": br, "bsa_by_type": bt,
                   "is_polymer": bool(ip)})
            for pid, ip, sv, br, bt in raw]


def residue_rows(rtab: Sequence[dict], polymer_only: bool = True):
    """Flatten the residue table into per-residue rows (with pdb_id).

    Accepts either the full residue table (interface records with a
    ``residues`` list) or the flat rows from :func:`load_residue_fit_rows`.
    """
    for rec in rtab:
        if isinstance(rec, tuple):
            pid, r = rec
            if polymer_only and not r.get("is_polymer", True):
                continue
            yield pid, r
            continue
        if polymer_only and not rec["is_polymer_pair"]:
            continue
        for r in rec["residues"]:
            yield rec["pdb_id"], r


def fit_sigma_residue_level(rtab: Sequence[dict], class_of,
                            classes: Sequence[str],
                            fine_types: Sequence[str] = (),
                            ridge_fine: float = 0.0,
                            polymer_only: bool = True,
                            ) -> Tuple[Dict[str, float], Dict[str, float], dict]:
    """Fit sigma per class (and optional per-fine-type deviations) to PISA's
    per-RESIDUE solvation energies.

    Model: solv_res = sum_c sigma_c * A_c(res) + sum_t delta_t * A_t(res),
    with an L2 penalty ``ridge_fine`` on the deltas only (hierarchical
    shrinkage: a fine type with little buried area falls back to its class).
    Returns ``(sigma, delta, diagnostics)``; ``delta`` is empty when no
    fine types are given.
    """
    rows = list(residue_rows(rtab, polymer_only))
    cidx = {c: j for j, c in enumerate(classes)}
    tidx = {t: len(classes) + j for j, t in enumerate(fine_types)}
    ncol = len(classes) + len(fine_types)
    X = np.zeros((len(rows), ncol))
    y = np.empty(len(rows))
    for i, (_, r) in enumerate(rows):
        y[i] = r["solv_ref"]
        for t, a in r["bsa_by_type"].items():
            j = cidx.get(class_of(t))
            if j is not None:
                X[i, j] += a
            jt = tidx.get(t)
            if jt is not None:
                X[i, jt] += a
    pen = np.zeros(ncol)
    pen[len(classes):] = ridge_fine
    A = X.T @ X + np.diag(pen)
    beta = np.linalg.solve(A, X.T @ y)
    resid = y - X @ beta
    sigma = {c: float(beta[cidx[c]]) for c in classes}
    delta = {t: float(beta[tidx[t]]) for t in fine_types}
    diag = {"n_residues": len(rows), "residual_rms": float(np.sqrt(np.mean(resid ** 2))),
            "median_abs_err": float(np.median(np.abs(resid)))}
    return sigma, delta, diag


def residue_geometry_audit(rtab: Sequence[dict], polymer_only: bool = True,
                           min_n: int = 200) -> Dict[str, dict]:
    """Per-residue-type agreement of our BSA / isolated ASA with PISA's."""
    acc: Dict[str, dict] = {}
    for _, r in residue_rows(rtab, polymer_only):
        d = acc.setdefault(r["name"], {"bsa_rel": [], "asa_rel": [], "bsa_signed": []})
        if r["bsa_ref"] and r["bsa_ref"] > 5.0:
            d["bsa_rel"].append(abs(r["bsa_fp"] - r["bsa_ref"]) / r["bsa_ref"])
            d["bsa_signed"].append((r["bsa_fp"] - r["bsa_ref"]) / r["bsa_ref"])
        if r["asa_ref"] and r["asa_ref"] > 5.0:
            d["asa_rel"].append(abs(r["asa_fp"] - r["asa_ref"]) / r["asa_ref"])
    out = {}
    for name, d in acc.items():
        if len(d["bsa_rel"]) < min_n:
            continue
        out[name] = {
            "n": len(d["bsa_rel"]),
            "bsa_median_rel_err": float(np.median(d["bsa_rel"])),
            "bsa_mean_signed": float(np.mean(d["bsa_signed"])),
            "asa_median_rel_err": float(np.median(d["asa_rel"])) if d["asa_rel"] else float("nan"),
        }
    return out


# ---------------------------------------------------------------------------
# Candidate solvation-class schemes (fine type -> class)
# ---------------------------------------------------------------------------
#: The shipped scheme (asp_table.class_of_fine) is scheme B of the design
#: spec; ``scheme_b`` below is kept as the explicit, importable definition
#: the CV experiments used and must stay identical to it (a test checks).
scheme_incumbent = class_of_fine

_NA_RES = frozenset({"A", "G", "C", "U", "DA", "DG", "DC", "DT", "DU"})
_AROM_C = {
    ("PHE", a) for a in ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")
} | {
    ("TYR", a) for a in ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")
} | {
    ("TRP", a) for a in ("CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2")
} | {("HIS", "CG"), ("HIS", "CD2"), ("HIS", "CE1")}


def scheme_b(fine: str) -> str:
    """Chemically motivated ~35-class scheme (spec section 3, candidate B).

    Protein atoms are split by chemical environment (backbone vs side
    chain, sp2 vs sp3 carbon, amide / carboxylate / guanidinium / amine /
    ring nitrogen and oxygen, Met vs Cys sulfur); nucleic acids by
    sugar / base / phosphate; hetero atoms keep the incumbent classes.
    """
    if fine == "H" or fine.startswith("het:"):
        return fine if fine == "H" else fine.split(":")[1]
    res, atom = fine.split(":", 1)
    el = atom[0] if not (res == "MSE" and atom == "SE") else "SE"
    if res in _NA_RES:
        if el == "P":
            return "NA_P"
        if atom in ("OP1", "OP2", "OP3", "O1P", "O2P"):
            return "NA_OP"
        if atom.endswith("'"):
            return "NA_C_sugar" if el == "C" else "NA_O_sugar"
        return {"C": "NA_C_base", "N": "NA_N_base", "O": "NA_O_base"}.get(el, "X")
    # protein
    if el == "C":
        if atom == "CA":
            return "C_CA"
        if atom == "C":
            return "C_bb"
        if (res, atom) in _AROM_C:
            return "C_aro"
        if (res, atom) in (("ASN", "CG"), ("GLN", "CD"),
                           ("ASP", "CG"), ("GLU", "CD"), ("ARG", "CZ")):
            return "C_sp2_polar"
        return "C_ali"
    if el == "N":
        if atom == "N":
            return "N_bb"
        if res == "ARG":
            return "N_arg"
        if res == "LYS":
            return "N_lys"
        if res == "HIS":
            return "N_his"
        if res == "TRP":
            return "N_trp"
        return "N_amide"          # ASN ND2, GLN NE2
    if el == "O":
        if atom in ("O", "OXT"):
            return "O_bb"
        if res in ("ASP", "GLU"):
            return "O_carbox"
        if res in ("ASN", "GLN"):
            return "O_amide"
        return "O_hyd"            # SER OG, THR OG1, TYR OH
    if el in ("S", "SE"):
        return "S_cys" if res == "CYS" else "S_met"
    return "X"


def classes_of_scheme(rtab: Sequence[dict], class_of) -> List[str]:
    """Sorted classes with any buried area under ``class_of`` (excl. H/X)."""
    seen = set()
    for rec in rtab:
        for r in rec["residues"]:
            for t in r["bsa_by_type"]:
                seen.add(class_of(t))
    return sorted(c for c in seen if c not in ("H", "X"))


def fit_sigma_hierarchical(records: Sequence[dict], class_of,
                           classes: Sequence[str], fine_types: Sequence[str],
                           ridge_fine: float,
                           ) -> Tuple[Dict[str, float], Dict[str, float], dict]:
    """Interface-level hierarchical fit: sigma per class + shrunk per-fine-type
    deviation (L2 penalty ``ridge_fine`` on the deviations only).

    Same model as :func:`fit_sigma_residue_level` but on interface records
    carrying ``bsa_by_type`` -- the form the committed feature table has, so
    the shipped constants can be re-derived offline.
    """
    cidx = {c: j for j, c in enumerate(classes)}
    tidx = {t: len(classes) + j for j, t in enumerate(fine_types)}
    ncol = len(classes) + len(fine_types)
    X = np.zeros((len(records), ncol))
    y = np.array([r["dg_ref"] for r in records], float)
    for i, r in enumerate(records):
        for t, a in r["bsa_by_type"].items():
            j = cidx.get(class_of(t))
            if j is not None:
                X[i, j] += a
            jt = tidx.get(t)
            if jt is not None:
                X[i, jt] += a
    pen = np.zeros(ncol)
    pen[len(classes):] = ridge_fine
    beta = np.linalg.solve(X.T @ X + np.diag(pen), X.T @ y)
    resid = y - X @ beta
    sigma = {c: float(beta[cidx[c]]) for c in classes}
    delta = {t: float(beta[tidx[t]]) for t in fine_types}
    # standard errors for the class level (ridge-free columns)
    dof = max(len(y) - ncol, 1)
    s2 = float(resid @ resid) / dof
    try:
        cov = s2 * np.linalg.inv(X.T @ X + np.diag(pen))
        se = {c: float(np.sqrt(cov[cidx[c], cidx[c]])) for c in classes}
    except np.linalg.LinAlgError:
        se = {c: float("nan") for c in classes}
    Xc = X[:, :len(classes)]
    return sigma, delta, {
        "n_interfaces": len(records), "columns": list(classes),
        "std_err": se, "residual_rms": float(np.sqrt(np.mean(resid ** 2))),
        "condition_number_classes": float(np.linalg.cond(Xc[:, Xc.any(axis=0)])),
    }


def predict_dg_typed(records: Sequence[dict], class_of,
                     sigma: Dict[str, float], delta: Dict[str, float],
                     ) -> np.ndarray:
    """dG from ``bsa_by_type`` under a class scheme + fine deviations."""
    out = np.empty(len(records))
    for i, r in enumerate(records):
        out[i] = sum((sigma.get(class_of(t), 0.0) + delta.get(t, 0.0)) * a
                     for t, a in r["bsa_by_type"].items())
    return out


#: Buried-area support a fine type needs before it earns a shrunk deviation.
FINE_TYPE_MIN_AREA = 2000.0
#: Ridge on the fine-type deviations (grouped-CV flat over 1e2 .. 1e4).
FINE_TYPE_RIDGE = 1000.0
#: Classes pinned at zero rather than fitted (phosphorus; see asp_table).
PINNED_CLASSES = ("P", "NA_P")


def fitted_fine_types(rows, class_of=class_of_fine,
                      min_area: float = FINE_TYPE_MIN_AREA) -> List[str]:
    """Fine types with enough buried area to get their own deviation."""
    tot: Dict[str, float] = {}
    for _, r in rows:
        for t, a in r["bsa_by_type"].items():
            tot[t] = tot.get(t, 0.0) + a
    # a hetero fine type earns a deviation only when it is element-resolved
    # ("het:MET:MG"); "het:C" IS its class and would be collinear with it
    return sorted(t for t, a in tot.items()
                  if a > min_area and t != "H"
                  and (not t.startswith("het:") or t.count(":") == 2)
                  and class_of(t) not in PINNED_CLASSES)


def fitted_classes(rows, class_of=class_of_fine) -> List[str]:
    seen = set()
    for _, r in rows:
        for t in r["bsa_by_type"]:
            seen.add(class_of(t))
    return sorted(c for c in seen if c not in ("H", "X") + PINNED_CLASSES)
