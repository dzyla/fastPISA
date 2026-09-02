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

from fastpisa.energy.asp_table import SIGMA
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
        for c, a in r["bsa_by_class"].items():
            j = idx.get(c)
            if j is not None:
                X[i, j] = a
    return X


def predict_dg(records: Sequence[dict], sigma: Dict[str, float]) -> np.ndarray:
    """dG_solv under a given sigma table (exactly the pipeline's value)."""
    return np.array([sum(sigma.get(c, 0.0) * a
                         for c, a in r["bsa_by_class"].items())
                     for r in records])


def _surface_sigma_moments(record: dict,
                           sigma: Dict[str, float]) -> Tuple[float, float]:
    w, s = [], []
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
