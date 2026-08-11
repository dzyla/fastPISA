"""Batch analysis of many structures (item 4.1 of fastpisa_improvements.md).

Provides a first-class entry point for analysing whole directories / lists of
structures, optionally in parallel using a process pool. Uses the standard
library ``concurrent.futures`` so there is no extra dependency (unlike a joblib
version would require).
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Union

PathLike = Union[str, os.PathLike]


def _analyze_one(args):
    """Module-level worker (must be picklable for the process pool)."""
    path, kwargs = args
    try:
        from fastpisa.api import PISAInterfaceAnalyzer
        ana = PISAInterfaceAnalyzer(path, **kwargs)
        result = ana.analyze()
        n = result["interfaces"]["assembly"]["interface_count"]
        return {
            "path": path, "ok": True, "result": result,
            "n_interfaces": n, "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - report per-file failures, keep going
        return {
            "path": path, "ok": False, "result": None,
            "n_interfaces": 0, "error": repr(exc),
        }


def analyze_many(paths: Iterable[PathLike], mode: str = "pisa",
                 n_jobs: int = 1, **kwargs) -> List[Dict[str, Any]]:
    """Analyze multiple structures and return one result dict per input.

    Parameters
    ----------
    paths : iterable of paths
        PDB/mmCIF files to analyse (globs / directories should be expanded by
        the caller, e.g. via :func:`expand_inputs`).
    mode : str
        ``"pisa"`` or ``"cocomaps"`` (forwarded to ``PISAInterfaceAnalyzer``).
    n_jobs : int
        ``1`` = serial (default). ``>1`` uses a process pool with that many
        workers; ``-1`` uses all CPUs. Serial is recommended for < ~10 files;
        the ASA machinery is the bottleneck and parallelises well.
    **kwargs
        Forwarded to ``PISAInterfaceAnalyzer`` (probe_radius, interface_cutoff,
        exclude_water, min_css, pdb_id, ...).

    Returns
    -------
    list of dict
        Same order as ``paths``. Each entry is ``{path, ok, result,
        n_interfaces, error}``; on failure ``ok=False`` and ``error`` holds the
        repr, so one bad file does not abort the batch.

    Example
    -------
    >>> from fastpisa.batch import analyze_many, expand_inputs
    >>> files = expand_inputs("models/*.cif")
    >>> for r in analyze_many(files, n_jobs=4):
    ...     print(r["path"], r["ok"], r["n_interfaces"])
    """
    paths = [os.fspath(p) for p in paths]
    kwargs = dict(kwargs)
    kwargs["mode"] = mode
    work = [(p, kwargs) for p in paths]

    if n_jobs == 1 or len(work) <= 1:
        return [_analyze_one(a) for a in work]

    workers: Optional[int] = -1 if n_jobs < 0 else n_jobs
    results: List[Optional[Dict]] = [None] * len(work)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_analyze_one, a): i for i, a in enumerate(work)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:  # worker itself died
                results[idx] = {
                    "path": work[idx][0], "ok": False, "result": None,
                    "n_interfaces": 0, "error": repr(exc),
                }
    return results  # type: ignore[return-value]


def expand_inputs(*patterns_or_paths: PathLike) -> List[str]:
    """Expand glob patterns / directories into a flat, de-duplicated list of
    structure files (``.pdb`` / ``.cif`` / ``.cif.gz``).

    Non-glob paths are passed through if they exist; directories are scanned
    recursively for structure files.
    """
    import glob
    out: List[str] = []
    for item in patterns_or_paths:
        item = os.fspath(item)
        if glob.has_magic(item):
            out.extend(glob.glob(item, recursive=True))
        elif os.path.isdir(item):
            for root, _dirs, files in os.walk(item):
                for fn in files:
                    if fn.endswith((".pdb", ".cif", ".cif.gz")):
                        out.append(os.path.join(root, fn))
        elif os.path.isfile(item):
            out.append(item)
    # keep order, drop duplicates
    seen = set()
    result = []
    for p in out:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result
