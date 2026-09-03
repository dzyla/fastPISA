"""Small pure helpers for the Streamlit app (no Streamlit calls)."""
from __future__ import annotations

import io
import re
from typing import Iterable

import pandas as pd


_INVALID_SHEET_CHARS = re.compile(r"[\[\]:*?/\\]")


def safe_sheet_names(names: Iterable[str]) -> list[str]:
    """Return valid, non-empty, case-insensitively unique Excel titles."""
    result = []
    used = set()
    for raw_name in names:
        base = " ".join(_INVALID_SHEET_CHARS.sub(" ", str(raw_name)).split())
        base = base.strip("'") or "Sheet"
        number = 1
        while True:
            suffix = "" if number == 1 else f" ({number})"
            candidate = base[:31 - len(suffix)].rstrip() + suffix
            if candidate.casefold() not in used:
                break
            number += 1
        used.add(candidate.casefold())
        result.append(candidate)
    return result


def excel_bytes(sheets: dict) -> bytes:
    """One workbook with a safe, unique worksheet name per table."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for (name, df), safe_name in zip(sheets.items(), safe_sheet_names(sheets)):
            df.to_excel(xw, sheet_name=safe_name, index=False)
    return buf.getvalue()


def apply_shared_side(current: dict, reference: dict, inventory: list[dict], matches) -> None:
    """Apply shared-side chain matches and invalidate a stale group digest."""
    matched = []
    for _reference_chain, mobile_chain, _identity in matches:
        if mobile_chain not in matched:
            matched.append(mobile_chain)
    current["group1"] = matched
    current["group2"] = [
        row["label"] for row in inventory
        if row["class"] != "Ligand" and row["label"] not in matched
    ]
    current["label1"] = reference["label1"]
    current["label2"] = reference["label2"]
    current.pop("gi", None)
