"""Small pure helpers for the Streamlit app (no Streamlit calls)."""
from __future__ import annotations

import io

import pandas as pd


def excel_bytes(sheets: dict) -> bytes:
    """One workbook, one sheet per table (sheet names truncated to 31)."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()
