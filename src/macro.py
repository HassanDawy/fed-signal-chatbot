"""FRED macro-indicator lookup, keyed by FOMC meeting date.

Used by ``generate_informational.py`` to enrich each retrieved chunk's context
block with a one-line macro snapshot taken from ``data/fred/macro_indicators.csv``.

Kept deliberately separate from ``retrieve.py``: the retrieval architecture
(Chroma + BM25 + RRF over text chunks) is unchanged. This module is a
post-retrieval CSV lookup, not a new retrieval leg, so the Tier 1 ablation
results stay valid.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from pathlib import Path

import pandas as pd

log = logging.getLogger("macro")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MACRO_PATH = PROJECT_ROOT / "data" / "fred" / "macro_indicators.csv"

SERIES = ("FEDFUNDS", "DGS10", "CPIAUCSL", "UNRATE")


@lru_cache(maxsize=1)
def load_macro() -> pd.DataFrame:
    """Read the macro CSV once, indexed by ``meeting_date`` (string YYYY-MM-DD)."""
    df = pd.read_csv(MACRO_PATH, dtype={"meeting_date": str})
    df = df.set_index("meeting_date")
    missing_cols = [c for c in SERIES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"macro CSV missing expected series: {missing_cols}")
    return df


def _fmt(val: float) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "NA"
    return f"{val:g}"


def format_macro_for_date(meeting_date: str) -> str | None:
    """Return a one-line macro snapshot for ``meeting_date``, or ``None`` if absent.

    Format::

        [Macro at this meeting: FEDFUNDS=4.29, DGS10=4.53, CPIAUCSL=199.3, UNRATE=4.7]

    Returns ``None`` when the date is not in the macro CSV (e.g., the meeting
    predates the macro window or the date string is malformed). Callers should
    skip the line entirely in that case rather than emit a placeholder.
    """
    df = load_macro()
    if meeting_date not in df.index:
        log.debug("no macro row for %s", meeting_date)
        return None
    row = df.loc[meeting_date]
    parts = [f"{s}={_fmt(row[s])}" for s in SERIES]
    return "[Macro at this meeting: " + ", ".join(parts) + "]"
