"""Query-side date extraction and snap-to-FOMC-meeting helpers.

Used only by ``generate_informational.py`` to address a known retrieval
failure mode: hybrid (RRF) retrieval drops the target meeting on
date-explicit queries because the dense leg has no representation of dates
and matches topically-similar prose from other eras instead. We extract
date/period mentions from the user's query, snap them to actual FOMC
meeting dates in the corpus, and force-inject those meetings' chunks into
the retrieved context regardless of what RRF surfaced.

This is a pre-retrieval preprocessor, **not** a new retrieval leg, so the
Tier 1 ablation results in ``data/eval_summary.csv`` stay reproducible.
``generate.py`` (the Tier 1 stance generator) deliberately does not consume
this module.
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import chromadb
import pandas as pd

from index import CHROMA_DIR, COLLECTION_NAME

log = logging.getLogger("query_dates")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "raw" / "_manifest.csv"

ISO_SNAP_WINDOW_DAYS = 90  # how far an ISO-date mention can drift to snap onto a meeting

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_MONTH_PAT = "|".join(sorted(MONTHS.keys(), key=len, reverse=True))
RE_MONTH_YEAR = re.compile(rf"\b({_MONTH_PAT})\.?\s+(\d{{4}})\b", re.IGNORECASE)
RE_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
RE_ISO_MONTH = re.compile(r"\b(\d{4})-(\d{2})\b(?!-\d)")  # YYYY-MM not followed by -DD
RE_QUARTER = re.compile(r"\bQ([1-4])\s*(\d{4})\b", re.IGNORECASE)


@lru_cache(maxsize=1)
def _meeting_dates() -> list[date]:
    df = pd.read_csv(MANIFEST_PATH, dtype={"meeting_date": str})
    return sorted(date.fromisoformat(d) for d in df["meeting_date"])


def _meetings_in_range(start: date, end: date) -> list[str]:
    return [d.isoformat() for d in _meeting_dates() if start <= d <= end]


def _snap_iso(target: date) -> str | None:
    """Snap an ISO date to the nearest actual meeting within the snap window."""
    candidates = sorted((abs((target - md).days), md) for md in _meeting_dates())
    if candidates and candidates[0][0] <= ISO_SNAP_WINDOW_DAYS:
        return candidates[0][1].isoformat()
    return None


def _month_range(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    end = date(year, 12, 31) if month == 12 else date(year, month + 1, 1) - timedelta(days=1)
    return start, end


def extract_meeting_dates_from_query(query: str) -> list[str]:
    """Return YYYY-MM-DD meeting dates referenced (explicitly or by period) in the query.

    Handles month-year ("July 2025"), ISO date ("2025-07-30"), ISO month
    ("2025-07"), and quarter ("Q3 2024"). Year-only mentions are deliberately
    ignored — they would inject 8+ meetings and overwhelm the context.
    """
    matches: set[str] = set()

    for m in RE_ISO_DATE.finditer(query):
        try:
            target = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue
        snapped = _snap_iso(target)
        if snapped is not None:
            matches.add(snapped)

    for m in RE_MONTH_YEAR.finditer(query):
        start, end = _month_range(int(m.group(2)), MONTHS[m.group(1).lower()])
        matches.update(_meetings_in_range(start, end))

    for m in RE_ISO_MONTH.finditer(query):
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            start, end = _month_range(y, mo)
            matches.update(_meetings_in_range(start, end))

    for m in RE_QUARTER.finditer(query):
        q, year = int(m.group(1)), int(m.group(2))
        start_month = 3 * (q - 1) + 1
        start = date(year, start_month, 1)
        end_month = start_month + 2
        end = date(year, 12, 31) if end_month == 12 else date(year, end_month + 1, 1) - timedelta(days=1)
        matches.update(_meetings_in_range(start, end))

    return sorted(matches)


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def chunks_for_meetings(meeting_dates: list[str]) -> list[dict]:
    """Fetch all chunks belonging to the given meeting dates from Chroma.

    Returns the same result-dict shape used by ``retrieve.py`` so the caller
    can merge these with retrieved results without reformatting. Score is set
    to a sentinel value because these are deterministic, not ranked.
    """
    if not meeting_dates:
        return []
    res = _collection().get(
        where={"meeting_date": {"$in": list(meeting_dates)}},
        include=["documents", "metadatas"],
    )
    out: list[dict] = []
    for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
        out.append(
            {
                "id": cid,
                "meeting_date": meta["meeting_date"],
                "source_url": meta["source_url"],
                "text": doc,
                "score": float("nan"),
            }
        )
    out.sort(key=lambda r: (r["meeting_date"], r["id"]))
    return out


def augment_with_date_snap(
    query: str, retrieved: list[dict]
) -> tuple[list[dict], set[str]]:
    """Force-include chunks for any FOMC meetings the query mentions by date.

    Addresses the RRF temporal-query failure mode: dense retrieval drops the
    target meeting because it has no date representation, and RRF promotes
    irrelevant topical neighbors over the correct meeting. We extract date
    mentions from the query, fetch all chunks for those meetings from Chroma,
    and prepend any that retrieval missed. The merged list keeps all
    originally-retrieved chunks (so retrieval-only metrics stay measurable)
    and the returned set of forced dates is used by the renderer to annotate
    those chunks for the model.
    """
    snapped = extract_meeting_dates_from_query(query)
    if not snapped:
        return retrieved, set()

    existing_ids = {r["id"] for r in retrieved}
    forced_chunks = chunks_for_meetings(snapped)
    new_chunks = [c for c in forced_chunks if c["id"] not in existing_ids]

    log.info(
        "date-snap: query mentions %s; %d forced chunks added (already in retrieval: %d)",
        snapped,
        len(new_chunks),
        len(forced_chunks) - len(new_chunks),
    )
    # Prepend forced-but-missing chunks so the model sees them first.
    merged = new_chunks + retrieved
    return merged, set(snapped)
