"""Scrape FOMC meeting statements from federalreserve.gov.

Discovers statement URLs from the Fed's own calendar pages (current + historical)
by regex-matching ``/newsevents/pressreleases/monetaryYYYYMMDDa.htm`` links,
downloads each page, strips the implementation-note / related-information
sections, and writes plain text to ``data/raw/{YYYY-MM-DD}.txt`` plus a
manifest CSV.

Run:
    python src/ingest.py

Scope: 2006 -> today. Meeting statements only (not minutes, not
implementation notes).
"""

from __future__ import annotations

import csv
import datetime as dt
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fredapi import Fred
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

FED_BASE = "https://www.federalreserve.gov"
CURRENT_CALENDAR = f"{FED_BASE}/monetarypolicy/fomccalendars.htm"
HISTORICAL_CALENDAR = f"{FED_BASE}/monetarypolicy/fomchistorical{{year}}.htm"

# The current calendar page covers roughly the last ~5 years; everything
# before that lives on per-year historical pages. The Fed releases the
# historical page for a given year ~5 years after the fact, so we pull a
# conservative window.
HISTORICAL_YEARS = range(2006, dt.date.today().year - 4)

# Statement URLs look like .../monetary20220504a.htm (2011+) or
# .../press/monetary/20080430a.htm (2006-2010). Implementation notes use a
# trailing digit (a1.htm); \b excludes those from the single-letter group.
STATEMENT_HREF_RE = re.compile(
    r"/newsevents/(?:pressreleases/monetary|press/monetary/)(\d{8})([a-z])\.htm\b",
    re.IGNORECASE,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "_manifest.csv"
URL_LIST_PATH = RAW_DIR / "_statement_urls.txt"
MACRO_CSV_PATH = PROJECT_ROOT / "data" / "macro_indicators.csv"

FRED_SERIES = ("FEDFUNDS", "DGS10", "CPIAUCSL", "UNRATE")
FRED_OBSERVATION_START = "2005-01-01"

REQUEST_DELAY_SEC = 1.0
USER_AGENT = (
    "fed-signal-chatbot/0.1 (DS593 academic project; "
    "contact: hz.dawy@gmail.com)"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("ingest")


@dataclass(frozen=True)
class StatementRef:
    meeting_date: dt.date
    url: str

    @property
    def filename(self) -> str:
        return f"{self.meeting_date.isoformat()}.txt"


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def polite_get(session: requests.Session, url: str) -> requests.Response | None:
    """GET with a fixed inter-request delay. Returns None on 404."""
    time.sleep(REQUEST_DELAY_SEC)
    resp = session.get(url, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp


# --------------------------------------------------------------------------
# Step 1: discover statement URLs from Fed calendar pages
# --------------------------------------------------------------------------


def extract_statement_urls(html: str) -> set[str]:
    """Return absolute statement URLs found in the given calendar page HTML.

    Only keeps the ``monetaryYYYYMMDDa.htm`` variant (the actual statement).
    Rejects implementation-note URLs like ``...a1.htm`` because the regex
    captures only a single trailing letter.
    """
    urls: set[str] = set()
    for match in STATEMENT_HREF_RE.finditer(html):
        path = match.group(0)
        urls.add(urljoin(FED_BASE, path))
    return urls


def discover_statement_urls(session: requests.Session) -> list[StatementRef]:
    calendar_pages = [CURRENT_CALENDAR] + [
        HISTORICAL_CALENDAR.format(year=y) for y in HISTORICAL_YEARS
    ]

    all_urls: set[str] = set()
    for page in calendar_pages:
        log.info("discover: %s", page)
        resp = polite_get(session, page)
        if resp is None:
            log.warning("discover: 404 for %s", page)
            continue
        found = extract_statement_urls(resp.text)
        log.info("discover: found %d statement links on %s", len(found), page)
        all_urls.update(found)

    # Multiple press releases can share a date (e.g. monetary20220504a.htm is
    # the FOMC statement, while ...b.htm is a separate Balance Sheet release).
    # Keep one URL per date, preferring the alphabetically earliest suffix:
    # 'a' is consistently the monetary-policy statement.
    by_date: dict[dt.date, tuple[str, str]] = {}  # date -> (suffix, url)
    for url in all_urls:
        m = STATEMENT_HREF_RE.search(url)
        if not m:
            continue
        date_str, suffix = m.group(1), m.group(2).lower()
        try:
            meeting_date = dt.datetime.strptime(date_str, "%Y%m%d").date()
        except ValueError:
            log.warning("discover: bad date in url %s", url)
            continue
        if meeting_date.year < 2006:
            continue
        prev = by_date.get(meeting_date)
        if prev is None or suffix < prev[0]:
            by_date[meeting_date] = (suffix, url)

    refs = [
        StatementRef(meeting_date=d, url=u)
        for d, (_, u) in sorted(by_date.items())
    ]
    return refs


# --------------------------------------------------------------------------
# Step 2: download and parse each statement
# --------------------------------------------------------------------------


def parse_statement(html: str) -> str:
    """Extract the plain-text statement body, dropping related-info/nav."""
    soup = BeautifulSoup(html, "html.parser")

    # Strip script/style/nav globally.
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    # The statement body lives inside the main article area. Fed pages have
    # varied over time; try a few selectors in priority order.
    candidates = [
        soup.select_one("div#article"),
        soup.select_one("div.col-xs-12.col-sm-8.col-md-8"),
        soup.select_one("div#content"),
        soup.select_one("main"),
    ]
    container = next((c for c in candidates if c is not None), soup.body or soup)

    # Drop any "Related Information" / "Implementation Note" trailing block.
    # These appear as a heading followed by links to other press releases.
    for heading in container.find_all(re.compile(r"^h[1-6]$")):
        text = heading.get_text(strip=True).lower()
        if "implementation note" in text or "related information" in text:
            # Remove this heading and every sibling after it.
            for sib in list(heading.find_all_next()):
                sib.decompose()
            heading.decompose()
            break

    # Also remove any standalone anchors pointing at implementation notes
    # (some pages list them inline without a heading).
    for a in container.find_all("a", href=True):
        if re.search(r"monetary\d{8}a1\.htm", a["href"]):
            # Remove the enclosing paragraph if it exists, else just the link.
            p = a.find_parent("p")
            (p or a).decompose()

    paragraphs = [
        p.get_text(" ", strip=True) for p in container.find_all("p")
    ]
    paragraphs = [p for p in paragraphs if p]

    # Belt-and-suspenders: drop any paragraph that is itself the impl-note
    # link (2016-era pages keep the link as a trailing paragraph whose text
    # is literally "Implementation Note issued Month DD, YYYY").
    paragraphs = [
        p for p in paragraphs
        if not re.match(r"implementation note\b", p, re.IGNORECASE)
    ]
    text = "\n\n".join(paragraphs)

    # Collapse runs of whitespace inside paragraphs.
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def fetch_statement(
    session: requests.Session, ref: StatementRef
) -> tuple[str, str] | None:
    """Return (final_url, body_text) or None if the URL 404s."""
    resp = polite_get(session, ref.url)
    if resp is None:
        return None
    body = parse_statement(resp.text)
    if not body or len(body) < 200:
        log.warning(
            "parse: suspiciously short body for %s (%d chars)",
            ref.url,
            len(body),
        )
    return resp.url, body


def write_outputs(
    ref: StatementRef, final_url: str, body: str
) -> dict[str, str | int]:
    out_path = RAW_DIR / ref.filename
    out_path.write_text(body, encoding="utf-8")
    return {
        "meeting_date": ref.meeting_date.isoformat(),
        "url": final_url,
        "char_count": len(body),
        "scraped_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


# --------------------------------------------------------------------------
# FRED macro indicators
# --------------------------------------------------------------------------


def pull_macro_indicators(meeting_dates: list[dt.date]) -> None:
    """Pull FRED series and join as-of each FOMC meeting date.

    For each meeting date, takes the most recent observation on or before the
    meeting date per series (no interpolation, no forward-fill past the last
    observation). Writes ``data/macro_indicators.csv`` with columns
    ``meeting_date, FEDFUNDS, DGS10, CPIAUCSL, UNRATE``.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set in environment (.env)")

    fred = Fred(api_key=api_key)

    meetings = pd.DataFrame(
        {"meeting_date": pd.to_datetime(sorted(set(meeting_dates)))}
    ).sort_values("meeting_date")

    out = meetings.copy()
    for series_id in FRED_SERIES:
        log.info("fred: pulling %s", series_id)
        s = fred.get_series(series_id, observation_start=FRED_OBSERVATION_START)
        series_df = (
            s.rename(series_id)
            .rename_axis("observation_date")
            .reset_index()
            .dropna(subset=[series_id])
            .sort_values("observation_date")
        )
        series_df["observation_date"] = pd.to_datetime(
            series_df["observation_date"]
        )
        merged = pd.merge_asof(
            out[["meeting_date"]],
            series_df,
            left_on="meeting_date",
            right_on="observation_date",
            direction="backward",
        )
        out[series_id] = merged[series_id].values

    out["meeting_date"] = out["meeting_date"].dt.date.astype(str)
    MACRO_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    out[["meeting_date", *FRED_SERIES]].to_csv(MACRO_CSV_PATH, index=False)
    log.info(
        "fred: wrote %d rows to %s", len(out), MACRO_CSV_PATH
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()

    refs = discover_statement_urls(session)
    log.info("discover: %d unique statement URLs (2006+)", len(refs))

    URL_LIST_PATH.write_text(
        "\n".join(f"{r.meeting_date.isoformat()}\t{r.url}" for r in refs),
        encoding="utf-8",
    )

    manifest_rows: list[dict[str, str | int]] = []
    for ref in refs:
        out_path = RAW_DIR / ref.filename
        if out_path.exists():
            log.info("skip (cached): %s", ref.filename)
            manifest_rows.append(
                {
                    "meeting_date": ref.meeting_date.isoformat(),
                    "url": ref.url,
                    "char_count": len(out_path.read_text(encoding="utf-8")),
                    "scraped_at": "cached",
                }
            )
            continue

        log.info("fetch: %s", ref.url)
        result = fetch_statement(session, ref)
        if result is None:
            log.warning("fetch: 404 for %s", ref.url)
            continue
        final_url, body = result
        manifest_rows.append(write_outputs(ref, final_url, body))

    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["meeting_date", "url", "char_count", "scraped_at"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    log.info(
        "done: %d statements written to %s", len(manifest_rows), RAW_DIR
    )

    meeting_dates = [
        dt.date.fromisoformat(str(row["meeting_date"]))
        for row in manifest_rows
    ]
    pull_macro_indicators(meeting_dates)


if __name__ == "__main__":
    main()
