"""Tier 2 evaluator: citation recall (objective) + LLM-as-judge groundedness/relevance.

Loops over ``data/golden_informational_set.csv`` × {baseline, semantic_only,
bm25_only, hybrid}, calls ``generate_informational.answer_informational``, then
issues a separate LLM judge call per (row, retrieval_mode) — baseline is
skipped from judging because ``citations=[]`` by construction.

Per-row results land in ``data/eval_informational_runs/{mode}.csv`` and a
summary table (per training_era × mode) in ``data/eval_informational_summary.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import get_args

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from generate_informational import (
    MODEL_NAME,
    Mode,
    _render_context,
    answer_informational,
)

log = logging.getLogger("evaluate_informational")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_informational_set.csv"
RUNS_DIR = PROJECT_ROOT / "data" / "eval_informational_runs"
SUMMARY_PATH = PROJECT_ROOT / "data" / "eval_informational_summary.csv"

ALL_MODES: list[Mode] = list(get_args(Mode))
RETRIEVAL_MODES = {"semantic_only", "bm25_only", "hybrid"}

CUTOFF = "2023-10-01"


JUDGE_SYSTEM = (
    "You are evaluating a retrieval-augmented assistant's answer. Score it "
    "ONLY on what can be verified from the retrieved chunks you are shown. "
    "Do not use outside knowledge. Return JSON."
)

JUDGE_USER_TEMPLATE = """\
[Question]
{query}

[Assistant's answer]
{answer}

[Assistant's citations]
{citations}

[Retrieved chunks the assistant saw]
{chunks_with_headers}

Score on a 1-5 scale:
- groundedness: 5 = every claim in the answer is directly supported by the
  retrieved chunks. 3 = most claims supported, some mild extrapolation. 1 =
  answer contradicts the chunks or invents facts not present.
- relevance: 5 = answer directly addresses the question. 3 = partially
  addresses it. 1 = does not address the question.

Output JSON: {{"groundedness": int 1-5, "relevance": int 1-5, "rationale": str}}
"""


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    import os

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment (.env)")
    return OpenAI(api_key=api_key)


def _classify_era(expected_dates: list[str]) -> str:
    """training_era: pre_cutoff / post_cutoff / mixed by 2023-10-01 boundary."""
    if not expected_dates:
        return "post_cutoff"  # shouldn't happen
    pre = [d for d in expected_dates if d <= CUTOFF]
    post = [d for d in expected_dates if d > CUTOFF]
    if pre and post:
        return "mixed"
    return "pre_cutoff" if pre else "post_cutoff"


def _load_golden() -> pd.DataFrame:
    df = pd.read_csv(GOLDEN_PATH, dtype=str).fillna("")
    required = {"query", "expected_meeting_dates", "expected_fred_series", "notes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"golden informational set missing columns: {missing}")
    df["expected_dates_list"] = df["expected_meeting_dates"].apply(
        lambda s: [d.strip() for d in s.split(";") if d.strip()]
    )
    df["training_era"] = df["expected_dates_list"].apply(_classify_era)
    return df


def _citation_metrics(citations: list[str], expected: list[str]) -> dict:
    expected_set = set(expected)
    cite_set = set(citations)
    hits = len(expected_set & cite_set)
    expected_count = len(expected_set)
    recall = hits / expected_count if expected_count else float("nan")
    precision = hits / len(cite_set) if cite_set else float("nan")
    if (
        not isinstance(precision, float)
        or math.isnan(precision)
        or math.isnan(recall)
        or precision + recall == 0.0
    ):
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "citation_hits": hits,
        "expected_count": expected_count,
        "citation_recall": recall,
        "citation_precision": precision,
        "citation_f1": f1,
    }


def _judge(query: str, answer: str, citations: list[str], retrieved: list[dict]) -> dict:
    """Run a fresh LLM-as-judge call. Returns groundedness, relevance, rationale.

    The judge sees the same macro-enriched context the generator saw — but
    never sees expected_meeting_dates.
    """
    chunks = _render_context(retrieved) if retrieved else "(none)"
    user = JUDGE_USER_TEMPLATE.format(
        query=query,
        answer=answer or "(empty)",
        citations=";".join(citations) if citations else "(none)",
        chunks_with_headers=chunks,
    )
    resp = _client().chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        obj = json.loads(raw)
        g = obj.get("groundedness")
        r = obj.get("relevance")
        rat = str(obj.get("rationale", "")).strip()
        if not (isinstance(g, int) and 1 <= g <= 5):
            log.warning("judge returned invalid groundedness=%r; storing NaN", g)
            g = float("nan")
        if not (isinstance(r, int) and 1 <= r <= 5):
            log.warning("judge returned invalid relevance=%r; storing NaN", r)
            r = float("nan")
        return {"groundedness": g, "relevance": r, "judge_rationale": rat}
    except (json.JSONDecodeError, TypeError) as e:
        log.warning("judge JSON parse failed (%s); raw=%s", e, raw)
        return {
            "groundedness": float("nan"),
            "relevance": float("nan"),
            "judge_rationale": raw.strip(),
        }


def _run_mode(df: pd.DataFrame, mode: Mode, k: int) -> pd.DataFrame:
    rows = []
    for i, row in enumerate(df.itertuples(index=False), 1):
        log.info("[%s] %d/%d  %s", mode, i, len(df), row.query[:60])
        out = answer_informational(row.query, mode=mode, k=k)
        retrieved_dates = [r["meeting_date"] for r in out["retrieved"]]
        cite_metrics = _citation_metrics(out["citations"], row.expected_dates_list)

        record = {
            "query_id": i - 1,
            "query": row.query,
            "training_era": row.training_era,
            "mode": mode,
            "answer": out["answer"],
            "citations": ";".join(out["citations"]),
            "retrieved_dates": ";".join(retrieved_dates),
            "expected_meeting_dates": row.expected_meeting_dates,
            **cite_metrics,
        }

        if mode in RETRIEVAL_MODES:
            judge = _judge(row.query, out["answer"], out["citations"], out["retrieved"])
        else:
            judge = {
                "groundedness": float("nan"),
                "relevance": float("nan"),
                "judge_rationale": "",
            }
        record.update(judge)
        rows.append(record)
    return pd.DataFrame(rows)


def _summarize(per_mode: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    eras = ("overall", "pre_cutoff", "post_cutoff", "mixed")
    for mode, df in per_mode.items():
        for era in eras:
            sub = df if era == "overall" else df[df["training_era"] == era]
            n = len(sub)
            if n == 0:
                continue
            rows.append(
                {
                    "training_era": era,
                    "mode": mode,
                    "n": n,
                    "citation_recall_mean": float(sub["citation_recall"].mean(skipna=True)),
                    "citation_precision_mean": float(
                        sub["citation_precision"].mean(skipna=True)
                    ),
                    "citation_f1_mean": float(sub["citation_f1"].mean(skipna=True)),
                    "groundedness_mean": (
                        float(sub["groundedness"].mean(skipna=True))
                        if mode in RETRIEVAL_MODES
                        else float("nan")
                    ),
                    "relevance_mean": (
                        float(sub["relevance"].mean(skipna=True))
                        if mode in RETRIEVAL_MODES
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def _print_summary(summary: pd.DataFrame) -> None:
    for era in ("overall", "pre_cutoff", "post_cutoff", "mixed"):
        sub = summary[summary["training_era"] == era]
        if sub.empty:
            continue
        print(f"\n=== summary — {era} ===")
        view = sub.drop(columns=["training_era"]).copy()
        for col in (
            "citation_recall_mean",
            "citation_precision_mean",
            "citation_f1_mean",
            "groundedness_mean",
            "relevance_mean",
        ):
            view[col] = view[col].map(lambda x: "—" if pd.isna(x) else f"{x:.3f}")
        print(view.to_string(index=False))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="Tier 2 informational evaluation.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=ALL_MODES,
        choices=ALL_MODES,
        help="subset of modes to evaluate (default: all 4)",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--limit", type=int, default=None, help="only evaluate first N golden rows"
    )
    args = parser.parse_args()

    df = _load_golden()
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    log.info(
        "golden informational set: %d rows, era counts: %s",
        len(df),
        df["training_era"].value_counts().to_dict(),
    )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    per_mode: dict[str, pd.DataFrame] = {}
    for mode in args.modes:
        log.info("running mode=%s", mode)
        results = _run_mode(df, mode, args.k)
        out_path = RUNS_DIR / f"{mode}.csv"
        results.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
        log.info("wrote %s", out_path)
        per_mode[mode] = results

    summary = _summarize(per_mode)
    summary.to_csv(SUMMARY_PATH, index=False)
    log.info("wrote %s", SUMMARY_PATH)
    _print_summary(summary)


if __name__ == "__main__":
    main()
