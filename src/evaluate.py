"""Tier 1 ablation: stance classification accuracy across 4 retrieval modes.

Loads ``data/golden_test_set.csv``, queries ``answer()`` once per (row, mode),
and writes per-mode raw outputs plus a summary table. The summary reports
stance accuracy and (for the 3 retrieval modes) precision@5 — the fraction of
retrieved chunks whose ``meeting_date`` equals the target meeting.

Usage:
    py -3.13 src/evaluate.py                      # full run, all 4 modes
    py -3.13 src/evaluate.py --limit 3            # smoke-test first 3 rows
    py -3.13 src/evaluate.py --modes baseline hybrid
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter
from pathlib import Path
from typing import get_args

import pandas as pd

from generate import Mode, VALID_STANCES, answer

log = logging.getLogger("evaluate")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_test_set.csv"
RUNS_DIR = PROJECT_ROOT / "data" / "eval_runs"
SUMMARY_PATH = PROJECT_ROOT / "data" / "eval_summary.csv"

ALL_MODES: list[Mode] = list(get_args(Mode))  # keeps modes in one source of truth
RETRIEVAL_MODES = {"semantic_only", "bm25_only", "hybrid"}

QUERY_TEMPLATE = "Classify the Fed's stance at the FOMC meeting on {meeting_date}."


VALID_ERAS = {"pre_cutoff", "post_cutoff"}


def _load_golden() -> pd.DataFrame:
    df = pd.read_csv(GOLDEN_PATH)
    required = {"meeting_date", "ground_truth_stance", "training_era"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"golden set missing columns: {missing}")
    df["ground_truth_stance"] = df["ground_truth_stance"].str.strip().str.lower()
    df["training_era"] = df["training_era"].str.strip().str.lower()
    invalid_stance = set(df["ground_truth_stance"]) - VALID_STANCES
    if invalid_stance:
        raise ValueError(f"golden set has invalid stances: {invalid_stance}")
    invalid_era = set(df["training_era"]) - VALID_ERAS
    if invalid_era:
        raise ValueError(f"golden set has invalid training_era values: {invalid_era}")
    return df


def _run_mode(df: pd.DataFrame, mode: Mode, k: int) -> pd.DataFrame:
    """Call answer() for every golden row in this mode; return a result frame."""
    rows = []
    for i, row in enumerate(df.itertuples(index=False), 1):
        query = QUERY_TEMPLATE.format(meeting_date=row.meeting_date)
        log.info("[%s] %d/%d  %s", mode, i, len(df), row.meeting_date)
        out = answer(query, mode=mode, k=k)
        retrieved_dates = [r["meeting_date"] for r in out["retrieved"]]
        rows.append(
            {
                "meeting_date": row.meeting_date,
                "training_era": row.training_era,
                "ground_truth_stance": row.ground_truth_stance,
                "predicted_stance": out["stance"],
                "correct": int(out["stance"] == row.ground_truth_stance),
                "citations": ";".join(out["citations"]),
                "retrieved_dates": ";".join(retrieved_dates),
                "precision_at_k": (
                    sum(1 for d in retrieved_dates if d == row.meeting_date)
                    / len(retrieved_dates)
                    if retrieved_dates
                    else 0.0
                ),
                "reasoning": out["reasoning"],
            }
        )
    return pd.DataFrame(rows)


def _confusion(results: pd.DataFrame) -> pd.DataFrame:
    labels = sorted(VALID_STANCES)
    # actual rows, predicted columns
    mat = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for gt, pred in zip(results["ground_truth_stance"], results["predicted_stance"]):
        if pred in VALID_STANCES:
            mat.loc[gt, pred] += 1
    return mat


def _summarize(per_mode: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (era, mode). Era "overall" aggregates across both eras."""
    rows = []
    for mode, df in per_mode.items():
        for era in ("overall", "pre_cutoff", "post_cutoff"):
            sub = df if era == "overall" else df[df["training_era"] == era]
            n = len(sub)
            if n == 0:
                continue
            correct = int(sub["correct"].sum())
            row = {
                "training_era": era,
                "mode": mode,
                "n": n,
                "accuracy": correct / n,
                "correct": correct,
                "invalid_stance_outputs": int((~sub["predicted_stance"].isin(VALID_STANCES)).sum()),
                "mean_precision_at_k": (
                    float(sub["precision_at_k"].mean()) if mode in RETRIEVAL_MODES else float("nan")
                ),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _print_confusion(mode: str, mat: pd.DataFrame) -> None:
    print(f"\n[confusion — {mode}]  rows=actual, cols=predicted")
    print(mat.to_string())


def _print_summary(summary: pd.DataFrame) -> None:
    for era in ("overall", "pre_cutoff", "post_cutoff"):
        sub = summary[summary["training_era"] == era]
        if sub.empty:
            continue
        print(f"\n=== summary — {era} ===")
        view = sub.drop(columns=["training_era"]).copy()
        view["accuracy"] = view["accuracy"].map(lambda x: f"{x:.3f}")
        view["mean_precision_at_k"] = view["mean_precision_at_k"].map(
            lambda x: "—" if pd.isna(x) else f"{x:.3f}"
        )
        print(view.to_string(index=False))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="Stance-classification ablation.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=ALL_MODES,
        choices=ALL_MODES,
        help="subset of modes to evaluate (default: all 4)",
    )
    parser.add_argument("--k", type=int, default=5, help="retrieved chunks per query")
    parser.add_argument(
        "--limit", type=int, default=None, help="only evaluate first N golden rows"
    )
    args = parser.parse_args()

    df = _load_golden()
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    log.info("golden set: %d rows, stance counts: %s", len(df), Counter(df["ground_truth_stance"]))

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    per_mode: dict[str, pd.DataFrame] = {}
    for mode in args.modes:
        log.info("running mode=%s", mode)
        results = _run_mode(df, mode, args.k)
        out_path = RUNS_DIR / f"{mode}.csv"
        results.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
        log.info("wrote %s", out_path)
        per_mode[mode] = results
        _print_confusion(mode, _confusion(results))

    summary = _summarize(per_mode)
    summary.to_csv(SUMMARY_PATH, index=False)
    log.info("wrote %s", SUMMARY_PATH)
    _print_summary(summary)


if __name__ == "__main__":
    main()
