"""Tier 2 informational generator: open-ended Fed questions with macro enrichment.

Mirrors :mod:`generate` in structure but returns a free-form 2-4 sentence
``answer`` instead of a stance label. Each retrieved chunk's context block is
augmented post-retrieval with a one-line macro snapshot via
``macro.format_macro_for_date``. Retrieval architecture is unchanged.

The Tier 1 stance generator deliberately does not consume macro enrichment, so
the existing Tier 1 ablation results stay reproducible.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI

from macro import format_macro_for_date
from query_dates import augment_with_date_snap, chunks_for_meetings, extract_meeting_dates_from_query
from retrieve import bm25_search, hybrid_search, semantic_search

log = logging.getLogger("generate_informational")

Mode = Literal["baseline", "semantic_only", "bm25_only", "hybrid"]
VALID_MODES = {"baseline", "semantic_only", "bm25_only", "hybrid"}
MODEL_NAME = "gpt-4o-mini"

SYSTEM_PROMPT_TEMPLATE = (
    "You are a macroeconomic research assistant. Given retrieved FOMC "
    "statement chunks (with macro indicator snapshots), answer the user's "
    "question concisely (2-4 sentences) and cite the meeting dates you drew "
    "from. If the retrieved context does not contain enough information, "
    "say so rather than guessing.\n\n"
    "Some meetings may be marked '(force-included: query references this date)' "
    "in the context header. Treat those as the primary source for date-specific "
    "questions — they were injected because the user explicitly referenced "
    "their period.\n\n"
    "If the user references a date or period that has no corresponding FOMC "
    "meeting in the retrieved context, mention the closest meeting available "
    "and ask the user to confirm whether they meant that one. Do not provide "
    "investment advice.\n"
    "Today's date is {today}."
)


def _system_prompt() -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(today=dt.date.today().isoformat())


OUTPUT_INSTRUCTION = (
    "Respond with a JSON object with three fields:\n"
    '  "reasoning": string — brief chain of thought.\n'
    '  "answer":    string — 2-4 sentence answer. If context is insufficient, say so.\n'
    '  "citations": array of YYYY-MM-DD strings from the retrieved context.'
)


# Few-shot examples — none of these meetings appear in the Tier 1 few-shots
# (2022-06-15, 2020-03-15, 2016-09-21) or in golden_informational_set.csv.
# Each example shows the model the macro-enriched context format.
FEW_SHOT_EXAMPLES = [
    {
        # Factual single-meeting.
        "query": "What did the Fed do at its September 2007 meeting and why?",
        "context": [
            (
                "2007-09-18",
                "The Federal Open Market Committee decided today to lower its target "
                "for the federal funds rate 50 basis points to 4-3/4 percent. "
                "Economic growth was moderate during the first half of the year, "
                "but the tightening of credit conditions has the potential to "
                "intensify the housing correction and to restrain economic growth "
                "more generally. Today's action is intended to help forestall some "
                "of the adverse effects on the broader economy that might otherwise "
                "arise from the disruptions in financial markets.",
            ),
        ],
        "answer": {
            "reasoning": (
                "Single-meeting factual question. The 2007-09-18 statement is "
                "explicit on both the action (50bp cut to 4.75%) and the rationale "
                "(credit tightening and housing correction risk)."
            ),
            "answer": (
                "At the September 18, 2007 meeting the FOMC cut the federal funds "
                "target by 50 basis points to 4.75 percent, citing the risk that "
                "the tightening of credit conditions could intensify the housing "
                "correction and weigh on broader growth. The Committee framed the "
                "move as forestalling adverse effects from financial-market "
                "disruptions while continuing to monitor inflation risks."
            ),
            "citations": ["2007-09-18"],
        },
    },
    {
        # Comparative multi-meeting.
        "query": "How did the Fed's stance on the federal funds rate differ between December 2014 and December 2018?",
        "context": [
            (
                "2014-12-17",
                "To support continued progress toward maximum employment and price "
                "stability, the Committee today reaffirmed its view that the current "
                "0 to 1/4 percent target range for the federal funds rate remains "
                "appropriate. Based on its current assessment, the Committee judges "
                "that it can be patient in beginning to normalize the stance of "
                "monetary policy.",
            ),
            (
                "2018-12-19",
                "In view of realized and expected labor market conditions and "
                "inflation, the Committee decided to raise the target range for the "
                "federal funds rate to 2-1/4 to 2-1/2 percent. The Committee judges "
                "that some further gradual increases in the target range for the "
                "federal funds rate will be consistent with sustained expansion of "
                "economic activity.",
            ),
        ],
        "answer": {
            "reasoning": (
                "Both statements are about the funds rate but describe opposite "
                "phases of the cycle: 2014-12-17 holds at the zero lower bound and "
                "promises patience; 2018-12-19 hikes to 2.25-2.50% and signals "
                "further gradual increases."
            ),
            "answer": (
                "In December 2014 the Fed kept the target range at 0-0.25 percent "
                "and signaled it could be 'patient' before beginning to normalize "
                "policy. By December 2018, after several years of hiking, the "
                "Committee raised the range to 2.25-2.50 percent and indicated some "
                "further gradual increases would be appropriate, marking a shift "
                "from accommodation to ongoing tightening."
            ),
            "citations": ["2014-12-17", "2018-12-19"],
        },
    },
    {
        # Insufficient-context refusal.
        "query": "What did the Fed announce about cryptocurrency regulation at its December 2018 meeting?",
        "context": [
            (
                "2018-12-19",
                "In view of realized and expected labor market conditions and "
                "inflation, the Committee decided to raise the target range for the "
                "federal funds rate to 2-1/4 to 2-1/2 percent. The Committee judges "
                "that some further gradual increases in the target range for the "
                "federal funds rate will be consistent with sustained expansion of "
                "economic activity, strong labor market conditions, and inflation "
                "near the Committee's symmetric 2 percent objective.",
            ),
            (
                "2014-12-17",
                "The Committee is maintaining its existing policy of reinvesting "
                "principal payments from its holdings of agency debt and agency "
                "mortgage-backed securities in agency mortgage-backed securities "
                "and of rolling over maturing Treasury securities at auction.",
            ),
        ],
        "answer": {
            "reasoning": (
                "The retrieved chunks describe the federal funds rate decision and "
                "balance-sheet reinvestment policy. Neither chunk mentions "
                "cryptocurrency or related regulatory action, so the question "
                "cannot be answered from this context."
            ),
            "answer": (
                "The retrieved context does not contain information about "
                "cryptocurrency regulation at the December 2018 FOMC meeting. The "
                "available statements address the federal funds rate decision and "
                "balance-sheet reinvestment, not regulatory matters."
            ),
            "citations": [],
        },
    },
]


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment (.env)")
    return OpenAI(api_key=api_key)


def _render_block(meeting_date: str, text: str, forced: bool = False) -> str:
    """Render one (date, text) block with optional macro snapshot.

    When ``forced`` is True the header is annotated to tell the model this
    chunk was force-included because the user's query mentioned its date.
    """
    marker = " (force-included: query references this date)" if forced else ""
    block = f"--- Meeting: {meeting_date}{marker} ---\n{text}"
    macro_line = format_macro_for_date(meeting_date)
    if macro_line is not None:
        block += f"\n{macro_line}"
    return block


def _render_context_pairs(chunks: list[tuple[str, str]]) -> str:
    return "\n\n".join(_render_block(d, t) for d, t in chunks)


def _render_context(retrieved: list[dict], forced_dates: set[str] | None = None) -> str:
    """Render retrieved chunks with `--- Meeting: {date} ---` headers + macro line."""
    forced_dates = forced_dates or set()
    return "\n\n".join(
        _render_block(r["meeting_date"], r["text"], forced=r["meeting_date"] in forced_dates)
        for r in retrieved
    )


def _render_few_shot(example: dict) -> list[dict]:
    user = (
        f"[Retrieved context]\n{_render_context_pairs(example['context'])}\n\n"
        f"[User query]\n{example['query']}\n\n"
        f"[Output format]\n{OUTPUT_INSTRUCTION}"
    )
    assistant = json.dumps(example["answer"])
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def _build_messages(
    query: str, retrieved: list[dict], forced_dates: set[str] | None = None
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": _system_prompt()}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.extend(_render_few_shot(ex))

    if retrieved:
        ctx_block = f"[Retrieved context]\n{_render_context(retrieved, forced_dates)}"
    else:
        ctx_block = (
            "[Retrieved context]\n(no retrieved context — answer from general knowledge "
            "or say so if unable)"
        )

    user = (
        f"{ctx_block}\n\n"
        f"[User query]\n{query}\n\n"
        f"[Output format]\n{OUTPUT_INSTRUCTION}"
    )
    messages.append({"role": "user", "content": user})
    return messages


def _retrieve(query: str, mode: Mode, k: int) -> list[dict]:
    if mode == "baseline":
        return []
    if mode == "semantic_only":
        return semantic_search(query, k=k)
    if mode == "bm25_only":
        return bm25_search(query, k=k)
    if mode == "hybrid":
        return hybrid_search(query, k=k)
    raise ValueError(f"unknown mode: {mode!r}")


_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def _parse_response(raw: str, retrieved_dates: set[str]) -> tuple[str, str, list[str]]:
    """Parse JSON output; fall back to raw text + regex-extracted dates."""
    try:
        obj = json.loads(raw)
        answer = str(obj.get("answer", "")).strip()
        reasoning = str(obj.get("reasoning", "")).strip()
        citations = [str(c) for c in obj.get("citations", []) if isinstance(c, str)]
        if answer:
            filtered = (
                [c for c in citations if c in retrieved_dates] if retrieved_dates else []
            )
            return answer, reasoning, filtered
        log.warning("model returned empty answer; falling back to raw text")
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        log.warning("JSON parse failed (%s); falling back to raw text", e)

    answer = raw.strip()
    reasoning = ""
    found = _DATE_RE.findall(raw)
    citations = sorted({d for d in found if d in retrieved_dates}) if retrieved_dates else []
    return answer, reasoning, citations


def answer_informational(query: str, mode: Mode = "hybrid", k: int = 5) -> dict:
    """Retrieve, prompt GPT-4o-mini, return parsed informational answer.

    Returns a dict with: query, mode, answer, reasoning, citations,
    retrieved, raw_response.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"unknown mode: {mode!r}")
    log.info("answer_informational(mode=%s, k=%d): %s", mode, k, query)

    retrieved = _retrieve(query, mode, k)
    forced_dates: set[str] = set()
    if mode != "baseline":
        retrieved, forced_dates = augment_with_date_snap(query, retrieved)
    messages = _build_messages(query, retrieved, forced_dates)

    resp = _client().chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or ""
    log.debug("raw model output: %s", raw)

    retrieved_dates = {r["meeting_date"] for r in retrieved}
    answer_text, reasoning, citations = _parse_response(raw, retrieved_dates)

    if mode == "baseline":
        citations = []

    return {
        "query": query,
        "mode": mode,
        "answer": answer_text,
        "reasoning": reasoning,
        "citations": citations,
        "retrieved": retrieved,
        "forced_dates": sorted(forced_dates),
        "raw_response": raw,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="Smoke-test generate_informational.py.")
    parser.add_argument("--query", default="What did the Fed do in March 2020?")
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["baseline", "semantic_only", "bm25_only", "hybrid", "all"],
    )
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    modes: list[Mode] = (
        ["baseline", "semantic_only", "bm25_only", "hybrid"]
        if args.mode == "all"
        else [args.mode]  # type: ignore[list-item]
    )
    for m in modes:
        print(f"\n=== mode={m} ===")
        out = answer_informational(args.query, mode=m, k=args.k)
        print(f"  citations: {out['citations']}")
        print(f"  answer:    {out['answer']}")


if __name__ == "__main__":
    main()
