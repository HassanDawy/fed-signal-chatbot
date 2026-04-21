"""Prompt building + GPT-4o-mini generation.

Public entry point is :func:`answer`, which ``evaluate.py`` will call for the
ablation (``mode`` in ``{"baseline", "semantic_only", "bm25_only", "hybrid"}``).

The system prompt and CoT instruction are copied verbatim from CLAUDE.md's
"Prompt Structure" section. Few-shot examples quote real passages from
``data/raw/{date}.txt`` so the model sees the same prose shape retrieval
will surface at inference time.
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

from retrieve import bm25_search, hybrid_search, semantic_search

log = logging.getLogger("generate")

Mode = Literal["baseline", "semantic_only", "bm25_only", "hybrid"]
VALID_STANCES = {"hawkish", "neutral", "dovish"}
MODEL_NAME = "gpt-4o-mini"

SYSTEM_PROMPT_TEMPLATE = (
    "You are a macroeconomic research assistant. Your job is to classify "
    "the Federal Reserve's stance and provide historical context. "
    "Always cite the specific FOMC meeting minutes you are drawing from. "
    "Do not provide investment advice.\n"
    "Today's date is {today}. When the user asks about the Fed's current, "
    "recent, or present stance, weight the most recent meeting dates in the "
    "retrieved context most heavily; older meetings are historical context, "
    "not the current stance."
)


def _system_prompt() -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(today=dt.date.today().isoformat())

COT_INSTRUCTION = (
    "Think step by step: (1) What language signals hawkish/neutral/dovish tone? "
    "(2) What does the retrieved context say? (3) What is your stance classification?"
)

OUTPUT_INSTRUCTION = (
    "Respond with a single JSON object with exactly three fields:\n"
    '  "reasoning": string — your step-by-step chain of thought answering the three questions above.\n'
    '  "stance":    string — exactly one of "hawkish", "neutral", or "dovish".\n'
    '  "citations": array of strings — the meeting_date values (YYYY-MM-DD) from the retrieved '
    "context that you actually drew on. Use only dates present in the retrieved context; "
    "return an empty array if no retrieved context was provided."
)

# Three few-shot examples — one per stance — drawn from real FOMC statements.
FEW_SHOT_EXAMPLES = [
    {
        "query": "Is the Fed tightening aggressively to fight inflation?",
        "context": [
            (
                "2022-06-15",
                "The Committee decided to raise the target range for the federal funds "
                "rate to 1-1/2 to 1-3/4 percent and anticipates that ongoing increases "
                "in the target range will be appropriate. The Committee is strongly "
                "committed to returning inflation to its 2 percent objective.",
            ),
        ],
        "answer": {
            "reasoning": (
                "(1) 'ongoing increases ... will be appropriate' and 'strongly committed "
                "to returning inflation' are classic hawkish signals — forward guidance "
                "toward more tightening plus an anti-inflation commitment. "
                "(2) The context shows a 75bp hike to 1-1/2 to 1-3/4 percent. "
                "(3) Aggressive tightening in progress — hawkish."
            ),
            "stance": "hawkish",
            "citations": ["2022-06-15"],
        },
    },
    {
        "query": "How is the Fed responding to the pandemic shock?",
        "context": [
            (
                "2020-03-15",
                "The Committee decided to lower the target range for the federal funds "
                "rate to 0 to 1/4 percent. Over coming months the Committee will "
                "increase its holdings of Treasury securities by at least $500 billion "
                "and its holdings of agency mortgage-backed securities by at least "
                "$200 billion.",
            ),
        ],
        "answer": {
            "reasoning": (
                "(1) Cutting to the zero lower bound and restarting large-scale asset "
                "purchases are unambiguously easing actions. "
                "(2) The context shows a cut to 0-0.25% plus $700B of asset purchases. "
                "(3) Emergency accommodation — dovish."
            ),
            "stance": "dovish",
            "citations": ["2020-03-15"],
        },
    },
    {
        "query": "Is the Fed leaning toward raising rates or keeping them steady?",
        "context": [
            (
                "2016-09-21",
                "The Committee decided to maintain the target range for the federal "
                "funds rate at 1/4 to 1/2 percent. The Committee judges that the case "
                "for an increase in the federal funds rate has strengthened but decided, "
                "for the time being, to wait for further evidence of continued progress "
                "toward its objectives.",
            ),
        ],
        "answer": {
            "reasoning": (
                "(1) 'Case has strengthened' leans hawkish but 'decided ... to wait' "
                "is a hold — the statement deliberately balances both sides. "
                "(2) No rate change; a tightening bias is acknowledged but not acted on. "
                "(3) Hold with a mild tightening bias — neutral."
            ),
            "stance": "neutral",
            "citations": ["2016-09-21"],
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


def _render_context(chunks: list[tuple[str, str]]) -> str:
    """Render (date, text) pairs with the `--- Meeting: {date} ---` header."""
    return "\n\n".join(f"--- Meeting: {d} ---\n{t}" for d, t in chunks)


def _render_few_shot(example: dict) -> list[dict]:
    """Turn one few-shot example into a user/assistant message pair."""
    user = (
        f"[Retrieved context]\n{_render_context(example['context'])}\n\n"
        f"[User query]\n{example['query']}\n\n"
        f"[CoT instruction]\n{COT_INSTRUCTION}\n\n"
        f"[Output format]\n{OUTPUT_INSTRUCTION}"
    )
    assistant = json.dumps(example["answer"])
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def _build_messages(query: str, retrieved: list[dict]) -> list[dict]:
    """Assemble the full message list: system, few-shots, live query."""
    messages: list[dict] = [{"role": "system", "content": _system_prompt()}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.extend(_render_few_shot(ex))

    if retrieved:
        ctx = _render_context([(r["meeting_date"], r["text"]) for r in retrieved])
        ctx_block = f"[Retrieved context]\n{ctx}"
    else:
        ctx_block = "[Retrieved context]\n(no retrieved context — answer from general knowledge)"

    user = (
        f"{ctx_block}\n\n"
        f"[User query]\n{query}\n\n"
        f"[CoT instruction]\n{COT_INSTRUCTION}\n\n"
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


_STANCE_RE = re.compile(r"\b(hawkish|neutral|dovish)\b", re.IGNORECASE)


def _parse_response(raw: str, retrieved_dates: set[str]) -> tuple[str, str, list[str]]:
    """Parse the model's JSON output; fall back to regex on malformed output.

    Returns (stance, reasoning, citations). Citations are filtered to dates
    that actually appeared in the retrieved context.
    """
    try:
        obj = json.loads(raw)
        stance = str(obj.get("stance", "")).strip().lower()
        reasoning = str(obj.get("reasoning", "")).strip()
        citations = [str(c) for c in obj.get("citations", []) if isinstance(c, str)]
        if stance in VALID_STANCES:
            filtered = [c for c in citations if c in retrieved_dates] if retrieved_dates else []
            return stance, reasoning, filtered
        log.warning("model returned invalid stance %r; falling back to regex", stance)
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        log.warning("JSON parse failed (%s); falling back to regex", e)

    matches = _STANCE_RE.findall(raw)
    stance = matches[-1].lower() if matches else "unknown"
    reasoning = raw.strip()
    citations = sorted(retrieved_dates) if retrieved_dates else []
    return stance, reasoning, citations


def answer(query: str, mode: Mode = "hybrid", k: int = 5) -> dict:
    """Retrieve (if applicable), prompt GPT-4o-mini, return parsed result.

    Returns a dict with keys: query, mode, stance, reasoning, citations,
    raw_response, retrieved.
    """
    log.info("answer(mode=%s, k=%d): %s", mode, k, query)

    retrieved = _retrieve(query, mode, k)
    messages = _build_messages(query, retrieved)

    resp = _client().chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or ""
    log.debug("raw model output: %s", raw)

    retrieved_dates = {r["meeting_date"] for r in retrieved}
    stance, reasoning, citations = _parse_response(raw, retrieved_dates)

    # Baseline has no retrieval, so no grounded citations are possible —
    # force the list empty even if the model echoed few-shot dates.
    if mode == "baseline":
        citations = []

    return {
        "query": query,
        "mode": mode,
        "stance": stance,
        "reasoning": reasoning,
        "citations": citations,
        "raw_response": raw,
        "retrieved": retrieved,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="Smoke-test generate.py.")
    parser.add_argument("--query", default="Is the Fed currently hawkish?")
    parser.add_argument(
        "--mode",
        default="all",
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
        out = answer(args.query, mode=m, k=args.k)
        print(f"  stance:    {out['stance']}")
        print(f"  citations: {out['citations']}")
        print(f"  reasoning: {out['reasoning'][:200]}")


if __name__ == "__main__":
    main()
