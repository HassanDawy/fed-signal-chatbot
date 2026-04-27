# Fed Signal Chatbot

A retrieval-augmented system that classifies the Federal Reserve's monetary policy stance — **hawkish, neutral, or dovish** — from FOMC statements, with cited evidence drawn from the actual source documents. The headline contribution is a four-way ablation that isolates the contribution of retrieval from the LLM's pretraining priors, with an era-stratified split that exposes a non-obvious result: hybrid retrieval is **not** strictly better than BM25 on date-targeted queries.

DS593 Applied LLMs · Solo Project · Spring 2026

---

## Problem and Motivation

Retail investors and finance students often want to understand where the Fed stands on monetary policy. A vanilla LLM produces plausible-sounding answers that are unverifiable and prone to hallucination — and worse, when asked about famous past meetings, it answers from memorized pretraining text rather than reasoning over evidence.

This system grounds every answer in actual FOMC statements, classifies the stance with chain-of-thought reasoning, and cites the meeting dates it drew on. It also lets us measure something the vanilla LLM hides: how much of its apparent "performance" is just memorized pretraining versus real retrieval.

---

## Approach

### Pipeline

1. **Ingest.** 175 FOMC statements (2006-01-31 → present) scraped from `federalreserve.gov`, deduped to one statement per meeting date. Macro indicators (FEDFUNDS, DGS10, CPIAUCSL, UNRATE) are pulled alongside via the FRED API and joined as-of each meeting date.
2. **Index.** Each statement is chunked into 256-token segments with 32-token overlap (sized to the MiniLM-L6-v2 context window so nothing gets silently truncated at embed time). 572 chunks total. Stored in two parallel indexes:
   - **Dense:** ChromaDB with `sentence-transformers/all-MiniLM-L6-v2` embeddings (cosine, L2-normalized).
   - **Lexical:** `rank_bm25.BM25Okapi` with a word-level regex tokenizer (BM25's IDF needs whole words, not subwords).
3. **Retrieve.** Top-5 chunks per query, in one of four modes:
   - `baseline` — no retrieval (LLM answers from priors only).
   - `semantic_only` — ChromaDB cosine similarity.
   - `bm25_only` — BM25 only.
   - `hybrid` — Reciprocal Rank Fusion of both, `1/(60 + rank)` per leg.
4. **Generate.** GPT-4o-mini with a system prompt + 3 few-shot examples (one per stance, drawn verbatim from real statements) + retrieved context + chain-of-thought prompt. Output is JSON-mode structured: `{stance, reasoning, citations}`.

### Key Design Decisions

| Decision | Why |
|---|---|
| MiniLM-L6-v2 (vs OpenAI embeddings) | Free, local, fits the corpus; embedding cost was not the bottleneck. |
| 256/32 chunking (vs 512) | Matches embedder context window — bigger chunks would be silently truncated. |
| BM25 alongside semantic | Dense embeddings have no representation of dates; lexical leg catches the meeting date as a literal token. |
| RRF (k=60) | Standard rank-fusion default; the surprising finding (below) is that this default is not always best. |
| GPT-4o-mini | Cost. Generation quality was not the bottleneck after retrieval was solid. |
| Era-split evaluation | LLM-pretraining-cutoff is the only way to disentangle memorized priors from real retrieval. |

---

## Evaluation

### Golden Test Set
38 hand-labeled FOMC meetings across `data/golden_test_set.csv`, split by `training_era`:
- **pre_cutoff** (n=26): on or before 2023-10-01 (GPT-4o-mini's approximate knowledge cutoff). Stance balance 10H / 8D / 8N.
- **post_cutoff** (n=12): after 2023-10-01. Stance balance 0H / 4D / 8N (the Fed's last hike was 2023-07-26 — no post-cutoff hawkish meetings exist).

The 3 few-shot meetings (2022-06-15, 2020-03-15, 2016-09-21) are excluded from the test set to prevent train/test leakage.

### Metrics
- **Stance accuracy** (primary): % of meetings the model classifies correctly.
- **hit@5** (primary retrieval metric): binary — did the top-5 retrieved chunks contain *any* chunk from the target meeting?
- **precision@5** (secondary): fraction of the top-5 whose `meeting_date` equals the target. Reported but not headlined — the ceiling is bounded by the number of chunks per statement (typically 2–5), so 1.0 is unreachable for most meetings.

### Headline Results (`data/eval_summary.csv`)

Stance accuracy by era × mode:

| mode | overall | pre_cutoff | post_cutoff |
|---|---|---|---|
| baseline | 50.0% | 73.1% | **0.0%** |
| semantic_only | 23.7% | 7.7% | 58.3% ⚠️ |
| bm25_only | 78.9% | 73.1% | **91.7%** |
| hybrid | 71.1% | 69.2% | 75.0% |

Retrieval quality (hit@5):

| mode | pre_cutoff | post_cutoff |
|---|---|---|
| semantic_only | 0.000 | 0.000 |
| bm25_only | 0.962 | 0.917 |
| hybrid | 0.769 | 0.750 |

### What These Numbers Show

**Baseline cliffs from 73.1% to 0.0% across the cutoff.** The model gets 19/26 pre-cutoff meetings right with no retrieval at all (memorized FOMC statements from pretraining), then refuses on 12/12 post-cutoff meetings (`invalid_stance` outputs). Without the era split, the original 60% baseline accuracy on a single mixed bucket would have looked like "the task is easy" rather than "the model has memorized the famous ones."

**BM25 beats hybrid on post-cutoff.** 91.7% vs 75.0% accuracy, hit@5 0.917 vs 0.750. RRF is supposed to combine two complementary signals into something at least as good as either alone — but on date-targeted queries, the dense leg ranks topical neighbors highly, those scores enter the fusion, and they push the BM25-correct meeting *out* of the top 5. Concrete failure on the query for 2025-07-30: BM25 had it ranked 5th (RRF score `1/65 ≈ 0.0154`); the dense leg ranked 2007-10-31 first because the query was about "rate cuts" and the 2007–08 easing cycle is semantically nearest; that wrong meeting won the fused list. **Hybrid retrieval is not strictly dominant, and naive RRF fusion can be worse than its better leg.** This is the most interesting finding in the project.

**Semantic retrieval collapses entirely on date queries.** hit@5 = 0.000 in both eras across 38 rows — dense embeddings never surfaced a chunk from the target meeting. The query template is `"Classify the Fed's stance at the FOMC meeting on {date}."` whose only content-bearing token is the date, and dense embeddings have no temporal representation. Lexical match catches the date literal in the chunk text; topical similarity does not. ⚠️ Note: the post-cutoff semantic accuracy of 58.3% is **not** a retrieval win — hit@5 is still 0.000. The accuracy bump comes from class imbalance (post_cutoff is 0H/4D/8N, so guessing "neutral" or "dovish" from any plausible Fed prose lands correct more often than in the 10H/8D/8N pre_cutoff bucket where hawkish must also be predicted). This is exactly why we report hit@5 alongside accuracy.

---

## Iteration and Reflection

- **`precision@5` was the original retrieval metric.** It looked low (0.20 for BM25) and read as a retrieval-quality failure. After noticing that most statements only chunk into 2–5 pieces (so precision@5 = 0.40–1.00 is the *ceiling*, not 1.0), we added **hit@5** as the primary retrieval metric. Lesson: bounded-by-construction metrics need a partner metric that doesn't share the bound.
- **The original baseline accuracy of ~60% on a mixed bucket was misleading.** The era split was added after noticing the baseline could "answer" famous meetings with no retrieval at all. The era-split numbers are the actual story.
- **The hybrid-below-BM25 finding was unexpected.** The original hypothesis was that hybrid would always win. We present it as an honest finding rather than tuning RRF until it disappears — naive fusion is what the literature recommends as the default, and showing where the default fails is more useful than showing where a tuned-to-the-test variant wins.

---

## Ethics and Limitations

- **Not financial advice.** The system disclaims this in every system prompt and does not produce buy/sell recommendations.
- **Label subjectivity.** "Hawkish/neutral/dovish" is a soft classification — meetings near transitions (e.g., 2016-09-21 "case has strengthened but waiting") are genuinely ambiguous and the labels reflect a single reader's judgment from statement text, not retrospective news coverage.
- **Post-cutoff class imbalance.** Zero post-cutoff hawkish meetings exist (the Fed's last hike was 2023-07-26), so per-class accuracy in that bucket is only defined for dovish/neutral. A model blindly predicting "neutral" would score well — flagged here so it cannot be cited as evidence the model is actually reading post-cutoff statements.
- **Domain bias.** The corpus is FOMC statement text only — it does not include meeting minutes, speeches, or testimony, all of which carry stance signal a real analyst would use.
- **Evaluator self-bias.** Stance labels were assigned by the same person designing the evaluation — a second labeler would surface where my reading is idiosyncratic.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.11+ (tested on 3.13) |
| LLM | GPT-4o-mini (snapshot `gpt-4o-mini-2024-07-18`) via `openai` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vector DB | `chromadb` |
| Lexical retrieval | `rank-bm25` |
| FOMC scraping | `requests` + `beautifulsoup4` |
| Macro data | `fredapi` |
| Env vars | `python-dotenv` |

---

## Project Structure

```
fed-signal-chatbot/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/                      # 175 scraped FOMC statements (YYYY-MM-DD.txt)
│   ├── fred/                     # Macro indicators from the FRED API
│   ├── chroma/                   # ChromaDB persistent collection
│   ├── bm25.pkl                  # Pickled BM25Okapi index
│   ├── golden_test_set.csv       # 38 hand-labeled meetings × era
│   ├── eval_runs/{mode}.csv      # Per-row eval outputs (4 modes)
│   └── eval_summary.csv          # Per-(era × mode) aggregate metrics
└── src/
    ├── ingest.py                 # FOMC scraper
    ├── index.py                  # Chunking + ChromaDB + BM25 build
    ├── retrieve.py               # semantic_search / bm25_search / hybrid_search (RRF)
    ├── generate.py               # Prompt assembly + GPT-4o-mini call → answer()
    └── evaluate.py               # Ablation runner over the golden test set
```

---

## Setup and Run

Requires Python 3.11+ and an OpenAI API key.

```bash
git clone https://github.com/<you>/fed-signal-chatbot.git
cd fed-signal-chatbot
pip install -r requirements.txt
cp .env.example .env  # then add OPENAI_API_KEY=sk-... and FRED_API_KEY=...
```

End-to-end pipeline:

```bash
python src/ingest.py        # Scrape FOMC statements → data/raw/  (optional, see below)
python src/index.py         # Build ChromaDB + BM25 indexes
python src/evaluate.py      # 38 rows × 4 modes = 152 GPT-4o-mini calls
```

`ingest.py` is only needed if you want to refresh the corpus from `federalreserve.gov` and FRED. The repo already includes the scraped statements in `data/raw/` and the built indexes in `data/chroma/` + `data/bm25.pkl`, so you can skip straight to `evaluate.py` if you only have an `OPENAI_API_KEY`.

`evaluate.py` writes per-row CSVs to `data/eval_runs/{mode}.csv` and the era-split summary to `data/eval_summary.csv`. Use `--limit N` for a smoke test on the first N rows, `--modes baseline hybrid` to subset.

Single-query smoke test:

```bash
python src/generate.py --query "Is the Fed currently hawkish?" --mode hybrid
```

---

## Reproducibility

- Model: pinned snapshot `gpt-4o-mini-2024-07-18`, temperature 0, JSON mode. Slight prose variation between runs is expected (OpenAI temperature=0 is not bit-exact across separate calls), but stance labels and retrieval results are stable.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, deterministic at inference.
- Chunk IDs: `{meeting_date}_{chunk_idx:03d}` with `collection.upsert` — re-running `index.py` is idempotent.
- The frozen evaluation outputs in `data/eval_runs/` and `data/eval_summary.csv` are the numbers reported above.

---

## References

- FOMC statements: <https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm>
- RRF: Cormack, Clarke, Büttcher (2009), *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods.*
- MiniLM: Wang et al. (2020), *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers.*
- BM25: Robertson & Zaragoza (2009), *The Probabilistic Relevance Framework: BM25 and Beyond.*
- LLMs used during development: Claude (Anthropic) for code review and documentation drafting; GPT-4o-mini as the system's generator.
