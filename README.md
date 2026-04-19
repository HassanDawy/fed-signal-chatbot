# Fed Signal Chatbot

A RAG-powered conversational agent that classifies Federal Reserve monetary policy stance and surfaces historical market context from FOMC meeting minutes.

---

## What It Does

Retail investors and finance students often want to understand where the Fed stands on monetary policy — and what similar past periods meant for markets. A vanilla LLM can produce plausible-sounding answers, but those answers are unverifiable and prone to hallucination.

The Fed Signal Chatbot solves this by grounding every response in actual Fed documents. Given a query like:

> *"Is the Fed leaning toward a rate cut soon, and what happened to markets in similar past periods?"*

The system retrieves relevant FOMC meeting minutes, reasons through the evidence step-by-step, and returns:

1. A **stance classification** — hawkish, neutral, or dovish
2. An **evidence summary** with citations to source documents
3. **Historical market context** from comparable past Fed periods

---

## System Architecture

The pipeline has four independently testable stages:

### Stage 1 — Data Ingestion
- **FOMC meeting minutes** scraped from [Fed.gov](https://www.federalreserve.gov/monetarypolicy/fomc_historical.htm) (~20+ years of meetings)
- **Macro indicators** from the [FRED API](https://fred.stlouisfed.org/): 10-year Treasury yield (DGS10), Fed Funds Rate (FEDFUNDS), CPI (CPIAUCSL), Unemployment (UNRATE)
- Each document is tagged with meeting date, ground-truth stance label, and macro snapshot at time of meeting

### Stage 2 — Hybrid Retrieval (RAG)
- FOMC minutes are chunked into ~500-token segments with 50-token overlap at paragraph boundaries
- Two retrieval methods run in parallel:
  - **Semantic search** via `sentence-transformers` (`all-MiniLM-L6-v2`) stored in ChromaDB
  - **Keyword search** via BM25 (`rank-bm25`) over the same corpus
- Results are merged using **Reciprocal Rank Fusion (RRF)**

### Stage 3 — Generation
- **Model:** GPT-4o-mini via OpenAI API
- **Prompt structure:** system prompt → few-shot examples → retrieved context → user query
- **Chain-of-thought (CoT)** reasoning before final stance classification
- Structured output: stance label + cited evidence + historical context

### Stage 4 — Evaluation
- **Golden test set:** 25–30 manually labeled FOMC meetings (hawkish / neutral / dovish)
- **Primary metric:** stance classification accuracy vs. ground truth
- **Secondary metric:** retrieval precision@5
- **Ablation study:** semantic-only vs. BM25-only vs. hybrid RRF vs. no-retrieval baseline

---

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.11+ |
| LLM API | `openai` (GPT-4o-mini) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB (local) |
| Keyword Search | `rank-bm25` |
| Macro Data | `fredapi` |
| FOMC Parsing | `requests` + `BeautifulSoup` |
| Notebooks | Jupyter |

---

## Project Structure

```
fed-signal-chatbot/
├── data/
│   ├── fomc_minutes/        # Raw FOMC minutes (text files, keyed by meeting date)
│   └── macro_indicators.csv # FRED data snapshot
├── src/
│   ├── ingest.py            # FOMC scraper + FRED pull
│   ├── index.py             # Chunking, embedding, ChromaDB + BM25 indexing
│   ├── retrieve.py          # Semantic search, BM25, RRF fusion
│   ├── generate.py          # Prompt construction + GPT-4o-mini call
│   └── evaluate.py          # Golden test set eval + ablation runner
├── notebooks/
│   ├── eda.ipynb            # Exploratory data analysis
│   └── ablation.ipynb       # Ablation results and charts
├── golden_test_set.json     # Manually labeled FOMC meetings
├── CLAUDE.md                # Design decisions and constraints for AI-assisted coding
├── requirements.txt
└── README.md
```

---

## Evaluation Results

*(To be filled in after ablation study is complete)*

| Retrieval Strategy | Stance Accuracy | Retrieval Prec@5 |
|---|---|---|
| No retrieval (baseline) | — | — |
| Semantic only | — | — |
| BM25 only | — | — |
| Hybrid RRF | — | — |

---

## Setup

```bash
git clone https://github.com/<your-username>/fed-signal-chatbot.git
cd fed-signal-chatbot
pip install -r requirements.txt
```

Set your API keys in a `.env` file:

```
OPENAI_API_KEY=your_key_here
FRED_API_KEY=your_key_here
```

Then run the pipeline:

```bash
python src/ingest.py       # Pull FOMC minutes + FRED data
python src/index.py        # Build ChromaDB + BM25 indexes
python src/evaluate.py     # Run ablation study against golden test set
```

---

## Ethics & Limitations

- **Not financial advice.** This system produces historical context and policy analysis only. All outputs include a disclaimer.
- **Hallucination risk.** RAG reduces but does not eliminate hallucination. Every response cites source chunks so users can verify claims independently.
- **Data staleness.** The knowledge base reflects FOMC documents at the time of ingestion. There is no live data feed.
- **Interpretive ambiguity.** Fed stance classification is inherently subjective. The model's label may differ from expert consensus — this is documented in the evaluation.

---

## Course Context

DS593 Applied LLMs | Solo Project | Spring 2026
