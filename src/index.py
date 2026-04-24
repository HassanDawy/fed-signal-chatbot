"""Chunk FOMC statements, embed with MiniLM, and build a ChromaDB index.

Chunk size (256 tokens, 32-token overlap) is sized to the ``all-MiniLM-L6-v2``
encoder's 256-token context so nothing gets silently truncated at embed time.
Chunks are split at paragraph boundaries where possible; oversized paragraphs
fall back to a sliding token window.

Run:
    python src/index.py

Idempotent: chunk ids are deterministic (``{meeting_date}_{chunk_idx:03d}``)
and we use ``collection.upsert``, so re-running updates in place.
"""

from __future__ import annotations

import csv
import logging
import pickle
import re
import statistics
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "_manifest.csv"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
BM25_PATH = PROJECT_ROOT / "data" / "bm25.pkl"

WORD_RE = re.compile(r"[a-z0-9]+")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "fomc_statements"

CHUNK_TOKENS = 256
OVERLAP_TOKENS = 32
EMBED_BATCH_SIZE = 64

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("index")


def load_manifest() -> list[dict]:
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def chunk_text(
    text: str,
    tokenizer,
    chunk_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[str]:
    """Split ``text`` into <=chunk_tokens chunks, preferring paragraph seams.

    Greedily packs paragraphs (split on blank lines) until the next paragraph
    would overflow, then starts a new chunk seeded with the last
    ``overlap_tokens`` tokens of the previous one. Paragraphs that individually
    exceed ``chunk_tokens`` fall back to a sliding token window.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_ids: list[int] = []

    def flush() -> None:
        if not current_ids:
            return
        chunks.append(_decode(tokenizer, current_ids))

    for para in paragraphs:
        para_ids = _encode(tokenizer, para)

        # Oversized paragraph: window it.
        if len(para_ids) > chunk_tokens:
            flush()
            current_ids = []
            stride = chunk_tokens - overlap_tokens
            for start in range(0, len(para_ids), stride):
                window = para_ids[start : start + chunk_tokens]
                if not window:
                    break
                chunks.append(_decode(tokenizer, window))
                if start + chunk_tokens >= len(para_ids):
                    break
            continue

        # Paragraph fits; add a separator token budget of 1 implicitly via decode.
        if len(current_ids) + len(para_ids) <= chunk_tokens:
            current_ids.extend(para_ids)
        else:
            flush()
            # Seed next chunk with overlap tail of previous chunk's tokens.
            current_ids = current_ids[-overlap_tokens:] if overlap_tokens else []
            current_ids.extend(para_ids)

    flush()
    return [c for c in chunks if c]


def bm25_tokenize(text: str) -> list[str]:
    """Lowercase word-level tokenizer used for BM25 (not the MiniLM tokenizer).

    BM25 works on whole-word term frequencies, so WordPiece subwords would
    hurt IDF weighting. Kept as a top-level function so `retrieve.py` uses
    the exact same tokenizer at query time.
    """
    return WORD_RE.findall(text.lower())


def build_index() -> tuple[int, dict[str, int]]:
    """Chunk every statement, embed, and upsert into the Chroma collection.

    Returns (total_chunks, {meeting_date: chunk_count}).
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("loading embedding model: %s", EMBED_MODEL_NAME)
    model = SentenceTransformer(EMBED_MODEL_NAME)
    tokenizer = model.tokenizer

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    manifest = load_manifest()
    log.info("manifest: %d statements", len(manifest))

    per_meeting: dict[str, int] = {}
    total = 0

    # Accumulate BM25 corpus alongside the Chroma upserts so both indexes
    # share the same ids and chunk text.
    bm25_ids: list[str] = []
    bm25_docs: list[str] = []
    bm25_metas: list[dict] = []
    bm25_tokens: list[list[str]] = []

    for row in manifest:
        meeting_date = row["meeting_date"]
        source_url = row["url"]
        txt_path = RAW_DIR / f"{meeting_date}.txt"
        if not txt_path.exists():
            log.warning("missing raw file: %s", txt_path)
            continue

        body = txt_path.read_text(encoding="utf-8")
        chunks = chunk_text(body, tokenizer)
        if not chunks:
            log.warning("no chunks produced for %s", meeting_date)
            continue

        ids = [f"{meeting_date}_{i:03d}" for i in range(len(chunks))]
        metadatas = [
            {
                "meeting_date": meeting_date,
                "source_url": source_url,
                "chunk_idx": i,
                "stance_label": "",
            }
            for i in range(len(chunks))
        ]

        embeddings = model.encode(
            chunks,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        bm25_ids.extend(ids)
        bm25_docs.extend(chunks)
        bm25_metas.extend(metadatas)
        bm25_tokens.extend(bm25_tokenize(c) for c in chunks)

        per_meeting[meeting_date] = len(chunks)
        total += len(chunks)
        log.info("indexed %s: %d chunks", meeting_date, len(chunks))

    log.info("fitting BM25 over %d chunks", len(bm25_tokens))
    bm25 = BM25Okapi(bm25_tokens)
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BM25_PATH.open("wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "ids": bm25_ids,
                "documents": bm25_docs,
                "metadatas": bm25_metas,
            },
            f,
        )
    log.info("wrote BM25 index to %s", BM25_PATH)

    log.info("index built: %d chunks across %d meetings", total, len(per_meeting))
    return total, per_meeting


def sanity_query(query: str = "rate hike inflation concerns", k: int = 3) -> None:
    model = SentenceTransformer(EMBED_MODEL_NAME)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    q_emb = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).tolist()
    res = collection.query(
        query_embeddings=q_emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    print(f"\nSanity query: {query!r}")
    for rank, (doc, meta, dist) in enumerate(
        zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), 1
    ):
        snippet = doc[:160].replace("\n", " ")
        print(f"  {rank}. [{meta['meeting_date']}] dist={dist:.3f}  {snippet}...")


def main() -> None:
    total, per_meeting = build_index()
    counts = list(per_meeting.values())
    if counts:
        print(f"\nTotal chunks: {total}")
        print(
            f"Chunks per meeting — min={min(counts)}, "
            f"median={int(statistics.median(counts))}, max={max(counts)}"
        )
    sanity_query()


if __name__ == "__main__":
    main()
