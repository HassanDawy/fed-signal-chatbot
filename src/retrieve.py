"""Retrieval: semantic (Chroma), BM25, and hybrid (RRF fusion).

All three paths return a unified list of result dicts so ``generate.py`` and
``evaluate.py`` can treat them interchangeably. The semantic-only and
BM25-only paths stay callable on their own — they are the ablation legs.

Result dict schema:
    {
        "id": str,              # chunk id, e.g. "2022-05-04_002"
        "meeting_date": str,
        "source_url": str,
        "text": str,            # raw chunk text
        "score": float,         # higher = better (cosine sim or BM25 or RRF)
    }
"""

from __future__ import annotations

import logging
import pickle
from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from index import (
    BM25_PATH,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL_NAME,
    bm25_tokenize,
)

log = logging.getLogger("retrieve")

DEFAULT_TOP_K = 5
RRF_K = 60  # standard


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME)


@lru_cache(maxsize=1)
def _bm25_bundle() -> dict:
    with Path(BM25_PATH).open("rb") as f:
        return pickle.load(f)


def semantic_search(query: str, k: int = DEFAULT_TOP_K) -> list[dict]:
    """Top-k chunks by cosine similarity in ChromaDB."""
    q_emb = _model().encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).tolist()
    res = _collection().query(
        query_embeddings=q_emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    out: list[dict] = []
    for _id, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        out.append(
            {
                "id": _id,
                "meeting_date": meta["meeting_date"],
                "source_url": meta["source_url"],
                "text": doc,
                "score": 1.0 - float(dist),  # cosine distance -> similarity
            }
        )
    return out


def bm25_search(query: str, k: int = DEFAULT_TOP_K) -> list[dict]:
    """Top-k chunks by BM25 score."""
    bundle = _bm25_bundle()
    bm25 = bundle["bm25"]
    ids, docs, metas = bundle["ids"], bundle["documents"], bundle["metadatas"]

    tokens = bm25_tokenize(query)
    scores = bm25.get_scores(tokens)
    # argsort desc
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {
            "id": ids[i],
            "meeting_date": metas[i]["meeting_date"],
            "source_url": metas[i]["source_url"],
            "text": docs[i],
            "score": float(scores[i]),
        }
        for i in top_idx
    ]


def hybrid_search(
    query: str,
    k: int = DEFAULT_TOP_K,
    pool: int = 20,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """Semantic + BM25 fused with Reciprocal Rank Fusion.

    Pulls ``pool`` candidates from each leg (wider than k), fuses by
    ``1/(rrf_k + rank)``, returns top k.
    """
    sem = semantic_search(query, k=pool)
    lex = bm25_search(query, k=pool)

    # RRF: each leg contributes 1/(k+rank). A doc that shows up in both legs
    # gets both contributions added, so cross-leg agreement boosts its rank.
    fused: dict[str, dict] = {}
    for rank, r in enumerate(sem):
        fused[r["id"]] = {**r, "score": 1.0 / (rrf_k + rank + 1)}
    for rank, r in enumerate(lex):
        if r["id"] in fused:
            fused[r["id"]]["score"] += 1.0 / (rrf_k + rank + 1)
        else:
            fused[r["id"]] = {**r, "score": 1.0 / (rrf_k + rank + 1)}

    ranked = sorted(fused.values(), key=lambda r: r["score"], reverse=True)
    return ranked[:k]


def _print_results(label: str, results: list[dict]) -> None:
    print(f"\n[{label}]")
    for i, r in enumerate(results, 1):
        snippet = r["text"][:140].replace("\n", " ")
        print(
            f"  {i}. [{r['meeting_date']}] score={r['score']:.4f}  {snippet}..."
        )


def main() -> None:
    queries = [
        "rate hike inflation concerns",
        "patient approach to policy accommodation",
        "ongoing increases in the target range are appropriate",
    ]
    for q in queries:
        print(f"\n=== query: {q!r} ===")
        _print_results("semantic", semantic_search(q, k=3))
        _print_results("bm25", bm25_search(q, k=3))
        _print_results("hybrid (RRF)", hybrid_search(q, k=3))


if __name__ == "__main__":
    main()
