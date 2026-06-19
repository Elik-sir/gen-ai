"""Hybrid RAG: ChromaDB + BM25 + RRF."""

from __future__ import annotations

import json
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "input" / "corpus"
INDEX_DIR = ROOT / "output" / "index"
CHROMA_PATH = str(INDEX_DIR / "chroma_db")
BM25_CACHE = INDEX_DIR / "bm25_cache.json"
CHUNK_MAP = INDEX_DIR / "chunk_map.json"

RECURSIVE_CHUNK_SIZE = 400
RECURSIVE_OVERLAP = 80
RRF_C = 60

print("Загружаю эмбеддер...", flush=True)
EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
)
_chroma = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _chroma.get_or_create_collection(
    name="habr_rag_corpus",
    embedding_function=EMBED_FN,
    metadata={"hnsw:space": "cosine"},
)


def tokenize_ru(text: str) -> list[str]:
    return re.findall(r"[а-яa-z0-9ё-]{2,}", text.lower())


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RECURSIVE_CHUNK_SIZE,
        chunk_overlap=RECURSIVE_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", " "],
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


def ingest(corpus_dir: Path = CORPUS_DIR) -> dict:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    existing = _collection.get()
    if existing["ids"]:
        _collection.delete(ids=existing["ids"])

    files = sorted(corpus_dir.glob("*.txt"))
    if not files:
        raise RuntimeError(f"Нет .txt в {corpus_dir}. Запустите: python scripts/fetch_habr.py")

    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_meta: list[dict] = []

    for f in files:
        text = f.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            cid = f"{f.stem}__{i}"
            all_chunks.append(c)
            all_ids.append(cid)
            all_meta.append({"source": f.stem, "chunk_id": i})
        print(f"  {f.stem}: {len(chunks)} чанков")

    _collection.add(documents=all_chunks, ids=all_ids, metadatas=all_meta)
    bm25_data = {
        "ids": all_ids,
        "tokens": [tokenize_ru(c) for c in all_chunks],
        "texts": all_chunks,
    }
    BM25_CACHE.write_text(json.dumps(bm25_data, ensure_ascii=False), encoding="utf-8")
    CHUNK_MAP.write_text(
        json.dumps(dict(zip(all_ids, all_chunks)), ensure_ascii=False),
        encoding="utf-8",
    )
    stats = {"files": len(files), "chunks": len(all_ids)}
    print(f"Индексировано: {stats['chunks']} чанков из {stats['files']} документов")
    return stats


def _load_bm25() -> tuple[BM25Okapi, list[str], list[str]]:
    data = json.loads(BM25_CACHE.read_text(encoding="utf-8"))
    return BM25Okapi(data["tokens"]), data["ids"], data["texts"]


def hybrid_retrieve(query: str, k: int = 5, top: int = 15) -> list[dict]:
    if _collection.count() == 0:
        raise RuntimeError("Индекс пуст. Запустите: python pipeline.py --ingest")

    dense = _collection.query(query_texts=[query], n_results=top)
    dense_ids = dense["ids"][0]

    bm25, bm25_ids, bm25_texts = _load_bm25()
    scores = bm25.get_scores(tokenize_ru(query))
    bm25_order = sorted(range(len(bm25_ids)), key=lambda i: scores[i], reverse=True)[:top]
    sparse_ids = [bm25_ids[i] for i in bm25_order]

    rrf: dict[str, float] = {}
    for rank, cid in enumerate(dense_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_C + rank)
    for rank, cid in enumerate(sparse_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_C + rank)

    ordered = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]
    text_by_id = dict(zip(bm25_ids, bm25_texts))
    for i, did in enumerate(dense["ids"][0]):
        text_by_id[did] = dense["documents"][0][i]

    return [
        {
            "chunk_id": cid,
            "source_id": cid.split("__")[0],
            "text": text_by_id[cid],
            "score": round(score, 4),
        }
        for cid, score in ordered
    ]


def get_chunk_text(chunk_id: str) -> str | None:
    if not CHUNK_MAP.exists():
        return None
    data = json.loads(CHUNK_MAP.read_text(encoding="utf-8"))
    return data.get(chunk_id)


def chunk_count() -> int:
    return _collection.count()
