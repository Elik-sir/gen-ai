"""
Наивный RAG: ChromaDB + OpenAI, fixed-size chunking, только dense-поиск.

Команды:
    python pipeline.py ingest
    python pipeline.py ask "Кто жаловался на push-уведомления?"

TODO для семинара:
    Блок 3, Фикс 1 — заменить фиксированные чанки на рекурсивные по абзацам
    Блок 3, Фикс 2 — обернуть ответ в Pydantic RAGAnswer
    Блок 3, Фикс 3 — добавить BM25-гибрид через rank-bm25 и RRF
"""

import os
import sys
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from llm_client import get_model, make_raw_client

# Блок 1 — наивный RAG: ответ модели идёт обычным текстом
client = make_raw_client()
MODEL = get_model()
chroma = chromadb.PersistentClient(path="./chroma_db")

print("Загружаю эмбеддер...", flush=True)
_t_embed = time.time()
EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
)
print(f"Эмбеддер готов за {time.time() - _t_embed:.1f}с", flush=True)
collection = chroma.get_or_create_collection(
    name="focus_groups",
    embedding_function=EMBED_FN,
    metadata={"hnsw:space": "cosine"},
)

DATA_DIR = Path(__file__).parent / "data"


# фиксированный чанкинг по символам
def chunk_text_naive(text: str, chunk_size: int = 2000) -> list[str]:
    """
    Примитивная нарезка: рубим каждые N символов.
    Проблема: граница может попасть в середину фразы «я ругался на |
    скорость» — и на запрос «скорость» не найдётся чанк про недовольство.
    """
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


# заполнение векторного хранилища: читаем data/, режем, кладём в ChromaDB
def ingest():
    # Чистим старую коллекцию перед переиндексацией
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    for f in sorted(DATA_DIR.glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        chunks = chunk_text_naive(text)
        ids = [f"{f.stem}__{i}" for i in range(len(chunks))]
        metadatas = [{"source": f.stem, "chunk_id": i} for i in range(len(chunks))]
        collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        print(f"  {f.stem}: {len(chunks)} чанков")

    total = collection.count()
    print(
        f"\nИндексировано: {total} чанков из {len(list(DATA_DIR.glob('*.txt')))} файлов"
    )


# Retrieve + generate
def retrieve(query: str, k: int = 5) -> dict:
    """Dense-поиск в ChromaDB."""
    return collection.query(query_texts=[query], n_results=k)


def build_prompt(query: str, hits: dict) -> str:
    docs = hits["documents"][0]
    ids = hits["ids"][0]
    ctx = "\n\n---\n\n".join(f"[{i}]\n{d}" for i, d in zip(ids, docs))
    return (
        "Ты отвечаешь на вопрос продакта по архиву фокус-групп. "
        "Опирайся ТОЛЬКО на контекст ниже. Если в контексте нет ответа — "
        "скажи об этом прямо. Перечисли имена участников.\n\n"
        f"Контекст:\n{ctx}\n\n"
        f"Вопрос: {query}\n\n"
        "Ответ:"
    )


def ask(query: str):
    # Эмбеддим запрос и ищем топ-5 в Chroma.
    print("Поиск по базе...", flush=True)
    t0 = time.time()
    hits = retrieve(query, k=5)
    found = hits["ids"][0]
    print(
        f"   нашёл {len(found)} чанков за {time.time() - t0:.1f}с: {', '.join(found)}",
        flush=True,
    )

    # Кладём найденное в промпт, спрашиваем модель.
    print("Генерация ответа...", flush=True)
    t1 = time.time()
    prompt = build_prompt(query, hits)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    print(f"   ответ за {time.time() - t1:.1f}с", flush=True)

    print("\n" + "=" * 60)
    print(f"ВОПРОС: {query}")
    print("=" * 60)
    print(resp.choices[0].message.content)
    print("\n--- источники ---")
    for i in found:
        print(f"  {i}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python pipeline.py {ingest|ask} [вопрос]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "ingest":
        ingest()
    elif cmd == "ask":
        if len(sys.argv) < 3:
            print('Нужен вопрос: python pipeline.py ask "..."')
            sys.exit(1)
        ask(sys.argv[2])
    else:
        print(f"Неизвестная команда: {cmd}")
        sys.exit(1)
