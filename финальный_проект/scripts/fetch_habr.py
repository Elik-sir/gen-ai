"""
Скачивает 10 статей Habr про RAG и агентов в input/corpus/.

Запуск из корня проекта:
    python scripts/fetch_habr.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "input" / "corpus"
META_PATH = ROOT / "input" / "corpus_meta.json"

ARTICLES = [
    {
        "id": "habr_rag_deep_dive",
        "url": "https://habr.com/ru/articles/931396/",
        "title": "RAG: глубокий технический обзор",
    },
    {
        "id": "habr_rag_intro",
        "url": "https://habr.com/ru/articles/841428/",
        "title": "Что такое RAG и как работает",
    },
    {
        "id": "habr_rag_simple",
        "url": "https://habr.com/ru/articles/779526/",
        "title": "RAG — простое объяснение",
    },
    {
        "id": "habr_rag_smart_search",
        "url": "https://habr.com/ru/articles/1016166/",
        "title": "RAG или умный поиск по документам",
    },
    {
        "id": "habr_rag_python",
        "url": "https://habr.com/ru/companies/otus/articles/979458/",
        "title": "Создаём простую систему RAG на Python",
    },
    {
        "id": "habr_react_chestnyznak",
        "url": "https://habr.com/ru/companies/chestnyznak/articles/1045460/",
        "title": "От Naive RAG до ReAct-агента",
    },
    {
        "id": "habr_agent_prod",
        "url": "https://habr.com/ru/companies/selectel/articles/1015508/",
        "title": "Готовим ИИ-агента к продакшену",
    },
    {
        "id": "habr_hybrid_rag",
        "url": "https://habr.com/ru/articles/1005776/",
        "title": "Hybrid RAG knowledge base",
    },
    {
        "id": "habr_react_sber",
        "url": "https://habr.com/ru/companies/sberbank/articles/934938/",
        "title": "ReAct-агент: руководство",
    },
    {
        "id": "habr_rag_eval",
        "url": "https://habr.com/ru/companies/otus/articles/1011464/",
        "title": "Как оценивать RAG-системы",
    },
]


def _clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def extract_article(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title_el = soup.select_one("h1")
    title = title_el.get_text(strip=True) if title_el else "Без заголовка"

    body = soup.select_one("div.tm-article-body") or soup.select_one(
        "div.article-formatted-body"
    )
    if body is None:
        raise RuntimeError("Не найден блок статьи")

    for tag in body.select("script, style, noscript"):
        tag.decompose()

    text = body.get_text("\n", strip=True)
    return title, _clean_text(text)


def fetch_one(client: httpx.Client, article: dict) -> dict:
    resp = client.get(article["url"], follow_redirects=True)
    resp.raise_for_status()
    title, body = extract_article(resp.text)
    if len(body) < 500:
        raise RuntimeError(f"Слишком короткий текст: {len(body)} символов")

    header = (
        f"# {title}\n\n"
        f"Источник: {article['url']}\n"
        f"ID: {article['id']}\n\n"
        f"---\n\n"
    )
    full_text = header + body
    out_path = CORPUS_DIR / f"{article['id']}.txt"
    out_path.write_text(full_text, encoding="utf-8")
    return {
        "id": article["id"],
        "title": title,
        "url": article["url"],
        "chars": len(full_text),
        "words": len(full_text.split()),
        "file": out_path.name,
    }


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    meta: list[dict] = []
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; gen-ai-final-project/1.0)",
        "Accept-Language": "ru-RU,ru;q=0.9",
    }

    with httpx.Client(timeout=60, headers=headers) as client:
        for i, article in enumerate(ARTICLES, 1):
            print(f"[{i}/{len(ARTICLES)}] {article['id']}...", flush=True)
            try:
                meta.append(fetch_one(client, article))
                print(f"  OK: {meta[-1]['chars']} символов", flush=True)
            except Exception as e:
                print(f"  FAIL: {e}", flush=True)
                raise
            time.sleep(1.0)

    total_chars = sum(m["chars"] for m in meta)
    META_PATH.write_text(
        json.dumps(
            {"articles": meta, "total_chars": total_chars, "count": len(meta)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nГотово: {len(meta)} статей, {total_chars:,} символов -> {CORPUS_DIR}")


if __name__ == "__main__":
    main()
