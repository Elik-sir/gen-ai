"""Проверка ghost-цитат и выдуманных chunk_id."""

from __future__ import annotations

import re

from schema import RAGAnswer


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def check_ghost_quotes(
    answer: RAGAnswer,
    retrieved_texts: dict[str, str],
) -> list[dict]:
    """Цитата считается ghost, если первые 30 символов не найдены ни в одном чанке."""
    if not answer.quotes:
        return []

    corpus = _normalize("\n".join(retrieved_texts.values()))
    ghosts: list[dict] = []
    for quote in answer.quotes:
        probe = _normalize(quote)[:30]
        if probe and probe not in corpus:
            ghosts.append({"type": "ghost_quote", "quote": quote, "probe": probe})
    return ghosts


def check_fake_chunk_ids(
    answer: RAGAnswer,
    valid_ids: set[str],
) -> list[dict]:
    bad = [cid for cid in answer.chunk_ids if cid not in valid_ids]
    return [{"type": "fake_chunk_id", "chunk_id": cid} for cid in bad]


def check_unsupported_numbers(
    answer: RAGAnswer,
    retrieved_texts: dict[str, str],
) -> list[dict]:
    """Числа в ответе, которых нет в retrieved chunks."""
    nums = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", answer.answer))
    if not nums:
        return []
    corpus = "\n".join(retrieved_texts.values())
    unsupported = [n for n in nums if n not in corpus and n.replace(".", ",") not in corpus]
    return [{"type": "unsupported_number", "value": n} for n in unsupported]


def run_all_checks(
    answer: RAGAnswer,
    retrieved_texts: dict[str, str],
) -> dict:
    valid_ids = set(retrieved_texts.keys())
    ghosts = check_ghost_quotes(answer, retrieved_texts)
    fake_ids = check_fake_chunk_ids(answer, valid_ids)
    bad_nums = check_unsupported_numbers(answer, retrieved_texts)
    issues = ghosts + fake_ids + bad_nums
    return {
        "ghost_quotes": len(ghosts),
        "fake_chunk_ids": len(fake_ids),
        "unsupported_numbers": len(bad_nums),
        "total_issues": len(issues),
        "issues": issues,
    }
