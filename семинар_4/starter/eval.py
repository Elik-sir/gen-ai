"""
Eval по 10 gold-вопросам. Метрика: hit-rate@5 на уровне документа-источника.

Правило: если в ТОП-5 чанков встретился хотя бы один чанк из gold_sources —
вопрос зачтён как HIT. Для вопросов, которым необходимы несколько чанков, считаем как долю найденных
источников (например, 2 из 3 → 0.67).

Команды:
    python eval.py --naive         # прогнать текущую конфигурацию pipeline.py
"""

import json
from pathlib import Path

from pipeline import collection, hybrid_retrieve, retrieve

DEFAULT_GOLD_PATH = Path(__file__).parent / "data" / "gold_homework.json"
EVAL_K = 5
EVAL_DENSE_ONLY = False
EVAL_VERBOSE = True


def load_gold(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def hit_rate(retrieved_ids: list[str], gold_sources: list[str]) -> float:
    """
    Для одного вопроса: сколько из gold_sources попали в ТОП-K чанков.
    retrieved_ids = ['olymp_anna__0', 'tinkoff_alex__2', ...]
    Мы смотрим только на префикс до '__' — это source_id.
    """
    retrieved_sources = {rid.split("__")[0] for rid in retrieved_ids}
    found = [g for g in gold_sources if g in retrieved_sources]
    return len(found) / len(gold_sources)


def run(
    gold_path: Path = DEFAULT_GOLD_PATH,
    dense_only: bool = False,
    k: int = 5,
    verbose: bool = True,
) -> dict:
    gold = load_gold(gold_path)
    total = 0.0
    results = []

    fn = retrieve if dense_only else hybrid_retrieve
    label = "DENSE-ONLY" if dense_only else "HYBRID (DENSE + BM25 + RRF)"
    print(f"\n==={label}===\n")

    for item in gold:
        q = item["question"]
        gold_sources = item["gold_sources"]

        hits = fn(q, k=k)
        retrieved_ids = hits["ids"][0]
        retrieved_sources = [rid.split("__")[0] for rid in retrieved_ids]

        score = hit_rate(retrieved_ids, gold_sources)
        total += score

        results.append(
            {
                "id": item["id"],
                "type": item["type"],
                "score": score,
                "gold": gold_sources,
                "retrieved_sources": retrieved_sources,
            }
        )

        if verbose:
            mark = "✓" if score == 1.0 else ("◐" if score > 0 else "✗")
            print(
                f"  [{item['id']:2d}] {item['type']:25s}  "
                f"hit@{k} = {score:.2f}  {mark}  {q}"
            )

    mean = total / len(gold)
    if verbose:
        print(f"\n  ИТОГО: hit-rate@{k} = {mean:.2f}  ({total:.1f} / {len(gold)})")
    return {"mean": mean, "results": results}


def main():
    # Проверка, что заполнили коллекцию
    if collection.count() == 0:
        print("⚠ Коллекция пустая. Запусти: python pipeline.py ingest ...")
        return

    run(
        gold_path=DEFAULT_GOLD_PATH,
        dense_only=EVAL_DENSE_ONLY,
        k=EVAL_K,
        verbose=EVAL_VERBOSE,
    )


if __name__ == "__main__":
    main()
