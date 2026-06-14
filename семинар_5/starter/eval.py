"""
Мини-оценка: 10 вопросов, проверяем:
1. Что агент завершает работу за разумное число шагов.
2. Что в трассе шагов есть ожидаемые инструменты.
3. Что в финальном ответе упомянуты ожидаемые ключевые числа (опционально).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import CACHE_STATS, run_agent

CASES = [
    {
        "id": 1,
        "query": "Какая сегодня ключевая ставка ЦБ?",
        "expected_tools": ["get_key_rate"],
        "must_have": [],  # число не фиксируем — зависит от живого запроса
        "comment": "Базовый тест — один инструмент, одно число.",
    },
    {
        "id": 2,
        "query": "Сколько стоит доллар сегодня и сколько стоил 1 января 2022?",
        "expected_tools": ["get_fx_rate"],
        "must_have": [],
        "comment": "Два вызова одного инструмента с разными аргументами.",
    },
    {
        "id": 3,
        "query": "Какая сейчас реальная ключевая ставка? (номинальная минус инфляция г/г)",
        "expected_tools": ["get_key_rate", "get_inflation", "calculate"],
        "must_have": ["%"],
        "comment": "Три разных инструмента + арифметика. Классический многостадийный кейс.",
    },
    {
        "id": 4,
        "query": "Посчитай, за сколько лет удвоится вклад 100 тыс руб при текущей ключевой ставке (формула 72).",
        "expected_tools": ["get_key_rate", "calculate"],
        "must_have": ["год"],
        "comment": "Вычисление с формулой: 72 / ставка = годы.",
    },
    {
        "id": 5,
        "query": "Во сколько раз вырос курс USD с января 2022 по апрель 2026?",
        "expected_tools": ["compare_periods"],
        "must_have": ["раз"],
        "comment": "Проверка нового инструмента compare_periods на основном сценарии домашки.",
    },
    {
        "id": 6,
        "query": "Сравни ключевую ставку в 2022-02 и в 2026-04: на сколько процентных пункта изменилась?",
        "expected_tools": ["compare_periods"],
        "must_have": [],
        "comment": "Второй обязательный кейс с compare_periods.",
    },
    {
        "id": 7,
        "query": "Что сейчас выше: ключевая ставка или индекс нищеты (инфляция + безработица)?",
        "expected_tools": ["get_key_rate", "get_inflation", "get_unemployment", "calculate"],
        "must_have": [],
        "comment": "Реальный макро-вопрос: сравнение денежно-кредитного и социального индикаторов.",
    },
    {
        "id": 8,
        "query": "Какова реальная доходность годового вклада под ключевую ставку с поправкой на инфляцию?",
        "expected_tools": ["get_key_rate", "get_inflation", "calculate"],
        "must_have": ["%"],
        "comment": "Реальный макро-вопрос: приближённая реальная доходность.",
    },
    {
        "id": 9,
        "query": "Сколько юаней за доллар по кросс-курсу ЦБ сегодня?",
        "expected_tools": ["get_fx_rate", "calculate"],
        "must_have": [],
        "comment": "Трудный кейс: легко перепутать порядок деления USD/CNY и единицы результата.",
    },
    {
        "id": 10,
        "query": "Сравни инфляцию за 2024-02 и 2024-03 и безработицу за те же месяцы: где изменение сильнее в относительных терминах?",
        "expected_tools": ["compare_periods"],
        "must_have": [],
        "comment": "Трудный кейс: сразу две метрики и относительное сравнение, возможна путаница в логике.",
    },
]


def _safe_console(text: str) -> str:
    # В Windows-консоли cp1251 не все символы печатаются (например, ₽).
    return text.encode("cp1251", errors="replace").decode("cp1251", errors="replace")


def run_case(case: dict, *, use_cache: bool = False, track_cost: bool = False) -> dict:
    print(f"\n{'=' * 70}\n[Q{case['id']}] {case['query']}\n{'-' * 70}")
    res = run_agent(
        case["query"],
        max_iter=8,
        verbose=True,
        use_cache=use_cache,
        track_cost=track_cost,
    )
    used_tools = [e["call"] for e in res["trace"] if "call" in e]
    answer = res.get("answer") or ""

    tool_match = all(t in used_tools for t in case["expected_tools"])
    text_match = all(s.lower() in answer.lower() for s in case["must_have"])
    ok = bool(answer) and tool_match and text_match

    print(_safe_console(f"\n  tools used : {used_tools}"))
    print(
        _safe_console(
            f"  expected    : {case['expected_tools']}  -> {'OK' if tool_match else 'MISS'}"
        )
    )
    print(_safe_console(f"  answer      : {answer[:200]}"))
    print(
        _safe_console(
            f"  must_have   : {case['must_have']}  -> {'OK' if text_match else 'MISS'}"
        )
    )
    print(_safe_console(f"  verdict     : {'PASS' if ok else 'FAIL'}"))

    return {
        "id": case["id"],
        "query": case["query"],
        "ok": ok,
        "tools_used": used_tools,
        "steps": res["steps"],
        "answer": answer,
    }


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Мини-оценка макро-агента")
    ap.add_argument(
        "--cache",
        action="store_true",
        help="Блок 9: общий кэш инструментов на все вопросы — видно повторные вызовы",
    )
    ap.add_argument(
        "--cost",
        action="store_true",
        help="Блок 10: показать токены и стоимость по шагам",
    )
    a = ap.parse_args()

    if a.cache:
        CACHE_STATS["hits"] = CACHE_STATS["misses"] = 0

    results = [run_case(c, use_cache=a.cache, track_cost=a.cost) for c in CASES]
    passed = sum(1 for r in results if r["ok"])

    print(_safe_console(f"\n{'=' * 70}\nИтого: {passed}/{len(CASES)} пройдено"))
    for r in results:
        mark = "[OK]  " if r["ok"] else "[FAIL]"
        print(_safe_console(f"  {mark} Q{r['id']} ({r['steps']} шагов) - {r['query'][:60]}"))

    if a.cache:
        h, m = CACHE_STATS["hits"], CACHE_STATS["misses"]
        print(
            _safe_console(
                f"\n[кэш] на {len(CASES)} вопросах: {h} попаданий из {h + m} обращений "
                f"к инструментам - столько вызовов ЦБ/Росстата сэкономлено."
            )
        )

    out = Path(__file__).parent / "eval_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(_safe_console(f"\nРезультаты: {out}"))


if __name__ == "__main__":
    main()
