"""Eval семинара 6: 6 кейсов × 3 конфигурации.

Конфигурации:
1) single          — одиночный агент из С5
2) pwc             — PWC без валидатора плана
3) pwc+validator   — PWC с валидатором плана
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_s5 import run_agent
from orchestrator import run_pwc


CASES = [
    {
        "id": "Q1",
        "query": "Во сколько раз USD подорожал с 1 января 2022 по сегодня?",
        "comment": (
            "Класс ошибки C: одиночный часто считает в уме, не зовёт calculate. "
            "PWC должен починить — Планировщик обязан добавить calculate-подвопрос."
        ),
        "expected_tools_pwc": {"get_fx_rate", "calculate"},
        "must_have_keywords": ["раз", "USD"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q2",
        "query": (
            "Какая сейчас реальная ключевая ставка, если инфляцию брать "
            "по последнему доступному месяцу, а не по году?"
        ),
        "comment": (
            "Класс ошибки B: одиночный не умеет искать «последний доступный» "
            "месяц, зацикливается. PWC должен разбить на шаги."
        ),
        "expected_tools_pwc": {"get_inflation", "get_key_rate", "calculate"},
        "must_have_keywords": ["%"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q3",
        "query": (
            "Какова накопленная инфляция с января 2022 по март 2026? "
            "Рассчитай как произведение всех (1 + ипц_м/100) по месяцам."
        ),
        "comment": (
            "Класс ошибки D (граница паттерна): требует get_inflation за много "
            "месяцев + большое calculate-выражение. Одиночный галлюцинирует "
            "get_cumulative_inflation; валидатор должен это ловить."
        ),
        "expected_tools_pwc": {"get_inflation", "calculate"},
        "must_have_keywords": ["%"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q4",
        "query": (
            "Посчитай накопленную инфляцию за 2024 год через произведение "
            "(1+ипц/100) по каждому месяцу и верни итог в процентах."
        ),
        "comment": (
            "Специальный кейс под валидатор: планировщик иногда пытается "
            "использовать выдуманный инструмент вроде get_cumulative_inflation."
        ),
        "expected_tools_pwc": {"get_inflation", "calculate"},
        "must_have_keywords": ["%", "инфляц"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q5",
        "query": (
            "Как изменился рубль к USD, EUR и CNY с 1 января 2022 по сегодня, "
            "и какая из валют изменилась сильнее всего?"
        ),
        "comment": (
            "Естественная параллельность: 3 независимых ветки по валютам + "
            "финальная агрегация."
        ),
        "expected_tools_pwc": {"get_fx_rate", "calculate"},
        "must_have_keywords": ["USD", "EUR", "CNY"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q6",
        "query": (
            "Какая сейчас реальная ключевая ставка в России и на сколько "
            "п.п. она отличается от уровня 1 января 2022?"
        ),
        "comment": (
            "Реальный макро-вопрос: текущая реальная ставка и сравнение "
            "с базовой датой."
        ),
        "expected_tools_pwc": {"get_key_rate", "get_inflation", "calculate"},
        "must_have_keywords": ["ставк", "%"],
        "forbid_hallucinated_tools": True,
    },
]


VALID_TOOL_NAMES = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}


def _check_single(case: dict, result: dict) -> dict:
    """Проверить результат одиночного прогона."""
    used = {e["call"] for e in result.get("trace", []) if "call" in e}
    ans = (result.get("answer") or "").lower()
    hallucinated = used - VALID_TOOL_NAMES
    must = all(kw.lower() in ans for kw in case["must_have_keywords"])
    arith_without_calc = (
        case["id"] in {"Q1", "Q2", "Q3"}
        and "calculate" not in used
        and bool(ans)
    )
    ok = bool(ans) and not hallucinated and must and not arith_without_calc
    return {
        "ok": ok,
        "used_tools": sorted(used),
        "hallucinated": sorted(hallucinated),
        "must_have_ok": must,
        "arith_without_calc": arith_without_calc,
        "answer_preview": (result.get("answer") or "")[:180],
    }


def _check_pwc(case: dict, result: dict) -> dict:
    """Проверить результат PWC-прогона."""
    used = set()
    for t in result.get("trace", []):
        if t.get("kind") == "worker":
            used.update(t.get("used_tools") or [])
    ans = (result.get("answer") or "").lower()
    hallucinated = used - VALID_TOOL_NAMES
    # Также проверим галлюцинации на этапе Планировщика (в плане expected_tools)
    plan_tools = set()
    plan = result.get("plan")
    if plan is not None:
        for sq in plan.subquestions:
            plan_tools.update(sq.expected_tools)
    plan_hallucinated = plan_tools - VALID_TOOL_NAMES

    must = all(kw.lower() in ans for kw in case["must_have_keywords"])
    ok = (
        bool(result.get("answer"))
        and not hallucinated
        and not plan_hallucinated
        and must
    )
    check = {
        "ok": ok,
        "used_tools": sorted(used),
        "plan_tools": sorted(plan_tools),
        "hallucinated_in_workers": sorted(hallucinated),
        "hallucinated_in_plan": sorted(plan_hallucinated),
        "must_have_ok": must,
        "iterations": result.get("iterations", -1),
        "answer_preview": (result.get("answer") or "")[:180],
    }
    if result.get("error"):
        check["error"] = result["error"]
    return check


def _run_many(
    runner: Callable[[str], dict],
    checker: Callable[[dict, dict], dict],
    case: dict,
    n: int,
    jobs: int = 1,
) -> dict:
    out = {"runs": [], "pass": 0}

    def _run_once() -> dict:
        try:
            raw = runner(case["query"])
        except Exception as e:
            raw = {"answer": None, "error": f"{type(e).__name__}: {e}", "trace": []}
        return checker(case, raw)

    if jobs <= 1:
        for _ in range(n):
            checked = _run_once()
            out["runs"].append(checked)
            out["pass"] += int(checked["ok"])
        return out

    with ThreadPoolExecutor(max_workers=min(jobs, n)) as pool:
        futures = [pool.submit(_run_once) for _ in range(n)]
        for fut in as_completed(futures):
            checked = fut.result()
            out["runs"].append(checked)
            out["pass"] += int(checked["ok"])
    return out


def run_case(
    case: dict,
    *,
    n: int = 5,
    jobs: int = 1,
    single_max_iter: int = 8,
    pwc_max_iter: int = 3,
) -> dict:
    def run_single(query: str) -> dict:
        return run_agent(query, max_iter=single_max_iter, verbose=False)

    def run_pwc_no_validator(query: str) -> dict:
        return run_pwc(
            query, max_iter=pwc_max_iter, verbose=False, use_validator=False
        )

    def run_pwc_with_validator(query: str) -> dict:
        return run_pwc(
            query, max_iter=pwc_max_iter, verbose=False, use_validator=True
        )

    single = _run_many(run_single, _check_single, case, n, jobs=jobs)
    pwc = _run_many(run_pwc_no_validator, _check_pwc, case, n, jobs=jobs)
    pwc_validator = _run_many(run_pwc_with_validator, _check_pwc, case, n, jobs=jobs)

    return {
        "id": case["id"],
        "query": case["query"],
        "comment": case["comment"],
        "n": n,
        "single": single,
        "pwc": pwc,
        "pwc_validator": pwc_validator,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true",
                    help="Только один прогон каждого кейса (быстро)")
    ap.add_argument("-n", type=int, default=5,
                    help="Сколько прогонов на кейс (default=5)")
    ap.add_argument(
        "--case-id",
        default=None,
        help="Запустить только один кейс по id (например, Q3).",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Параллельные повторы внутри одной конфигурации (default=1).",
    )
    ap.add_argument(
        "--single-max-iter",
        type=int,
        default=8,
        help="max_iter для одиночного агента (default=8).",
    )
    ap.add_argument(
        "--pwc-max-iter",
        type=int,
        default=3,
        help="max_iter для PWC (default=3).",
    )
    args = ap.parse_args()
    n = 1 if args.single else args.n

    print(f"Eval С6: {len(CASES)} кейсов x 3 конфигурации x {n} прогонов\n")
    results = []
    selected_cases = CASES
    if args.case_id:
        selected_cases = [c for c in CASES if c["id"] == args.case_id]
        if not selected_cases:
            raise ValueError(f"Неизвестный case-id: {args.case_id}")

    out = Path(__file__).parent / "eval_pwc_results.json"

    for case in selected_cases:
        print(f"=== {case['id']}: {case['query'][:70]}...")
        r = run_case(
            case,
            n=n,
            jobs=max(1, args.jobs),
            single_max_iter=args.single_max_iter,
            pwc_max_iter=args.pwc_max_iter,
        )
        results.append(r)
        s = r["single"]
        p = r["pwc"]
        pv = r["pwc_validator"]
        print(
            f"   single: {s['pass']}/{n}    pwc: {p['pass']}/{n}    "
            f"pwc+validator: {pv['pass']}/{n}"
        )
        for label, bucket in (("pwc", p), ("pwc+validator", pv)):
            sample = bucket["runs"][0]
            if sample.get("hallucinated_in_plan"):
                print(
                    f"   [warn] {label}: план содержит выдуманные инструменты: "
                    f"{sample['hallucinated_in_plan']}"
                )
            if sample.get("error"):
                print(f"   [warn] {label}: {sample['error']}")
        print()

        # Сохраняем прогресс после каждого кейса.
        out.write_text(
            json.dumps(results, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    # Итог
    print("=" * 60)
    print("ИТОГО:")
    for r in results:
        print(
            f"  {r['id']}: single {r['single']['pass']}/{n}  "
            f"pwc {r['pwc']['pass']}/{n}  "
            f"pwc+validator {r['pwc_validator']['pass']}/{n}  "
            f"— {r['query'][:60]}"
        )

    out.write_text(json.dumps(results, ensure_ascii=False, indent=2,
                              default=str), encoding="utf-8")
    print(f"\nРезультаты: {out}")


if __name__ == "__main__":
    main()
