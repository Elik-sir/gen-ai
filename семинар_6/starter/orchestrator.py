"""
Оркестратор: главный цикл Планировщик-Исполнитель-Критик.

На семинаре нужно:
- реализовать topological_sort (TODO 1),
- реализовать replan/rework-ветки цикла (TODO 2),
- написать synthesize для финального ответа (TODO 3).

Важно: max_iter защищает от бесконечного цикла, если Критик
постоянно говорит «переделай».
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from llm_client import get_model, make_raw_client
from planner import planner
from schemas_pwc import Plan, SubQuestion, WorkerAnswer
from worker import worker

VALID_TOOLS = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}


def validate_plan(plan: Plan) -> list[str]:
    """Вернуть список ошибок плана (пустой — всё ок)."""
    errors: list[str] = []
    for sq in plan.subquestions:
        for tool in sq.expected_tools:
            if tool not in VALID_TOOLS:
                errors.append(f"sq#{sq.id}:{tool}")
    return sorted(set(errors))


def _ensure_valid_plan(question: str, plan: Plan) -> tuple[Plan, list[str]]:
    """Проверить план и при необходимости один раз перепланировать."""
    errors = validate_plan(plan)
    if not errors:
        return plan, []
    repaired = planner(question, feedback=f"Инструменты не существуют: {errors}")
    return repaired, errors


def _topological_levels(subqs: list[SubQuestion]) -> list[list[SubQuestion]]:
    """Сгруппировать подвопросы в уровни независимого исполнения."""
    by_id = {s.id: s for s in subqs}
    indegree = {sq.id: 0 for sq in subqs}
    dependents: dict[int, list[int]] = {sq.id: [] for sq in subqs}

    for sq in subqs:
        valid_deps = [dep for dep in sq.depends_on if dep in by_id]
        indegree[sq.id] = len(valid_deps)
        for dep in valid_deps:
            dependents[dep].append(sq.id)

    current = [sq_id for sq_id, deg in indegree.items() if deg == 0]
    levels: list[list[SubQuestion]] = []
    processed = 0

    while current:
        current.sort()
        levels.append([by_id[sq_id] for sq_id in current])
        next_level: list[int] = []
        for sq_id in current:
            processed += 1
            for child_id in dependents[sq_id]:
                indegree[child_id] -= 1
                if indegree[child_id] == 0:
                    next_level.append(child_id)
        current = next_level

    if processed != len(subqs):
        raise ValueError("Цикл в depends_on, невозможно построить уровни.")
    return levels


def execute_level(
    level: list[SubQuestion], prev_answers: dict[int, WorkerAnswer]
) -> dict[int, WorkerAnswer]:
    """Прогнать все подвопросы уровня параллельно."""
    out: dict[int, WorkerAnswer] = {}
    if not level:
        return out

    with ThreadPoolExecutor(max_workers=len(level)) as pool:
        futures = {
            pool.submit(worker, sq, prev_answers=prev_answers): sq.id for sq in level
        }
        for fut in as_completed(futures):
            sq_id = futures[fut]
            out[sq_id] = fut.result()
    return out


def _synthesize(
    question: str,
    plan: Plan,
    answers: dict[int, WorkerAnswer],
) -> str:
    """Собрать финальный ответ одним LLM-вызовом без tools."""
    parts = [f"{i}. {answers[i].answer}" for i in sorted(answers)]
    client = make_raw_client()
    resp = client.chat.completions.create(
        model=get_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "Собери итоговый ответ пользователю по промежуточным фактам. "
                    "Пиши 1-2 короткие фразы, без выдумывания новых чисел."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Вопрос: {question}\n\n"
                    f"План: {plan.reasoning}\n"
                    "Промежуточные ответы:\n"
                    + "\n".join(parts)
                ),
            },
        ],
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip()


def run_pwc(
    question: str,
    *,
    max_iter: int = 3,
    verbose: bool = True,
    use_validator: bool = True,
    use_parallel: bool = True,
    critic_temperature: float = 0.7,
) -> dict[str, Any]:
    """Запустить цикл Планировщик-Исполнитель-Критик."""
    trace: list[dict[str, Any]] = []

    plan = planner(question)
    if use_validator:
        plan, validation_errors = _ensure_valid_plan(question, plan)
        if validation_errors:
            trace.append(
                {
                    "iter": 0,
                    "kind": "plan_validation",
                    "errors": validation_errors,
                }
            )

    final_validation_errors = validate_plan(plan) if use_validator else []
    if final_validation_errors:
        return {
            "answer": None,
            "error": f"план не прошёл валидацию: {final_validation_errors}",
            "plan": plan,
            "answers": {},
            "trace": trace,
            "iterations": 0,
        }

    trace.append(
        {
            "iter": 0,
            "kind": "plan",
            "reasoning": plan.reasoning,
            "subquestions": [sq.model_dump() for sq in plan.subquestions],
        }
    )

    if verbose:
        print(f"\n[plan] {plan.reasoning}")
        for sq in plan.subquestions:
            print(f"  {sq.id}. [{','.join(sq.expected_tools)}] {sq.question}")

    for iter_num in range(1, max_iter + 1):
        answers: dict[int, WorkerAnswer] = {}
        levels = _topological_levels(plan.subquestions)
        for level in levels:
            level_answers = (
                execute_level(level, prev_answers=answers)
                if use_parallel
                else {sq.id: worker(sq, prev_answers=answers) for sq in level}
            )
            for sq in sorted(level, key=lambda s: s.id):
                ans = level_answers[sq.id]
                answers[sq.id] = ans
                trace.append(
                    {
                        "iter": iter_num,
                        "kind": "worker",
                        "sq_id": sq.id,
                        "used_tools": ans.used_tools,
                        "answer": ans.answer,
                    }
                )
                if verbose:
                    print(f"  [{sq.id}] → {ans.answer}   tools={ans.used_tools}")

        verdict = critic(
            question, plan, answers, temperature=critic_temperature
        )
        trace.append(
            {
                "iter": iter_num,
                "kind": "verdict",
                "ok": verdict.ok,
                "action": verdict.action,
                "reason": verdict.reason,
                "rework_ids": verdict.rework_ids,
            }
        )

        if verbose:
            mark = "✅" if verdict.ok else "❌"
            print(f"  [critic {mark}] {verdict.action}: {verdict.reason}")

        if verdict.ok:
            final = _synthesize(question, plan, answers)
            return {
                "answer": final,
                "plan": plan,
                "answers": answers,
                "trace": trace,
                "iterations": iter_num,
            }

        if verdict.action == "replan":
            plan = planner(question, feedback=verdict.reason)
        elif verdict.action == "rework":
            rework_ids = sorted(set(verdict.rework_ids))
            feedback = (
                f"Переделай подвопросы {rework_ids}. "
                f"Причина критика: {verdict.reason}"
            )
            plan = planner(question, feedback=feedback)
        else:
            break

        if use_validator:
            plan, validation_errors = _ensure_valid_plan(question, plan)
            if validation_errors:
                trace.append(
                    {
                        "iter": iter_num,
                        "kind": "plan_validation",
                        "errors": validation_errors,
                    }
                )
            final_validation_errors = validate_plan(plan)
            if final_validation_errors:
                return {
                    "answer": None,
                    "error": f"план не прошёл валидацию: {final_validation_errors}",
                    "plan": plan,
                    "answers": answers,
                    "trace": trace,
                    "iterations": iter_num,
                }
        trace.append(
            {
                "iter": iter_num,
                "kind": "replan",
                "reasoning": plan.reasoning,
                "subquestions": [sq.model_dump() for sq in plan.subquestions],
            }
        )

    return {
        "answer": None,
        "error": f"не удалось получить вердикт 'accept' за {max_iter} итераций",
        "plan": plan,
        "answers": answers,
        "trace": trace,
        "iterations": max_iter,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+", help="Вопрос к агенту")
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--no-validator",
        action="store_true",
        help="Отключить валидацию инструментов в плане.",
    )
    ap.add_argument(
        "--sequential",
        action="store_true",
        help="Исполнять подвопросы последовательно (без параллельности по уровням).",
    )
    ap.add_argument(
        "--critic-temperature",
        type=float,
        default=0.7,
        help="Температура критика (по умолчанию 0.7).",
    )
    ap.add_argument(
        "--trace", type=Path, default=None, help="Куда сохранить JSON-лог (если задан)"
    )
    args = ap.parse_args()

    q = " ".join(args.query)
    res = run_pwc(
        q,
        max_iter=args.max_iter,
        verbose=not args.quiet,
        use_validator=not args.no_validator,
        use_parallel=not args.sequential,
        critic_temperature=args.critic_temperature,
    )

    print("\n=== ВОПРОС ===")
    print(q)
    print("\n=== ОТВЕТ ===")
    print(res.get("answer") or res.get("error"))
    print(f"\n(итераций: {res.get('iterations', '?')})")

    if args.trace:
        args.trace.write_text(
            json.dumps(
                {"query": q, **_serialize(res)},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"Трейс сохранён: {args.trace}")


def _serialize(res: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in res.items():
        if k == "plan" and v is not None:
            out[k] = v.model_dump()
        elif k == "answers":
            out[k] = {i: a.model_dump() for i, a in v.items()}
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
