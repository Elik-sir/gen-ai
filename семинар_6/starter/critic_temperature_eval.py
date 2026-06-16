from __future__ import annotations

import json
from pathlib import Path

from critic import critic
from schemas_pwc import Plan, SubQuestion, WorkerAnswer

RUNS_PER_TEMP = 10
TEMPERATURES = [0.0, 0.7]

FAKE_BROKEN = [
    {
        "name": "арифметика без calculate",
        "question": "На сколько EUR дороже USD сегодня?",
        "plan": Plan(
            reasoning="Получим два курса, затем найдем разницу.",
            subquestions=[
                SubQuestion(
                    id=1,
                    question="Какой курс USD сегодня?",
                    expected_tools=["get_fx_rate"],
                    depends_on=[],
                ),
                SubQuestion(
                    id=2,
                    question="Какой курс EUR сегодня?",
                    expected_tools=["get_fx_rate"],
                    depends_on=[],
                ),
            ],
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="курс USD",
                answer="USD=82.5, EUR=89.0, разница=6.5 руб.",
                used_tools=["get_fx_rate"],
                raw_trace=[],
            )
        },
    },
    {
        "name": "выдуманное число",
        "question": "Какой курс USD на 2024-12-31?",
        "plan": Plan(
            reasoning="Один вызов get_fx_rate.",
            subquestions=[
                SubQuestion(
                    id=1,
                    question="Курс USD на 2024-12-31",
                    expected_tools=["get_fx_rate"],
                    depends_on=[],
                )
            ],
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="курс USD",
                answer="Курс USD = 999 руб.",
                used_tools=[],
                raw_trace=[],
            )
        },
    },
    {
        "name": "несогласованные подвопросы",
        "question": "Во сколько раз USD подорожал с 2022-01-01 по сегодня?",
        "plan": Plan(
            reasoning="Нужны два курса и отношение.",
            subquestions=[
                SubQuestion(
                    id=1,
                    question="Курс USD на 2022-01-01",
                    expected_tools=["get_fx_rate"],
                    depends_on=[],
                ),
                SubQuestion(
                    id=2,
                    question="Курс USD сегодня",
                    expected_tools=["get_fx_rate"],
                    depends_on=[],
                ),
                SubQuestion(
                    id=3,
                    question="Во сколько раз вырос курс",
                    expected_tools=["calculate"],
                    depends_on=[1, 2],
                ),
            ],
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="курс на 2022-01-01",
                answer="70.0",
                used_tools=["get_fx_rate"],
                raw_trace=[],
            ),
            2: WorkerAnswer(
                subquestion_id=2,
                question_snippet="курс сегодня",
                answer="84.0",
                used_tools=["get_fx_rate"],
                raw_trace=[],
            ),
            3: WorkerAnswer(
                subquestion_id=3,
                question_snippet="во сколько раз",
                answer="Рост в 2.8 раза.",
                used_tools=["calculate"],
                raw_trace=[],
            ),
        },
    },
    {
        "name": "ошибка в одном из ответов",
        "question": "Какова реальная ставка?",
        "plan": Plan(
            reasoning="Взять ключевую ставку и инфляцию, затем вычесть.",
            subquestions=[
                SubQuestion(
                    id=1,
                    question="Ключевая ставка сейчас",
                    expected_tools=["get_key_rate"],
                    depends_on=[],
                ),
                SubQuestion(
                    id=2,
                    question="Инфляция за последний месяц",
                    expected_tools=["get_inflation"],
                    depends_on=[],
                ),
                SubQuestion(
                    id=3,
                    question="Реальная ставка",
                    expected_tools=["calculate"],
                    depends_on=[1, 2],
                ),
            ],
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="ставка",
                answer="21.0%",
                used_tools=["get_key_rate"],
                raw_trace=[],
            ),
            2: WorkerAnswer(
                subquestion_id=2,
                question_snippet="инфляция",
                answer="(ошибка: timeout)",
                used_tools=["get_inflation"],
                raw_trace=[],
            ),
            3: WorkerAnswer(
                subquestion_id=3,
                question_snippet="реальная ставка",
                answer="11.0%",
                used_tools=["calculate"],
                raw_trace=[],
            ),
        },
    },
    {
        "name": "неполное покрытие вопроса",
        "question": "Сравни реальную ставку сейчас и на 2022-01-01.",
        "plan": Plan(
            reasoning="Посчитаем только текущую реальную ставку.",
            subquestions=[
                SubQuestion(
                    id=1,
                    question="Текущая ключевая ставка",
                    expected_tools=["get_key_rate"],
                    depends_on=[],
                ),
                SubQuestion(
                    id=2,
                    question="Текущая инфляция",
                    expected_tools=["get_inflation"],
                    depends_on=[],
                ),
                SubQuestion(
                    id=3,
                    question="Текущая реальная ставка",
                    expected_tools=["calculate"],
                    depends_on=[1, 2],
                ),
            ],
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="текущая ставка",
                answer="21.0%",
                used_tools=["get_key_rate"],
                raw_trace=[],
            ),
            2: WorkerAnswer(
                subquestion_id=2,
                question_snippet="текущая инфляция",
                answer="10.5%",
                used_tools=["get_inflation"],
                raw_trace=[],
            ),
            3: WorkerAnswer(
                subquestion_id=3,
                question_snippet="текущая реальная",
                answer="10.5 п.п.",
                used_tools=["calculate"],
                raw_trace=[],
            ),
        },
    },
]


def main() -> None:
    summary = []
    for case in FAKE_BROKEN:
        row = {"case": case["name"]}
        print(f"\nCase: {case['name']}")
        for temp in TEMPERATURES:
            false_accepts = 0
            for _ in range(RUNS_PER_TEMP):
                verdict = critic(
                    case["question"],
                    case["plan"],
                    case["answers"],
                    temperature=temp,
                )
                false_accepts += int(verdict.ok)
            key = f"t_{temp}"
            row[key] = {
                "false_accepts": false_accepts,
                "runs": RUNS_PER_TEMP,
            }
            print(f"  T={temp}: {false_accepts}/{RUNS_PER_TEMP}")
        summary.append(row)

    out = Path(__file__).parent / "critic_temperature_results.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
