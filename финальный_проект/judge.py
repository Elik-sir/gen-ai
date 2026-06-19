"""LLM-as-judge: подтверждён ли ответ retrieved chunks."""

from __future__ import annotations

from llm_client import get_model, make_client
from schema import AnswerStatus, JudgeReport, JudgeVerdict, RAGAnswer

client = make_client()
MODEL = get_model()

JUDGE_SYSTEM = """Ты — строгий ревизор RAG-системы. Тебе дают вопрос, ответ агента и фрагменты
документов (retrieved chunks). Оцени, подтверждается ли ответ контекстом.

Вердикты:
- supported — каждый ключевой факт есть в chunks
- partially_supported — часть фактов подтверждена, часть нет
- unsupported — ответ не следует из chunks (галлюцинация или общие знания модели)
- abstain_ok — status=insufficient и агент честно отказался ответить

Не оправдывай модель. Ищи противоречия и выдуманные детали.
score: 1.0 = полностью подтверждён; 0.0 = полная галлюцинация."""


def judge_answer(
    question: str,
    answer: RAGAnswer,
    chunks: list[str],
    *,
    expect_abstain: bool = False,
) -> JudgeReport:
    if expect_abstain and answer.status == AnswerStatus.INSUFFICIENT:
        return JudgeReport(
            verdict=JudgeVerdict.ABSTAIN_OK,
            reasoning="Агент корректно отказался: в корпусе нет ответа.",
            score=1.0,
        )
    if expect_abstain and answer.status == AnswerStatus.ANSWERED:
        return JudgeReport(
            verdict=JudgeVerdict.UNSUPPORTED,
            reasoning="Ожидался отказ (out-of-scope), но агент дал ответ.",
            score=0.0,
        )

    ctx = "\n\n---\n\n".join(f"[{i}]\n{c}" for i, c in enumerate(chunks))
    user = (
        f"Вопрос: {question}\n\n"
        f"Ответ агента (status={answer.status.value}, confidence={answer.confidence}):\n"
        f"{answer.answer}\n\n"
        f"Цитаты: {answer.quotes}\n\n"
        f"Retrieved chunks:\n{ctx}"
    )
    return client.chat.completions.create(
        model=MODEL,
        response_model=JudgeReport,
        max_retries=2,
        temperature=0.0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
