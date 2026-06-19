"""Pydantic-схемы ответа RAG-агента."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    INSUFFICIENT = "insufficient"


class RAGAnswer(BaseModel):
    status: AnswerStatus = Field(description="answered — ответ есть; insufficient — данных нет")
    answer: str = Field(min_length=5, max_length=3000, description="Итоговый ответ")
    quotes: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Дословные цитаты из retrieved chunks (пусто при insufficient)",
    )
    chunk_ids: list[str] = Field(
        default_factory=list,
        description="ID чанков-источников, например habr_rag_intro__3",
    )
    confidence: float = Field(ge=0, le=1, description="Уверенность 0..1")

    @field_validator("confidence")
    @classmethod
    def confidence_matches_status(cls, v: float, info) -> float:
        status = info.data.get("status")
        if status == AnswerStatus.INSUFFICIENT and v > 0.35:
            raise ValueError("insufficient → confidence должна быть ≤ 0.35")
        if status == AnswerStatus.ANSWERED and v < 0.45:
            raise ValueError("answered → confidence должна быть ≥ 0.45")
        return v

    @field_validator("chunk_ids")
    @classmethod
    def chunks_required_when_answered(cls, v: list[str], info) -> list[str]:
        if info.data.get("status") == AnswerStatus.ANSWERED and not v:
            raise ValueError("answered требует хотя бы один chunk_id")
        return v

    @field_validator("quotes")
    @classmethod
    def quotes_required_when_answered(cls, v: list[str], info) -> list[str]:
        if info.data.get("status") == AnswerStatus.ANSWERED and not v:
            raise ValueError("answered требует хотя бы одну цитату")
        return v


class JudgeVerdict(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partially_supported"
    UNSUPPORTED = "unsupported"
    ABSTAIN_OK = "abstain_ok"


class JudgeReport(BaseModel):
    verdict: JudgeVerdict
    reasoning: str = Field(min_length=10, max_length=1000)
    score: float = Field(ge=0, le=1)

    @field_validator("verdict")
    @classmethod
    def abstain_verdict_for_insufficient(cls, v: JudgeVerdict, info) -> JudgeVerdict:
        # Дополнительная проверка делается в judge.py по status ответа
        return v
