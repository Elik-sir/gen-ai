from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


AppSource = Literal["app_store", "google_play", "rustore", "kaggle", "synthetic", "other"]
IssueCategory = Literal["performance", "design", "support", "price", "ads", "reliability"]
AspectName = Literal["performance", "design", "support", "price", "ads", "reliability"]
SentimentLabel = Literal["positive", "negative", "neutral"]


class RawReview(BaseModel):
    review_id: str
    source: AppSource = "other"
    rating: Optional[int] = None
    review_date: Optional[date] = None
    app_name: str
    text: str = Field(min_length=8)

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and not (1 <= value <= 5):
            raise ValueError("rating must be in [1, 5]")
        return value

    @field_validator("review_date")
    @classmethod
    def validate_review_date(cls, value: Optional[date]) -> Optional[date]:
        if value is not None and value > date.today():
            raise ValueError("review_date cannot be in the future")
        return value


class ReviewIssue(BaseModel):
    category: IssueCategory
    severity: int = Field(ge=1, le=5)
    quote: str = Field(min_length=5)
    rationale: str = Field(min_length=8)


class ReviewExtraction(BaseModel):
    review_id: str
    app_name: str
    sentiment: SentimentLabel
    issues: list[ReviewIssue] = Field(default_factory=list)
    short_summary: str = Field(min_length=10)


class AspectAssessment(BaseModel):
    aspect: AspectName
    sentiment: SentimentLabel
    confidence: float = Field(ge=0, le=1)
    quote: str = Field(min_length=5)
    score: float = Field(ge=-1, le=1)


class ReviewAspects(BaseModel):
    review_id: str
    aspects: list[AspectAssessment] = Field(default_factory=list)


class MapChunkSummary(BaseModel):
    chunk_id: str
    key_points: list[str] = Field(min_length=2, max_length=8)
    risks: list[str] = Field(default_factory=list, max_length=5)
    opportunities: list[str] = Field(default_factory=list, max_length=5)


class ReviewSummary(BaseModel):
    headline: str
    key_findings: list[str] = Field(min_length=3, max_length=8)
    action_items: list[str] = Field(min_length=2, max_length=8)


class ActionVerdict(BaseModel):
    action: str
    support: Literal["supported", "weakly_supported", "not_supported"]
    evidence: list[str] = Field(default_factory=list)
    comment: str


class JudgeReport(BaseModel):
    verdicts: list[ActionVerdict]
    overall_score: float = Field(ge=0, le=1)
    summary: str
