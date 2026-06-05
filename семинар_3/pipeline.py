from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from pydantic import ValidationError

from prompts import ASPECTS_SYSTEM, IE_SYSTEM, JUDGE_SYSTEM, MAP_SYSTEM, REDUCE_SYSTEM
from schema import (
    JudgeReport,
    MapChunkSummary,
    RawReview,
    ReviewAspects,
    ReviewExtraction,
    ReviewSummary,
)

ASPECT_ORDER = ["performance", "design", "support", "price", "ads", "reliability"]


@dataclass
class QuoteRecord:
    review_id: str
    quote: str


@dataclass
class CostReport:
    baseline_input_cost_usd: float
    actual_input_cost_usd: float
    output_cost_usd: float
    total_actual_cost_usd: float
    savings_usd: float
    cache_hit_rate: float


@dataclass
class UsageAggregate:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0


def _usage_to_dict(usage: Any) -> dict[str, int]:
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    cache_hit_tokens = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
    cache_miss_tokens = int(
        getattr(usage, "prompt_cache_miss_tokens", prompt_tokens - cache_hit_tokens) or 0
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_hit_tokens": cache_hit_tokens,
        "cache_miss_tokens": cache_miss_tokens,
    }


def _accumulate_usage(total: UsageAggregate, usage: Any) -> None:
    stats = _usage_to_dict(usage)
    total.prompt_tokens += stats["prompt_tokens"]
    total.completion_tokens += stats["completion_tokens"]
    total.cache_hit_tokens += stats["cache_hit_tokens"]
    total.cache_miss_tokens += stats["cache_miss_tokens"]


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    raw_lines = Path(path).read_text(encoding="utf-8").splitlines()
    result = []
    for idx, line in enumerate(raw_lines, 1):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL line {idx}: {exc}") from exc
    return result


def load_reviews(path: str) -> list[RawReview]:
    rows = _read_jsonl(path)
    reviews: list[RawReview] = []
    for row in rows:
        reviews.append(RawReview.model_validate(row))
    return reviews


def load_reviews_with_errors(path: str) -> tuple[list[RawReview], list[dict[str, Any]]]:
    rows = _read_jsonl(path)
    valid: list[RawReview] = []
    invalid: list[dict[str, Any]] = []
    for row in rows:
        try:
            valid.append(RawReview.model_validate(row))
        except ValidationError as exc:
            invalid.append({"review_id": row.get("review_id"), "error": str(exc)})
    return valid, invalid


def chunked(items: Iterable[Any], size: int) -> list[list[Any]]:
    arr = list(items)
    return [arr[i : i + size] for i in range(0, len(arr), size)]


def check_quotes_equivalent(
    quotes: list[QuoteRecord], source_texts: dict[str, str]
) -> list[QuoteRecord]:
    ghosts: list[QuoteRecord] = []
    for quote in quotes:
        base = source_texts.get(quote.review_id, "").lower()
        probe = quote.quote.strip().lower()[:30]
        if probe and probe not in base:
            ghosts.append(quote)
    return ghosts


def build_heatmap(aspects: list[ReviewAspects], out_path: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sent_to_num = {"positive": 1, "neutral": 0, "negative": -1}
    names = [a.review_id for a in aspects]
    matrix = np.full((len(names), len(ASPECT_ORDER)), np.nan)
    for i, review in enumerate(aspects):
        for item in review.aspects:
            if item.aspect in ASPECT_ORDER:
                j = ASPECT_ORDER.index(item.aspect)
                matrix[i, j] = sent_to_num[item.sentiment]
    plt.figure(figsize=(10, max(4, len(names) * 0.25)))
    sns.heatmap(
        matrix,
        xticklabels=ASPECT_ORDER,
        yticklabels=names,
        center=0,
        cmap="coolwarm",
        cbar_kws={"label": "Sentiment"},
    )
    plt.title("Aspect sentiment by review")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def compute_cost_report(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cache_hit_tokens: int,
    cache_miss_tokens: int,
    prompt_price_per_1m: float,
    cache_hit_price_per_1m: float,
    completion_price_per_1m: float,
) -> CostReport:
    baseline_input_cost = (prompt_tokens / 1_000_000) * prompt_price_per_1m
    actual_input_cost = (
        (cache_miss_tokens / 1_000_000) * prompt_price_per_1m
        + (cache_hit_tokens / 1_000_000) * cache_hit_price_per_1m
    )
    output_cost = (completion_tokens / 1_000_000) * completion_price_per_1m
    total_actual = actual_input_cost + output_cost
    savings = baseline_input_cost - actual_input_cost
    hit_rate = cache_hit_tokens / prompt_tokens if prompt_tokens else 0.0
    return CostReport(
        baseline_input_cost_usd=baseline_input_cost,
        actual_input_cost_usd=actual_input_cost,
        output_cost_usd=output_cost,
        total_actual_cost_usd=total_actual,
        savings_usd=savings,
        cache_hit_rate=hit_rate,
    )


def _make_completion(client: Any, *, model: str, response_model: Any, messages: list[dict[str, str]]):
    return client.chat.completions.create(
        model=model,
        response_model=response_model,
        max_retries=3,
        temperature=0.0,
        with_completion=True,
        messages=messages,
    )


def _extract_reviews(client: Any, model: str, reviews: list[RawReview], usage: UsageAggregate) -> list[ReviewExtraction]:
    extracted: list[ReviewExtraction] = []
    for review in reviews:
        user_payload = (
            f"review_id: {review.review_id}\n"
            f"app_name: {review.app_name}\n"
            f"rating: {review.rating}\n"
            f"source: {review.source}\n"
            f"review_date: {review.review_date}\n\n"
            f"Текст отзыва:\n{review.text}"
        )
        result, completion = _make_completion(
            client,
            model=model,
            response_model=ReviewExtraction,
            messages=[
                {"role": "system", "content": IE_SYSTEM},
                {"role": "user", "content": user_payload},
            ],
        )
        extracted.append(result)
        _accumulate_usage(usage, completion.usage)
    return extracted


def _extract_aspects(client: Any, model: str, reviews: list[RawReview], usage: UsageAggregate) -> list[ReviewAspects]:
    per_review: list[ReviewAspects] = []
    for review in reviews:
        result, completion = _make_completion(
            client,
            model=model,
            response_model=ReviewAspects,
            messages=[
                {"role": "system", "content": ASPECTS_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"review_id: {review.review_id}\n"
                        f"Текст отзыва:\n{review.text}"
                    ),
                },
            ],
        )
        if result.review_id != review.review_id:
            result.review_id = review.review_id
        per_review.append(result)
        _accumulate_usage(usage, completion.usage)
    return per_review


def _map_reduce_summary(
    client: Any,
    model: str,
    extractions: list[ReviewExtraction],
    aspects: list[ReviewAspects],
    usage: UsageAggregate,
) -> ReviewSummary:
    by_review_aspects = {item.review_id: item for item in aspects}
    chunks = chunked(extractions, size=8)
    map_summaries: list[MapChunkSummary] = []

    for idx, chunk in enumerate(chunks, 1):
        packet = []
        for item in chunk:
            packet.append(
                {
                    "review_id": item.review_id,
                    "sentiment": item.sentiment,
                    "issues": [issue.model_dump() for issue in item.issues],
                    "aspects": [
                        aspect.model_dump()
                        for aspect in by_review_aspects.get(
                            item.review_id, ReviewAspects(review_id=item.review_id)
                        ).aspects
                    ],
                }
            )
        result, completion = _make_completion(
            client,
            model=model,
            response_model=MapChunkSummary,
            messages=[
                {"role": "system", "content": MAP_SYSTEM},
                {"role": "user", "content": json.dumps({"chunk_id": f"chunk_{idx}", "items": packet}, ensure_ascii=False)},
            ],
        )
        if not result.chunk_id:
            result.chunk_id = f"chunk_{idx}"
        map_summaries.append(result)
        _accumulate_usage(usage, completion.usage)

    reduce_input = {"chunks": [item.model_dump() for item in map_summaries]}
    summary, completion = _make_completion(
        client,
        model=model,
        response_model=ReviewSummary,
        messages=[
            {"role": "system", "content": REDUCE_SYSTEM},
            {"role": "user", "content": json.dumps(reduce_input, ensure_ascii=False)},
        ],
    )
    _accumulate_usage(usage, completion.usage)
    return summary


def _run_judge(
    client: Any,
    model: str,
    extractions: list[ReviewExtraction],
    summary: ReviewSummary,
    usage: UsageAggregate,
) -> JudgeReport:
    evidence = []
    for extraction in extractions:
        for issue in extraction.issues:
            evidence.append(
                {
                    "review_id": extraction.review_id,
                    "category": issue.category,
                    "severity": issue.severity,
                    "quote": issue.quote,
                }
            )
    report, completion = _make_completion(
        client,
        model=model,
        response_model=JudgeReport,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    {"action_items": summary.action_items, "evidence": evidence},
                    ensure_ascii=False,
                ),
            },
        ],
    )
    _accumulate_usage(usage, completion.usage)
    return report


def analyze(input_path: str, out_dir: str = "output") -> dict[str, Any]:
    from starter.llm_client import get_model, make_client

    started = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    reviews, invalid_rows = load_reviews_with_errors(input_path)
    if not reviews:
        raise ValueError("No valid reviews loaded from input.")

    model = get_model()
    client = make_client()
    usage = UsageAggregate()

    extractions = _extract_reviews(client, model, reviews, usage)
    aspects = _extract_aspects(client, model, reviews, usage)
    summary = _map_reduce_summary(client, model, extractions, aspects, usage)
    judge_report = _run_judge(client, model, extractions, summary, usage)

    source_texts = {review.review_id: review.text for review in reviews}
    quote_records: list[QuoteRecord] = []
    for extraction in extractions:
        for issue in extraction.issues:
            quote_records.append(QuoteRecord(review_id=extraction.review_id, quote=issue.quote))
    for review_aspects in aspects:
        for aspect in review_aspects.aspects:
            quote_records.append(QuoteRecord(review_id=review_aspects.review_id, quote=aspect.quote))

    ghosts = check_quotes_equivalent(quote_records, source_texts)
    ghost_rate = len(ghosts) / len(quote_records) if quote_records else 0.0

    build_heatmap(aspects, out_path=str(out / "heatmap.png"))

    prompt_price = float(os.getenv("PRICE_PROMPT_PER_1M", "0.27"))
    cache_hit_price = float(os.getenv("PRICE_CACHE_HIT_PER_1M", "0.07"))
    completion_price = float(os.getenv("PRICE_COMPLETION_PER_1M", "1.10"))
    cost = compute_cost_report(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cache_hit_tokens=usage.cache_hit_tokens,
        cache_miss_tokens=usage.cache_miss_tokens,
        prompt_price_per_1m=prompt_price,
        cache_hit_price_per_1m=cache_hit_price,
        completion_price_per_1m=completion_price,
    )

    (out / "reviews.json").write_text(
        json.dumps([item.model_dump() for item in extractions], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "aspects.json").write_text(
        json.dumps([item.model_dump() for item in aspects], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    (out / "judge_report.json").write_text(
        judge_report.model_dump_json(indent=2),
        encoding="utf-8",
    )

    report = {
        "model": model,
        "input_count": len(reviews),
        "validation_errors": invalid_rows,
        "validation_error_count": len(invalid_rows),
        "ghost_quotes_count": len(ghosts),
        "ghost_quote_rate": ghost_rate,
        "judge_overall_score": judge_report.overall_score,
        "usage": asdict(usage),
        "cost_report": asdict(cost),
        "elapsed_sec": time.time() - started,
    }
    (out / "run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "ghost_quotes.json").write_text(
        json.dumps([asdict(item) for item in ghosts], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run seminar 3 app review pipeline")
    parser.add_argument("input_path", nargs="?", default="input/reviews.jsonl")
    parser.add_argument("--out-dir", default="output")
    args = parser.parse_args()
    result = analyze(args.input_path, out_dir=args.out_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
