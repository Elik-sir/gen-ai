"""
Eval на gold.json: правильность (retrieval + judge) и путь (шаги, tools, токены).

Запуск:
    python eval.py
    python eval.py --case-id 3
    python eval.py --retrieval-only   # только hit-rate@5, без LLM-агента
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from agent import run_agent
from hallucination import run_all_checks
from judge import judge_answer
from rag import chunk_count, hybrid_retrieve, ingest
from schema import AnswerStatus, RAGAnswer

ROOT = Path(__file__).resolve().parent
GOLD_PATH = ROOT / "input" / "gold.json"
OUTPUT_PATH = ROOT / "output" / "eval_results.json"
CSV_PATH = ROOT / "output" / "eval_results.csv"
EVAL_K = 5


def load_gold() -> list[dict]:
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def hit_rate(retrieved_ids: list[str], gold_sources: list[str]) -> float:
    if not gold_sources:
        return 1.0 if not retrieved_ids else 0.0
    retrieved_sources = {rid.split("__")[0] for rid in retrieved_ids}
    found = [g for g in gold_sources if g in retrieved_sources]
    return len(found) / len(gold_sources)


def eval_retrieval(case: dict) -> dict:
    hits = hybrid_retrieve(case["question"], k=EVAL_K)
    retrieved_ids = [h["chunk_id"] for h in hits]
    retrieved_sources = [h["source_id"] for h in hits]
    hr = hit_rate(retrieved_ids, case.get("gold_sources", []))
    return {
        "hit_rate": hr,
        "retrieved_ids": retrieved_ids,
        "retrieved_sources": retrieved_sources,
    }


def eval_case(case: dict, *, skip_agent: bool = False) -> dict:
    retrieval = eval_retrieval(case)
    if skip_agent:
        return {
            "id": case["id"],
            "type": case["type"],
            "question": case["question"],
            "retrieval": retrieval,
            "pass": retrieval["hit_rate"] >= 0.5,
        }

    t0 = time.time()
    agent_result = run_agent(case["question"], verbose=False)
    answer = RAGAnswer.model_validate(agent_result["answer"])
    retrieved = agent_result["retrieved"]
    hallucination = run_all_checks(answer, retrieved)
    judge = judge_answer(
        case["question"],
        answer,
        list(retrieved.values()),
        expect_abstain=case.get("expect_abstain", False),
    )
    elapsed = round(time.time() - t0, 2)

    expect_abstain = case.get("expect_abstain", False)
    if expect_abstain:
        correctness = answer.status == AnswerStatus.INSUFFICIENT and judge.verdict.value == "abstain_ok"
    else:
        judge_ok = judge.verdict.value in ("supported", "partially_supported")
        retrieval_ok = retrieval["hit_rate"] >= 0.5 or not case.get("gold_sources")
        correctness = judge_ok and retrieval_ok and answer.status == AnswerStatus.ANSWERED

    no_hallucination = hallucination["total_issues"] == 0
    passed = correctness and no_hallucination

    return {
        "id": case["id"],
        "type": case["type"],
        "question": case["question"],
        "pass": passed,
        "correctness": correctness,
        "hit_rate": retrieval["hit_rate"],
        "judge_verdict": judge.verdict.value,
        "judge_score": judge.score,
        "status": answer.status.value,
        "confidence": answer.confidence,
        "ghost_quotes": hallucination["ghost_quotes"],
        "fake_chunk_ids": hallucination["fake_chunk_ids"],
        "agent_steps": agent_result["agent_steps"],
        "search_calls": agent_result["search_calls"],
        "tools_called": agent_result["tools_called"],
        "prompt_tokens": agent_result["usage"]["prompt_tokens"],
        "completion_tokens": agent_result["usage"]["completion_tokens"],
        "cost_usd": agent_result["usage"]["cost_usd"],
        "latency_sec": elapsed,
        "retrieved_sources": retrieval["retrieved_sources"],
    }


def summarize(cases: list[dict]) -> dict:
    n = len(cases)
    if n == 0:
        return {}
    hit_rates = [
        c.get("hit_rate", c.get("retrieval", {}).get("hit_rate", 0)) for c in cases
    ]
    return {
        "n": n,
        "pass_rate": round(sum(1 for c in cases if c.get("pass")) / n, 3),
        "avg_hit_rate": round(sum(hit_rates) / n, 3),
        "total_ghost_quotes": sum(c.get("ghost_quotes", 0) for c in cases),
        "avg_steps": round(sum(c.get("agent_steps", 0) for c in cases) / n, 2),
        "avg_tokens": round(
            sum(c.get("prompt_tokens", 0) + c.get("completion_tokens", 0) for c in cases) / n
        ),
        "avg_latency_sec": round(sum(c.get("latency_sec", 0) for c in cases) / n, 2),
    }


def run_ghost_checker_tests() -> list[dict]:
    """Синтетические битые ответы — проверяем, что checker ловит."""
    from schema import AnswerStatus

    chunks = {
        "habr_rag_intro__0": "RAG дополняет LLM внешними документами через retrieval.",
    }
    tests = [
        {
            "name": "ghost_quote",
            "answer": RAGAnswer(
                status=AnswerStatus.ANSWERED,
                answer="RAG использует блокчейн для хранения.",
                quotes=["блокчейн для хранения embeddings навсегда"],
                chunk_ids=["habr_rag_intro__0"],
                confidence=0.9,
            ),
            "expect_issues": 1,
        },
        {
            "name": "fake_chunk_id",
            "answer": RAGAnswer(
                status=AnswerStatus.ANSWERED,
                answer="RAG дополняет LLM.",
                quotes=["RAG дополняет LLM"],
                chunk_ids=["nonexistent__99"],
                confidence=0.9,
            ),
            "expect_issues": 1,
        },
        {
            "name": "valid_answer",
            "answer": RAGAnswer(
                status=AnswerStatus.ANSWERED,
                answer="RAG дополняет LLM внешними документами.",
                quotes=["RAG дополняет LLM внешними"],
                chunk_ids=["habr_rag_intro__0"],
                confidence=0.85,
            ),
            "expect_issues": 0,
        },
    ]
    results = []
    for t in tests:
        h = run_all_checks(t["answer"], chunks)
        ok = (h["total_issues"] >= t["expect_issues"]) if t["expect_issues"] else h["total_issues"] == 0
        results.append({"name": t["name"], "issues": h["total_issues"], "pass": ok, **h})
    return results


def write_csv(cases: list[dict]) -> None:
    fields = [
        "id", "type", "pass", "hit_rate", "judge_verdict", "status",
        "ghost_quotes", "agent_steps", "search_calls", "prompt_tokens",
        "completion_tokens", "latency_sec", "question",
    ]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(cases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", type=int, help="Только один кейс по id")
    parser.add_argument("--retrieval-only", action="store_true", help="Без агента, только hit-rate")
    parser.add_argument("--ghost-tests", action="store_true", help="Тест checker галлюцинаций")
    args = parser.parse_args()

    if args.ghost_tests:
        ghost = run_ghost_checker_tests()
        print(json.dumps(ghost, ensure_ascii=False, indent=2))
        passed = sum(1 for g in ghost if g["pass"])
        print(f"\nGhost checker: {passed}/{len(ghost)} passed")
        return

    if chunk_count() == 0:
        print("Индекс пуст — ingest...", flush=True)
        ingest()

    gold = load_gold()
    if args.case_id is not None:
        gold = [c for c in gold if c["id"] == args.case_id]
        if not gold:
            raise SystemExit(f"Кейс id={args.case_id} не найден")

    results = []
    for i, case in enumerate(gold, 1):
        print(f"[{i}/{len(gold)}] Q{case['id']}: {case['question'][:60]}...", flush=True)
        results.append(eval_case(case, skip_agent=args.retrieval_only))

    summary = summarize(results)
    payload = {"summary": summary, "cases": results}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.retrieval_only:
        write_csv(results)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n-> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
