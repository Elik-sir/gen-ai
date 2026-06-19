"""
Главный конвейер: ingest → agent → hallucination check → judge.

Запуск:
    python pipeline.py --ingest
    python pipeline.py --question "Что такое RAG?"
    python pipeline.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from agent import run_agent
from hallucination import run_all_checks
from judge import judge_answer
from rag import chunk_count, ingest
from schema import AnswerStatus, RAGAnswer

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
ANSWERS_DIR = OUTPUT_DIR / "run_answers"
TRACE_PATH = OUTPUT_DIR / "trace.jsonl"
HALLUCINATION_PATH = OUTPUT_DIR / "hallucination_report.json"


def ensure_index() -> None:
    if chunk_count() == 0:
        print("Индекс пуст — запускаю ingest...", flush=True)
        ingest()


def answer_one(question: str, *, case_id: str | None = None, expect_abstain: bool = False) -> dict:
    ensure_index()
    t0 = time.time()

    agent_result = run_agent(question, trace_path=TRACE_PATH, verbose=True)
    answer = RAGAnswer.model_validate(agent_result["answer"])
    retrieved = agent_result["retrieved"]

    hallucination = run_all_checks(answer, retrieved)
    chunks_text = list(retrieved.values())
    judge = judge_answer(question, answer, chunks_text, expect_abstain=expect_abstain)

    elapsed = round(time.time() - t0, 2)
    payload = {
        "case_id": case_id,
        "question": question,
        "answer": answer.model_dump(),
        "hallucination": hallucination,
        "judge": judge.model_dump(),
        "path": {
            "agent_steps": agent_result["agent_steps"],
            "tools_called": agent_result["tools_called"],
            "search_calls": agent_result["search_calls"],
            "latency_sec": elapsed,
            **agent_result["usage"],
        },
        "retrieved_chunk_ids": list(retrieved.keys()),
    }

    ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
    fname = case_id or f"q_{abs(hash(question)) % 10_000}"
    out_file = ANSWERS_DIR / f"{fname}.json"
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _update_hallucination_report(hallucination, case_id or fname)

    print("\n" + "=" * 60)
    print(f"Вопрос: {question}")
    print(f"Status: {answer.status.value} | confidence: {answer.confidence}")
    print(f"Judge: {judge.verdict.value} (score={judge.score})")
    print(f"Ghosts: {hallucination['ghost_quotes']} | Steps: {agent_result['agent_steps']}")
    print(f"-> {out_file}")
    return payload


def _update_hallucination_report(h: dict, case_id: str) -> None:
    report: dict = {"cases": [], "totals": {"ghost_quotes": 0, "fake_chunk_ids": 0}}
    if HALLUCINATION_PATH.exists():
        report = json.loads(HALLUCINATION_PATH.read_text(encoding="utf-8"))
    report["cases"] = [c for c in report.get("cases", []) if c.get("case_id") != case_id]
    report["cases"].append({"case_id": case_id, **h})
    report["totals"] = {
        "ghost_quotes": sum(c.get("ghost_quotes", 0) for c in report["cases"]),
        "fake_chunk_ids": sum(c.get("fake_chunk_ids", 0) for c in report["cases"]),
        "unsupported_numbers": sum(c.get("unsupported_numbers", 0) for c in report["cases"]),
    }
    HALLUCINATION_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Q&A по статьям Habr")
    parser.add_argument("--ingest", action="store_true", help="Переиндексировать корпус")
    parser.add_argument("--question", "-q", type=str, help="Один вопрос")
    args = parser.parse_args()

    if args.ingest:
        ingest()
        return

    if args.question:
        answer_one(args.question)
        return

    # Демо-прогон по умолчанию
    demo = "Что такое RAG и зачем он нужен языковым моделям?"
    if chunk_count() == 0:
        ingest()
    answer_one(demo, case_id="demo")


if __name__ == "__main__":
    main()
