"""
ReAct-агент с инструментами search_corpus и submit_answer.

Техники: агент с tools (S5) + structured output (S4) + RAG (S4).
"""

from __future__ import annotations

import datetime
import json
import uuid
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import Any

from llm_client import get_model, make_raw_client
from rag import hybrid_retrieve
from schema import AnswerStatus, RAGAnswer

PRICE_IN_PER_MTOK = 0.14
PRICE_OUT_PER_MTOK = 0.28

SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_corpus",
        "description": "Гибридный поиск (BM25 + dense + RRF) по корпусу статей Habr про RAG.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Поисковый запрос"},
                "k": {"type": "integer", "description": "Число чанков (1-8)", "default": 5},
            },
            "required": ["query"],
        },
    },
}

SUBMIT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": "Финальный структурированный ответ. Вызывай только после search_corpus.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["answered", "insufficient"]},
                "answer": {"type": "string"},
                "quotes": {"type": "array", "items": {"type": "string"}},
                "chunk_ids": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
            },
            "required": ["status", "answer", "confidence"],
        },
    },
}

SYSTEM_PROMPT = """\
Ты — ассистент по корпусу статей Habr про RAG и LLM-агентов.

Правила:
1. Сначала вызови search_corpus — не отвечай без поиска.
2. Опирайся ТОЛЬКО на найденные чанки. Не используй общие знания.
3. Если в чанках нет ответа — submit_answer со status=insufficient, confidence≤0.3.
4. В quotes — дословные короткие фрагменты из чанков (не пересказ).
5. В chunk_ids — только ID из результатов search_corpus.
6. Для сложных вопросов можно вызвать search_corpus несколько раз с разными запросами.
7. Финал — только через submit_answer.

Текущая дата: {today}.
"""


def _exec_search(args: dict) -> dict:
    query = args.get("query", "")
    k = min(max(int(args.get("k", 5)), 1), 8)
    hits = hybrid_retrieve(query, k=k)
    return {
        "query": query,
        "hits": [
            {
                "chunk_id": h["chunk_id"],
                "source_id": h["source_id"],
                "score": h["score"],
                "text": h["text"][:1200],
            }
            for h in hits
        ],
    }


def _parse_submit(args: dict) -> RAGAnswer:
    return RAGAnswer.model_validate(args)


def run_agent(
    question: str,
    *,
    max_iter: int = 6,
    trace_path: Path | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    client = make_raw_client()
    model = get_model()
    tools = [SEARCH_SCHEMA, SUBMIT_SCHEMA]
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(today=datetime.date.today().isoformat()),
        },
        {"role": "user", "content": question},
    ]

    trace: list[dict[str, Any]] = []
    usage_log: list[dict[str, Any]] = []
    retrieved: dict[str, str] = {}
    final_answer: RAGAnswer | None = None
    run_id = str(uuid.uuid4())

    for step in range(1, max_iter + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        u = getattr(resp, "usage", None)
        pin = pout = 0
        cost = 0.0
        if u is not None:
            pin, pout = u.prompt_tokens, u.completion_tokens
            cost = pin / 1e6 * PRICE_IN_PER_MTOK + pout / 1e6 * PRICE_OUT_PER_MTOK
            usage_log.append(
                {
                    "step": step,
                    "prompt_tokens": pin,
                    "completion_tokens": pout,
                    "cost_usd": round(cost, 6),
                }
            )

        if not msg.tool_calls:
            trace.append({"step": step, "event": "text_final", "content": msg.content})
            break

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except JSONDecodeError as e:
                obs = {"error": f"битый JSON: {e}"}
                args = {}
            else:
                if name == "search_corpus":
                    obs = _exec_search(args)
                    for h in obs.get("hits", []):
                        retrieved[h["chunk_id"]] = h["text"]
                elif name == "submit_answer":
                    try:
                        final_answer = _parse_submit(args)
                        obs = {"ok": True, "status": final_answer.status.value}
                    except Exception as e:
                        obs = {"error": f"validation: {e}"}
                else:
                    obs = {"error": f"unknown tool: {name}"}

            trace.append(
                {
                    "step": step,
                    "tool": name,
                    "args": args,
                    "obs_summary": (
                        f"{len(obs.get('hits', []))} hits"
                        if name == "search_corpus"
                        else str(obs)[:200]
                    ),
                    "prompt_tokens": pin,
                    "completion_tokens": pout,
                }
            )
            if verbose:
                print(f"[step {step}] {name}({args})")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(obs, ensure_ascii=False),
                }
            )

        if final_answer is not None:
            break

    if final_answer is None:
        final_answer = RAGAnswer(
            status=AnswerStatus.INSUFFICIENT,
            answer="Не удалось сформировать структурированный ответ за отведённые шаги.",
            quotes=[],
            chunk_ids=[],
            confidence=0.1,
        )

    result = {
        "run_id": run_id,
        "question": question,
        "answer": final_answer.model_dump(),
        "retrieved": retrieved,
        "trace": trace,
        "agent_steps": len(trace),
        "tools_called": [t["tool"] for t in trace if "tool" in t],
        "search_calls": sum(1 for t in trace if t.get("tool") == "search_corpus"),
        "usage": {
            "prompt_tokens": sum(u["prompt_tokens"] for u in usage_log),
            "completion_tokens": sum(u["completion_tokens"] for u in usage_log),
            "cost_usd": round(sum(u["cost_usd"] for u in usage_log), 6),
            "by_step": usage_log,
        },
    }

    if trace_path:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"run_id": run_id, "question": question, "trace": trace},
                    ensure_ascii=False,
                )
                + "\n"
            )

    return result
