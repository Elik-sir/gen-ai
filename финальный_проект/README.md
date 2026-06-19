# Финальный проект — RAG Q&A по статьям Habr

**Трек B:** прикладной конвейер — ответы по корпусу статей про RAG и LLM-агентов.

## Техники курса

| Техника                                     | Модуль             |
| ------------------------------------------- | ------------------ |
| RAG (hybrid BM25 + dense + RRF)             | `rag.py`           |
| Агент с инструментами (ReAct)               | `agent.py`         |
| Структурированный вывод + `field_validator` | `schema.py`        |
| LLM-as-judge                                | `judge.py`         |
| Проверка галлюцинаций (ghost-цитаты)        | `hallucination.py` |

## Быстрый старт

```bash
cd финальный_проект
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy ..\семинар_6\.env .env     # или скопируйте .env.example → .env и впишите токен

# 1. Скачать 10 статей Habr
python scripts/fetch_habr.py

# 2. Индекс + демо-вопрос
python pipeline.py --ingest
python pipeline.py --question "Что такое RAG?"

# 3. Eval (18 кейсов, ~15–20 мин)
python eval.py
```

**Одна команда (после настройки `.env`):**

```bash
python pipeline.py --ingest && python eval.py
```

## Структура

```
финальный_проект/
├── input/
│   ├── corpus/          ← 10 статей Habr (.txt)
│   ├── corpus_meta.json
│   └── gold.json        ← 18 eval-кейсов
├── output/
│   ├── index/           ← ChromaDB + BM25 cache
│   ├── run_answers/     ← ответы pipeline
│   ├── eval_results.json
│   ├── hallucination_report.json
│   └── trace.jsonl
├── pipeline.py          ← главный entrypoint
├── eval.py
└── отчёт.md
```

## Переменные окружения

См. `.env.example`: `LLM_BASE_URL`, `LLM_AUTH_TOKEN`, `LLM_MODEL`.

## Eval

- **Правильность:** hit-rate@5 по `gold_sources` + LLM-as-judge + abstain для out-of-scope
- **Путь:** agent_steps, search_calls, tokens, latency
- **Галлюцинации:** ghost_quotes, fake_chunk_ids в `hallucination_report.json`
