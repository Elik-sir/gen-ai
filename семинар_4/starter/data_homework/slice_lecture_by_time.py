"""
Нарезка транскрипта лекции на документы по фиксированному временному окну.

Пример:
    uv run --no-project slice_lecture_by_time.py
    uv run --no-project slice_lecture_by_time.py --minutes 10 --output-dir ..\\data
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def ms_to_mmss(value_ms: int) -> str:
    total_seconds = max(0, value_ms // 1000)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def speaker_name(item: dict[str, Any]) -> str:
    user_info = item.get("speaker", {}).get("userInfo", {})
    first = str(user_info.get("firstname", "")).strip()
    last = str(user_info.get("surname", "")).strip()
    full_name = " ".join(part for part in [first, last] if part)
    if full_name:
        return full_name
    return str(user_info.get("login") or item.get("trackId") or "unknown_speaker")


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description="Режет raw_data.json на .txt документы по временным окнам."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=base_dir / "raw_data.json",
        help="Путь к исходному транскрипту JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir.parent / "data",
        help="Папка для сохранения документов",
    )
    parser.add_argument(
        "--minutes",
        type=int,
        default=10,
        help="Размер окна в минутах",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="lecture_10min",
        help="Префикс имён выходных файлов",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.minutes <= 0:
        raise ValueError("--minutes должен быть больше 0")
    if not args.input.exists():
        raise FileNotFoundError(f"Не найден входной файл: {args.input}")

    raw_items = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(raw_items, list):
        raise ValueError("Ожидался JSON-массив")

    events: list[dict[str, Any]] = []
    for item in raw_items:
        name = speaker_name(item)
        for chunk in item.get("chunks", []):
            text = normalize_spaces(str(chunk.get("text", "")))
            if not text:
                continue

            start_ms = chunk.get("startTimeOffsetInMillis", chunk.get("timeOffsetInMillis"))
            if not isinstance(start_ms, int):
                continue
            end_ms = chunk.get("endTimeOffsetInMillis")
            if not isinstance(end_ms, int):
                end_ms = start_ms

            events.append(
                {
                    "speaker": name,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": text,
                }
            )

    if not events:
        raise RuntimeError("Не найдено ни одного текстового чанка для нарезки")

    events.sort(key=lambda x: x["start_ms"])
    lecture_start = events[0]["start_ms"]
    window_ms = args.minutes * 60 * 1000

    buckets: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        bucket_idx = (event["start_ms"] - lecture_start) // window_ms
        buckets.setdefault(bucket_idx, []).append(event)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    created_files: list[Path] = []
    for bucket_idx in sorted(buckets):
        rows = buckets[bucket_idx]
        chunk_lines = []
        for row in rows:
            ts = ms_to_mmss(row["start_ms"] - lecture_start)
            chunk_lines.append(f"[{ts}] {row['speaker']}: {row['text']}")

        window_start = bucket_idx * window_ms
        window_end = window_start + window_ms
        body = "\n".join(chunk_lines).strip()
        if not body:
            continue

        file_id = bucket_idx + 1
        file_path = args.output_dir / f"{args.prefix}_{file_id:03d}.txt"
        header = (
            f"Окно: {ms_to_mmss(window_start)} - {ms_to_mmss(window_end)}\n"
            f"Длительность окна: {args.minutes} минут\n"
            f"Записей: {len(rows)}\n\n"
        )
        file_path.write_text(header + body + "\n", encoding="utf-8")
        created_files.append(file_path)

    print(f"Готово. Создано файлов: {len(created_files)}")
    print(f"Выходная директория: {args.output_dir.resolve()}")
    if created_files:
        print("Примеры:")
        for path in created_files[:3]:
            print(f"  - {path.name}")


if __name__ == "__main__":
    main()
