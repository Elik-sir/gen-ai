from __future__ import annotations

import json
import time
from pathlib import Path

from orchestrator import run_pwc

CASES = [
    ("Q1", "Во сколько раз USD подорожал с 1 января 2022 по сегодня?"),
    (
        "Q5",
        "Как изменился рубль к USD, EUR и CNY с 1 января 2022 по сегодня, "
        "и какая из валют изменилась сильнее всего?",
    ),
]


def _timed_run(query: str, use_parallel: bool) -> tuple[float, dict]:
    t0 = time.perf_counter()
    res = run_pwc(
        query,
        max_iter=3,
        verbose=False,
        use_validator=True,
        use_parallel=use_parallel,
    )
    elapsed = time.perf_counter() - t0
    return elapsed, res


def main() -> None:
    rows = []
    for case_id, query in CASES:
        seq_t, seq_res = _timed_run(query, use_parallel=False)
        par_t, par_res = _timed_run(query, use_parallel=True)
        speedup = (seq_t / par_t) if par_t > 0 else 0.0

        row = {
            "id": case_id,
            "query": query,
            "sequential_sec": round(seq_t, 3),
            "parallel_sec": round(par_t, 3),
            "speedup_x": round(speedup, 3),
            "sequential_ok": bool(seq_res.get("answer")),
            "parallel_ok": bool(par_res.get("answer")),
        }
        rows.append(row)
        print(
            f"{case_id}: sequential={row['sequential_sec']}s  "
            f"parallel={row['parallel_sec']}s  speedup={row['speedup_x']}x"
        )

    out = Path(__file__).parent / "benchmark_parallel_results.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
