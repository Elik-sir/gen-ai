from __future__ import annotations

import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from llm_client import get_model, make_client
from schema import Application, CITIES

N_APPLICATIONS = 50
MAX_WORKERS = 8
STRATIFIED_CITIES_COUNT = 10
MODEL = get_model()
client = make_client()

SPECIALITIES = [
    "аналитик",
    "менеджер",
    "ui дизайнер",
    "frontend разработчик",
    "backend разработчик",
    "BI-аналитик",
    "схемотехник",
    "девопсер",
]

SPECIALITY_TO_COURSES = {
    "аналитик": ["Python для продвинутых", "ML-инженер", "Python для начинающих"],
    "менеджер": ["1C", "Python для начинающих", "ML-инженер"],
    "ui дизайнер": ["Python для начинающих", "JS"],
    "frontend разработчик": ["JS", "Java", "Python для продвинутых"],
    "backend разработчик": ["Java", "C++", "Python для продвинутых", "ML-инженер"],
    "BI-аналитик": ["Python для продвинутых", "ML-инженер", "1C"],
    "схемотехник": ["C++", "Python для начинающих"],
    "девопсер": ["Python для продвинутых", "Java", "C++"],
}

SYSTEM_PROMPT = """Ты генерируешь синтетические заявки на курсы повышения
квалификации (ДПО) для взрослых специалистов из России.

Верни строго один JSON-объект с полями Application.
Делай данные правдоподобными и внутренне согласованными:
- years_of_experience примерно соответствует age и graduation_year;
- возраст от 22 до 65;
- graduation_year не позже 2024;
- speciality и desired_course должны быть логично связаны;
- город и район выглядят реалистично."""

USER_PROMPT_TEMPLATE = """Сгенерируй одну заявку на ДПО.
Используй этот seed_city как мягкую подсказку для разнообразия: {seed_city}
Требуемая текущая специальность (используй точно это значение): {seed_speciality}
Желаемый курс (используй точно это значение): {seed_course}

Важно: это только подсказка для разнообразия. Город в ответе должен быть одним
из разрешённых в схеме."""


def generate_one(seed_city: str, seed_speciality: str, seed_course: str) -> Application:
    """Один запрос к LLM -> одна валидная заявка."""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        seed_city=seed_city,
        seed_speciality=seed_speciality,
        seed_course=seed_course,
    )
    for _ in range(3):
        app = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_model=Application,
            max_retries=3,
            temperature=0.8,
        )
        if app.speciality == seed_speciality and app.desired_course == seed_course:
            return app
    return app


def to_flat_rows(applications: list[Application]) -> list[dict]:
    """Распаковать address в отдельные колонки city/district для CSV."""
    rows: list[dict] = []
    for app in applications:
        row = app.model_dump()
        addr = row.pop("address", {})
        row["city"] = addr.get("city")
        row["district"] = addr.get("district")
        rows.append(row)
    return rows


def plot_distribution(series: pd.Series, title: str, out_path: str, color: str) -> Counter:
    """Построить bar chart и вернуть счётчик распределения."""
    counts = Counter(series.tolist())
    ordered = pd.Series(counts).sort_values(ascending=False)
    plt.figure(figsize=(10, 4))
    ordered.plot.bar(color=color, edgecolor="white")
    plt.title(title)
    plt.ylabel("Количество")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return counts


def print_quality_checks(
    n_generated: int, city_counts: Counter, speciality_counts: Counter, target_n: int
) -> None:
    """Проверить соответствие критериям ДЗ и вывести предупреждения."""
    print("\nПроверка критериев ДЗ:")
    print(f"- Валидных заявок: {n_generated}/{target_n}")
    if n_generated < target_n:
        print("Меньше 50 валидных заявок.")
    else:
        print("Количество валидных заявок выполнено.")

    top_city, top_city_n = city_counts.most_common(1)[0]
    top_city_pct = top_city_n / n_generated * 100
    print(f"- Топ-город: {top_city} ({top_city_n}/{n_generated}, {top_city_pct:.1f}%)")
    if top_city_pct > 40:
        print("Порог по городам превышен (должно быть <= 40%).")
    else:
        print("Порог по городам выполнен.")

    top_spec, top_spec_n = speciality_counts.most_common(1)[0]
    top_spec_pct = top_spec_n / n_generated * 100
    print(
        f"- Топ-специальность: {top_spec} ({top_spec_n}/{n_generated}, {top_spec_pct:.1f}%)"
    )
    if top_spec_pct > 35:
        print("Порог по специальностям превышен (должно быть <= 35%).")
    else:
        print("Порог по специальностям выполнен.")


def print_stratification_summary(city_counts: Counter, speciality_counts: Counter) -> None:
    """Короткая сводка для раздела выводов (влияние стратификации городов)."""
    city_values = list(city_counts.values())
    spec_values = list(speciality_counts.values())
    if city_values:
        print(
            f"\nСтратификация по городам: min={min(city_values)}, "
            f"max={max(city_values)}, категорий={len(city_values)}."
        )
    if spec_values:
        print(
            "Распределение по специальностям после городской стратификации: "
            f"min={min(spec_values)}, max={max(spec_values)}, категорий={len(spec_values)}."
        )




def build_generation_plan(n_applications: int) -> list[tuple[str, str, str]]:
    """Сформировать стратифицированный план: city + speciality + course."""
    # Для критерия "отлично": квотируем по 10 городам.
    # При 50 заявках получаем ровно по 5 заявок на город.
    stratified_cities = sorted(CITIES)[:STRATIFIED_CITIES_COUNT]
    city_base, city_remainder = divmod(n_applications, len(stratified_cities))
    city_plan: list[str] = []
    for idx, city in enumerate(stratified_cities):
        city_count = city_base + (1 if idx < city_remainder else 0)
        city_plan.extend([city] * city_count)
    random.shuffle(city_plan)

    base, remainder = divmod(n_applications, len(SPECIALITIES))
    speciality_plan: list[str] = []
    for idx, speciality in enumerate(SPECIALITIES):
        count = base + (1 if idx < remainder else 0)
        speciality_plan.extend([speciality] * count)
    random.shuffle(speciality_plan)

    counters: Counter = Counter()
    plan: list[tuple[str, str, str]] = []
    for city, speciality in zip(city_plan, speciality_plan):
        course_options = SPECIALITY_TO_COURSES[speciality]
        course = course_options[counters[speciality] % len(course_options)]
        counters[speciality] += 1
        plan.append((city, speciality, course))
    return plan


def generate_parallel(n_applications: int) -> list[Application]:
    """Сгенерировать заявки параллельно в несколько потоков."""
    generation_plan = build_generation_plan(n_applications)
    applications: list[Application] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(generate_one, city, speciality, course): (city, speciality, course)
            for city, speciality, course in generation_plan
        }
        done = 0
        for future in as_completed(futures):
            app = future.result()
            applications.append(app)
            done += 1
            print(
                f"  [{done:02d}/{n_applications}] "
                f"{app.full_name} | {app.speciality} -> {app.desired_course} | {app.city}"
            )

    return applications


def main():
    print(f"Модель: {MODEL}")
    print(f"Генерируем {N_APPLICATIONS} заявок в {MAX_WORKERS} потоков...")
    applications = generate_parallel(N_APPLICATIONS)

    rows = to_flat_rows(applications)
    df = pd.DataFrame(rows)
    df.to_csv("applications.csv", index=False, encoding="utf-8-sig")

    city_counts = plot_distribution(
        df["city"], "Распределение заявок по городам", "cities.png", "#7AB66E"
    )
    speciality_counts = plot_distribution(
        df["speciality"],
        "Распределение заявок по специальностям",
        "specialities.png",
        "#D97A4A",
    )

    print("\nСохранено:")
    print("  - applications.csv")
    print("  - cities.png")
    print("  - specialities.png")
    print_stratification_summary(city_counts=city_counts, speciality_counts=speciality_counts)
    print_quality_checks(
        n_generated=len(df),
        city_counts=city_counts,
        speciality_counts=speciality_counts,
        target_n=N_APPLICATIONS,
    )


if __name__ == "__main__":
    main()

