"""
Pydantic-схема для персоны.
==========================
Сейчас здесь только комментарии.
(раскомментируй и допиши на семинаре):
"""

# ───── Раунды 2–4: плоская Persona ─────
from typing import Literal
from pydantic import BaseModel, Field, field_validator

CITIES = {
    "Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург",
    "Казань", "Нижний Новгород", "Самара", "Краснодар", "Пушкин", "Павловск","Гатчина"
}


# ───── Раунд 4.5: вложенная Address ─────
class Address(BaseModel):
    city: str
    district: str = Field(min_length=2, max_length=40)

    @field_validator("city")
    @classmethod
    def city_must_be_in_list(cls, v: str) -> str:
        if v not in CITIES:
            raise ValueError(f"Город «{v}» не из утверждённого списка")
        return v



class Application(BaseModel):
    full_name: str
    age:int = Field(ge=22, le=65)
    years_of_experience:int = Field(ge=0, le=40)
    graduation_year: int = Field(ge=1980, le=2024)
    address: Address
    speciality: Literal["аналитик", "менеджер", "ui дизайнер","frontend разработчик","backend разработчик","BI-аналитик", "схемотехник","девопсер"]
    desired_course: Literal["ML-инженер", "JS", "Java", "C++", "Python для продвинутых", "1C", "Python для начинающих"]

    @property
    def city(self) -> str:
        return self.address.city
