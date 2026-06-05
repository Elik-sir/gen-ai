"""
Pydantic-схема ответа RAG.

TODO (Блок 3 семинара): заполните поля RAGAnswer.
Сейчас схема пустая — модель возвращает строку, и это как раз проблема.
"""

from pydantic import BaseModel, Field


class RAGAnswer(BaseModel):
    # TODO: добавьте поля
    #   answer: str — что показать пользователю
    #   quotes: list[str] — точные цитаты из ретрива (min_length=1, max_length=5)
    #   confidence: float — уверенность модели (ge=0, le=1)
    #   sources: list[str] — id-чанков, откуда взяли
    pass
