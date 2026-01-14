from __future__ import annotations

import json

from pydantic import BaseModel
from pydantic import Field


class _QAPair(BaseModel):
    context: str
    question: str
    short_answers: list[str]
    wikipage: str | None = None


class _WikiPage(BaseModel):
    title: str
    url: str | None = None


class _Knowledge(BaseModel):
    content: str | None = None
    wikipage: str | None = None


class _Annotation(BaseModel):
    knowledge: list[_Knowledge] = Field(default_factory=list)
    long_answer: str


class _Document(BaseModel):
    id: str
    title: str
    text: str
    score: float
    summary: str | None = None
    extraction: str | None = None
    answers_found: list[int] | None = None


class _ALCEExample(BaseModel):
    qa_pairs: list[_QAPair]
    wikipages: list[_WikiPage]
    annotations: list[_Annotation]
    sample_id: str
    question: str
    docs: list[_Document]
    answer: str


def load_alce_data(filepath: str) -> list[_ALCEExample]:
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return [_ALCEExample(**item) for item in data]
