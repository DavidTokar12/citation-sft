from __future__ import annotations

from pydantic import BaseModel


class InputDoc(BaseModel):
    index: int
    title: str
    text: str


class SFTExample(BaseModel):
    id: str
    question: str
    docs: list[InputDoc]
    label: str
    precision_score: float
    recall_score: float
    augmented: bool = False
    num_inserted: int = 0
