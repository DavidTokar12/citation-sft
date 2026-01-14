from __future__ import annotations

import re

from pydantic import BaseModel
from pydantic import model_validator


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

    @model_validator(mode="after")
    def validate_citations(self) -> SFTExample:
        citations = [int(c) for c in re.findall(r"\[(\d+)\]", self.label)]
        num_docs = len(self.docs)
        invalid = [c for c in citations if c < 1 or c > num_docs]
        if invalid:
            raise ValueError(
                f"Invalid citations {invalid} in label. Must be 1-{num_docs}."
            )
        return self
