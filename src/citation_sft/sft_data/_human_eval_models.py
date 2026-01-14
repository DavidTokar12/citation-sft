from __future__ import annotations

import json

from pydantic import BaseModel


class _Citation(BaseModel):
    title: str
    text: str


class _Sentence(BaseModel):
    text: str
    citations: list[_Citation]


class _HumanEvalExample(BaseModel):
    id: str
    question: str
    output: str
    overall_precision_score: float
    overall_recall_score: float
    sentences: list[_Sentence]


def load_human_eval(filepath: str) -> list[_HumanEvalExample]:
    with open(filepath) as f:
        data = json.load(f)
    examples = {}
    for _, model_examples in data.get("asqa", {}).items():
        for ex_id, ex_data in model_examples.items():
            if ex_id == "overall_results":
                continue
            if ex_id not in examples:
                examples[ex_id] = _HumanEvalExample(**ex_data)
    return list(examples.values())
