from __future__ import annotations

import json
import os
import random
import re
import tarfile

import requests

from tqdm import tqdm

from citation_sft.settings import settings
from citation_sft.sft_data._alce_models import _ALCEExample
from citation_sft.sft_data._alce_models import load_alce_data
from citation_sft.sft_data._human_eval_models import _HumanEvalExample
from citation_sft.sft_data._human_eval_models import load_human_eval
from citation_sft.sft_data.sft_data import InputDoc
from citation_sft.sft_data.sft_data import SFTExample


random.seed(42)

DATA_DIR = settings.data_dir
MAX_AUGMENT = settings.max_augment


def _download_file(url: str, path: str):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with (
        open(path, "wb") as f,
        tqdm(total=total, unit="B", unit_scale=True, desc=path.split("/")[-1]) as pbar,
    ):
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def download():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(f"{DATA_DIR}/alce"):
        tar_path = f"{DATA_DIR}/ALCE-data.tar"
        _download_file(
            "https://huggingface.co/datasets/princeton-nlp/ALCE-data/resolve/main/ALCE-data.tar",
            tar_path,
        )
        print("Extracting...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(DATA_DIR)
        os.rename(f"{DATA_DIR}/ALCE-data", f"{DATA_DIR}/alce")
        os.remove(tar_path)
    else:
        print("ALCE data already exists, skipping")

    if not os.path.exists(f"{DATA_DIR}/human_eval_citations_completed.json"):
        _download_file(
            "https://raw.githubusercontent.com/princeton-nlp/ALCE/main/human_eval/human_eval_citations_completed.json",
            f"{DATA_DIR}/human_eval_citations_completed.json",
        )
    else:
        print("Human eval data already exists, skipping")


def load_joined():
    alce = load_alce_data(f"{DATA_DIR}/alce/asqa_eval_gtr_top100.json")
    human = load_human_eval(f"{DATA_DIR}/human_eval_citations_completed.json")
    human_by_id = {ex.id: ex for ex in human}
    return [
        (ex, human_by_id[ex.sample_id]) for ex in alce if ex.sample_id in human_by_id
    ]


def filter_perfect(joined: list[tuple[_ALCEExample, _HumanEvalExample]]):
    return [
        (a, h)
        for a, h in joined
        if h.overall_precision_score == 1.0 and h.overall_recall_score == 1.0
    ]


def collect_all_docs(joined: list[tuple[_ALCEExample, _HumanEvalExample]]):
    docs = []
    for alce, _ in joined:
        docs.extend(alce.docs[:5])
    return docs


def get_short_answers(alce: _ALCEExample) -> list[str]:
    answers = []
    for qa in alce.qa_pairs:
        answers.extend([a.lower() for a in qa.short_answers])
    return answers


def find_irrelevant_doc(alce: _ALCEExample, all_docs: list, max_attempts: int = 100):
    """
    Find a document irrelevant to this question.

    Note: Uses substring matching which is shallow.
    For production I would use something like GTR embeddings (gtr-t5-large) as used in the ALCE paper(it's a 2023 paper so something more recent probably exists)
    to compute semantic similarity and filter out related documents.
    """
    short_answers = get_short_answers(alce)

    current_doc_ids = {d.id for d in alce.docs[:5]}

    for _ in range(max_attempts):
        doc = random.choice(all_docs)
        if doc.id in current_doc_ids:
            continue
        if not any(ans in doc.text.lower() for ans in short_answers):
            return doc

    return None


def to_sft_example(alce: _ALCEExample, human: _HumanEvalExample) -> SFTExample:
    return SFTExample(
        id=alce.sample_id,
        question=alce.question,
        docs=[
            InputDoc(index=i + 1, title=d.title, text=d.text)
            for i, d in enumerate(alce.docs[:5])
        ],
        label=human.output,
        precision_score=human.overall_precision_score,
        recall_score=human.overall_recall_score,
        augmented=False,
        num_inserted=0,
    )


def remap_citations(label: str, insert_pos: int, max_citation: int) -> str:
    for i in range(max_citation, insert_pos - 1, -1):
        label = re.sub(rf"\[{i}\]", f"[{i + 1}]", label)
    return label


def augment_example(
    example: SFTExample, all_docs: list, alce: _ALCEExample, num_insert: int
) -> SFTExample | None:
    docs = [InputDoc(index=d.index, title=d.title, text=d.text) for d in example.docs]
    label = example.label
    max_citation = len(docs)  # starts at 5

    for _ in range(num_insert):
        irr_doc = find_irrelevant_doc(alce, all_docs)
        if irr_doc is None:
            return None
        insert_pos = random.randint(1, len(docs) + 1)
        label = remap_citations(label, insert_pos, max_citation)
        max_citation += 1
        new_doc = InputDoc(index=insert_pos, title=irr_doc.title, text=irr_doc.text)
        docs.insert(insert_pos - 1, new_doc)
        for i, d in enumerate(docs):
            docs[i] = InputDoc(index=i + 1, title=d.title, text=d.text)

    return SFTExample(
        id=f"{example.id}_aug{num_insert}",
        question=example.question,
        docs=docs,
        label=label,
        precision_score=example.precision_score,
        recall_score=example.recall_score,
        augmented=True,
        num_inserted=num_insert,
    )


def build_dataset():
    download()
    joined = load_joined()
    print(f"Total joined: {len(joined)}")

    perfect = filter_perfect(joined)
    print(f"Perfect scores: {len(perfect)}")

    all_docs = collect_all_docs(joined)
    print(f"Total docs pool: {len(all_docs)}")

    examples = []
    for alce, human in perfect:
        base = to_sft_example(alce, human)
        examples.append(base)

        for n in range(1, MAX_AUGMENT + 1):
            aug = augment_example(base, all_docs, alce, n)
            if aug:
                examples.append(aug)

    print(f"\nFinal dataset: {len(examples)}")
    print(f"  Base: {sum(1 for e in examples if not e.augmented)}")
    print(f"  Augmented: {sum(1 for e in examples if e.augmented)}")

    return examples


def save_dataset(examples: list[SFTExample], path: str):
    with open(path, "w") as f:
        json.dump([e.model_dump() for e in examples], f, indent=2)


if __name__ == "__main__":
    examples = build_dataset()
    save_dataset(examples, f"{DATA_DIR}/sft_dataset.json")
    print(f"Saved to {DATA_DIR}/sft_dataset.json")
